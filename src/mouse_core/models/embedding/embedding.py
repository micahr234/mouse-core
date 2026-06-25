"""StepEmbedder — converts a batch of enriched step records into the per-token
embedding sequence consumed by the sequence backbone.

Each modality declares its own token count via ``tokens`` in its spec
(falling back to the constructor ``token_data_len`` default).  Each step produces
exactly ``tokens_per_step`` embedding vectors.

**Sum mode** (default, ``concat_modalities=False``):

    Modality contributions are summed into a shared block of width equal to the
    *maximum* per-modality token count.  Modalities with fewer tokens affect only the
    leading positions.

        tokens_per_step = max(Tc)

**Concat mode** (``concat_modalities=True``):

    Modality blocks are concatenated in declaration order:

        [modA × Ta | modB × Tb | ...]

    ``tokens_per_step = sum(Tc)``.

**Scratch tokens**:

    Declare a learnable modality anywhere in the ``modalities`` list:

        {"type": "learnable", "tokens": N}

    Such modalities contribute ``N`` learned scratch tokens at the position they
    appear in the list.  They never read from input rows — the embedding is a
    pure learned parameter the backbone can freely write into.  Placing one last
    in the list makes it the step's prediction token (the one the head reads).

After the backbone runs over the full ``[B, S*tokens_per_step, D]`` sequence,
``forward`` emits a ``step_token_indices [B, S]`` tensor identifying the single
flat-sequence position that represents each step.  ``pool_step_reprs`` gathers
those positions from the backbone output to produce one ``[D]``-vector per step.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any, ClassVar

import numpy as np
import torch
import torch.nn as nn
from tensordict import TensorDict

from mouse_core.models.embedding.encoding import NormalizedPixel, RandomFourierFeatures
from mouse_core.models.embedding.linear import ScaledEmbedding, ScaledPosLinear


# ---------------------------------------------------------------------------
# Modality config helpers  (dtype → type)
# ---------------------------------------------------------------------------


def action_modality(dtype: torch.dtype, *, vocab_size: int | None = None, **kwargs: Any) -> dict:
    """Return a modality config dict for the flat ``"action"`` key.

    ``dtype`` comes from ``env.input_spec.action.dtype``:

    * ``torch.int64``   → discrete action  (``"type": "discrete"``; requires *vocab_size*)
    * ``torch.float32`` → continuous action (``"type": "continuous"``)

    Extra keyword arguments are merged into the returned dict and forwarded to
    :class:`ModalitySpec` (e.g. ``tokens``, ``std``, ``in_min``, ``in_max``).
    """
    if dtype == torch.int64:
        if vocab_size is None:
            raise ValueError(
                "vocab_size is required for discrete (int64) actions; "
                "pass vocab_size=env.action_dim or the known action-space size."
            )
        return {"field": "action", "type": "discrete", "vocab_size": vocab_size, **kwargs}
    return {"field": "action", "type": "continuous", **kwargs}


def observation_modalities(
    obs_spec: Any,
    *,
    vocab_sizes: int | dict[str, int] | None = None,
    **shared_kwargs: Any,
) -> list[dict]:
    """Return a list of modality config dicts for the observation field(s).

    ``obs_spec`` comes from ``env.output_spec.observation``:

    * A single ``FieldSpec`` (non-Dict obs space) → produces one modality named
      ``"observation"``.
    * A ``dict[str, FieldSpec]`` (``gym.spaces.Dict`` obs space) → the subspace
      keys land directly on each output dict, so one modality is produced per key.

    ``dtype`` drives the modality type:

    * ``torch.int64``             → ``"type": "discrete"`` (requires *vocab_sizes*)
    * ``torch.float32``, 1-D     → ``"type": "continuous"``
    * ``torch.float32``, 2-D/3-D → ``"type": "image"``

    ``vocab_sizes`` is used for discrete sub-spaces:

    * Single obs: pass an ``int``.
    * Dict obs:   pass a ``dict[key → int]``.

    Extra keyword arguments are forwarded to every modality dict (e.g. ``tokens``,
    ``std``).  Override per-key values after calling this function if needed.
    """
    if isinstance(obs_spec, dict):
        result: list[dict] = []
        for key, spec in obs_spec.items():
            vs = vocab_sizes.get(key) if isinstance(vocab_sizes, dict) else None
            result.append(_obs_modality(key, spec.dtype, getattr(spec, "shape", ()), vocab_size=vs, **shared_kwargs))
        return result
    return [_obs_modality("observation", obs_spec.dtype, getattr(obs_spec, "shape", ()), vocab_size=vocab_sizes if isinstance(vocab_sizes, int) else None, **shared_kwargs)]


def _obs_modality(
    name: str,
    dtype: torch.dtype,
    shape: tuple[int, ...],
    *,
    vocab_size: int | None,
    **kwargs: Any,
) -> dict:
    if dtype == torch.int64:
        if vocab_size is None:
            raise ValueError(
                f"vocab_size is required for discrete (int64) observation {name!r}; "
                "pass vocab_sizes=<int> (or a dict for Dict obs spaces)."
            )
        return {"field": name, "type": "discrete", "vocab_size": vocab_size, **kwargs}
    if len(shape) >= 2:
        return {"field": name, "type": "image", "dim": int(np.prod(shape)), **kwargs}
    return {"field": name, "type": "continuous", **kwargs}


class Encoder(nn.Module, ABC):
    """Abstract base for encoders.

    An encoder is the first stage of a MOUSE model. It converts a batch of
    raw step records (a ``list[list[dict]]`` of shape ``[B][S]``) into a flat
    token embedding sequence ``[B, T, D]`` that a backbone can process, and
    simultaneously exposes the per-modality tensors it extracted so that
    objectives and heads can use them.

    The encoder also defines how to extract per-step representations
    (the vectors fed to heads for action output) from the backbone's
    token-level hidden states via :meth:`pool_step_reprs`.

    Subclasses must implement :meth:`forward` and :meth:`pool_step_reprs`,
    and expose :attr:`hidden_dim` and :attr:`tokens_per_step`.
    """

    @property
    @abstractmethod
    def hidden_dim(self) -> int:
        """Hidden dimension ``D`` of the produced embeddings."""
        ...

    @property
    @abstractmethod
    def tokens_per_step(self) -> int:
        """Number of tokens produced per step record.

        The backbone receives a sequence of length ``S * tokens_per_step``.
        """
        ...

    @abstractmethod
    def forward(
        self, batch: list[list[dict]]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
        """Encode a batch of raw step records into token embeddings.

        Args:
            batch: ``[B][S]`` list of row dicts. Each dict contains whatever
                fields the data source produced.  Required modality keys depend
                on the concrete encoder configuration.

        Returns:
            ``(embeds, col_values, step_token_indices)`` where

            * ``embeds`` is ``[B, T, D]`` with ``T = S * tokens_per_step``.
            * ``col_values`` maps each non-learnable modality name to its
              extracted tensor (shape ``[B, S]`` for scalars, ``[B, S, D]``
              for vectors). These are the same tensors used for embedding;
              :class:`~mouse_core.models.base.Model` wraps them into the
              ``objective_data`` TensorDict returned alongside the head outputs.
            * ``step_token_indices`` is a ``[B, S]`` int64 tensor.  Entry
              ``[b, s]`` is the absolute position within the flat ``T``-length
              sequence of the single token that represents step ``s`` for batch
              element ``b``.  :meth:`pool_step_reprs` gathers these positions
              from the backbone output to produce one ``[D]``-vector per step.
        """
        ...

    @abstractmethod
    def pool_step_reprs(self, h: torch.Tensor, step_token_indices: torch.Tensor) -> torch.Tensor:
        """Extract per-step representations from backbone hidden states.

        After the backbone processes ``[B, T, D]`` tokens, gathers the single
        token identified by ``step_token_indices`` for each step to produce one
        ``[D]``-vector per step.

        Args:
            h: Backbone output of shape ``[B, T, D]``.
            step_token_indices: ``[B, S]`` int64 tensor of absolute positions
                within the flat sequence, as returned by :meth:`forward`.

        Returns:
            Step representations ``[B, S, D]``.
        """
        ...


@dataclass
class ModalitySpec:
    """Specification for how to embed one modality from the input rows.

    The encoder (StepEmbedder) builds the necessary sub-modules based on this
    list (passed as the ``modalities`` argument) so that each declared modality
    (action, reward, observations, scratch/compute, etc.) is turned into one or
    more token embeddings.

    Each modality can declare its own ``tokens`` (number of tokens it contributes
    per step). If omitted, the constructor's ``token_data_len`` default is used.
    This allows different modalities to map to different numbers of tokens.
    For modalities whose token count can vary at runtime (e.g. images of varying
    size), declare the maximum with ``tokens`` and unused slots can be marked
    as PAD within that modality's block.

    ``include_type_token`` can be set per modality to control whether the learned
    type embedding is added for that modality's tokens (defaults to the global
    ``include_type_token`` value if omitted from the spec).

    Example::

        modalities = [
            {"field": "action", "type": "discrete", "vocab_size": 18, "tokens": 1},
            {"field": "reward", "type": "rff", "tokens": 1},
            {"type": "learnable", "tokens": 2},  # learned scratch tokens; never reads input data
            {"field": "obs", "type": "continuous", "dim": 8, "tokens": 2},
            {"field": "img", "type": "image", "dim": 7056, "tokens": 16},  # e.g. patches
            {"field": "my_time", "type": "discrete", "vocab_size": 1000, "absent": -1},
        ]
        enc = StepEmbedder(hidden_dim=128, modalities=modalities, include_type_token=False)
    """

    type: str
    field: str | Sequence[str] | None = None
    vocab_size: int | None = None
    dim: int | None = None
    size: int | None = None
    tokens: int | None = None
    in_min: float | None = None
    in_max: float | None = None
    std: float | None = None
    include_type_token: bool | None = None
    method: str = "rff"
    absent: Any = None
    required: bool = True
    allow_none: bool = False

    # Valid values for the ``type`` field in ModalitySpec.
    # These name the embedding *technique*, not the semantic role of the modality.
    _VALID_TYPES: ClassVar[tuple[str, ...]] = (
        "discrete",    # integer id → learned table (DiscreteEmbedder)
        "rff",         # scalar → Random Fourier Features (ScalarRFFEmbedder)
        "continuous",  # vector of scalars; ``method="rff"`` (default) or ``"linear"``
        "image",       # pixel/patch values → normalized linear (ImageEmbedder)
        "learnable",   # learned scratch tokens; input data ignored (LearnableEmbedder)
    )
    _METHOD_USING: ClassVar[tuple[str, ...]] = ("continuous",)
    _DISCRETE_LIKE: ClassVar[tuple[str, ...]] = ("discrete",)

    def __post_init__(self) -> None:
        k = (self.type or "").lower()
        if k not in self._VALID_TYPES:
            raise ValueError(
                f"unknown modality type {self.type!r} for modality {self.field!r}; "
                f"expected one of {self._VALID_TYPES}"
            )
        if self.method not in (None, "rff", "linear"):
            m = str(self.method).lower()
            if m not in ("rff", "linear"):
                raise ValueError(
                    f"unknown method {self.method!r} for modality {self.field!r}; expected 'rff' or 'linear'"
                )
            object.__setattr__(self, "method", m)
        else:
            if self.method is None:
                object.__setattr__(self, "method", "rff")
            elif isinstance(self.method, str):
                object.__setattr__(self, "method", self.method.lower())
        if self.type != k:
            object.__setattr__(self, "type", k)
        if k == "learnable":
            object.__setattr__(self, "required", False)

        # If method is a non-default on a kind that doesn't use it, callers will enforce;
        # here we just ensure the value itself is valid (done above).


def _field_names(field: str | Sequence[str] | None) -> tuple[str, ...]:
    if field is None:
        return ()
    if isinstance(field, str):
        return (field,)
    return tuple(field)


def _expand_modality_spec(spec: ModalitySpec, fallback_field: str) -> list[ModalitySpec]:
    fields = _field_names(spec.field)
    if not fields and spec.type == "learnable":
        return [replace(spec, field=fallback_field)]
    if not fields:
        raise ValueError("input-backed modalities must set a field")
    return [replace(spec, field=field) for field in fields]


# ---------------------------------------------------------------------------
# Per-modality content embedders
# ---------------------------------------------------------------------------


class TypeEmbedder(nn.Module):
    """Shared token-type embedding table.

    Maps a small integer id (from the encoder-assigned token_type tensor)
    to a learned [D] vector that is added to tokens of that id when
    ``include_type_token`` is enabled for the modality (or globally).
    """

    def __init__(self, hidden_dim: int, embedding_std: float = 0.02) -> None:
        super().__init__()
        # Per-token type embedding (added to each token of that type).
        self.embed = ScaledEmbedding(num_embeddings=64, embedding_dim=hidden_dim, scale=embedding_std)

    def forward(self, type_id: int, shape: tuple[int, ...], device: torch.device) -> torch.Tensor:
        ids = torch.full(shape, int(type_id), device=device, dtype=torch.long)
        return self.embed(ids)  # [..., D]


class ScalarRFFEmbedder(nn.Module):
    """Embeds a scalar via Random Fourier Features → flat content vector ``[N, T*D]``.

    Used when ``type="rff"`` (canonical name for scalar RFF). The embedding
    technique is independent of the modality's semantic ``name``.
    """

    def __init__(
        self,
        hidden_dim: int,
        token_data_len: int,
        in_min: float,
        in_max: float,
        embedding_std: float = 0.02,
    ) -> None:
        super().__init__()
        rff_scale = embedding_std / 0.5 ** 0.5
        self.rff = RandomFourierFeatures(
            num_features=hidden_dim * token_data_len, in_min=in_min, in_max=in_max, output_scale=rff_scale
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args:
            x: ``[N]`` float32 scalar values.
        Returns:
            ``[N, T*D]`` content embedding.
        """
        return self.rff(x, 0)


class VectorRFFEmbedder(nn.Module):
    """Embeds a vector of scalars via per-element Random Fourier Features → ``[N, T*D]``.

    Each coordinate is projected with its own frequency set (position-indexed RFF).
    All contributions are combined (summed or arranged) to produce the content embedding.
    This is the technique used for continuous/vector modalities when ``method="rff"``.
    The semantic role (obs, state, etc.) does not affect the embedding code.
    """

    def __init__(
        self,
        hidden_dim: int,
        max_num_obs: int,
        token_data_len: int,
        in_min: float,
        in_max: float,
        embedding_std: float = 0.02,
    ) -> None:
        super().__init__()
        self.max_num_obs = max_num_obs
        # cos has std ≈ 1/√2; sum over max_num_obs dims grows by √max_num_obs → divide both out
        rff_scale = embedding_std / (0.5 ** 0.5 * max_num_obs ** 0.5)
        self.rff = RandomFourierFeatures(
            num_features=hidden_dim * token_data_len, in_min=in_min, in_max=in_max,
            num_freq_sets=max_num_obs, output_scale=rff_scale,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args:
            x: ``[*batch, d]`` float32 vector, ``d`` may be <= ``max_num_obs``.
               Extra capacity is zero-padded internally.
        Returns:
            ``[*batch, T*D]`` content embedding.
        """
        x = x.float()
        d = x.shape[-1]
        m = self.max_num_obs
        if d >= m:
            x_use = x[..., :m]
        else:
            pad = torch.zeros((*x.shape[:-1], m - d), device=x.device, dtype=x.dtype)
            x_use = torch.cat([x, pad], dim=-1)
        positions = torch.arange(m, device=x.device).expand_as(x_use)
        return self.rff(x_use, positions).sum(dim=-2)


class VectorLinearEmbedder(nn.Module):
    """Embeds a vector of scalars via per-element learned linear projections → ``[N, T*D]``.

    Each coordinate is mapped by a position-specific linear layer applied to the scalar;
    contributions are combined to form the content embedding. No random features are used;
    the input value directly scales a learned direction.

    This is the technique for continuous/vector modalities when ``method="linear"``.

    Args:
        hidden_dim: Model hidden dimension ``D``.
        max_num_obs: Length of the input vector.
        token_data_len: Number of tokens ``T`` per step.
        input_std: Expected std of the incoming values, used to normalise
            the linear initialisation.  Defaults to ``1.0``.
        embedding_std: Desired output std of the embedding.  Defaults to ``0.02``.
    """

    def __init__(
        self,
        hidden_dim: int,
        max_num_obs: int,
        token_data_len: int,
        input_std: float = 1.0,
        embedding_std: float = 0.02,
    ) -> None:
        super().__init__()
        self.max_num_obs = max_num_obs
        self._one_token_per_elem = (token_data_len == max_num_obs)
        out_tok = 1 if self._one_token_per_elem else token_data_len
        # When one_token_per_elem, each position maps to its own D (no sum).
        # Otherwise we sum over positions into T tokens.
        _kaiming_std = 3.0 ** -0.5
        scale = embedding_std / (_kaiming_std * input_std)
        if not self._one_token_per_elem:
            scale = scale / (max_num_obs ** 0.5)
        self.projs = ScaledPosLinear(
            num_positions=max_num_obs,
            in_features=1,
            out_features=hidden_dim * out_tok,
            scale=scale,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args:
            x: ``[*batch, d]`` float32 vector, ``d`` may be <= ``max_num_obs``.
               Extra capacity is zero-padded internally.
        Returns:
            ``[*batch, T*D]`` content embedding. If token_data_len == max_num_obs,
            this is arranged as one token per input element (no summing).
        """
        x = x.float()
        d = x.shape[-1]
        m = self.max_num_obs
        if d >= m:
            x_use = x[..., :m]
        else:
            pad = torch.zeros((*x.shape[:-1], m - d), device=x.device, dtype=x.dtype)
            x_use = torch.cat([x, pad], dim=-1)
        positions = torch.arange(m, device=x.device).expand_as(x_use)
        out = self.projs(x_use.unsqueeze(-1), positions)  # [*, m, out_tok * D]
        if self._one_token_per_elem:
            # Each position maps to its own D token. out is [*, m, D]
            flat = out.reshape(*out.shape[:-2], m * (out.shape[-1]))
            return flat
        return out.sum(dim=-2)  # sum over elements into T*D


class ImageEmbedder(nn.Module):
    """Embeds a vector of (normalized) pixel / patch values via per-position linear maps → ``[N, T*D]``.

    Each element is normalized then projected by a position-specific linear layer.
    Contributions are combined (sum or one-per-element) to form the content embedding.
    Used for any image/pixel modality regardless of whether it is called "obs", "pixels", etc.
    """

    def __init__(
        self,
        hidden_dim: int,
        max_num_obs: int,
        token_data_len: int,
        embedding_std: float = 0.02,
    ) -> None:
        super().__init__()
        self.max_num_obs = max_num_obs
        self._one_token_per_elem = (token_data_len == max_num_obs)
        out_tok = 1 if self._one_token_per_elem else token_data_len
        # When one_token_per_elem, each pixel/patch maps to its own token of D.
        pixel_norm_std = 3.0 ** -0.5
        _kaiming_std = 3.0 ** -0.5
        self.norm = NormalizedPixel()
        scale = embedding_std / (_kaiming_std * pixel_norm_std)
        if not self._one_token_per_elem:
            scale = scale / (max_num_obs ** 0.5)
        self.projs = ScaledPosLinear(
            num_positions=max_num_obs, in_features=1, out_features=hidden_dim * out_tok,
            scale=scale,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args:
            x: ``[*batch, d]`` int64/float pixel values, ``d`` may be <= ``max_num_obs``.
               Extra capacity is zero-padded internally.
        Returns:
            ``[*batch, T*D]`` content embedding. If token_data_len == max_num_obs,
            this yields one token per input element (no summing across elements).
        """
        x = x.float()
        d = x.shape[-1]
        m = self.max_num_obs
        if d >= m:
            x_use = x[..., :m]
        else:
            pad = torch.zeros((*x.shape[:-1], m - d), device=x.device, dtype=x.dtype)
            x_use = torch.cat([x, pad], dim=-1)
        positions = torch.arange(m, device=x.device).expand_as(x_use)
        normalized = self.norm(x_use).unsqueeze(-1)            # [*batch, m, 1]
        out = self.projs(normalized, positions)  # [*, m, out_tok * D]
        if self._one_token_per_elem:
            return out.reshape(*out.shape[:-2], m * out.shape[-1])
        return out.sum(dim=-2)


class DiscreteEmbedder(nn.Module):
    """Embeds discrete integer indices via a learned table → ``[N, T*D]``.

    If ``absent_value`` is provided, positions holding that value yield the
    zero vector (they are mapped through embed(0) then zeroed).

    This is the technique for modalities declared with ``type="discrete"``.
    The modality ``name`` does not affect how the indices are embedded.
    """

    def __init__(
        self,
        hidden_dim: int,
        token_data_len: int,
        vocab_size: int,
        absent_value: int | None = None,
        embedding_std: float = 0.02,
    ) -> None:
        super().__init__()
        if vocab_size <= 0:
            raise ValueError("vocab_size must be > 0 for discrete embedder")
        self.absent_value = absent_value
        self.embed = ScaledEmbedding(
            num_embeddings=vocab_size,
            embedding_dim=hidden_dim * token_data_len,
            scale=embedding_std,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.long()
        if self.absent_value is not None:
            mask = x == self.absent_value
            safe = x.masked_fill(mask, 0)
            out = self.embed(safe)
            out = torch.where(mask.unsqueeze(-1), torch.zeros_like(out), out)
            return out
        return self.embed(x)


class LearnableEmbedder(nn.Module):
    """Learned scratch tokens for a declared learnable modality.

    The input value for this modality (if any) is ignored. The tokens are
    pure learned parameters placed at the position of the modality in the
    ``modalities`` declaration order. Technique is independent of the
    modality's ``name``.
    """

    def __init__(self, num_tokens: int, hidden_dim: int, std: float = 0.02):
        super().__init__()
        self.num_tokens = int(num_tokens)
        self.embed = nn.Parameter(torch.empty(self.num_tokens, hidden_dim))
        nn.init.normal_(self.embed, std=std)

    def forward(self, x: torch.Tensor | None = None) -> torch.Tensor:
        # Return base [Tc, D]. Expansion to [B, S, Tc, D] is done by caller.
        return self.embed


# ---------------------------------------------------------------------------
# StepEmbedder
# ---------------------------------------------------------------------------


class StepEmbedder(Encoder):
    """Converts a batch of step records ``[B, S]`` into embedding sequences ``[B, S*T, D]``.

    ``tokens_per_step`` is fixed at construction time so that the backbone always
    receives a consistently-shaped input.

    Modalities declare their token counts individually (see :class:`ModalitySpec`).
    The global ``token_data_len`` is used only as a default for modalities that omit
    an explicit ``tokens`` value.

    Two embedding modes are available:

    **Sum mode** (``concat_modalities=False``, default):
        Contributions are summed into a block whose size is the *max* per-modality
        token count.  A modality with ``Tc`` tokens writes to the first ``Tc``
        positions of that shared block.  ``tokens_per_step = max(Tc)``.

    **Concat mode** (``concat_modalities=True``):
        The data tokens for each step are the concatenation of per-modality blocks,
        in the exact order the modalities appear in the ``modalities`` list.
        ``tokens_per_step = sum(Tc)``.

    The step representation passed to heads is always the **last** token of each
    step block.  Place a ``{"type": "learnable", "tokens": 1}``
    modality last in the list to use it as a dedicated prediction token.

    Args:
        hidden_dim: Model hidden dimension ``D``.
        modalities: Declarative list of modality specs. Each may include a ``tokens``
            field to set how many tokens that modality contributes per step. A
            spec's ``field`` may also be a tuple/list of field names; in that case
            each field is encoded as its own modality with the same settings.
        token_data_len: Default number of tokens per modality when the modality spec
            does not specify its own ``tokens``.
        concat_modalities: When ``True``, modality embeddings are concatenated
            sequentially rather than summed.
        include_type_token: Add the learned type embedding to every token.
            Set to ``False`` to disable type embeddings entirely (recommended
            when all modality content stds are already balanced).
            Can be overridden per modality via ``include_type_token`` in the modality spec.
        fourier_min: Smallest input value the RFF resolves.
        fourier_max: Largest input value the RFF covers.
        std: Initialisation std for content embedding tables.
        type_embedding_std: Initialisation std for the shared type embedding table.
            **Required when** ``include_type_token=True``; raises ``ValueError``
            if omitted.  Set to the same value as ``std`` to keep type and
            content signals balanced, or to a smaller value to reduce type
            influence on the summed token.  Ignored when
            ``include_type_token=False``.
    """

    def __init__(
        self,
        hidden_dim: int,
        modalities: list[dict[str, Any] | ModalitySpec] | None = None,
        token_data_len: int = 1,
        concat_modalities: bool = False,
        include_type_token: bool = True,
        fourier_min: float = 0.01,
        fourier_max: float = 10.0,
        std: float = 0.02,
        type_embedding_std: float | None = None,
    ) -> None:
        super().__init__()

        if modalities is None:
            modalities = []

        self.modalities: list[ModalitySpec] = []
        for i, c in enumerate(modalities or []):
            if isinstance(c, dict):
                spec = ModalitySpec(**c)
            else:
                spec = c
            self.modalities.extend(_expand_modality_spec(spec, fallback_field=f"__learnable_{i}"))

        # Validation for sized modalities (learnable modalities need no vocab/dim)
        for cs in self.modalities:
            k = cs.type
            if k == "learnable":
                continue
            if k == "discrete":
                vs = cs.vocab_size or cs.size or 0
                if vs <= 0:
                    raise ValueError(f"modality {cs.field!r} (discrete) requires positive vocab_size")
            if k in {"continuous", "image"}:
                d = cs.dim or cs.size or 0
                if d <= 0:
                    raise ValueError(f"modality {cs.field!r} requires positive dim/size")

        # Applicability checks: error on arguments supplied for the wrong modality kind
        # rather than silently ignoring them.
        for cs in self.modalities:
            k = cs.type
            is_discrete = k in ModalitySpec._DISCRETE_LIKE
            is_method_using = k in ModalitySpec._METHOD_USING

            if cs.method != "rff" and not is_method_using:
                raise ValueError(
                    f"modality {cs.field!r} (type={k}) does not support method={cs.method!r}; "
                    f"method only applies to continuous (with linear vs rff choice)"
                )

            if cs.absent is not None and not is_discrete:
                raise ValueError(
                    f"modality {cs.field!r} (type={k}) does not support 'absent'; "
                    f"'absent' only applies to discrete modalities"
                )
            # in_min/in_max are only meaningful for RFF-using embedders (rff or continuous with rff)
            uses_rff_ranges = (k == "rff") or (is_method_using and cs.method != "linear")
            if (cs.in_min is not None or cs.in_max is not None) and not uses_rff_ranges:
                raise ValueError(
                    f"modality {cs.field!r} (type={k}) does not use in_min/in_max; "
                    f"those apply to rff or continuous (when not using linear)"
                )

        self._hidden_dim = int(hidden_dim)
        self.include_type_token = bool(include_type_token)
        self.concat_modalities = bool(concat_modalities)
        self.token_data_len = int(token_data_len)  # default for modalities without explicit tokens
        self.fourier_min = float(fourier_min)
        self.fourier_max = float(fourier_max)
        self.std = float(std)

        if self.include_type_token and type_embedding_std is None:
            raise ValueError(
                "type_embedding_std is required when include_type_token=True. "
                "Set it explicitly to control the type-to-content signal ratio "
                "(e.g. type_embedding_std=0.02 to match content std, or a smaller "
                "value to reduce type influence). Use include_type_token=False to "
                "disable type embeddings entirely."
            )
        self.type_embedding_std: float = float(type_embedding_std) if type_embedding_std is not None else 0.0

        # Per-modality token counts (fall back to global default)
        self._modality_tokens: dict[str, int] = {}
        self._modality_include_type: dict[str, bool] = {}
        self._learnable_modalities: set[str] = set()
        for cs in self.modalities:
            tc = cs.tokens if (cs.tokens is not None) else self.token_data_len
            if tc <= 0:
                raise ValueError(f"modality {cs.field!r} has non-positive tokens ({tc})")
            assert isinstance(cs.field, str)
            self._modality_tokens[cs.field] = int(tc)

            # Per-modality include_type_token overrides the global default
            col_type = cs.include_type_token
            self._modality_include_type[cs.field] = bool(col_type) if col_type is not None else self.include_type_token

            if cs.type.lower() == "learnable":
                self._learnable_modalities.add(cs.field)

        if concat_modalities:
            self._tokens_per_step: int = sum(self._modality_tokens.values())
        else:
            self._tokens_per_step = max(self._modality_tokens.values()) if self._modality_tokens else 0

        # Type embedder now produces per-token [D] vectors
        self.type_embedder = TypeEmbedder(hidden_dim=hidden_dim, embedding_std=self.type_embedding_std)

        self.modality_embedders = nn.ModuleDict()
        self._modality_token_types: dict[str, int] = {}
        tid = 1
        # Assign token-type ids in the exact order of the modalities list.
        for cs in self.modalities:
            assert isinstance(cs.field, str)
            Tc = self._modality_tokens[cs.field]
            emb = self._create_embedder_for_modality(cs, hidden_dim, Tc, std)
            self.modality_embedders[cs.field] = emb
            self._modality_token_types[cs.field] = tid
            tid += 1

    # ------------------------------------------------------------------
    # Encoder interface
    # ------------------------------------------------------------------

    @property
    def hidden_dim(self) -> int:
        return self._hidden_dim

    @property
    def tokens_per_step(self) -> int:
        return self._tokens_per_step

    def pool_step_reprs(self, h: torch.Tensor, step_token_indices: torch.Tensor) -> torch.Tensor:
        """Gather one representation vector per step from backbone hidden states.

        Args:
            h: Backbone output ``[B, T, D]``.
            step_token_indices: ``[B, S]`` int64 tensor of absolute positions
                within the flat sequence, as returned by :meth:`forward`.

        Returns:
            Step representations ``[B, S, D]``.
        """
        D = self._hidden_dim
        B, S = step_token_indices.shape
        idx = step_token_indices.unsqueeze(-1).expand(B, S, D)
        return h.gather(1, idx)

    def _create_embedder_for_modality(self, spec: ModalitySpec, hidden_dim: int, T: int, std: float) -> nn.Module:
        k = spec.type.lower()
        assert isinstance(spec.field, str)
        field = spec.field

        # Per-modality range (for RFF/continuous/image); falls back to the
        # embedder's global fourier_min/fourier_max defaults.
        im = spec.in_min if spec.in_min is not None else self.fourier_min
        ix = spec.in_max if spec.in_max is not None else self.fourier_max

        # Per-modality std overrides the global one passed for this embedder
        mod_std = spec.std if spec.std is not None else std

        if k == "discrete":
            vs = spec.vocab_size or spec.size or 0
            absv = spec.absent
            if absv is None and field == "time":
                absv = -1
            return DiscreteEmbedder(hidden_dim, T, vs, absent_value=absv, embedding_std=mod_std)

        if k == "rff":
            return ScalarRFFEmbedder(hidden_dim, T, in_min=im, in_max=ix, embedding_std=mod_std)

        if k == "continuous":
            d = spec.dim or spec.size or 0
            if (spec.method or "rff").lower() == "linear":
                return VectorLinearEmbedder(hidden_dim, max_num_obs=d, token_data_len=T, input_std=1.0, embedding_std=mod_std)
            return VectorRFFEmbedder(hidden_dim, max_num_obs=d, token_data_len=T, in_min=im, in_max=ix, embedding_std=mod_std)

        if k == "image":
            p = spec.dim or spec.size or 0
            return ImageEmbedder(hidden_dim, max_num_obs=p, token_data_len=T, embedding_std=mod_std)

        if k == "learnable":
            return LearnableEmbedder(T, hidden_dim, std=mod_std)

        raise ValueError(f"unknown modality type {spec.type!r} for modality {field!r}")

    @staticmethod
    def infer_max_num_actions(embedding_kwargs: dict | object | None) -> int:
        """Extract action cardinality from the modalities list (action modality's vocab_size).

        Accepts:
          - the original embedding_kwargs dict (with "modalities"), or
          - a StepEmbedder instance (uses its .modalities), or
          - None / empty (returns 0).
        """
        if embedding_kwargs is None:
            return 0
        # If it's (or behaves like) a StepEmbedder, use its stored modalities
        mods = getattr(embedding_kwargs, "modalities", None)
        if mods is not None:
            for c in mods:
                if "action" in _field_names(getattr(c, "field", "")):
                    return int(getattr(c, "vocab_size", 0) or getattr(c, "size", 0) or 0)
            return 0
        # Otherwise treat as dict
        d = embedding_kwargs or {}
        if isinstance(d, dict) and "modalities" in d:
            for c in d["modalities"]:
                if isinstance(c, dict) and "action" in _field_names(c.get("field", "")):
                    return int(c.get("vocab_size") or c.get("size") or 0)
                if "action" in _field_names(getattr(c, "field", "")):
                    return int(getattr(c, "vocab_size", 0) or getattr(c, "size", 0) or 0)
        return 0

    def _default_value_for(self, spec: ModalitySpec, B: int, S: int, device: torch.device) -> torch.Tensor:
        """Synthesize a default (absent) tensor when a declared modality is missing from the batch."""
        k = spec.type.lower()
        if k == "learnable":
            # Learnable modalities never read data; return a dummy (should not be used).
            return torch.zeros((B, S), device=device)
        dtype = torch.long if k == "discrete" else torch.get_default_dtype()
        if k == "discrete":
            absv = spec.absent
            if absv is not None:
                return torch.full((B, S), int(absv), device=device, dtype=torch.long)
            return torch.zeros((B, S), device=device, dtype=torch.long)
        if k == "rff":
            return torch.zeros((B, S), device=device, dtype=dtype)
        if k == "continuous":
            d = spec.dim or spec.size or 0
            return torch.zeros((B, S, max(int(d), 0)), device=device, dtype=dtype)
        if k == "image":
            p = spec.dim or spec.size or 0
            return torch.zeros((B, S, max(int(p), 0)), device=device, dtype=dtype)
        # fallback scalar 0
        return torch.zeros((B, S), device=device, dtype=dtype)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        batch: list[list[dict]],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
        """Embed a batch of raw step records.

        Args:
            batch: ``[B][S]`` list of row dicts.  Each dict contains the fields
                produced by the data source (DataLoader or a manual rollout).

        Returns:
            ``(embeds, col_values, step_token_indices)`` where

            * ``embeds`` is ``[B, S*tokens_per_step, D]``.
            * ``col_values`` maps each non-learnable modality name to the tensor
              extracted from the batch (``[B, S]`` for scalars, ``[B, S, D]``
              for vectors), ready to be wrapped into a ``objective_data`` TensorDict
              by :class:`~mouse_core.models.base.Model`.
            * ``step_token_indices`` is ``[B, S]`` int64 — the absolute position
              in the flat sequence of the last token of each step block (the token
              used by heads for per-step prediction).
        """
        device = next(self.parameters()).device
        B = len(batch)
        S = len(batch[0]) if B > 0 else 0
        D = self._hidden_dim
        dtype = torch.get_default_dtype()

        # ------------------------------------------------------------------ #
        # Single-pass extraction: iterate each row once, fill all modality    #
        # buffers simultaneously.  This keeps dict-lookup count at B*S        #
        # instead of num_modalities * B * S (separate passes).                #
        # ------------------------------------------------------------------ #
        non_learnable = [s for s in self.modalities if s.field not in self._learnable_modalities]

        # Pre-allocate one Python list per modality, length B*S.
        raw: dict[str, list] = {str(spec.field): [None] * (B * S) for spec in non_learnable}

        idx = 0
        for b in range(B):
            for s in range(S):
                row = batch[b][s]
                for spec in non_learnable:
                    assert isinstance(spec.field, str)
                    raw[spec.field][idx] = row.get(spec.field)
                idx += 1

        # Convert raw lists to tensors per modality.
        col_values: dict[str, torch.Tensor] = {}
        for spec in non_learnable:
            assert isinstance(spec.field, str)
            values = raw[spec.field]
            all_none = all(v is None for v in values)
            any_none = any(v is None for v in values)

            if all_none:
                if spec.required:
                    raise KeyError(
                        f"Required modality {spec.field!r} is missing from all "
                        f"rows in the batch."
                    )
                col_values[spec.field] = self._default_value_for(spec, B, S, device)
                continue
            if spec.required and any_none:
                missing = sum(v is None for v in values)
                raise KeyError(
                    f"Required modality {spec.field!r} is missing from {missing} "
                    f"of {B * S} rows in the batch."
                )

            k = spec.type.lower()

            if k == "discrete":
                absent_val = int(spec.absent) if spec.absent is not None else 0
                arr = np.array(
                    [int(v) if v is not None else absent_val for v in values],
                    dtype=np.int64,
                )
                col_values[spec.field] = torch.from_numpy(arr).to(device).reshape(B, S)

            elif k == "rff":
                arr = np.array(
                    [float(v) if v is not None else 0.0 for v in values],
                    dtype=np.float32,
                )
                col_values[spec.field] = torch.from_numpy(arr).to(device).reshape(B, S)

            elif k in ("continuous", "image"):
                dim = spec.dim or spec.size or 0
                np_dtype = np.float32 if k == "continuous" else np.int64
                buf = np.zeros((B * S, dim), dtype=np_dtype)
                for i, v in enumerate(values):
                    if v is not None:
                        a = np.asarray(v, dtype=np_dtype).ravel()
                        d = min(a.size, dim)
                        buf[i, :d] = a[:d]
                col_values[spec.field] = (
                    torch.from_numpy(buf).to(device).reshape(B, S, dim)
                )

        # ------------------------------------------------------------------ #
        # Embed extracted tensors → token sequence [B, S, T, D].              #
        # ------------------------------------------------------------------ #
        if self.concat_modalities:
            embeds = self._forward_concat(B, S, D, device, dtype, col_values)
        else:
            embeds = self._forward_sum(B, S, D, device, dtype, col_values)

        T = embeds.shape[2]
        # Last token of each step block is the prediction token for that step.
        step_indices = torch.arange(S, device=device) * T + (T - 1)       # [S]
        step_token_indices = step_indices.unsqueeze(0).expand(B, -1).contiguous()  # [B, S]

        return embeds.reshape(B, S * T, D), col_values, step_token_indices

    # ------------------------------------------------------------------
    # Internal helpers (generalized over modalities)
    # ------------------------------------------------------------------

    def _forward_sum(
        self,
        B: int, S: int, D: int,
        device: torch.device, dtype: torch.dtype,
        col_values: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Sum mode: add contributions of all modalities into a shared block of size max(Tc)."""
        if not self.modalities:
            T = 0
            return torch.empty(B, S, 0, D, device=device, dtype=dtype)

        T = max(self._modality_tokens.values())
        total = torch.zeros(B, S, T, D, device=device, dtype=dtype)

        for spec in self.modalities:
            assert isinstance(spec.field, str)
            Tc = self._modality_tokens[spec.field]
            mod = self.modality_embedders[spec.field]
            if spec.field in self._learnable_modalities:
                # Learnable modalities ignore input data entirely.
                cmod = mod  # narrowed by membership
                assert isinstance(cmod, LearnableEmbedder)
                base = cmod.embed.to(dtype=dtype)  # [Tc, D]
                contrib = base.view(1, 1, Tc, D).expand(B, S, Tc, D)
            else:
                flat = mod(col_values[spec.field]).to(dtype=dtype)  # [B*S, Tc*D]
                contrib = flat.view(B, S, Tc, D)
            if self._modality_include_type[spec.field]:
                typ = self.type_embedder(self._modality_token_types[spec.field], (B, S), device).to(dtype=dtype)
                contrib = contrib + typ.unsqueeze(2)  # broadcast over Tc
            if Tc < T:
                pad = torch.zeros(B, S, T - Tc, D, device=device, dtype=dtype)
                contrib = torch.cat([contrib, pad], dim=2)
            total.add_(contrib)

        return total

    def _forward_concat(
        self,
        B: int, S: int, D: int,
        device: torch.device, dtype: torch.dtype,
        col_values: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Concat mode: concatenate per-modality blocks (each with its own Tc)."""
        parts: list[torch.Tensor] = []

        for spec in self.modalities:
            assert isinstance(spec.field, str)
            Tc = self._modality_tokens[spec.field]
            mod = self.modality_embedders[spec.field]
            if spec.field in self._learnable_modalities:
                # Learnable modalities ignore any input data.
                cmod = mod  # narrowed by membership
                assert isinstance(cmod, LearnableEmbedder)
                base = cmod.embed.to(dtype=dtype)  # [Tc, D]
                block = base.view(1, 1, Tc, D).expand(B, S, Tc, D)
            else:
                flat = mod(col_values[spec.field]).to(dtype=dtype)  # [B*S, Tc*D]
                block = flat.view(B, S, Tc, D)
            if self._modality_include_type[spec.field]:
                typ = self.type_embedder(self._modality_token_types[spec.field], (B, S), device).to(dtype=dtype)
                block = block + typ.unsqueeze(2)
            parts.append(block)

        return torch.cat(parts, dim=2) if parts else torch.empty(B, S, 0, D, device=device, dtype=dtype)
