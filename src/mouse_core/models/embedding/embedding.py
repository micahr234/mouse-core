"""StepEmbedder — converts a batch of enriched step records into the per-token
embedding sequence consumed by the sequence backbone.

Each modality declares its own token count via ``tokens`` in its spec
(falling back to the constructor ``token_data_len`` default).  Each step produces
exactly ``tokens_per_step`` embedding vectors.

**Sum mode** (default, ``concat_modalities=False``):

    Modality contributions are summed into a shared block of width equal to the
    *maximum* per-modality token count.  Modalities with fewer tokens affect only the
    leading positions.

        tokens_per_step = max(Tc) + K

**Concat mode** (``concat_modalities=True``):

    Modality blocks are concatenated in declaration order:

        [modA × Ta | modB × Tb | ... | compute × K]

    ``tokens_per_step = sum(Tc) + K``.

**Compute tokens**:

    You can append ``K`` trailing scratch tokens with the top-level
    ``num_compute_tokens=K`` (they are always added after all declared modalities).

    You can also declare learnable blocks anywhere in the sequence by including
    modalities with ``{"name": "...", "embed": "learnable", "tokens": N}``.  Such
    modalities contribute learned scratch tokens at the position they appear in
    the ``modalities`` list; the named modality does **not** need to be present in
    the input rows (its data is ignored).  These are independent of the
    trailing global ``num_compute_tokens``.

After the backbone runs over the full ``[B, S*tokens_per_step, D]`` sequence, the
last token within each step is used to produce one ``[D]``-vector per step.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar

import torch
import torch.nn as nn
from tensordict import TensorDict

from mouse_core.models.embedding.encoding import NormalizedPixel, RandomFourierFeatures
from mouse_core.models.embedding.linear import ScaledEmbedding, ScaledPosLinear


class Encoder(nn.Module, ABC):
    """Abstract base for encoders.

    An encoder is the first stage of a MOUSE model. It converts a ``TensorDict``
    of step records ``[B, S]`` into a flat token embedding sequence
    ``[B, T, D]`` that a backbone can process.

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
    def forward(self, step_stream: TensorDict) -> torch.Tensor:
        """Encode a batch of step records into token embeddings.

        Args:
            step_stream: TensorDict ``[B, S]`` of step records. Required keys
                depend on the concrete encoder configuration.

        Returns:
            ``embeds``: ``[B, T, D]`` with ``T = S * tokens_per_step``.
        """
        ...

    @abstractmethod
    def pool_step_reprs(self, h: torch.Tensor, batch_size: tuple[int, int]) -> torch.Tensor:
        """Extract per-step representations from backbone hidden states.

        After the backbone processes ``[B, T, D]`` tokens, this method maps
        them back to one vector per step ``[B, S, D]``. These are the vectors
        passed to output heads.

        Args:
            h: Backbone output of shape ``[B, T, D]`` where
               ``T = S * tokens_per_step``.
            batch_size: ``(B, S)`` — original batch dimensions.

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
            {"name": "action", "embed": "discrete", "vocab_size": 18, "tokens": 1},
            {"name": "reward", "embed": "rff", "tokens": 1},
            {"name": "scratch", "embed": "learnable", "tokens": 2},  # learned scratch tokens; need not exist in data
            {"name": "obs", "embed": "continuous", "dim": 8, "tokens": 2},
            {"name": "img", "embed": "image", "dim": 7056, "tokens": 16},  # e.g. patches
            {"name": "my_time", "embed": "discrete", "vocab_size": 1000, "absent": -1},
        ]
        enc = StepEmbedder(**{"hidden_dim": 128, "modalities": modalities})  # or pass hidden_dim + modalities directly; hidden_dim is part of the embedding config you feed to StepEmbedder
    """

    name: str
    embed: str
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

    # Valid values for the ``embed`` field in ModalitySpec.
    # These name the embedding *technique*, not the semantic role of the modality.
    _VALID_EMBEDS: ClassVar[tuple[str, ...]] = (
        "discrete",    # integer id → learned table (DiscreteEmbedder)
        "rff",         # scalar → Random Fourier Features (ScalarRFFEmbedder)
        "continuous",  # vector of scalars; ``method="rff"`` (default) or ``"linear"``
        "image",       # pixel/patch values → normalized linear (ImageEmbedder)
        "learnable",   # learned scratch tokens; input data ignored (LearnableEmbedder)
    )
    _METHOD_USING: ClassVar[tuple[str, ...]] = ("continuous",)
    _DISCRETE_LIKE: ClassVar[tuple[str, ...]] = ("discrete",)

    def __post_init__(self) -> None:
        k = (self.embed or "").lower()
        if k not in self._VALID_EMBEDS:
            raise ValueError(
                f"unknown embed kind {self.embed!r} for modality {self.name!r}; "
                f"expected one of {self._VALID_EMBEDS}"
            )
        if self.method not in (None, "rff", "linear"):
            m = str(self.method).lower()
            if m not in ("rff", "linear"):
                raise ValueError(
                    f"unknown method {self.method!r} for modality {self.name!r}; expected 'rff' or 'linear'"
                )
            object.__setattr__(self, "method", m)
        else:
            if self.method is None:
                object.__setattr__(self, "method", "rff")
            elif isinstance(self.method, str):
                object.__setattr__(self, "method", self.method.lower())
        if self.embed != k:
            object.__setattr__(self, "embed", k)

        # If method is a non-default on a kind that doesn't use it, callers will enforce;
        # here we just ensure the value itself is valid (done above).


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

    Used when ``embed="rff"`` (canonical name for scalar RFF). The embedding
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

    This is the technique for modalities declared with ``embed="discrete"``.
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
        positions of that shared block.

    **Concat mode** (``concat_modalities=True``):
        The data tokens for each step are the concatenation of per-modality blocks,
        in the exact order the modalities appear in the ``modalities`` list.
        ``tokens_per_step = sum(Tc) + num_compute_tokens``.

    **Compute tokens** (``num_compute_tokens > 0``):
        ``K`` learned scratch tokens are appended after the data tokens in every
        step block.  The backbone can attend to and write into them as working
        memory.  The step representation is always pooled from the **last** token
        (the last compute token when ``K > 0``).

    Each modality may specify its own ``tokens`` count (see :class:`ModalitySpec`).
    If a modality omits it, the constructor's ``token_data_len`` acts as the default.

    **Sum mode** (``concat_modalities=False``, default):
        Modality contributions are summed into a shared block whose size is the
        *maximum* per-modality token count.  A modality with fewer tokens contributes
        only to the leading positions within that block (remaining positions are
        unaffected by that modality).

    **Concat mode** (``concat_modalities=True``):
        Modality blocks are concatenated in modality order.  A modality with ``Tc``
        tokens occupies exactly ``Tc`` positions.
        ``tokens_per_step = sum(Tc for modalities) + num_compute_tokens``.

    Args:
        hidden_dim: Model hidden dimension ``D``.
        modalities: Declarative list of modality specs. Each may include a ``tokens``
            field to set how many tokens that modality contributes per step.
        token_data_len: Default number of tokens per modality when the modality spec
            does not specify its own ``tokens``.
        num_compute_tokens: Number of learned scratch tokens ``K`` appended after
            the data tokens within each step block.  ``0`` disables compute tokens.
        concat_modalities: When ``True``, modality embeddings are concatenated
            sequentially rather than summed.
        include_type_token: Add the learned type embedding to every token.
            Can be overridden per modality via ``include_type_token`` in the modality spec.
        fourier_min: Smallest input value the RFF resolves.
        fourier_max: Largest input value the RFF covers.
        std: Initialisation std for embedding tables.
    """

    def __init__(
        self,
        hidden_dim: int,
        modalities: list[dict[str, Any] | ModalitySpec] | None = None,
        token_data_len: int = 1,
        num_compute_tokens: int = 0,
        concat_modalities: bool = False,
        include_type_token: bool = True,
        fourier_min: float = 0.01,
        fourier_max: float = 10.0,
        std: float = 0.02,
    ) -> None:
        super().__init__()

        if int(num_compute_tokens) < 0:
            raise ValueError(f"num_compute_tokens must be >= 0, got {num_compute_tokens}.")

        if modalities is None:
            modalities = []

        self.modalities: list[ModalitySpec] = []
        for c in (modalities or []):
            if isinstance(c, dict):
                self.modalities.append(ModalitySpec(**c))
            else:
                self.modalities.append(c)

        # Validation for sized modalities (learnable modalities need no vocab/dim)
        for cs in self.modalities:
            k = cs.embed
            if k == "learnable":
                continue
            if k == "discrete":
                vs = cs.vocab_size or cs.size or 0
                if vs <= 0:
                    raise ValueError(f"modality {cs.name!r} (discrete) requires positive vocab_size")
            if k in {"continuous", "image"}:
                d = cs.dim or cs.size or 0
                if d <= 0:
                    raise ValueError(f"modality {cs.name!r} requires positive dim/size")

        # Applicability checks: error on arguments supplied for the wrong modality kind
        # rather than silently ignoring them.
        for cs in self.modalities:
            k = cs.embed
            is_discrete = k in ModalitySpec._DISCRETE_LIKE
            is_method_using = k in ModalitySpec._METHOD_USING

            if cs.method != "rff" and not is_method_using:
                raise ValueError(
                    f"modality {cs.name!r} (embed={k}) does not support method={cs.method!r}; "
                    f"method only applies to continuous (with linear vs rff choice)"
                )

            if cs.absent is not None and not is_discrete:
                raise ValueError(
                    f"modality {cs.name!r} (embed={k}) does not support 'absent'; "
                    f"'absent' only applies to discrete modalities"
                )

            # in_min/in_max are only meaningful for RFF-using embedders (rff or continuous with rff)
            uses_rff_ranges = (k == "rff") or (is_method_using and cs.method != "linear")
            if (cs.in_min is not None or cs.in_max is not None) and not uses_rff_ranges:
                raise ValueError(
                    f"modality {cs.name!r} (embed={k}) does not use in_min/in_max; "
                    f"those apply to rff or continuous (when not using linear)"
                )

        self._hidden_dim = int(hidden_dim)
        self.include_type_token = bool(include_type_token)
        self.num_compute_tokens = int(num_compute_tokens)
        self.concat_modalities = bool(concat_modalities)
        self.token_data_len = int(token_data_len)  # default for modalities without explicit tokens
        self.fourier_min = float(fourier_min)
        self.fourier_max = float(fourier_max)
        self.std = float(std)

        # Per-modality token counts (fall back to global default)
        self._modality_tokens: dict[str, int] = {}
        self._modality_include_type: dict[str, bool] = {}
        self._learnable_modalities: set[str] = set()
        for cs in self.modalities:
            tc = cs.tokens if (cs.tokens is not None) else self.token_data_len
            if tc <= 0:
                raise ValueError(f"modality {cs.name!r} has non-positive tokens ({tc})")
            self._modality_tokens[cs.name] = int(tc)

            # Per-modality include_type_token overrides the global default
            col_type = cs.include_type_token
            self._modality_include_type[cs.name] = bool(col_type) if col_type is not None else self.include_type_token

            if cs.embed.lower() == "learnable":
                self._learnable_modalities.add(cs.name)

        K = self.num_compute_tokens
        if concat_modalities:
            data_slots = sum(self._modality_tokens.values())
        else:
            data_slots = max(self._modality_tokens.values()) if self._modality_tokens else 0
        self._tokens_per_step: int = data_slots + K

        # Type embedder now produces per-token [D] vectors
        self.type_embedder = TypeEmbedder(hidden_dim=hidden_dim, embedding_std=std)

        self.modality_embedders = nn.ModuleDict()
        self._modality_token_types: dict[str, int] = {}
        tid = 1
        # Assign token-type ids in the exact order of the modalities list.
        for cs in self.modalities:
            Tc = self._modality_tokens[cs.name]
            emb = self._create_embedder_for_modality(cs, hidden_dim, Tc, std)
            self.modality_embedders[cs.name] = emb
            self._modality_token_types[cs.name] = tid
            tid += 1

        if K > 0:
            self.compute_embed = nn.Parameter(torch.empty(K, int(hidden_dim)))
            nn.init.normal_(self.compute_embed, std=std)
        else:
            self.compute_embed = None  # type: ignore[assignment]

    # ------------------------------------------------------------------
    # Encoder interface
    # ------------------------------------------------------------------

    @property
    def hidden_dim(self) -> int:
        return self._hidden_dim

    @property
    def tokens_per_step(self) -> int:
        return self._tokens_per_step

    def pool_step_reprs(self, h: torch.Tensor, batch_size: tuple[int, int]) -> torch.Tensor:
        """Extract step representations: last token of each step block.

        The step representation is always the final token within each step's
        token block (the last compute token when ``num_compute_tokens > 0``,
        otherwise the last data token). This is the vector passed to heads.
        """
        B, S = batch_size
        T = self._tokens_per_step
        D = self._hidden_dim
        return h.view(B, S, T, D)[:, :, -1, :]

    def _create_embedder_for_modality(self, spec: ModalitySpec, hidden_dim: int, T: int, std: float) -> nn.Module:
        k = spec.embed.lower()
        nm = spec.name

        # Per-modality range (for RFF/continuous/image); falls back to the
        # embedder's global fourier_min/fourier_max defaults.
        im = spec.in_min if spec.in_min is not None else self.fourier_min
        ix = spec.in_max if spec.in_max is not None else self.fourier_max

        # Per-modality std overrides the global one passed for this embedder
        mod_std = spec.std if spec.std is not None else std

        if k == "discrete":
            vs = spec.vocab_size or spec.size or 0
            absv = spec.absent
            if absv is None and nm == "time":
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

        raise ValueError(f"unknown embed kind {spec.embed!r} for modality {nm!r}")

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
                if getattr(c, "name", None) == "action":
                    return int(getattr(c, "vocab_size", 0) or getattr(c, "size", 0) or 0)
            return 0
        # Otherwise treat as dict
        d = embedding_kwargs or {}
        if isinstance(d, dict) and "modalities" in d:
            for c in d["modalities"]:
                if isinstance(c, dict) and c.get("name") == "action":
                    return int(c.get("vocab_size") or c.get("size") or 0)
                if getattr(c, "name", None) == "action":
                    return int(getattr(c, "vocab_size", 0) or getattr(c, "size", 0) or 0)
        return 0

    def _default_value_for(self, spec: ModalitySpec, B: int, S: int, device: torch.device) -> torch.Tensor:
        """Synthesize a default (absent) tensor when a declared modality is missing from the batch."""
        k = spec.embed.lower()
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
        step_stream: TensorDict,
    ) -> torch.Tensor:
        """Embed a batch of steps.

        Args:
            step_stream: TensorDict of shape ``[B, S]``.

        Returns:
            embeds: ``[B, S*tokens_per_step, D]``

            Padding information, if any, is not returned as part of the embedding
            stream. If certain positions within the produced sequence should be
            masked (e.g. unused slots for a variable-length modality), supply an
            explicit ``attention_mask`` when calling the model or backbone.
        """
        device = next(self.parameters()).device
        step_stream = step_stream.to(device)

        B, S = int(step_stream.batch_size[0]), int(step_stream.batch_size[1])
        D = self._hidden_dim
        dtype = torch.get_default_dtype()

        # Collect values (or synthesized defaults) for every declared modality.
        # Compute modalities do not read from (or require) any data in the step_stream;
        # we simply omit them from col_values.
        col_values: dict[str, torch.Tensor] = {}
        for spec in self.modalities:
            key = spec.name
            if key in self._learnable_modalities:
                continue
            if key in step_stream.keys():
                col_values[key] = step_stream[key]
            else:
                col_values[key] = self._default_value_for(spec, B, S, device)

        if self.concat_modalities:
            data_embeds = self._forward_concat(B, S, D, device, dtype, col_values)
        else:
            data_embeds = self._forward_sum(B, S, D, device, dtype, col_values)

        # Append the global trailing compute tokens (declared learnable modalities
        # are already included in the data_* blocks above).
        K = self.num_compute_tokens
        if K > 0:
            assert self.compute_embed is not None
            c = self.compute_embed.to(dtype=dtype)
            c = c.view(1, 1, K, D).expand(B, S, K, D)
            embeds = torch.cat([data_embeds, c], dim=2)
        else:
            embeds = data_embeds

        total_T = embeds.shape[2]
        return embeds.reshape(B, S * total_T, D)

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
            Tc = self._modality_tokens[spec.name]
            mod = self.modality_embedders[spec.name]
            if spec.name in self._learnable_modalities:
                # Learnable modalities ignore input data entirely.
                cmod = mod  # narrowed by membership
                assert isinstance(cmod, LearnableEmbedder)
                base = cmod.embed.to(dtype=dtype)  # [Tc, D]
                contrib = base.view(1, 1, Tc, D).expand(B, S, Tc, D)
            else:
                flat = mod(col_values[spec.name]).to(dtype=dtype)  # [B*S, Tc*D]
                contrib = flat.view(B, S, Tc, D)
            if self._modality_include_type[spec.name]:
                typ = self.type_embedder(self._modality_token_types[spec.name], (B, S), device).to(dtype=dtype)
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
            Tc = self._modality_tokens[spec.name]
            mod = self.modality_embedders[spec.name]
            if spec.name in self._learnable_modalities:
                # Learnable modalities ignore any input data.
                cmod = mod  # narrowed by membership
                assert isinstance(cmod, LearnableEmbedder)
                base = cmod.embed.to(dtype=dtype)  # [Tc, D]
                block = base.view(1, 1, Tc, D).expand(B, S, Tc, D)
            else:
                flat = mod(col_values[spec.name]).to(dtype=dtype)  # [B*S, Tc*D]
                block = flat.view(B, S, Tc, D)
            if self._modality_include_type[spec.name]:
                typ = self.type_embedder(self._modality_token_types[spec.name], (B, S), device).to(dtype=dtype)
                block = block + typ.unsqueeze(2)
            parts.append(block)

        return torch.cat(parts, dim=2) if parts else torch.empty(B, S, 0, D, device=device, dtype=dtype)
