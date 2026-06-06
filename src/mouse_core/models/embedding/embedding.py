"""StepEmbedder — converts a batch of enriched step records into the per-token
embedding sequence consumed by the sequence backbone.

Each environment transition (step) produces exactly ``tokens_per_step`` embedding
vectors.  Two modes are supported:

**Sum mode** (default, ``concat_modalities=False``):

    Every active modality maps its data to a flat ``T * D`` vector, reshaped into
    ``T`` tokens of dimension ``D`` and **added** to the running sum:

        token[i] = Σ_modality  ( type_embed(modality) + content_embed(modality, i) )

    ``tokens_per_step = T + K`` where ``K = num_compute_tokens``.

**Concat mode** (``concat_modalities=True``):

    Each active modality gets its own dedicated block of ``T`` tokens in the
    sequence, laid out sequentially rather than summed:

        [modality_A × T | modality_B × T | ... | compute × K]

    ``tokens_per_step = M * T + K`` where ``M`` is the number of active modalities
    and ``K = num_compute_tokens``.

**Compute tokens** (``num_compute_tokens > 0``):

    ``K`` learned scratch tokens are appended after the data tokens within every
    step block.  They carry no input data — their embedding is a shared
    ``nn.Parameter`` of shape ``[K, D]`` broadcast across ``(B, S)``.  The
    backbone can use them as working memory.  The step representation is always
    taken from the **last** token (the last compute token when ``K > 0``).

After the backbone runs over the full ``[B, S*tokens_per_step, D]`` sequence, the
last token within each step is used to produce one ``[D]``-vector per step.
"""

from __future__ import annotations

from enum import IntEnum

import torch
import torch.nn as nn
from tensordict import TensorDict

from mouse_core.models.embedding.encoding import NormalizedPixel, RandomFourierFeatures
from mouse_core.models.embedding.linear import ScaledEmbedding, ScaledPosLinear


class TokenType(IntEnum):
    """Token type identifiers used by StepEmbedder when building the embedding sequence."""

    PAD          = 0  # unused / padding
    ACTION       = 1  # int64 action index (discrete)
    REWARD       = 2  # float64 reward (continuous)
    DONE         = 3  # int64 done flag: 0=not done, 1=terminal, 2=truncated
    OBS_IMAGE    = 4  # image obs pixel
    OBS_CONTINUOUS = 5  # continuous vector obs dimension
    TIME         = 6  # int64 episode step index (the `time` field)
    OBS_DISCRETE = 7  # discrete vector obs dimension
    COMPUTE      = 8  # learned scratch token; carries no input data


# ---------------------------------------------------------------------------
# Per-modality content embedders
# ---------------------------------------------------------------------------


class TypeEmbedder(nn.Module):
    """Shared token-type embedding table. Maps a ``TokenType`` → ``[N, T*D]``."""

    def __init__(self, hidden_dim: int, token_data_len: int, embedding_std: float = 0.02) -> None:
        super().__init__()
        self.embed = ScaledEmbedding(num_embeddings=8, embedding_dim=hidden_dim * token_data_len, scale=embedding_std)

    def forward(self, token_type: TokenType, shape: tuple[int, ...], device: torch.device) -> torch.Tensor:
        ids = torch.full(shape, int(token_type), device=device, dtype=torch.long)
        return self.embed(ids)


class ActionEmbedder(nn.Module):
    """Embeds a discrete action id → flat content vector ``[N, T*D]``."""

    def __init__(self, hidden_dim: int, token_data_len: int, max_num_actions: int, embedding_std: float = 0.02) -> None:
        super().__init__()
        self.embed = ScaledEmbedding(
            num_embeddings=max_num_actions, embedding_dim=hidden_dim * token_data_len, scale=embedding_std
        )

    def forward(self, action: torch.Tensor) -> torch.Tensor:
        """Args:
            action: ``[N]`` int64 action indices.
        Returns:
            ``[N, T*D]`` content embedding.
        """
        return self.embed(action)


class TimeEmbedder(nn.Module):
    """Embeds episode step index → flat content vector ``[N, T*D]``.

    Positions with ``time_idx < 0`` are treated as absent and produce a zero vector.
    """

    def __init__(
        self, hidden_dim: int, token_data_len: int, max_num_time_steps: int, embedding_std: float = 0.02
    ) -> None:
        super().__init__()
        self.embed = ScaledEmbedding(
            num_embeddings=max_num_time_steps, embedding_dim=hidden_dim * token_data_len, scale=embedding_std
        )

    def forward(self, time_idx: torch.Tensor) -> torch.Tensor:
        """Args:
            time_idx: ``[N]`` int64; negative values mean the field is absent.
        Returns:
            ``[N, T*D]`` content embedding (zero where time_idx < 0).
        """
        return self.embed(time_idx)


class DoneEmbedder(nn.Module):
    """Embeds a ternary done flag → flat content vector ``[N, T*D]``."""

    def __init__(self, hidden_dim: int, token_data_len: int, embedding_std: float = 0.02) -> None:
        super().__init__()
        self.embed = ScaledEmbedding(num_embeddings=3, embedding_dim=hidden_dim * token_data_len, scale=embedding_std)

    def forward(self, done: torch.Tensor) -> torch.Tensor:
        """Args:
            done: ``[N]`` int64 in {0, 1, 2}.
        Returns:
            ``[N, T*D]`` content embedding.
        """
        return self.embed(done)


class RewardEmbedder(nn.Module):
    """Embeds a scalar reward via Random Fourier Features → flat content vector ``[N, T*D]``."""

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

    def forward(self, reward: torch.Tensor) -> torch.Tensor:
        """Args:
            reward: ``[N]`` float32 scalar rewards.
        Returns:
            ``[N, T*D]`` content embedding.
        """
        return self.rff(reward, 0)


class ObsContinuousEmbedder(nn.Module):
    """Embeds continuous observations → flat content vector ``[N, T*D]``.

    Each obs dimension is projected via a position-indexed RFF; all contributions
    are summed to give the final ``[N, T*D]`` output.
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

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Args:
            obs: ``[*batch, max_num_obs]`` float32 observations.
        Returns:
            ``[*batch, T*D]`` content embedding.
        """
        positions = torch.arange(self.max_num_obs, device=obs.device).expand_as(obs)
        return self.rff(obs.float(), positions).sum(dim=-2)


class ObsContinuousLinearEmbedder(nn.Module):
    """Embeds continuous observations → flat content vector ``[N, T*D]``.

    Each obs dimension is projected via a position-specific learned linear map
    applied directly to the scalar value; all contributions are summed to give
    the final ``[N, T*D]`` output.  Unlike :class:`ObsContinuousEmbedder` this
    uses no random features — the obs value scales a learned direction.

    Args:
        hidden_dim: Model hidden dimension ``D``.
        max_num_obs: Length of the continuous obs vector.
        token_data_len: Number of tokens ``T`` per step.
        input_std: Expected std of the incoming obs values, used to normalise
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
        # Kaiming uniform for in_features=1 has std = 1/√3 (Uniform[-1,1]).
        # ScaledPosLinear multiplies those weights by scale, so per-dim output std =
        # scale × (1/√3) × input_std.  Divide scale by (1/√3) × √max_num_obs to
        # hit embedding_std after summing max_num_obs independent dims.
        _kaiming_std = 3.0 ** -0.5
        self.projs = ScaledPosLinear(
            num_positions=max_num_obs,
            in_features=1,
            out_features=hidden_dim * token_data_len,
            scale=embedding_std / (_kaiming_std * input_std),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Args:
            obs: ``[*batch, max_num_obs]`` float32 observations.
        Returns:
            ``[*batch, T*D]`` content embedding.
        """
        positions = torch.arange(self.max_num_obs, device=obs.device).expand_as(obs)
        return self.projs(obs.float().unsqueeze(-1), positions).sum(dim=-2)


class ObsDiscreteEmbedder(nn.Module):
    """Embeds a scalar discrete state index → flat content vector ``[N, T*D]``.

    The state index is looked up in a learned embedding table of size
    ``max_num_obs`` (the state-space cardinality).
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
        # Summing max_num_obs independent N(0, scale) rows inflates std by √max_num_obs.
        # Note: if obs values are uniform integers in [0, max_num_obs-1], collisions further
        # inflate by ≈√(2·max_num_obs-1)/√max_num_obs; √max_num_obs is a reasonable approximation.
        self.embed = ScaledEmbedding(
            num_embeddings=max_num_obs, embedding_dim=hidden_dim * token_data_len,
            scale=embedding_std / max_num_obs ** 0.5,
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Args:
            obs: ``[*batch]`` int64 discrete state index.
        Returns:
            ``[*batch, T*D]`` content embedding.
        """
        return self.embed(obs)


class ObsImageEmbedder(nn.Module):
    """Embeds image pixels → flat content vector ``[N, T*D]``.

    Each pixel is projected via a position-specific linear map on the normalised
    pixel value; all contributions are summed to give the final ``[N, T*D]`` output.
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
        # pixel_norm_std: std of NormalizedPixel output  ≈ std of Uniform[-1,1] = 1/√3
        # _kaiming_std:   std of Kaiming-uniform weights for in_features=1 = 1/√3
        # per-dim output std = scale × _kaiming_std × pixel_norm_std
        # Divide scale by that product × √max_num_obs to hit embedding_std after summing.
        pixel_norm_std = 3.0 ** -0.5
        _kaiming_std = 3.0 ** -0.5
        self.norm = NormalizedPixel()
        self.projs = ScaledPosLinear(
            num_positions=max_num_obs, in_features=1, out_features=hidden_dim * token_data_len,
            scale=embedding_std / (_kaiming_std * pixel_norm_std * max_num_obs ** 0.5),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Args:
            obs: ``[*batch, max_num_obs]`` int64/float pixel values.
        Returns:
            ``[*batch, T*D]`` content embedding.
        """
        positions = torch.arange(self.max_num_obs, device=obs.device).expand_as(obs)
        normalized = self.norm(obs.float()).unsqueeze(-1)            # [*batch, max_num_obs, 1]
        return self.projs(normalized, positions).sum(dim=-2)


# ---------------------------------------------------------------------------
# StepEmbedder
# ---------------------------------------------------------------------------


class StepEmbedder(nn.Module):
    """Converts a batch of step records ``[B, S]`` into embedding sequences ``[B, S*T, D]``.

    ``tokens_per_step`` is fixed at construction time so that the backbone always
    receives a consistently-shaped input.

    Two embedding modes are available:

    **Sum mode** (``concat_modalities=False``, default):
        Every modality contributes ``token_data_len`` tokens; contributions are
        **summed** at each position so the output is always exactly
        ``token_data_len`` data tokens per step regardless of which modalities
        are active.

    **Concat mode** (``concat_modalities=True``):
        Each active modality occupies its own dedicated block of ``token_data_len``
        tokens, laid out sequentially.
        ``tokens_per_step = num_active_modalities * token_data_len + num_compute_tokens``.

    **Compute tokens** (``num_compute_tokens > 0``):
        ``K`` learned scratch tokens are appended after the data tokens in every
        step block.  The backbone can attend to and write into them as working
        memory.  The step representation is always pooled from the **last** token
        (the last compute token when ``K > 0``).

    Args:
        hidden_dim: Model hidden dimension ``D``.
        max_num_actions: Size of the action embedding table.
        max_num_obs_continuous: Continuous obs vector length; must be > 0 when include_obs_continuous is True.
        max_num_obs_discrete: Discrete obs vector length; must be > 0 when include_obs_discrete is True.
        max_num_obs_image: Total pixel count per image; must be > 0 when include_obs_image is True.
        max_num_time_steps: TIME embedding table size; must be > 0 when include_time_token is True.
        include_action_token: Emit an ACTION token per step.
        include_done_token: Emit a DONE token per step.
        include_reward_token: Emit a REWARD token per step.
        include_obs_continuous: Emit an OBS_CONTINUOUS token per step.
        include_obs_discrete: Emit an OBS_DISCRETE token per step.
        include_obs_image: Emit an OBS_IMAGE token per step.
        include_time_token: Emit a TIME token per step.
        include_type_token: Add the learned type embedding to every token.
        token_data_len: Number of tokens ``T`` produced per modality.
        num_compute_tokens: Number of learned scratch tokens ``K`` appended after
            the data tokens within each step block.  ``0`` disables compute tokens.
        concat_modalities: When ``True``, modality embeddings are concatenated
            sequentially rather than summed.  Each active modality occupies its
            own ``token_data_len`` slots.
        fourier_in_min: Smallest input value the RFF resolves.
        fourier_in_max: Largest input value the RFF covers.
        std: Initialisation std for embedding tables.
    """

    def __init__(
        self,
        hidden_dim: int,
        max_num_actions: int,
        max_num_obs_continuous: int,
        max_num_obs_discrete: int,
        max_num_obs_image: int,
        max_num_time_steps: int,
        include_action_token: bool,
        include_done_token: bool,
        include_reward_token: bool,
        include_obs_continuous: bool,
        include_obs_discrete: bool,
        include_obs_image: bool,
        include_time_token: bool,
        include_type_token: bool,
        token_data_len: int,
        num_compute_tokens: int = 0,
        concat_modalities: bool = False,
        fourier_in_min: float = -1.0,
        fourier_in_max: float = 1.0,
        std: float = 0.02,
    ) -> None:
        super().__init__()

        _size_checks = [
            ("include_action_token", include_action_token, "max_num_actions", max_num_actions),
            ("include_obs_continuous", include_obs_continuous, "max_num_obs_continuous", max_num_obs_continuous),
            ("include_obs_discrete", include_obs_discrete, "max_num_obs_discrete", max_num_obs_discrete),
            ("include_obs_image", include_obs_image, "max_num_obs_image", max_num_obs_image),
            ("include_time_token", include_time_token, "max_num_time_steps", max_num_time_steps),
        ]
        for inc_name, inc_val, size_name, size_val in _size_checks:
            if inc_val and int(size_val) <= 0:
                raise ValueError(f"{inc_name} is True but {size_name} is {size_val} (must be > 0).")

        if int(num_compute_tokens) < 0:
            raise ValueError(f"num_compute_tokens must be >= 0, got {num_compute_tokens}.")

        self.hidden_dim = int(hidden_dim)
        self.include_action_token = bool(include_action_token)
        self.include_time_token = bool(include_time_token)
        self.include_done_token = bool(include_done_token)
        self.include_reward_token = bool(include_reward_token)
        self.include_obs_continuous = bool(include_obs_continuous)
        self.include_obs_discrete = bool(include_obs_discrete)
        self.include_obs_image = bool(include_obs_image)
        self.include_type_token = bool(include_type_token)
        self.num_compute_tokens = int(num_compute_tokens)
        self.concat_modalities = bool(concat_modalities)
        self.token_data_len = int(token_data_len)

        # Count active data modalities (determines tokens_per_step in concat mode).
        self._num_data_modalities: int = sum([
            include_time_token,
            include_action_token,
            include_obs_continuous,
            include_obs_discrete,
            include_obs_image,
            include_reward_token,
            include_done_token,
        ])

        # Compute tokens_per_step.
        T = self.token_data_len
        K = self.num_compute_tokens
        if concat_modalities:
            data_slots = self._num_data_modalities * T
        else:
            data_slots = T
        self.tokens_per_step: int = data_slots + K

        # Shared type embedding (only used internally, not in returned token_types).
        self.type_embedder = TypeEmbedder(hidden_dim=hidden_dim, token_data_len=T, embedding_std=std)

        # Action (optional)
        self.action_embedder = (
            ActionEmbedder(hidden_dim=hidden_dim, token_data_len=T, max_num_actions=int(max_num_actions), embedding_std=std)
            if include_action_token else None
        )

        # Time (optional)
        self.time_embedder = (
            TimeEmbedder(hidden_dim=hidden_dim, token_data_len=T, max_num_time_steps=int(max_num_time_steps), embedding_std=std)
            if include_time_token else None
        )

        # Done (optional)
        self.done_embedder = (
            DoneEmbedder(hidden_dim=hidden_dim, token_data_len=T, embedding_std=std)
            if include_done_token else None
        )

        # Reward (optional)
        self.reward_embedder = (
            RewardEmbedder(hidden_dim=hidden_dim, token_data_len=T, in_min=fourier_in_min, in_max=fourier_in_max, embedding_std=std)
            if include_reward_token else None
        )

        # Continuous obs (optional)
        if include_obs_continuous:
            self.obs_continuous_embedder = ObsContinuousEmbedder(
                hidden_dim=hidden_dim, max_num_obs=int(max_num_obs_continuous), token_data_len=T,
                in_min=fourier_in_min, in_max=fourier_in_max, embedding_std=std,
            )
        else:
            self.obs_continuous_embedder = None

        # Discrete obs (optional)
        self.obs_discrete_embedder = (
            ObsDiscreteEmbedder(
                hidden_dim=hidden_dim, max_num_obs=int(max_num_obs_discrete), token_data_len=T, embedding_std=std
            )
            if include_obs_discrete else None
        )

        # Image obs (optional)
        self.obs_image_embedder = (
            ObsImageEmbedder(
                hidden_dim=hidden_dim, max_num_obs=int(max_num_obs_image), token_data_len=T, embedding_std=std
            )
            if include_obs_image else None
        )

        # Compute tokens — one shared learned embedding per compute slot, broadcast over (B, S).
        if K > 0:
            self.compute_embed = nn.Parameter(torch.empty(K, int(hidden_dim)))
            nn.init.normal_(self.compute_embed, std=std)
        else:
            self.compute_embed = None  # type: ignore[assignment]

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        step_stream: TensorDict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Embed a batch of steps.

        Args:
            step_stream: TensorDict of shape ``[B, S]``.

        Returns:
            embeds:      ``[B, S*tokens_per_step, D]`` — per-position embedding vectors.
            token_types: ``[B, S*tokens_per_step]`` int64 — ``TokenType`` id at each
                         position (used by the backbone to build the attention mask).
                         Data positions carry their modality's ``TokenType``; compute
                         positions carry ``TokenType.COMPUTE``; unused padding positions
                         carry ``TokenType.PAD`` (0).
        """
        device = next(self.parameters()).device
        step_stream = step_stream.to(device)

        B, S = int(step_stream.batch_size[0]), int(step_stream.batch_size[1])
        T, D = self.token_data_len, self.hidden_dim

        # Fetch each active modality from the step stream.
        action   = step_stream["action"]          if self.include_action_token   else None
        reward   = step_stream["reward"]          if self.include_reward_token   else None
        done     = step_stream["done"]            if self.include_done_token     else None
        time_idx = step_stream["time"]            if self.include_time_token     else None
        obs_cont = step_stream["obs_continuous"]  if self.include_obs_continuous else None
        obs_disc = step_stream["obs_discrete"]    if self.include_obs_discrete   else None
        obs_img  = step_stream["obs_image"]       if self.include_obs_image      else None

        dtype = torch.get_default_dtype()

        if self.concat_modalities:
            data_embeds, data_types = self._forward_concat(
                B, S, T, D, device, dtype,
                action, reward, done, time_idx, obs_cont, obs_disc, obs_img,
            )
        else:
            data_embeds, data_types = self._forward_sum(
                B, S, T, D, device, dtype,
                action, reward, done, time_idx, obs_cont, obs_disc, obs_img,
            )

        # Append compute tokens.
        K = self.num_compute_tokens
        if K > 0:
            assert self.compute_embed is not None
            c = self.compute_embed.to(dtype=dtype)          # [K, D]
            c = c.view(1, 1, K, D).expand(B, S, K, D)
            embeds = torch.cat([data_embeds, c], dim=2)     # [B, S, data_slots+K, D]

            c_types = torch.full((B, S, K), int(TokenType.COMPUTE), device=device, dtype=torch.long)
            token_types = torch.cat([data_types, c_types], dim=2)  # [B, S, total]
        else:
            embeds = data_embeds
            token_types = data_types

        total_T = embeds.shape[2]
        return embeds.reshape(B, S * total_T, D), token_types.reshape(B, S * total_T)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _embed_modality(
        self,
        flat: torch.Tensor,
        ttype: TokenType,
        B: int,
        S: int,
        T: int,
        D: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Apply optional type embedding and return ``[B, S, T, D]``."""
        flat = flat.to(dtype=dtype)
        if self.include_type_token:
            flat = flat + self.type_embedder(ttype, (B, S), device).to(dtype=dtype)
        return flat.view(B, S, T, D)

    def _forward_sum(
        self,
        B: int, S: int, T: int, D: int,
        device: torch.device, dtype: torch.dtype,
        action, reward, done, time_idx, obs_cont, obs_disc, obs_img,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sum mode: all modalities accumulated into T slots. Returns ([B,S,T,D], [B,S,T])."""

        total = torch.zeros(B, S, T * D, device=device, dtype=dtype)

        def _add(flat: torch.Tensor, ttype: TokenType) -> None:
            flat = flat.to(dtype=dtype)
            if self.include_type_token:
                total.add_(flat + self.type_embedder(ttype, (B, S), device).to(dtype=dtype))
            else:
                total.add_(flat)

        if self.include_time_token:
            assert self.time_embedder is not None
            _add(self.time_embedder(time_idx), TokenType.TIME)
        if self.include_action_token:
            assert self.action_embedder is not None
            _add(self.action_embedder(action), TokenType.ACTION)
        if self.include_obs_continuous:
            assert self.obs_continuous_embedder is not None
            _add(self.obs_continuous_embedder(obs_cont), TokenType.OBS_CONTINUOUS)
        if self.include_obs_discrete:
            assert self.obs_discrete_embedder is not None
            _add(self.obs_discrete_embedder(obs_disc), TokenType.OBS_DISCRETE)
        if self.include_obs_image:
            assert self.obs_image_embedder is not None
            _add(self.obs_image_embedder(obs_img), TokenType.OBS_IMAGE)
        if self.include_reward_token:
            assert self.reward_embedder is not None
            _add(self.reward_embedder(reward), TokenType.REWARD)
        if self.include_done_token:
            assert self.done_embedder is not None
            _add(self.done_embedder(done), TokenType.DONE)

        data_embeds = total.view(B, S, T, D)
        # All data positions are valid (non-PAD); use ACTION=1 as a generic non-zero label.
        data_types = torch.ones(B, S, T, device=device, dtype=torch.long)
        return data_embeds, data_types

    def _forward_concat(
        self,
        B: int, S: int, T: int, D: int,
        device: torch.device, dtype: torch.dtype,
        action, reward, done, time_idx, obs_cont, obs_disc, obs_img,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Concat mode: each modality gets its own T slots. Returns ([B,S,M*T,D], [B,S,M*T])."""

        parts: list[torch.Tensor] = []
        type_parts: list[torch.Tensor] = []

        def _push(flat: torch.Tensor, ttype: TokenType) -> None:
            block = self._embed_modality(flat, ttype, B, S, T, D, device, dtype)  # [B, S, T, D]
            parts.append(block)
            type_parts.append(
                torch.full((B, S, T), int(ttype), device=device, dtype=torch.long)
            )

        # Fixed canonical order (matches sum-mode _add order for consistency).
        if self.include_time_token:
            assert self.time_embedder is not None
            _push(self.time_embedder(time_idx), TokenType.TIME)
        if self.include_action_token:
            assert self.action_embedder is not None
            _push(self.action_embedder(action), TokenType.ACTION)
        if self.include_obs_continuous:
            assert self.obs_continuous_embedder is not None
            _push(self.obs_continuous_embedder(obs_cont), TokenType.OBS_CONTINUOUS)
        if self.include_obs_discrete:
            assert self.obs_discrete_embedder is not None
            _push(self.obs_discrete_embedder(obs_disc), TokenType.OBS_DISCRETE)
        if self.include_obs_image:
            assert self.obs_image_embedder is not None
            _push(self.obs_image_embedder(obs_img), TokenType.OBS_IMAGE)
        if self.include_reward_token:
            assert self.reward_embedder is not None
            _push(self.reward_embedder(reward), TokenType.REWARD)
        if self.include_done_token:
            assert self.done_embedder is not None
            _push(self.done_embedder(done), TokenType.DONE)

        data_embeds = torch.cat(parts, dim=2)       # [B, S, M*T, D]
        data_types  = torch.cat(type_parts, dim=2)  # [B, S, M*T]
        return data_embeds, data_types
