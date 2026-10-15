"""Discrete Q-values with a fixed table of continuous action vectors."""

from __future__ import annotations

import torch

from mouse_core.models.heads.regression import RegressionHead


class DiscretizedActionHead(RegressionHead):
    """One Q-value per row of a fixed ``[num_actions, action_dim]`` table.

    ``forward`` returns raw Q-values, just like :class:`RegressionHead`.
    ``discrete_to_continuous`` decodes selected IDs; ``continuous_to_discrete``
    quantizes vectors to their nearest row by Euclidean distance in the
    table's original units. Ties choose the lowest ID. For differently scaled
    coordinates, choose suitable units before constructing the table.

    The table is copied into a non-trainable float32 buffer, moves with the
    model, and is included in its state dict and ``save_model`` checkpoint.
    Rows must be finite and distinct after conversion to float32. Keep the
    table and its ordering fixed throughout training, replay, and inference:
    changing them changes the meaning of the Q-values and recorded IDs.

    This table gives the categories physical meaning; it does not make the
    Q-value projections share parameters according to action distance.
    """

    action_vectors: torch.Tensor

    def __init__(
        self,
        *,
        in_features: int,
        action_vectors: torch.Tensor,
        hidden_dim: int,
        num_layers: int,
        propagate_gradient: float,
        scale: float = 1.0,
    ) -> None:
        if not isinstance(action_vectors, torch.Tensor) or not action_vectors.is_floating_point():
            raise TypeError("action_vectors must be a floating-point tensor.")
        if action_vectors.ndim != 2 or min(action_vectors.shape) == 0:
            raise ValueError("action_vectors must have nonempty shape [num_actions, action_dim].")
        # Like the other heads, construct on CPU in float32; Model.to moves it.
        table = action_vectors.detach().to(device="cpu", dtype=torch.float32).clone()
        if not torch.isfinite(table).all():
            raise ValueError("action_vectors must contain only finite float32 values.")
        if torch.unique(table, dim=0).shape[0] != table.shape[0]:
            raise ValueError("action_vectors must contain distinct rows in float32.")
        super().__init__(
            in_features=in_features,
            out_features=table.shape[0],
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            propagate_gradient=propagate_gradient,
            scale=scale,
        )
        self.register_buffer("action_vectors", table)

    @property
    def action_dim(self) -> int:
        """Number of continuous components in each action vector."""
        return self.action_vectors.shape[1]

    @torch.no_grad()
    def discrete_to_continuous(self, action_ids: int | torch.Tensor) -> torch.Tensor:
        """Map integer IDs ``[...]`` to vectors ``[..., action_dim]``.

        Returns a tensor on the head's device, with no gradient. A Python
        integer or scalar tensor returns one vector. Negative/out-of-range
        IDs raise instead of wrapping around the table.
        """
        ids = torch.as_tensor(action_ids, device=self.action_vectors.device)
        if ids.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
            raise TypeError("action_ids must contain integers (not bools or floats).")
        ids = ids.long()
        if ((ids < 0) | (ids >= self.out_features)).any():
            raise ValueError(f"action_ids must be in [0, {self.out_features}).")
        # index_select returns independent storage even for a scalar ID.
        return self.action_vectors.index_select(0, ids.reshape(-1)).reshape(
            *ids.shape, self.action_dim,
        )

    @torch.no_grad()
    def continuous_to_discrete(self, actions: torch.Tensor) -> torch.Tensor:
        """Quantize vectors ``[..., action_dim]`` to int64 IDs ``[...]``.

        Inputs must be finite floating-point tensors. Values outside the
        table's range also select the nearest row; no clipping is performed.
        Distances use float32, with a ``[num_vectors, num_actions]`` temporary
        matrix. Results are on the head's device. This is a lossy quantizer,
        not a differentiable inverse for arbitrary continuous actions.
        """
        if not isinstance(actions, torch.Tensor) or not actions.is_floating_point():
            raise TypeError("actions must be a floating-point tensor.")
        if actions.ndim == 0 or actions.shape[-1] != self.action_dim:
            raise ValueError(f"actions must have shape [..., {self.action_dim}].")
        actions = actions.to(device=self.action_vectors.device, dtype=torch.float32)
        if not torch.isfinite(actions).all():
            raise ValueError("actions must contain only finite float32 values.")
        # Avoid the matrix-multiplication distance identity: cancellation
        # can make distinct, finely spaced vectors appear equally distant.
        distances = torch.cdist(
            actions.reshape(-1, self.action_dim), self.action_vectors,
            compute_mode="donot_use_mm_for_euclid_dist",
        )
        return distances.argmin(dim=-1).reshape(actions.shape[:-1])
