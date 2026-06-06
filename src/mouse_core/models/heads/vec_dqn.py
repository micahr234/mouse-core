"""VecDQNHead: vector-valued DQN head with RoPE-based angular scoring."""

from __future__ import annotations

import math
from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from mouse_core.models.heads.base import BaseHeadWithTarget
from mouse_core.models.heads.swiglu import SwiGLUHead


def rope_rotate(x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """Rotate each consecutive pair of dimensions in ``x`` by ``theta``.

    Args:
        x:     ``[..., D]`` where ``D`` is even.
        theta: ``[...]`` — same leading shape as ``x``, one angle per vector.

    Returns:
        Rotated tensor of the same shape as ``x``.
    """
    rotated = torch.empty_like(x)
    rotated[..., ::2]  = x[..., ::2]  * theta.unsqueeze(-1).cos() - x[..., 1::2] * theta.unsqueeze(-1).sin()
    rotated[..., 1::2] = x[..., ::2]  * theta.unsqueeze(-1).sin() + x[..., 1::2] * theta.unsqueeze(-1).cos()
    return rotated


def vec_dqn_scores(vecs: torch.Tensor) -> torch.Tensor:
    """Compute pairwise angular action scores from vec-DQN vectors.

    For each pair of actions ``(i, a)``, computes the full signed angle
    ``φ_a − φ_i`` via ``atan2(sin, cos)``, then sums over all ``i`` to give
    action ``a``'s total angular lead.

    The sin component is ``dot(rot90(vᵢ), vₐ)`` and the cos component is
    ``dot(vᵢ, vₐ)``.  Using both avoids the aliasing of a sin-only score,
    which saturates at ±90° and folds back toward zero toward 180°.  The
    atan2 score is monotone across the full (−π, +π) range — aliasing only
    occurs if two action vectors rotate past ±180° apart, which is twice as
    hard to reach.

    For ``D = 2`` (a single rotation plane) this is geometrically exact.
    For ``D > 2`` (RoPE with multiple planes) it is a well-conditioned
    approximation; the D = 2 case is recommended for exact geometry.

    Self-terms contribute ``atan2(0, 1) = 0`` and require no masking.

    Args:
        vecs: ``[..., A, D]`` — raw (un-normalised) action vectors.

    Returns:
        scores: ``[..., A]`` — summed angular lead per action, in radians.
    """
    leading = vecs.shape[:-2]
    A, D = vecs.shape[-2], vecs.shape[-1]
    vecs_norm = F.normalize(vecs, dim=-1)                              # [..., A, D]
    flat = vecs_norm.reshape(-1, D)
    theta90 = torch.full((flat.shape[0],), math.pi / 2, device=flat.device, dtype=flat.dtype)
    rot90 = rope_rotate(x=flat, theta=theta90).reshape(*leading, A, D)
    sin_ia = torch.einsum("...id,...ad->...ia", rot90, vecs_norm)      # [..., A, A]  sin(φ_a − φ_i)
    cos_ia = torch.einsum("...id,...ad->...ia", vecs_norm, vecs_norm)  # [..., A, A]  cos(φ_a − φ_i)
    return torch.atan2(sin_ia, cos_ia).sum(dim=-2)                     # [..., A]


class VecDQNHead(BaseHeadWithTarget):
    """SwiGLUHead paired with an EMA target copy and Polyak averaging.

    Like ``DQNHead`` but each action produces a ``vec_dim``-dimensional vector
    instead of a single scalar.  Output shape is ``[..., max_num_actions, vec_dim]``.

    ``forward`` runs the online head. ``target_forward`` runs the target head
    (no gradient tracking). Call ``polyak_update(tau)`` after each optimizer
    step to soft-update the target:  θ_target ← τ·θ_online + (1−τ)·θ_target.
    Initialize with ``tau=1.0`` to copy online weights into the target.
    """

    def __init__(
        self,
        in_features: int,
        max_num_actions: int,
        vec_dim: int,
        hidden_dim: int,
        num_layers: int,
        scale: float = 1.0,
        bias_scale: float | None = None,
        use_norm: bool = True,
    ):
        super().__init__()
        self.A = max_num_actions
        self.D = vec_dim
        self.online = SwiGLUHead(
            in_features=in_features,
            out_features=max_num_actions * vec_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            scale=scale,
            use_norm=use_norm,
        )
        if bias_scale is not None:
            out_layer = cast(nn.Linear, self.online.layers[-1])
            if out_layer.bias is not None:
                with torch.no_grad():
                    out_layer.bias.fill_(float(bias_scale))
        self._init_target(self.online)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.online(h).unflatten(-1, (self.A, self.D))

    def target_forward(self, h: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.target(h).unflatten(-1, (self.A, self.D))

