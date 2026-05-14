from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict

from mouse.models.heads import DQNHead, SwiGLUHead, VecDQNHead, vec_dqn_scores
from mouse.models.embedding import StepEmbedder, TokenType


MODEL_CARD_TEMPLATE = """\
---
{{ card_data }}
---

# MOUSE — Meta-Optimization Using Sequential Experiences

A context-conditioned sequence model for reinforcement learning. It reads a
history of environment transitions and outputs action logits.

## Install (requires private repo access)

```bash
pip install "git+https://github.com/micahr234/MOUSE.git"
```

## Load

```python
from mouse.models.base import load_model

model = load_model("{{ model_id if model_id else 'your-org/your-model' }}")
model.eval()
```

## Step stream

The model takes a `TensorDict[B, S]` — `B` parallel sequences of `S` timesteps each.
This model was trained with **S = {{ sequence_length if sequence_length else 'see config.json' }}**; keep context close to that.

```python
import torch
from tensordict import TensorDict

B, S = 1, 1   # S grows each step when using the cache

step_stream = TensorDict(
    {
{%- if embedding_kwargs is defined %}
{%- if embedding_kwargs.include_action_token %}
        "action":         torch.zeros(B, S, dtype=torch.int64),
{%- endif %}
{%- if embedding_kwargs.include_reward_token %}
        "reward":         torch.zeros(B, S, dtype=torch.float32),
{%- endif %}
{%- if embedding_kwargs.include_done_token %}
        "done":           torch.zeros(B, S, dtype=torch.int64),  # 0=alive 1=terminal 2=truncated
{%- endif %}
{%- if embedding_kwargs.include_time_token %}
        "time":           torch.arange(S).unsqueeze(0).expand(B, S).contiguous(),
{%- endif %}
{%- if embedding_kwargs.include_obs_continuous %}
        "obs_continuous": torch.zeros(B, S, {{ embedding_kwargs.max_num_obs_continuous }}, dtype=torch.float32),
{%- endif %}
{%- if embedding_kwargs.include_obs_discrete %}
        "obs_discrete":   torch.zeros(B, S, dtype=torch.int64),
{%- endif %}
{%- if embedding_kwargs.include_obs_image %}
        "obs_image":      torch.zeros(B, S, {{ embedding_kwargs.max_num_obs_image }}, dtype=torch.int64),
{%- endif %}
{%- else %}
        # see embedding_kwargs in config.json for the required fields
{%- endif %}
    },
    batch_size=(B, S),
)
```

## Inference

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with torch.no_grad():
    out, cache = model(step_stream.to(device))
```

`out` is a `TensorDict[B, S]` with one key per enabled head (`A = model.max_num_actions`, `D = vec_dim`):

| Key | Shape | Description |
|---|---|---|
{%- if sp_head_kwargs is not defined or sp_head_kwargs.num_layers > 0 %}
| `sp` | `[B, S, A]` | Supervised-policy logits |
{%- endif %}
{%- if dqn_head_kwargs is not defined or dqn_head_kwargs.num_layers > 0 %}
| `dqn` | `[B, S, A]` | Q-value logits (online) |
| `dqn_target` | `[B, S, A]` | Q-value logits (target) |
{%- endif %}
{%- if vec_dqn_head_kwargs is not defined or vec_dqn_head_kwargs.num_layers > 0 %}
| `vec_dqn` | `[B, S, A, D]` | Action vectors (online); use `get_action` or `vec_dqn_scores` |
| `vec_dqn_target` | `[B, S, A, D]` | Action vectors (target) |
{%- endif %}
{%- if sv_head_kwargs is not defined or sv_head_kwargs.num_layers > 0 %}
| `sv` | `[B, S, A]` | Q-star regression logits |
{%- endif %}

Select an action from the last timestep (works for all heads, handles temperature):

```python
# greedy (temperature=0) or stochastic (temperature>0)
{%- if vec_dqn_head_kwargs is not defined or vec_dqn_head_kwargs.num_layers > 0 %}
action = model.get_action(out, head="vec_dqn", temperature=0.0)  # [B]
{%- elif dqn_head_kwargs is not defined or dqn_head_kwargs.num_layers > 0 %}
action = model.get_action(out, head="dqn", temperature=0.0)      # [B]
{%- elif sp_head_kwargs is not defined or sp_head_kwargs.num_layers > 0 %}
action = model.get_action(out, head="sp", temperature=0.0)       # [B]
{%- elif sv_head_kwargs is not defined or sv_head_kwargs.num_layers > 0 %}
action = model.get_action(out, head="sv", temperature=0.0)       # [B]
{%- endif %}
```

## Online rollouts with KV-cache

Pass one new step at a time (`S=1`) and carry the `cache` forward to avoid
re-processing the full history on every call:

```python
cache = None

while not done:
    step_stream = TensorDict(
        {
{%- if embedding_kwargs is defined %}
{%- if embedding_kwargs.include_action_token %}
            "action":         last_action.unsqueeze(1),
{%- endif %}
{%- if embedding_kwargs.include_reward_token %}
            "reward":         last_reward.unsqueeze(1),
{%- endif %}
{%- if embedding_kwargs.include_done_token %}
            "done":           last_done.unsqueeze(1),
{%- endif %}
{%- if embedding_kwargs.include_time_token %}
            "time":           torch.full((B, 1), step_idx, dtype=torch.long),
{%- endif %}
{%- if embedding_kwargs.include_obs_continuous %}
            "obs_continuous": obs.unsqueeze(1).float(),  # [B, 1, {{ embedding_kwargs.max_num_obs_continuous }}]
{%- endif %}
{%- if embedding_kwargs.include_obs_discrete %}
            "obs_discrete":   obs_disc.unsqueeze(1),     # [B, 1]
{%- endif %}
{%- if embedding_kwargs.include_obs_image %}
            "obs_image":      obs_img.unsqueeze(1),      # [B, 1, {{ embedding_kwargs.max_num_obs_image }}]
{%- endif %}
{%- else %}
            # see embedding_kwargs in config.json for the required fields
{%- endif %}
        },
        batch_size=(B, 1),
    )

    with torch.no_grad():
        out, cache = model(step_stream.to(device), cache=cache, use_cache=True)

{%- if vec_dqn_head_kwargs is not defined or vec_dqn_head_kwargs.num_layers > 0 %}
    action = model.get_action(out, head="vec_dqn", temperature=0.0)
{%- elif dqn_head_kwargs is not defined or dqn_head_kwargs.num_layers > 0 %}
    action = model.get_action(out, head="dqn", temperature=0.0)
{%- elif sp_head_kwargs is not defined or sp_head_kwargs.num_layers > 0 %}
    action = model.get_action(out, head="sp", temperature=0.0)
{%- else %}
    action = model.get_action(out, head="sv", temperature=0.0)
{%- endif %}
    step_idx += 1
```

> **Cache warning.** This model was trained on sequences of length {{ sequence_length if sequence_length else '`sequence_length`' }}.
> Quality degrades once the cache exceeds roughly **2×** that length — reset it (`cache = None`) before that limit.
"""


def load_model(
    repo_id_or_path: str,
    force_download: bool = False,
    local_dir: str | Path | None = None,
    **kwargs: Any,
) -> "Model":
    """Load a MOUSE model from a local directory or HuggingFace Hub repo.

    Automatically selects the correct model class (ModelLlama, ModelQwen3, or
    ModelNone) by inspecting ``backbone_kwargs`` in ``config.json`` — no need
    to know the class up front.

    Detection logic:
        - ``backbone_kwargs`` is empty  → ModelNone
        - ``backbone_kwargs`` has ``head_dim`` → ModelQwen3
        - ``backbone_kwargs`` is non-empty without ``head_dim`` → ModelLlama

    Args:
        repo_id_or_path: A local path to a checkpoint directory or a HF Hub
            repo id (e.g. ``"your-org/your-model"``).
        force_download: If ``True``, bypass the HF Hub cache and re-download.
            Ignored for local paths.
        local_dir: Directory where Hub files are saved after download.  When
            set, ``hf_hub_download`` writes files there and
            ``from_pretrained`` loads from that directory instead of the
            Hub cache.  Ignored for local paths.
        **kwargs: Forwarded verbatim to ``cls.from_pretrained`` (e.g.
            ``map_location``, ``revision``, ``token``).

    Returns:
        The loaded model instance.
    """
    local = Path(repo_id_or_path)
    if local.exists():
        with (local / "config.json").open() as fh:
            config = json.load(fh)
    else:
        from huggingface_hub import hf_hub_download
        hf_kwargs: dict[str, Any] = {"force_download": force_download}
        if local_dir is not None:
            hf_kwargs["local_dir"] = str(local_dir)
        config_file = hf_hub_download(repo_id=repo_id_or_path, filename="config.json", **hf_kwargs)
        with open(config_file) as fh:
            config = json.load(fh)
        kwargs = {**kwargs, "force_download": force_download}
        if local_dir is not None:
            repo_id_or_path = str(local_dir)

    backbone_kwargs = config.get("backbone_kwargs", {})

    if not backbone_kwargs:
        from mouse.models.none import ModelNone
        return ModelNone.from_pretrained(repo_id_or_path, **kwargs)
    if "head_dim" in backbone_kwargs:
        from mouse.models.qwen3 import ModelQwen3
        return ModelQwen3.from_pretrained(repo_id_or_path, **kwargs)
    from mouse.models.llama import ModelLlama
    return ModelLlama.from_pretrained(repo_id_or_path, **kwargs)


class Model(nn.Module):
    """Base for context-conditioned model: StepEmbedder, backbone, and output heads.

    The forward pass takes a TensorDict ``[B, S]`` of step records.
    ``StepEmbedder`` maps each step to ``tokens_per_step`` embedding vectors,
    producing a flat token sequence ``[B, S * tokens_per_step, D]`` for the
    backbone.  The last token of each step is pooled as the step representation
    and passed to each enabled output head.

    Enabled heads are determined by ``num_layers > 0`` in their kwargs:

    - ``sp``       — SwiGLU MLP → logits ``[B, S, A]``
    - ``dqn``      — DQN twin-head → logits ``[B, S, A]`` (+ ``dqn_target``)
    - ``vec_dqn``  — VecDQN head → vectors ``[B, S, A, D]`` (+ ``vec_dqn_target``);
                     use ``get_action`` or ``vec_dqn_scores`` to get scalar scores
    - ``sv``       — SwiGLU MLP → logits ``[B, S, A]``

    Use ``get_action(out, head, temperature, num_actions)`` to sample or
    greedily select actions from any head without manual score conversion.
    """

    backbone: nn.Module  # Set by subclasses (ModelLlama, ModelQwen3)

    def __init__(
        self,
        hidden_dim: int,
        backbone_kwargs: dict,
        embedding_kwargs: dict,
        sp_head_kwargs: dict,
        dqn_head_kwargs: dict,
        sv_head_kwargs: dict,
        vec_dqn_head_kwargs: dict,
    ):
        super().__init__()

        self.hidden_dim = int(hidden_dim)
        embedding_kwargs = {k: v for k, v in embedding_kwargs.items() if k != "obs_continuous_encoder"}
        self.max_num_actions = int(embedding_kwargs["max_num_actions"])

        self.embedder = StepEmbedder(hidden_dim=hidden_dim, **embedding_kwargs)

        self.sp_head = (
            SwiGLUHead(in_features=hidden_dim, out_features=self.max_num_actions, **sp_head_kwargs)
            if sp_head_kwargs.get("num_layers", 0) > 0 else None
        )

        self.dqn_head = (
            DQNHead(in_features=hidden_dim, out_features=self.max_num_actions, **dqn_head_kwargs)
            if dqn_head_kwargs.get("num_layers", 0) > 0 else None
        )

        self.vec_dqn_head = (
            VecDQNHead(
                in_features=hidden_dim,
                max_num_actions=self.max_num_actions,
                **vec_dqn_head_kwargs,
            )
            if vec_dqn_head_kwargs.get("num_layers", 0) > 0 else None
        )

        self.sv_head = (
            SwiGLUHead(in_features=hidden_dim, out_features=self.max_num_actions, **sv_head_kwargs)
            if sv_head_kwargs.get("num_layers", 0) > 0 else None
        )

        self._init_backbone(backbone_kwargs)

    def _init_backbone(self, backbone_kwargs: dict) -> None:
        """Build and assign self.backbone from the backbone config dict."""
        raise NotImplementedError("Subclasses must implement _init_backbone.")

    def backbone_forward(
        self,
        embeds: torch.Tensor,
        token_type: torch.Tensor,
        cache: dict[str, Any] | None = None,
        use_cache: bool = False,
        cache_position: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, dict[str, Any] | None]:
        """Run backbone; return (hidden states [B, T, D], cache dict or None)."""
        raise NotImplementedError("Subclasses must implement backbone_forward.")

    def head(self, h: torch.Tensor, batch_size: tuple[int, int]) -> TensorDict:
        """Run all enabled heads on pooled step representations.

        Args:
            h: Step representations ``[B, S, D]``.
            batch_size: ``(B, S)`` used to set the TensorDict batch dimensions.

        Returns:
            TensorDict ``[B, S]`` with a key for each enabled head.
            Logit heads (``sp``, ``dqn``, ``dqn_target``, ``sv``) have shape
            ``[B, S, A]``; vector heads (``vec_dqn``, ``vec_dqn_target``) have
            shape ``[B, S, A, D]``.  Disabled heads are absent.
        """
        tensors: dict[str, torch.Tensor] = {}
        if self.sp_head is not None:
            tensors["sp"] = self.sp_head(h)
        if self.dqn_head is not None:
            tensors["dqn"] = self.dqn_head(h)
            tensors["dqn_target"] = self.dqn_head.target_forward(h)
        if self.vec_dqn_head is not None:
            tensors["vec_dqn"] = self.vec_dqn_head(h)
            tensors["vec_dqn_target"] = self.vec_dqn_head.target_forward(h)
        if self.sv_head is not None:
            tensors["sv"] = self.sv_head(h)
        return TensorDict(tensors, batch_size=batch_size)

    def forward(
        self,
        step_stream: TensorDict,
        cache: dict[str, Any] | None = None,
        use_cache: bool = False,
        cache_position: torch.Tensor | None = None,
    ) -> tuple[TensorDict, dict[str, Any] | None]:
        """Run a full forward pass over a batch of step sequences.

        Args:
            step_stream: TensorDict ``[B, S]`` of step records (observations,
                actions, rewards, etc. as configured by the embedder).
            cache: KV-cache dict from a previous call, or ``None`` for a full
                prefill.  Only meaningful when ``use_cache=True``.
            use_cache: If ``True`` the backbone returns an updated cache that
                can be passed back on the next call for incremental decoding.
            cache_position: Token position indices ``[T]`` for incremental
                decoding; leave ``None`` for full prefill.

        Returns:
            out: TensorDict ``[B, S]`` with one key per enabled head.
                 Logit heads — ``sp``, ``dqn``, ``dqn_target``, ``sv`` —
                 have shape ``[B, S, A]``.  Vector heads — ``vec_dqn``,
                 ``vec_dqn_target`` — have shape ``[B, S, A, D]``.
                 Use ``get_action`` or ``vec_dqn_scores`` for the vector heads.
            cache: Updated KV-cache dict, or ``None`` when ``use_cache=False``.
        """
        B, S = int(step_stream.batch_size[0]), int(step_stream.batch_size[1])

        embeds, token_type = self.embedder(step_stream)
        h, new_cache = self.backbone_forward(
            embeds=embeds,
            token_type=token_type,
            cache=cache,
            use_cache=use_cache,
            cache_position=cache_position,
        )

        # Take the last token per step as the step representation
        T = self.embedder.tokens_per_step
        h_step = h.view(B, S, T, self.hidden_dim)[:, :, -1, :]  # [B, S, D]

        return self.head(h_step.float(), batch_size=(B, S)), new_cache

    def polyak_update(self, dqn_tau: float = 0.0, vec_dqn_tau: float = 0.0) -> None:
        """Soft-update all target heads toward their online counterparts."""
        if self.dqn_head is not None:
            self.dqn_head.polyak_update(tau=dqn_tau)
        if self.vec_dqn_head is not None:
            self.vec_dqn_head.polyak_update(tau=vec_dqn_tau)

    def get_action(
        self,
        out: TensorDict,
        head: str,
        temperature: float = 1.0,
        num_actions: int | None = None,
    ) -> torch.Tensor:
        """Select an action from model output at the last step position.

        Args:
            out: Model output TensorDict ``[B, S, ...]``.
            head: Which head to read — ``'sp'``, ``'dqn'``, ``'sv'`` produce
                  logits ``[B, S, A]``; ``'vec_dqn'`` produces vectors
                  ``[B, S, A, D]`` that are converted to scores automatically.
            temperature: Sampling temperature. ``0.0`` → greedy argmax;
                         ``> 0`` → softmax sampling.
            num_actions: If given, trim scores to the first ``num_actions``
                         columns before sampling (useful when the environment
                         has fewer actions than the model's maximum).

        Returns:
            ``[B]`` int64 tensor of selected actions.
        """
        raw = out[head][:, -1]  # [B, A] or [B, A, D] for vec_dqn
        scores: torch.Tensor = vec_dqn_scores(raw) if head == "vec_dqn" else raw  # [B, A]
        if num_actions is not None:
            scores = scores[:, :num_actions]
        if temperature == 0.0:
            return scores.argmax(dim=-1)
        scores = scores - scores.max(dim=-1, keepdim=True).values  # numerical stability
        probs = F.softmax(scores / temperature, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)