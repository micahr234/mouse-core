from mouse_core.models.heads.base import BaseHead, BaseHeadWithTarget, HeadSpec
from mouse_core.models.heads.swiglu import SwiGLU, SwiGLUHead
from mouse_core.models.heads.layerwise_dqn import LayerwiseDiscreteActionValueHead
from mouse_core.models.heads.dqn import DiscreteActionValueHead
from mouse_core.models.heads.vec_dqn import VectorActionValueHead, vector_action_scores, rope_rotate

__all__ = [
    "BaseHead",
    "BaseHeadWithTarget",
    "HeadSpec",
    "SwiGLU",
    "SwiGLUHead",
    "DiscreteActionValueHead",
    "LayerwiseDiscreteActionValueHead",
    "VectorActionValueHead",
    "vector_action_scores",
    "rope_rotate",
    "build_heads",
]


def build_heads(
    hidden_dim: int, max_num_actions: int, head_kwargs: dict | None
) -> dict[str, BaseHead | None]:
    """Build head instances from a declarative head_kwargs dict.

    New semantic head names (preferred):

        head_kwargs = {
            "heads": [
                {"name": "action_value", "num_layers": 1, "hidden_dim": 32, "scale": 0.01},
                {"name": "action", "num_layers": 0},
            ]
        }

    Supported names:
      - "action_value": DiscreteActionValueHead (per-discrete-action values, has target net)
      - "action_vector": VectorActionValueHead (vector per action, has target net)
      - "action": plain head for direct discrete action logits/policy
      - "value": plain head for value regression

    If ``head_kwargs`` is None or has no ``heads`` list, all heads are disabled.
    A head with ``num_layers`` of 0 (or falsy) is disabled.
    """
    hk = head_kwargs or {}
    specs: list[HeadSpec] = []
    for h in hk.get("heads", []) or []:
        if isinstance(h, dict):
            specs.append(HeadSpec(**h))
        elif isinstance(h, HeadSpec):
            specs.append(h)
        else:
            specs.append(HeadSpec(**dict(h)))

    built: dict[str, BaseHead | None] = {
        "action_value": None,
        "action_value_layerwise": None,
        "action_vector": None,
        "action": None,
        "value": None,
    }

    for spec in specs:
        nm = spec.name
        if nm not in built:
            raise ValueError(
                f"unknown head name {nm!r}; expected one of action_value, action_vector, action, value"
            )
        if not spec.num_layers or int(spec.num_layers) <= 0:
            built[nm] = None
            continue
        hd = spec.hidden_dim if spec.hidden_dim is not None else hidden_dim
        sc = spec.scale if spec.scale is not None else 1.0
        un = spec.use_norm if spec.use_norm is not None else True
        if nm == "action_value_layerwise":
            if spec.num_backbone_layers is None:
                raise ValueError(
                    "action_value_layerwise head requires num_backbone_layers in HeadSpec."
                )
            built[nm] = LayerwiseDiscreteActionValueHead(
                num_backbone_layers=int(spec.num_backbone_layers),
                in_features=hidden_dim,
                out_features=max_num_actions,
                hidden_dim=hd,
                num_layers=int(spec.num_layers),
                scale=sc,
                use_norm=un,
            )
        elif nm == "action_value":
            built[nm] = DiscreteActionValueHead(
                in_features=hidden_dim,
                out_features=max_num_actions,
                hidden_dim=hd,
                num_layers=int(spec.num_layers),
                scale=sc,
                use_norm=un,
            )
        elif nm == "action_vector":
            vd = spec.vec_dim if spec.vec_dim is not None else 2
            bs = spec.bias_scale
            built[nm] = VectorActionValueHead(
                in_features=hidden_dim,
                max_num_actions=max_num_actions,
                vec_dim=int(vd),
                hidden_dim=hd,
                num_layers=int(spec.num_layers),
                scale=sc,
                bias_scale=bs,
                use_norm=un,
            )
        else:
            built[nm] = SwiGLUHead(
                in_features=hidden_dim,
                out_features=max_num_actions,
                hidden_dim=hd,
                num_layers=int(spec.num_layers),
                scale=sc,
                use_norm=un,
            )
    return built
