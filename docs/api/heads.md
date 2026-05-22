# Heads

Output heads map pooled step representations to logits, Q-values, or action vectors.

| Name | Source |
|------|--------|
| [`DQNHead`](../../src/models/heads/dqn.py) | Q-values with EMA target and Polyak update |
| [`VecDQNHead`](../../src/models/heads/vec_dqn.py) | Per-action vector outputs |
| [`vec_dqn_scores`](../../src/models/heads/vec_dqn.py) | Score candidate actions from vector head outputs |
| [`rope_rotate`](../../src/models/heads/vec_dqn.py) | RoPE rotation helper for vector DQN |
| [`SwiGLUHead`](../../src/models/heads/swiglu.py) | MLP head (policy / SV) |
| [`SwiGLU`](../../src/models/heads/swiglu.py) | SwiGLU activation block |

Base classes: [`BaseHead`](../../src/models/heads/base.py), [`BaseHeadWithTarget`](../../src/models/heads/base.py).

Package exports: [`mouse.models.heads`](../../src/models/heads/__init__.py).
