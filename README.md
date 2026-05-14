# Meta-Optimization Using Sequential Experiences

![MOUSE](mouse.png)

**MOUSE** is a PyTorch library for in-context meta-reinforcement learning. It reads a history of environment transitions, runs a transformer over the sequence, and outputs actions.

---

## Documentation

📖 Documentation available **[here](https://micahr234.github.io/mouse-core/)**.

---

## Install

```bash
pip install "git+https://github.com/micahr234/mouse-core.git"
```

---

## Example

```python
import torch
from tensordict import TensorDict
from mouse.models.base import load_model

model = load_model("your-org/your-model").eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

B, S = 4, 32
step_stream = TensorDict(
    {
        "action":         torch.zeros(B, S, dtype=torch.int64),
        "reward":         torch.zeros(B, S, dtype=torch.float32),
        "done":           torch.zeros(B, S, dtype=torch.int64),
        "time":           torch.arange(S).unsqueeze(0).expand(B, S).contiguous(),
        "obs_continuous": torch.zeros(B, S, 8, dtype=torch.float32),
    },
    batch_size=(B, S),
)

with torch.no_grad():
    out, _ = model(step_stream.to(device))

action = model.get_action(out, temperature=0.0)  # [B]
```

---

## License

GNU General Public License v3.0 — see [LICENSE](LICENSE).
