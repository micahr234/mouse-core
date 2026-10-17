# MOUSE Core 🧠

<p align="center"><img src="https://raw.githubusercontent.com/micahr234/mouse-core/main/mouse-core.png" width="400"/></p>

> **Warning:** MOUSE is in early development and is not yet ready for production use. APIs may change without notice.

**mouse-core** is the core library for the Meta-Optimization Using Sequential Experience (MOUSE) learning system — a modular PyTorch stack for <u>in-context reinforcement learning (ICRL)</u>. It provides data utilities, embedding frameworks, transformer backbones, output heads, and objective functions for training and deploying agents that adapt from transition history at inference time, **without weight updates**.

**[mouse-gym](https://github.com/micahr234/mouse-gym)** sits alongside mouse-core and handles the environment side: `EnvConfig` turns any Gymnasium env into a **reset-free** stream. `step(action)` returns `(observation, reward, terminated, truncated, info)`. `terminated` and `truncated` are the wrapped env's flags. The first step, and the step after either flag, is a reset frame. There is no public `reset()`. Repeated inner episodes are implemented by **[repeat-task-gym](https://github.com/micahr234/repeat-task-gym)**: wrap the Gymnasium env in `RepeatTaskEnv`, then pass that to mouse-gym. Mouse-gym `terminated` / `truncated` then mark the outer episode; inner episode ends stay on the observation (`episode_terminated`, `episode_truncated`). The examples use **[procedural-frozenlake](https://github.com/micahr234/procedural-frozenlake)**, a FrozenLake variant with procedurally generated maps and optimal-Q supervision signals. **mouse-core** learns from those trajectories — data utilities, models, and objectives for training and deploying in-context RL agents. DQN, PPO, and GRPO read `terminated` and `truncated`. A false flag contributes scale `1`, shift `0`, and gamma `1`; both flags can be true, and the extras multiply.


## News 📰

- **2026-10-08 — mouse-gym only.** Live notebooks talk to mouse-gym and nothing else. `step` is the Gymnasium tuple. Objectives read `terminated` and `truncated` (bool, or `0`/`1`). Episode grouping moved to `RepeatTaskEnv`: the examples wrap FrozenLake and CartPole there, then pass `env_fn` to `EnvConfig`. Inner episode flags are copied onto the row; the collector assigns `episode_index` and `episode_index`. `SpObjective` requires `mask_key` (`"episode_ended"` skips an inner episode end).
- **2026-09-17 — mouse-gym 1.1.0.** `EnvConfig.episodes_per_task` is now `max_task_episodes` (episode-length timeout, `episode_done=2`). `terminate_episode` ends an episode with `episode_done=1`; `reward_transform` rewrites each step reward before the output and metrics. Live-env notebooks pass `max_task_episodes=`.
- **2026-08-18 — mouse-gym 1.0.0.** The env step contract is now `episode_done` / `episode_done` (each `0`/`1`/`2`) instead of a single 5-code `done` field. `EnvConfig.seed` advances once per episode and is passed to `reset(seed=...)` only at episode start. The `examples` extra installs `mouse-gym` from GitHub `main`.
- **2026-06-26 — Offline training works.** [`examples/02_train_offline_dqn.ipynb`](examples/02_train_offline_dqn.ipynb) now trains a full `Qwen/Qwen3-0.6B` MOUSE model from Hub replay data and reaches strong FrozenLake performance. Push the checkpoint, then evaluate in [`examples/15_inference.ipynb`](examples/15_inference.ipynb).

See [CHANGELOG.md](CHANGELOG.md) for the full release history.


## Why MOUSE exists 💡

MOUSE is built around two observations:

1. General learning systems that scale tend to outperform hand-crafted solutions in the long run. This idea is captured in Rich Sutton's essay [The Bitter Lesson](https://web.archive.org/web/20260409023855/https://www.incompleteideas.net/IncIdeas/BitterLesson.html). MOUSE takes that lesson seriously: it meta-learns during training how to solve episodes, so that at deployment time it can adapt to new situations from experience.

2. Learning must not stop at deployment time. The [Big World Hypothesis](http://incompleteideas.net/papers/The_Big_World_Hypothesis.pdf) says that real environments are too vast to model completely ahead of time, so agents cannot be given all the information they will need before they act. MOUSE adapts by conditioning on prior history rather than updating its weights. Because the weights remain fixed at deployment, this avoids plasticity loss, a common continual-learning failure mode where repeated updates gradually reduce an agent's ability to learn.

In the video below, an agent plays FrozenLake on a map it has never seen before, with the map hidden from the agent. Without gradient updates, using only in-context learning, it tries different paths until it finds one that leads directly to the goal. You can train one yourself using the [example notebooks](examples/).

<p align="center"><img src="frozenlake.gif" width="400" alt="MOUSE agent on Procedural FrozenLake"/></p>


## Install 📦

```bash
pip install mouse-core --extra-index-url https://download.pytorch.org/whl/test/cu132
```

`torch` 2.15 wheels for Python 3.15 are not on PyPI yet. The extra index
above is the CUDA 13.2 build. Use `https://download.pytorch.org/whl/test/cpu`
instead when you do not have a CUDA 13.2 driver.

For development (free-threaded Python **3.15t** via uv — required for
`DataLoader(num_workers>0)`):

```bash
git clone https://github.com/micahr234/mouse-core.git
cd mouse-core
source scripts/install.sh
```


## Core components 🧩

mouse-core gives you three building blocks for in-context RL. Compose them in your own training loop:

* **Data** (`mouse_core.data`) — stores sequential rows in `Datastore` and batches them with `DataLoader`. `sample_field` names a scalar column. A sample is every contiguous row that shares one value in that column, and each sample is one group, so attention and loss stop there. Every sample kept in a batch gets a unique `group_id`. Pass exactly one of `samples_budget` or `token_budget`, and `batch_size`. `batch_size` is how many times that budget is filled; that count does not depend on the budget. Each fill adds whole samples until the next sample would pass the budget. `token_budget` counts packed tokens (including each sample's group-start tokens). `samples_budget` counts steps. A sample that does not fit is left out, and a first sample that alone is over the budget raises. Training examples pass `sample_field="episode_index"`, `token_budget`, and `batch_size=1`. `next_batch` and `pack_token_batch` return `(inputs, objective_data, group_id)`. `group_id` is int64 `[N]`, one id per step, and it is not a key in `objective_data`. DQN, PPO, and GRPO take it as `group_id=`. Tokenizer field `when=` is a callable `ctx → bool`; the tokenizer injects boolean `group_start` into `ctx` and evaluates ordinary vs pack-time routes (define predicates inline in the notebook). `pack_token_batch` / `Tokenizer.pack_rows` take required `continuing`: `None` emits group-start tokens on the first step of each sequence, and `True` skips that prefix because the sequence already has cached tokens.
* **Models** (`mouse_core.models`) — ordinary PyTorch modules: subclass `Model` define the flow in `forward`, and supply every module in your implementation.
* **Objectives** (`mouse_core.objectives`) — training losses such as DQN, PPO, GRPO, SP, and SV, plus `CosineObjective` for direct latent matching against stop-gradient targets. Add its `cosine` loss to DQN with an explicit auxiliary weight; callers own the action-conditioned transition model. `CosineObjective.latent_cosine_loss` pairs sequential encodings, applies group/reset boundaries, and runs that transition internally.

Backbone loading has one public path: instantiate `Backbone`. For example, `Backbone(train_kernel="flex", decode_kernel="flex", dtype=preferred_dtype(device=device), pretrained="Qwen/Qwen3-0.6B")` reads the pretrained config, loads matching decoder weights, and exposes `backbone.hidden_dim` for the heads. Packed Flex / varlen / padded kernels run when every layer is a Llama/Qwen3-shaped softmax block; other maskable softmax decoders use HuggingFace `forward` with the sequence mask and require `train_kernel="reference"`; hybrid / linear / sliding-window stacks raise. From-scratch stacks pass `architecture="qwen3"` or `architecture="llama"` plus width. `IdentityBackbone(hidden_dim=...)` is parameterless and returns its input embeddings unchanged. Three arguments are required on every transformer backbone because they describe how the model runs on your machine, not what it is, so they are configured in your implementation: `train_kernel` for the uncached forward, `decode_kernel` for cached decode (`"flex"`, paged FlexAttention, the only kernel that reads K/V through a page table), and `dtype` for the base weights. The transformer returns its final residual stream without an output norm. Each prediction head applies its own input RMSNorm; normalization has no on/off switch. `model.to(device)` moves and never casts; every part other than the backbone base is float32.


Each notebook defines an `Encoder(Model)` that builds its embedding and backbone
internally. It returns raw backbone readouts as `out["z"]`. Separate `QHead`,
`PolicyHead`, or `ValueHead` modules build their own MLPs and consume those encodings.
There is no intermediate projection. The heads retain their input RMSNorm.

```python
import torch
from torch import nn
from mouse_core.models import Model, Embedding, Backbone, MLP

class Encoder(Model):
    def __init__(self, *, pretrained, train_kernel, decode_kernel, dtype):
        super().__init__()
        self.embedding = Embedding(pretrained=pretrained)
        self.backbone = Backbone(
            pretrained=pretrained, train_kernel=train_kernel,
            decode_kernel=decode_kernel, dtype=dtype,
        )

    def forward(self, tokens):
        tensors = tokens.to_tensors(next(self.parameters()).device)
        embed = self.embedding(
            tensors["token_ids"], numeric_values=tensors["numeric_values"],
            numeric_mask=tensors["numeric_mask"],
        )
        features = self.backbone(embed, sequence_ids=tensors["sequence_ids"])
        return {"z": features[tensors["head_token_ids"]]}

class QHead(nn.Module):
    def __init__(self, *, hidden_dim, num_actions, num_layers, scale):
        super().__init__()
        self.mlp = MLP(
            in_features=hidden_dim, out_features=num_actions,
            hidden_dim=hidden_dim, num_layers=num_layers,
            scale=scale, propagate_gradient=1.0,
        )

    def forward(self, z):
        return self.mlp(z)

encoder = Encoder(
    pretrained="Qwen/Qwen3-0.6B", train_kernel="flex",
    decode_kernel="flex", dtype=torch.float32,
)
q_head = QHead(
    hidden_dim=encoder.backbone.hidden_dim, num_actions=4,
    num_layers=1, scale=0.1,
)
# With a TokenBatch named inputs:
# out = encoder(inputs)
# q = q_head(out["z"])
```

`Embedding` owns the token table and optional numeric Fourier features. The table
can be frozen with `encoder.embedding.embed_tokens.requires_grad_(False)`.
Embedding and backbone calls use plain tensors, and the encoder prepares them on
the model device. `sequence_ids` isolates packed samples; `head_token_ids` selects
readouts. The notebook encoders additionally return `last_hidden_state`, readout
indices, `cache`, and `head_valid_mask`. Cached forwards use `prepare_decode` and a
separate decode session for each backbone. `select_discrete_action` consumes head
scores and the valid-readout mask outside the encoder.

Learning rates refer to each component explicitly:

```python
from mouse_core import AdamW

optimizer = AdamW(
    params=[
        *encoder.parameter_groups(lr={"embedding": 1e-5, "backbone": 1e-5}),
        {"params": q_head.parameters(), "lr": 1e-5},
    ],
    lr=1e-5,
)
```

Training and rollout helpers use the notebook objects already constructed above
them. Calls pass only changing values, such as `run_train(num_steps=TRAIN_STEPS)`.
Branch-specific environments and cursors remain explicit arguments in GRPO.

For DQN, copy the encoder and heads independently:

```python
from copy import deepcopy
from mouse_core import polyak_update

delayed_encoder = deepcopy(encoder).requires_grad_(False).eval()
backup_value_head = deepcopy(q_head).requires_grad_(False).eval()
backup_action_head = deepcopy(q_head).requires_grad_(False).eval()

# After each optimizer update:
polyak_update(target=delayed_encoder.embedding, source=encoder.embedding, tau=0.1)
polyak_update(target=delayed_encoder.backbone, source=encoder.backbone, tau=0.1)
polyak_update(target=backup_value_head, source=q_head, tau=0.001)
polyak_update(target=backup_action_head, source=q_head, tau=1.0)
```

The examples use `nn.ModuleDict({"encoder": encoder, "q_head": q_head})` as a
checkpoint container. `save_model`, `load_model`, and `push_model_to_hub` accept
ordinary PyTorch modules, including this container. Forward calls still use the
encoder and head separately. The Hub bundle stores weights and tokenizer together;
example 15 reconstructs the same components as example 02 before loading them.
Old checkpoints containing the removed projection and combined agent layout are
not compatible with the revised examples.

Register a `LatentReasoner` on the model and call `reasoner.insert(embeds=embed, tokens=tokens, sequence_ids=sequence_ids, head_token_ids=head_token_ids, splits=reasoning, backbone=backbone)` explicitly before the final backbone pass. It returns the extended embedding tensor, sequence IDs, and shifted head-token IDs. Its parameters have their own `reasoner` group.

DQN takes three explicit prediction tensors: the online `predictions`, `backup_value_predictions`, and `backup_action_predictions`. Both backup inputs are detached float32 Q-score tensors with the same `[P, A]` shape. At zero temperature, the action scores choose the argmax and the value scores evaluate it. At positive temperature, the action scores define the Boltzmann policy, and the target uses its expected backup value plus entropy. The separate backup value and backup action heads are separate frozen modules with independent Polyak rates toward the same online Q head. There is no `double` switch.

For continuous environments with discrete DQN outputs, `DiscretizedActionHead(action_vectors=table, ...)` extends `MLP` with a float tensor of shape `[num_actions, action_dim]` and infers its output width. `head.discrete_to_continuous(ids)` maps selected IDs to full vectors; `head.continuous_to_discrete(vectors)` selects the nearest table row by Euclidean distance in the original units (ties choose the lowest ID). The latter is lossy for off-table inputs. The fixed table is a model buffer, copied with delayed heads, moved with the model, and saved/restored by `save_model`/`load_model`. Keep its rows and ordering fixed. Example `19` selects discrete IDs from the separate Q head and converts them to continuous vectors for the environment. The table does not impose similarity between neighboring actions' Q-values.

For one online/delayed forward pair, `with reuse_token_tensors():` (from `mouse_core.data`) converts each `TokenBatch` once per device and inference mode. Treat the batch and returned tensors as read-only inside the scope. The cache ends with the scope; later batches/scopes convert afresh. This reuses input tensors only: each model still computes its own activations through its own weights.


## Quick start 🚀

The [example notebooks](examples/) are short usage docs, not full experiments. Work through them in order, then evaluate a saved checkpoint in `15`. Full training runs live in **[mouse-experiment](https://github.com/micahr234/mouse-experiment)**.

| Notebook | What it covers |
|----------|----------------|
| [01 — Collect dataset](examples/01_collect_dataset.ipynb) | `Datastore`, collecting transitions, pushing to the Hub |
| [02 — Train offline DQN](examples/02_train_offline_dqn.ipynb) | Offline replay baseline: `Tokenizer` text fields + pretrained `embed_tokens`, model architecture, DQN training |
| [03 — Train online DQN](examples/03_train_online_dqn.ipynb) | Live `mouse-gym` rollouts, in-memory replay, DQN updates |
| [05 — Train offline SV](examples/05_train_offline_sv.ipynb) | Same offline loop as `02`, regressing the action-value head onto `info_q_star` with `SvObjective` |
| [07 — Train online PPO](examples/07_train_online_ppo.ipynb) | Online on-policy PPO (`MLP` + value head, `PpoObjective` with GAE) |
| [08 — Train online GRPO](examples/08_train_online_grpo.ipynb) | Branched GRPO: fork env+context at many `L`, group-relative advantages, `GrpoObjective` |
| [09 — Train offline SP](examples/09_train_offline_sp.ipynb) | Same offline loop as `05`, but `best_action(info_q_star)` then `SpObjective` CE onto those ids with `MLP` *(ranking check)* |
| [10 — Offline reasoning DQN](examples/10_train_offline_reasoning_dqn.ipynb) | Same offline loop as `02` (including the trailing `value` readout), plus Coconut-style latent reasoning bursts (`LatentReasoner`, `sample_reasoning_splits`) trained through the DQN loss |
| [13 — Offline n-step DQN](examples/13_train_offline_n_step_dqn.ipynb) | Same offline loop as `02`, with one `DqnObjective` per Q head (`nstep_gate(n=1, 3, 5)`): each head trains toward its own n-step return; the three losses are summed; action selection reads the longest-horizon head. The delayed encoder and both sets of Q heads have separate interpolation rates. |
| [14 — Offline value-gap DQN](examples/14_train_offline_value_gap_dqn.ipynb) | Same offline loop as `02`, with `gate=value_gap_gate(beta=, normalize=, policy_delayed=False, value_delayed=False, bias=, eps=)`: `a*` and the taken action are read from the current head. The signed gap is `Q(s, a_taken) - V`. `normalize=True` divides by `|max_a Q| + eps` on the value Q and requires `eps`. `normalize=False` uses the raw signed gap and takes no `eps`. `bias` is subtracted, then the gap is capped at `0`. `c = 2 ** (gap / beta)`. A gap of `0` continues at `c = 1`. For positive `beta`, a gap of `-beta` continues at `1/2`. A larger `beta` is a softer cut. `beta=0` is the hard limit: adjusted gap zero continues with `1`, and a negative gap cuts with `0`, including exact ties without division. A negative gap moves the return toward the delayed state value. With `beta=0`, `bias=0`, and both delayed flags `False`, this is the Watkins hard cut. Combine it with `lambda_gate(td_lambda=...)` using `general_gate(gates=(...))` to multiply the continuation by TD λ. |
| [17 — Collect CartPole](examples/17_collect_cartpole.ipynb) | Random `CartPole-v1` rollouts into one `Datastore` per stream. `observation` is the length-4 float vector (cart position, cart velocity, pole angle, pole angular velocity). |
| [18 — Train offline CartPole DQN](examples/18_train_offline_cartpole_dqn.ipynb) | Same offline DQN loop as `02` on the `17` dataset. Each observation element is a `numeric` field (`x`, `v`, `t`, `w`) with a Fourier add. A `token` comma sits between the action and those characters. |
| [19 — Train online Pendulum DQN](examples/19_train_online_pendulum_dqn.ipynb) | Continuous actions in, discrete actions out: `DiscretizedActionHead` owns 33 fixed torque vectors for `Pendulum-v1`, both conversion methods, and checkpoint persistence. Applied torque enters history through numeric Fourier embeddings; category IDs go only to the DQN objective. A small CPU example with replay and a delayed model. |
| [20 — Sequential action DQN](examples/20_train_sequential_actions_dqn.ipynb) | Control Reacher-v5’s two arm joints with one shared categorical head and one decision token per torque. Compare next-component, fixed n-action, and environment-boundary backups; internal choices have discount 1. |
| [21 — DQN with latent cosine prediction](examples/21_train_dqn_cosine.ipynb) | Offline replay workflow from `02`, with an encoder, separate Q head, and no projection. Residual transitions use their own action embeddings, fixed two-layer 4× blocks, and a final RMSNorm; direct cosine matching against delayed encodings is added to normal DQN. Zero cosine weight omits the auxiliary path. |
| [15 — Inference](examples/15_inference.ipynb) | Evaluation: construct the encoder and Q head, then `load_model(model=modules, ...)` on the model repo and `load_tokenizer` on the same repository, then batched FlexAttention cached inference (`max_cache` / `start_cache`) |

### Example dependencies

The notebooks need the `examples` extra (`mouse-gym`, `repeat-task-gym`, `procedural-frozenlake`, `matplotlib`, …). A full `source scripts/install.sh` already installs it via `mouse-core[all]`. From a lighter venv:

```bash
uv pip install -e ".[examples]" --python .venv/bin/python
```

Example `20` uses `Reacher-v5`; the `examples` extra includes `gymnasium[mujoco]`. It runs without rendering and does not need a display.

The extra also pulls `gymnasium[toy-text]` → `pygame-ce`. On free-threaded Python there is often no prebuilt `pygame-ce` wheel, so the install builds from source and needs SDL2 / PortMidi headers. On Debian/Ubuntu:

```bash
sudo apt install libsdl2-dev libsdl2-image-dev libsdl2-mixer-dev libsdl2-ttf-dev libportmidi-dev libfreetype6-dev
```

Without these, install fails with errors like `library 'SDL2' not found` or `library 'portmidi' not found`.

Each notebook explains the relevant concepts inline. API details live in the Python docstrings (`load_model`, `Datastore`, `DqnObjective`, etc.).

Each notebook starts with imports and grouped settings for its data or artifacts, environment, model, rollout, training, objective, learning rates, target interpolation, and seeds. Shared settings keep the same names and order across examples. Model construction uses an encoder (embedding → backbone) and separate prediction heads.

Every trainable parameter is float32, so plain `AdamW(...)` steps it in place and the delayed encoders and heads interpolate in fp32 — no master weights or shadows. The training notebooks pass separate groups: the embedding at `LR_EMBEDDING`, the backbone at `LR_BACKBONE`, and prediction heads at `LR_HEADS` (example 10's reasoner uses `LR_BACKBONE`). Copied delayed embeddings use `POLYAK_TAU_EMBEDDING`, separately from `POLYAK_TAU_BACKBONE`. Backbone training options:

- **Full fp32 fine-tuning** — `Backbone(train_kernel="flex", decode_kernel="flex", dtype=torch.float32, pretrained="Qwen/Qwen3-0.6B")`; the whole model is float32. Use `"flex"` here: it is compiled and block-sparse in fp32, whereas `"varlen"` on a plain fp32 base raises (the flash kernel needs bf16/fp16 q/k/v — there is no fallback). The training notebooks use this path.
- **Mixed-precision fp32 fine-tuning** — add `train_autocast_dtype=torch.bfloat16` (and, if eval should match, `decode_autocast_dtype=torch.bfloat16`) to a fp32 backbone: that path runs its matmuls in bf16 under `torch.autocast` while weights, optimizer state, norms, RoPE, and the residual stream stay fp32; decode also allocates its KV pool in bf16 (half the fp32 size). The two are independent, so bf16 training can pair with exact fp32 eval decode. `train_autocast_dtype` makes `"varlen"` valid (and the fastest kernel) on an fp32 base. Precision is declared at construction only; calling either forward inside an ambient `torch.autocast` raises.
- **fp32 LoRA on a frozen bf16 base** — `Backbone(train_kernel="flex", decode_kernel="flex", dtype=preferred_dtype(device), pretrained="Qwen/Qwen3-0.6B", lora=LoRAConfig(rank=16, alpha=32))`: on CUDA the frozen base is **bfloat16** and runs FlexAttention, while LoRA adapters on every backbone `nn.Linear`, plus the embedding, reasoner, and heads, are float32.

Then `model.to(device)`. `AdamW` rejects a trainable non-fp32 parameter, so a fully trainable backbone built in bf16 fails loudly at training time. The example notebooks construct their models with `dtype=torch.float32` before loading weights.


## Contributing 🔧

See [CONTRIBUTING.md](CONTRIBUTING.md).


## License 🔑

GNU General Public License v3.0 — see [LICENSE](LICENSE).

### Sequential components within one environment step

Record each component choice as a decision row with its own head-output token.
The next row holds the selected action ID and physical value, so subsequent
choices condition on the earlier choices through causal history. Set reward to
zero and done flags to false on internal rows. Mark the row reached by an actual
environment transition with an objective bool column such as `environment_step`.

Use `boundary_discount(..., step_key="environment_step")` to apply `gamma_step`
only when the environment advances. Terminal/truncation factors still apply.
With `DqnObjective(gate=None, ...)`, each decision bootstraps from the next
component. `nstep_gate(n=K)` skips K decisions (the same component in the next
step for fixed K-component actions). `nstep_gate(n=1, step_key="environment_step")`
instead reaches the next environment boundary from every component, including
variable-length assemblies. Increasing n in this latter form counts environment
transitions. Rewards, actions, and markers all describe the transition **into**
their row. Existing group-boundary behavior is unchanged.

These are sampled n-step returns: intermediate actions are taken from replay,
not maximized over. Off-policy suffix choices can therefore affect earlier
component targets. Do not represent different components as repeated readouts
of a single decision row: those readouts share one selected action ID.
