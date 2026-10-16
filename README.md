# MOUSE Core 🧠

<p align="center"><img src="https://raw.githubusercontent.com/micahr234/mouse-core/main/mouse-core.png" width="400"/></p>

> **Warning:** MOUSE is in early development and is not yet ready for production use. APIs may change without notice.

**mouse-core** is the core library for the Meta-Optimization Using Sequential Experience (MOUSE) learning system — a modular PyTorch stack for <u>in-context reinforcement learning (ICRL)</u>. It provides data utilities, embedding frameworks, transformer backbones, output heads, and objective functions for training and deploying agents that adapt from transition history at inference time, **without weight updates**.

**[mouse-gym](https://github.com/micahr234/mouse-gym)** sits alongside mouse-core and handles the environment side: `EnvConfig` turns any Gymnasium env into a **reset-free** stream. `step(action)` returns `(observation, reward, terminated, truncated, info)`. `terminated` and `truncated` are the wrapped env's flags. The first step, and the step after either flag, is a reset frame. There is no public `reset()`. Multi-episode tasks live in **[repeat-task-gym](https://github.com/micahr234/repeat-task-gym)**: wrap the Gymnasium env in `RepeatTaskEnv`, then pass that to mouse-gym. Mouse-gym `terminated` / `truncated` are then the task; inner episode ends stay on the observation (`episode_terminated`, `episode_truncated`). The examples use **[procedural-frozenlake](https://github.com/micahr234/procedural-frozenlake)**, a FrozenLake variant with procedurally generated maps and optimal-Q supervision signals. **mouse-core** learns from those trajectories — data utilities, models, and objectives for training and deploying in-context RL agents. DQN, PPO, and GRPO read `terminated` and `truncated`. A false flag contributes scale `1`, shift `0`, and gamma `1`; both flags can be true, and the extras multiply.


## News 📰

- **2026-10-08 — mouse-gym only.** Live notebooks talk to mouse-gym and nothing else. `step` is the Gymnasium tuple. Objectives read `terminated` and `truncated` (bool, or `0`/`1`). Task grouping moved to `RepeatTaskEnv`: the examples wrap FrozenLake and CartPole there, then pass `env_fn` to `EnvConfig`. Inner episode flags are copied onto the row; the collector assigns `task_index` and `episode_index`. `SpObjective` requires `mask_key` (`"episode_ended"` skips an inner episode end).
- **2026-09-17 — mouse-gym 1.1.0.** `EnvConfig.episodes_per_task` is now `max_task_episodes` (task-length timeout, `task_done=2`). `terminate_task` ends a task with `task_done=1`; `reward_transform` rewrites each step reward before the output and metrics. Live-env notebooks pass `max_task_episodes=`.
- **2026-08-18 — mouse-gym 1.0.0.** The env step contract is now `episode_done` / `task_done` (each `0`/`1`/`2`) instead of a single 5-code `done` field. `EnvConfig.seed` advances once per task and is passed to `reset(seed=...)` only at task start. The `examples` extra installs `mouse-gym` from GitHub `main`.
- **2026-06-26 — Offline training works.** [`examples/02_train_offline_dqn.ipynb`](examples/02_train_offline_dqn.ipynb) now trains a full `Qwen/Qwen3-0.6B` MOUSE model from Hub replay data and reaches strong FrozenLake performance. Push the checkpoint, then evaluate in [`examples/15_inference.ipynb`](examples/15_inference.ipynb).

See [CHANGELOG.md](CHANGELOG.md) for the full release history.


## Why MOUSE exists 💡

MOUSE is built around two observations:

1. General learning systems that scale tend to outperform hand-crafted solutions in the long run. This idea is captured in Rich Sutton's essay [The Bitter Lesson](https://web.archive.org/web/20260409023855/https://www.incompleteideas.net/IncIdeas/BitterLesson.html). MOUSE takes that lesson seriously: it meta-learns during training how to solve tasks, so that at deployment time it can adapt to new situations from experience.

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

* **Data** (`mouse_core.data`) — stores sequential rows in `Datastore` and batches them with `DataLoader`. `sample_field` names a scalar column. A sample is every contiguous row that shares one value in that column, and each sample is one group, so attention and loss stop there. Every sample kept in a batch gets a unique `group_id`. Pass exactly one of `samples_budget` or `token_budget`, and `batch_size`. `batch_size` is how many times that budget is filled; that count does not depend on the budget. Each fill adds whole samples until the next sample would pass the budget. `token_budget` counts packed tokens (including each sample's group-start tokens). `samples_budget` counts steps. A sample that does not fit is left out, and a first sample that alone is over the budget raises. Training examples pass `sample_field="task_index"`, `token_budget`, and `batch_size=1`. `next_batch` and `pack_token_batch` return `(inputs, objective_data, group_id)`. `group_id` is int64 `[N]`, one id per step, and it is not a key in `objective_data`. DQN, PPO, and GRPO take it as `group_id=`. Tokenizer field `when=` is a callable `ctx → bool`; the tokenizer injects boolean `group_start` into `ctx` and evaluates ordinary vs pack-time routes (define predicates inline in the notebook). `pack_token_batch` / `Tokenizer.pack_rows` take required `continuing`: `None` emits group-start tokens on the first step of each sequence, and `True` skips that prefix because the sequence already has cached tokens.
* **Models** (`mouse_core.models`) — ordinary PyTorch modules: subclass `Model` define the flow in `forward`, and supply every module in your implementation.
* **Objectives** (`mouse_core.objectives`) — training losses such as DQN, PPO, GRPO, SP, and SV.

Backbone loading has one public path: instantiate `Backbone`. For example, `Backbone(train_kernel="flex", decode_kernel="flex", dtype=preferred_dtype(device=device), pretrained="Qwen/Qwen3-0.6B")` reads the pretrained config, loads matching transformer weights including `embed_tokens`, and exposes `backbone.hidden_dim` for the heads. Packed Flex / varlen / padded kernels run when every layer is a Llama/Qwen3-shaped softmax block; other maskable softmax decoders use HuggingFace `forward` with the sequence mask and require `train_kernel="reference"`; hybrid / linear / sliding-window stacks raise. From-scratch stacks pass `architecture="qwen3"` or `architecture="llama"` plus width. `IdentityBackbone` takes `hidden_dim=` and `vocab_size=` and owns its own `embed_tokens` table. Three arguments are required on every transformer backbone because they describe how the model runs on your machine, not what it is, so they are configured in your implementation: `train_kernel` for the uncached forward, `decode_kernel` for cached decode (`"flex"`, paged FlexAttention, the only kernel that reads K/V through a page table), and `dtype` for the base weights. The transformer returns its final residual stream without an output norm. The projection and each prediction head always apply their own input RMSNorm; normalization has no on/off switch. `model.to(device)` moves and never casts; every part other than the backbone base is float32.


Define the architecture in Python, like any PyTorch module. `Model` supplies parameter groups and delayed copies; a subclass's `forward()` defines the flow. There is no default architecture, and the base class cannot be instantiated directly. `TokenEncoder` handles packed tokens, stacked backbones, cached decoding, and optional latent reasoning. Calling `encoded.pool()` selects only head tokens before projection or prediction work.

```python
from mouse_core.models import Model, TokenEncoder, select_discrete_action

class Agent(Model):
    def __init__(self, backbone, projection, policy, value):
        super().__init__()
        self.encoder = TokenEncoder([backbone])
        self.projection = projection
        self.policy = policy
        self.value = value

    def forward(self, batch, **kwargs):
        encoded = self.encoder(batch, **kwargs)
        h = self.projection(encoded.pool())
        return encoded.with_predictions({
            "action": self.policy(h),
            "value": self.value(h),
        })

    def get_action(self, *, out, temperature=0.0):
        return select_discrete_action(
            scores=out.predictions["action"], valid=out.head_output_valid,
            temperature=temperature,
        )
```

`get_action()` also belongs to your implementation: choose which predictions it reads and whether it returns discrete IDs, continuous values, or another action representation. `select_discrete_action` is an optional helper for greedy or categorical selection.

Branches, residual additions, and conditional paths are ordinary Python and tensor operations in `forward()`. Use `nn.Sequential` for a projection stack. The library has no separate connection schema, graph executor, or diagram generator.

Each training notebook defines its own `Agent(Model)` implementation. For a projection stack, explicitly supply `nn.Sequential(...)`; to disable projection, supply `nn.Identity()`. All training notebooks expose `USE_PROJECTION`. Stack backbones with `TokenEncoder([first, second])`; only the first embeds tokens, and their hidden widths must match. Each cached backbone owns a separate decode session.

Projection, regression, and classification use separate instances of the same `MLP` class. Set the output width explicitly: `out_features=1` for scalar regression, the number of actions for Q-values or logits, or the desired feature width for a projection. Every instance owns a mandatory input RMSNorm. Pass `scale` only on prediction instances to scale the last linear layer's initial weight and bias; omit it for feature projections. `propagate_gradient` controls the gradient into upstream modules while the MLP's own parameters receive the full gradient.

```python
import torch
from mouse_core.models import Backbone, MLP

backbone = Backbone(
    pretrained="Qwen/Qwen3-0.6B", dtype=torch.float32,
    train_kernel="flex", decode_kernel="flex",
)
d = backbone.hidden_dim
projection = MLP(in_features=d, out_features=d, hidden_dim=d)
policy = MLP(in_features=d, out_features=4, hidden_dim=d, scale=0.1)
value = MLP(in_features=d, out_features=1, hidden_dim=d, scale=0.1)
model = Agent(backbone, projection, policy, value)
```

`Backbone` is the concrete transformer; subclass `BaseBackbone` to implement another sequence processor. There are no legacy aliases for the previous backbone or head names.

Learning and interpolation rates reference the module attributes you chose:

```python
from mouse_core import AdamW, model_polyak

optimizer = AdamW(
    params=model.parameter_groups(lr={"encoder": 1e-5, "projection": 1e-4, "policy": 1e-4, "value": 1e-4}),
    lr=1e-5,
)
delayed_model = model.copy()
update_delayed = model_polyak(online=model, delayed=delayed_model)
# After each optimizer.step():
update_delayed(tau={"encoder": 0.1, "projection": 0.01, "policy": 0.001, "value": 0.001})
```

Only parameterized modules need rates: omit `projection` when it is `nn.Identity()`.

`out.pool()` and `out.last_hidden_state` preserve unnormalized backbone states. The notebooks define a `head(h=...)` method for applying their readout to already-selected states; custom implementations can structure this however they need.

Checkpoints contain only the online model's `state_dict` and identifying metadata; optimizer state and delayed models are never saved. The implementation owns the architecture and construction settings. Loading requires a constructed instance:

```python
from mouse_core.models import load_model, save_model

save_model(model=model, path="checkpoint")
restored = MyAgent(...)  # your implementation, with its modules and runtime settings
load_model(model=restored, repo_id_or_path="checkpoint")
restored.eval()
```

Loading fills that instance's weights and preserves its class, precision, and trainability. There is no default model, automatic architecture reconstruction, or loading of Python forward code. The checkpoint format is `mouse-core-model-v3`, with no legacy loader.

Training runs the transformer over the flat packed token stream (`packed_forward` in `mouse_core.models.backbone`): tokens are regrouped by group id and attention is causal within each group. `train_kernel` picks the kernel over those sequences, so they can be compared on the same weights and streams, and the choice is strict — a kernel that cannot run on the current device/dtype raises instead of silently substituting another one: `"varlen"` is PyTorch's FlashAttention varlen kernel and requires CUDA with bf16/fp16 q/k/v (a bf16/fp16 base — the LoRA setup — or a fp32 base with `train_autocast_dtype=`); `"padded"` right-pads each sequence to `max_seqlen` and runs dense causal SDPA on the rectangular `[n_seg, H, S, Dh]` tensor (Flash when every sequence is already that long); `"flex"` is FlexAttention with a block-sparse mask over the sequences, compiled on CUDA in every dtype (forward-only on CPU, where FlexAttention has no backward), which makes it the kernel for full fp32 fine-tuning; `"reference"` is the same decoder body with masked SDPA over the packed stream (O(L²) memory, any device/dtype), the ground-truth implementation the fused kernels are tested against. Call `install_compiled_decoder()` before training to `torch.compile` the per-layer train decoder body once (idempotent, a few seconds of warmup, used by every layer and stream length); on CUDA that call also compiles the cached-decode layer. Set `backbone.gradient_checkpointing = True` to recompute layers in backward: roughly a 10x cut in activation memory for about 1.4x the step time. `bench/bench_train.py` measures the packed train path on your GPU; `bench/bench_inference.py` times paged FlexAttention prefill and a decode step of `--step` new tokens per row. `bench/bench_dataloader.py` times `DataLoader.next_batch`. Pass `--profile` on any of them for a `torch.profiler` CPU/CUDA breakdown and a `cProfile` host stack after warmup, and `--profile-trace DIR` to write Chrome traces. Env step rate is `bench/bench_env.py` in [mouse-gym](https://github.com/micahr234/mouse-gym).

Rollout and inference decode incrementally with `model(batch, cache=out.cache, use_cache=True)`: on every call each sequence may add any number of steps, including zero. The `DecodeCache` holds one `FlexDecodeSession` per backbone, each of whose KV caches is a paged pool shared by the whole batch: each sequence owns as many 128-token pages as its own history needs, so a batch with one long stream and many short ones costs the sum of their lengths rather than `batch × longest`, and `cache.reset_rows(rows)` hands a finished stream's pages back to the pool. On CUDA the pool grows by mapping more physical pages under a reserved virtual address (existing K/V are not copied); it grows by the pages actually needed, not by doubling. Call `cache.close()` when the rollout ends so those mappings unmap before train (do not rely on `__del__`). Attention runs through FlexAttention over only the pages a row owns, compiled on CUDA in every dtype, so each row's decode cost tracks its own history rather than the batch maximum. Each decode layer is compiled (attention inside, graph break after the KV scatter). After warmup a repeated incremental `[B, S]` — whatever `S` the caller keeps sending — is CUDA-graphed as one stack replay. Page growth, mask build, and address setup stay eager so a growing cache does not recompile. fp32 decode is as fast as bf16 once `torch.set_float32_matmul_precision("high")` (TF32: fp32 storage and accumulation, 10-bit-mantissa matmul inputs) is set; without it a long fp32 row is compute-bound at up to 2.5x the bf16 step time.

`model.copy()` creates a frozen delayed copy while preserving the Python subclass. `model.copy(modules=["projection", "heads"])` copies just those modules and shares the encoder. Already frozen parameters, including LoRA base weights, remain shared. The n-step notebook reuses online backbone states with `delayed_model.head(h=out.pool())` under `torch.no_grad()`.

LR and tau groups use standard dotted module paths, such as `encoder.backbones.0`, `encoder.reasoner`, `projection.0`, and `heads.action_value`. `parameter_groups(lr={...})` checks that every trainable parameter belongs to exactly one group. `model_polyak` defaults to parameterized direct children; pass `modules=[...]` when binding to choose finer groups. The resulting updater takes one `tau` per bound group: `0` keeps a copied group frozen, `1` copies online weights, and `None` marks a shared group. Parameter pairing happens once when binding. Rebind after changing parameter objects, trainability, dtype, or device. Reasoning is configured on `TokenEncoder(reasoner=...)` and can have its own group through `encoder.reasoner`.

For continuous environments with discrete DQN outputs, `DiscretizedActionHead(action_vectors=table, ...)` extends `MLP` with a float tensor of shape `[num_actions, action_dim]` and infers its output width. `head.discrete_to_continuous(ids)` maps selected IDs to full vectors; `head.continuous_to_discrete(vectors)` selects the nearest table row by Euclidean distance in the original units (ties choose the lowest ID). The latter is lossy for off-table inputs. The fixed table is a model buffer, copied with delayed heads, moved with the model, and saved/restored by `save_model`/`load_model`. Keep its rows and ordering fixed. Example `19` defines `Agent.get_action()` to return IDs for DQN and converts them to continuous vectors for the environment. The table does not impose similarity between neighboring actions' Q-values.

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
| [13 — Offline n-step DQN](examples/13_train_offline_n_step_dqn.ipynb) | Same offline loop as `02`, with one `DqnObjective` per Q head (`nstep_gate(n=1, 3, 5)`): each head trains toward its own n-step return; the three losses are summed; `get_action` reads the longest-horizon head. The delayed model shares the backbone and delays the projection and Q heads. |
| [14 — Offline value-gap DQN](examples/14_train_offline_value_gap_dqn.ipynb) | Same offline loop as `02`, with `gate=value_gap_gate(beta=, normalize=, policy_delayed=False, value_delayed=False, bias=, eps=)`: `a*` and the taken action are read from the current head. The signed gap is `Q(s, a_taken) - V`. `normalize=True` divides by `|max_a Q| + eps` on the value Q and requires `eps`. `normalize=False` uses the raw signed gap and takes no `eps`. `bias` is subtracted, then the gap is capped at `0`. `c = 2 ** (gap / beta)`. A gap of `0` continues at `c = 1`. For positive `beta`, a gap of `-beta` continues at `1/2`. A larger `beta` is a softer cut. `beta=0` is the hard limit: adjusted gap zero continues with `1`, and a negative gap cuts with `0`, including exact ties without division. A negative gap moves the return toward the delayed state value. With `beta=0`, `bias=0`, and both delayed flags `False`, this is the Watkins hard cut. Combine it with `lambda_gate(td_lambda=...)` using `general_gate(gates=(...))` to multiply the continuation by TD λ. |
| [17 — Collect CartPole](examples/17_collect_cartpole.ipynb) | Random `CartPole-v1` rollouts into one `Datastore` per stream. `observation` is the length-4 float vector (cart position, cart velocity, pole angle, pole angular velocity). |
| [18 — Train offline CartPole DQN](examples/18_train_offline_cartpole_dqn.ipynb) | Same offline DQN loop as `02` on the `17` dataset. Each observation element is a `numeric` field (`x`, `v`, `t`, `w`) with a Fourier add. A `token` comma sits between the action and those characters. |
| [19 — Train online Pendulum DQN](examples/19_train_online_pendulum_dqn.ipynb) | Continuous actions in, discrete actions out: `DiscretizedActionHead` owns 33 fixed torque vectors for `Pendulum-v1`, both conversion methods, and checkpoint persistence. Applied torque enters history through numeric Fourier embeddings; category IDs go only to Double DQN. A small CPU example with replay and a delayed model. |
| [15 — Inference](examples/15_inference.ipynb) | Evaluation: construct the example agent explicitly, then `load_model(model=agent, ...)` on the model repo and `load_tokenizer` on a separate tokenizer repo, then batched FlexAttention cached inference (`max_cache` / `start_cache`) |

### Example dependencies

The notebooks need the `examples` extra (`mouse-gym`, `repeat-task-gym`, `procedural-frozenlake`, `matplotlib`, …). A full `source scripts/install.sh` already installs it via `mouse-core[all]`. From a lighter venv:

```bash
uv pip install -e ".[examples]" --python .venv/bin/python
```

That pulls `gymnasium[toy-text]` → `pygame-ce`. On free-threaded Python there is often no prebuilt `pygame-ce` wheel, so the install builds from source and needs SDL2 / PortMidi headers. On Debian/Ubuntu:

```bash
sudo apt install libsdl2-dev libsdl2-image-dev libsdl2-mixer-dev libsdl2-ttf-dev libportmidi-dev libfreetype6-dev
```

Without these, install fails with errors like `library 'SDL2' not found` or `library 'portmidi' not found`.

Each notebook explains the relevant concepts inline. API details live in the Python docstrings (`load_model`, `Datastore`, `DqnObjective`, etc.).

Each notebook starts with imports and grouped settings for its data or artifacts, environment, model, rollout, training, objective, learning rates, target interpolation, and seeds. Shared settings keep the same names and order across examples. Model construction follows backbone → projection → prediction heads.

Every trainable parameter is float32, so plain `AdamW(...)` steps it in place and `model_polyak` interpolates in fp32 — no master weights or shadows. The training notebooks pass separate groups: the backbone at `LR_BACKBONE`, the projection at `LR_PROJECTION`, and prediction heads at `LR_HEADS` (example 10's reasoner uses `LR_BACKBONE`). Backbone training options:

- **Full fp32 fine-tuning** — `Backbone(train_kernel="flex", decode_kernel="flex", dtype=torch.float32, pretrained="Qwen/Qwen3-0.6B")`; the whole model is float32. Use `"flex"` here: it is compiled and block-sparse in fp32, whereas `"varlen"` on a plain fp32 base raises (the flash kernel needs bf16/fp16 q/k/v — there is no fallback). The training notebooks use this path.
- **Mixed-precision fp32 fine-tuning** — add `train_autocast_dtype=torch.bfloat16` (and, if eval should match, `decode_autocast_dtype=torch.bfloat16`) to a fp32 backbone: that path runs its matmuls in bf16 under `torch.autocast` while weights, optimizer state, norms, RoPE, and the residual stream stay fp32; decode also allocates its KV pool in bf16 (half the fp32 size). The two are independent, so bf16 training can pair with exact fp32 eval decode. `train_autocast_dtype` makes `"varlen"` valid (and the fastest kernel) on an fp32 base. Precision is declared at construction only; calling either forward inside an ambient `torch.autocast` raises.
- **fp32 LoRA on a frozen bf16 base** — `Backbone(train_kernel="flex", decode_kernel="flex", dtype=preferred_dtype(device), pretrained="Qwen/Qwen3-0.6B", lora=LoRAConfig(rank=16, alpha=32))`: on CUDA the frozen base is **bfloat16** and runs FlexAttention, while LoRA adapters on every backbone `nn.Linear`, plus the encoder, reasoner, and heads, are float32.

Then `model.to(device)`. `AdamW` and `model_polyak` reject a trainable non-fp32 parameter, so a fully trainable backbone built in bf16 fails loudly at training time. The example notebooks construct their models with `dtype=torch.float32` before loading weights.


## Contributing 🔧

See [CONTRIBUTING.md](CONTRIBUTING.md).


## License 🔑

GNU General Public License v3.0 — see [LICENSE](LICENSE).
