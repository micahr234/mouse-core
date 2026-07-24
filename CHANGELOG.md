# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Removed
- ``DataLoader`` ``pack`` / ``pad`` flags and train-time ``segment_ids`` plumbing.
  Packing/padding short windows and pack-seam isolation are gone (one public path).
- ``TokenBatch.lengths``: per-sequence step counts are derived from
  ``col_values["sequence_id"]`` + ``B`` via ``TokenBatch.step_counts()``
  (``step_counts_from_sequence_id``).
- Dead packing helpers (``pack_and_pad_rows``, ``pack_live_step_tokens``,
  ``counts_from_prediction_indices``, …); only ``left_align_content`` remains.
- Compat aliases / unused modules: ``DiscreteEmbedder``, ``RandomFourierFeatures``,
  ``NormalizedPixel``, ``PosLinear`` / ``ScaledPosLinear``, ``build_heads``,
  TextEmbedder vision-tower path, ``DataLoader.set_preparer``, and
  ``_segment_*`` attention helpers.
- Removed NumericEmbedder kwargs (``modality_fusion``, ``include_type_token``, …)
  and unused ``ModalitySpec`` fields (``size``, ``in_min``, ``in_max``).

### Changed
- Public constructors and helpers are **keyword-only** (``def f(*, ...)`` / ``def m(self, *, ...)``):
  ``Model``, encoders, backbones, objectives, ``DataLoader``, hub push/load helpers,
  ``Model.head``, ``FlexDecodeSession.forward``, and related utilities. Call sites must
  use keyword arguments (``Model(encoder=..., backbone=..., heads=...)``).
  ``nn.Module.forward`` methods that are invoked via ``module(...)`` (e.g. ``Model.forward``,
  head ``forward``) keep a leading positional tensor/batch argument so PyTorch's
  ``__call__`` still works.
- Training batches are **ragged**: ``sequence_length`` is a max window length;
  each of ``B`` sequences is one contiguous store slice of length ``1 .. S_max``.
  ``DataLoader.next_batch()`` with a preparer returns a ``TokenBatch``; without a
  preparer it returns ``list[list[dict]]`` only (no ``segment_ids``).
- ``step_token_indices`` renamed to ``prediction_indices`` (and
  ``counts_from_step_token_indices`` → ``counts_from_prediction_indices``).
- ``TokenBatch`` / ``Encoder.prepare``: variable per-sequence step counts; flat
  ``prediction_indices`` ``[N]``, ``col_values`` ``[N]`` (+ ``sequence_id``).
  No stored ``lengths`` / ``segment_ids`` fields.
- ``Model.forward`` training path: predictions and ``objective_data`` are flat
  ``TensorDict[N]`` (``N = len(prediction_indices)``). Flex / CPU document
  attention masks by ``sequence_ids`` only (causal within sequence). Cached
  decode stays rectangular ``[B, S]``.
- Objectives (DQN / Layerwise / Vec / PPO / GRPO): flat ``x[:-1]`` / ``x[1:]`` with
  valid pairs where ``sequence_id[i] == sequence_id[i+1]``.

### Fixed
- Flex train / flat RoPE positions use a sync-free ``cummax`` run reset (no
  ``.item()`` for run count). Embedder type scatters skip host ``.any()`` checks.

### Changed
- CUDA training/inference examples use ``preferred_dtype(device)`` (``bfloat16`` on
  GPU, ``float32`` on CPU) via ``Model.to(device=..., dtype=...)``. Encoder and
  backbone run in bf16 so FlexAttention compiles; heads stay float32. Training
  in float32 on CUDA falls back to the unfused Flex path and warns once.
- ``flex_packed_forward`` caches the compiled FlexAttention kernel per device/dtype
  instead of recreating ``torch.compile(flex_attention)`` every step.
- Example notebooks: inference moved to ``09_inference.ipynb`` (was ``04``);
  layerwise / vec / text / PPO / GRPO training notebooks renumbered to ``04``–``08``.
  All training notebooks run greedy held-out eval on a separate ``GroupEnv``
  (seed stream offset by ``EVAL_SEED_OFFSET``), with eval helpers defined locally
  in each notebook.
- All example notebooks build environments in a dedicated **Build Environment**
  section immediately after imports (train and held-out eval envs together when
  both are used).
- Example notebook import / constant cells use one compact formatting style
  (grouped imports, aligned trailing comments, no sparse blank lines).
- Training example notebooks use a uniform cycle loop: ``NUM_CYCLES`` drives the
  outer loop; ``run_train`` / ``run_eval`` / ``run_rollout`` each take
  ``num_steps`` from ``TRAIN_STEPS`` / ``EVAL_STEPS`` / ``ROLLOUT_STEPS``.
  Eval runs once per cycle (``EVAL_EVERY*`` schedules removed).
  Each of ``run_eval`` / ``run_rollout`` is defined in its own notebook cell;
  ``run_train`` shares a cell with optimizer / objective setup. Those helpers
  are grouped together after model / buffer setup and immediately before the
  outer ``## Run`` cycle loop.
  ``run_eval`` / ``run_rollout`` leave the model in ``eval``; ``run_train``
  calls ``model.train()``.
- Offline training notebooks use the shared ``EPISODES_PER_TASK`` /
  ``MAX_EPISODE_STEPS`` constants (removed ``EVAL_EPISODES_PER_TASK`` /
  ``EVAL_MAX_EPISODE_STEPS``).
- Example ``run_eval`` / ``run_rollout`` keep row **contexts** across cycle
  calls and rebuild the **KV cache** from those contexts each call (cache does
  not persist between calls).

### Added
- ``preferred_dtype(device)``: returns ``torch.bfloat16`` on CUDA and
  ``torch.float32`` otherwise (exported from ``mouse_core.models``).
- ``TokenBatch``: flat concatenated token stream (no padding) with parallel
  ``token_types`` / ``token_ids`` / ``scalars`` / ``sequence_ids`` / ``step_ids`` /
  ``prediction_indices`` (step counts via ``sequence_id`` + ``B``). Built by
  ``Encoder.prepare`` / ``make_preparer()`` and consumed by embedders and
  ``Model.forward``.
- ``StaticFourierFeatures``: deterministic log-spaced Fourier features (buffers
  only; no ``nn.Parameter``). One token per real scalar.
- ``DataLoader(..., preparer=encoder.make_preparer())``: workers run
  sample → augment → prepare and return a ``TokenBatch``.
- ``flex_packed_forward``: FlexAttention training over a packed flat sequence
  masked by ``sequence_ids`` (CUDA). CPU uses an equivalent SDPA document mask
  (Flex has no backward on CPU).
- ``mouse_core.models.kv_policy``: grow-then-rebuild helpers
  (``resolve_cache_bounds``, ``rebuild_starts``, ``cache_needs_rebuild``) for
  cached inference. Examples ``03``, ``07``, and ``09`` use ``max_cache`` +
  ``start_cache`` instead of a sliding ``deque(maxlen=...)`` paired with a
  never-rebased KV session.

### Removed
- ``modality_fusion`` / ``include_type_token`` / ``type_embedding_std`` /
  ``token_data_len`` / continuous ``method="linear"`` / learnable RFF weights /
  numeric raw-pixel ``ImageEmbedder``. Tokens are always concatenated; reals use
  static Fourier; images require an image tokenizer (``image_tokenizer=``).
- `action_modality` / `observation_modalities` helpers and `NumericEmbedder.infer_max_num_actions`.
  Modalities are declared with generic ``ModalitySpec`` field names and types
  (``discrete`` / ``continuous`` / ``image`` / …); env-specific field naming belongs
  at the call site.
- Padded ``embeds [B, L, D]`` training path and SDPA segment float masks for
  training (replaced by flat concat + Flex / document mask).

### Changed
- ``NumericEmbedder`` / ``TextEmbedder``: CPU ``prepare`` builds ``TokenBatch``;
  ``forward`` only does typed GPU embedding gathers / static Fourier →
  ``embeds [L, D]``. ``Model.forward`` accepts ``TokenBatch`` or raw steps
  (online); pools via ``prediction_indices`` to flat ``[N, D]`` (train) or
  rectangular ``[B, S, D]`` (cached decode).
- `DataLoader` background workers are now **threads** on free-threaded CPython
  (`3.14t`) so `num_workers > 1` uses multiple CPU cores while sharing the HF
  Dataset snapshot and prepared ``TokenBatch`` objects in-process (no
  multiprocessing queue pickle). ``num_workers > 0`` requires a free-threaded
  build with the GIL disabled. Install/CI temporarily set ``PYTHON_GIL=0``
  because current ``transformers`` pins ``tokenizers<=0.23.0``, which re-enables
  the GIL on import; drop that once ``transformers`` allows ``tokenizers>=0.23.1``
  (see ``CONTRIBUTING.md``). ``refresh()`` restarts worker threads against a
  fresh store snapshot. Minimum Python raised to **3.14**.
- Development setup (`scripts/install.sh`), CI, publish workflow, and Pyright
  now target free-threaded Python 3.14t; Python 3.13 is no longer supported.

### Added
- `GrpoObjective`: clipped policy loss with **group-relative** advantages (no critic). Caller stamps `advantage` via `group_relative_advantages` over G branch returns; inject with `batch_field` alongside `old_log_prob`.
- `examples/09_train_online_grpo.ipynb`: branched online GRPO — trunk growth, `deepcopy` env forks at many context lengths `L`, stochastic completions, group z-score advantages.
- `PpoObjective`: clipped PPO with GAE over the same done-code discounts and pack-segment masking as the DQN family. Reads `predictions["action"]` (logits) and `predictions["value"]` (scalar V); optional `objective_data["old_log_prob"]` for multi-epoch updates (inject via `batch_field`). Helpers: `sample_discrete_action`, `batch_field`.
- `examples/08_train_online_ppo.ipynb`: online on-policy PPO (`DiscreteActionHead` + scalar `value` head, stochastic rollouts with stored behavior log-probs).
- `TextEmbedder`: step fields → pretrained embeddings via a whole-step ``format`` template plus modality kinds ``text`` (value → string via per-field ``format`` → tokenizer), ``token`` (integer id → single ``embed_tokens`` row), or ``image`` (vision span). No learnable scratch tokens (those remain ``NumericEmbedder``-only); heads read the last token of each step. ``col_values`` dtypes are inferred from the batch data. Optional `skip` omits that field's fragment/span (literals such as commas stay). Emits flat `[B, L, D]` embeds plus `prediction_indices`. See `examples/07_train_offline_text.ipynb`.
- `examples/07_train_offline_text.ipynb`: offline DQN with `TextEmbedder` (`token` action + `text` fields); documents `type: "image"` for vision-language checkpoints.

### Changed
- `StepEmbedder` renamed to `NumericEmbedder` (no alias). Modality field `absent` renamed to `skip`: matching values omit that modality from the step's fusion instead of embedding a sentinel. Both `NumericEmbedder` and `TextEmbedder` pack variable-length per-step spans into flat `embeds [B, L, D]` with `prediction_indices [B, S]` (no fixed tokens-per-step / `[B, S, T, D]` axis). `Model` / Flex decode derive token counts and pad masks from indices only. `TextEmbedder` does not truncate per step (no `max_tokens_per_step`); sequence-length limits belong at the sequence / context level if needed.
- `DataLoader` accepts ``pad=True`` as an alternative to ``pack=True`` (mutually exclusive). Short store suffixes are right-padded by repeating the last real row; each pad step gets a unique ``segment_id`` so attention/RoPE and TD objectives isolate pads the same way as pack seams. Short stores are allowed at construction when either ``pack`` or ``pad`` is set.
- `DataLoader.next_batch()` now returns `(batch, segment_ids)` instead of injecting an `is_seam` flag into row dicts. With `pack=True`, `segment_ids` label which independently sampled pack slice each step came from (`0`, `1`, … within a sequence); with `pad=True`, the real suffix shares id `0` and each pad step gets its own id; with neither, every id is `0`. `Model.forward(..., segment_ids=...)` builds a same-segment causal attention mask, resets RoPE `position_ids` at each new segment, and places `segment_id` in `objective_data`. TD objectives (`DqnObjective`, `LayerwiseDqnObjective`, `VecDqnObjective`) exclude pairs whose adjacent steps have different `segment_id` values. Example notebooks updated (no `"is_seam"` in `keep_fields`).

### Added
- Experiment scripts under `scripts/experiments/`: `train_offline_dqn.py` mirrors `examples/02_train_offline.ipynb`; `compare_offline_datasets.py` trains on both an oracle-ramp Hub dataset and an always-random Hub dataset and writes a side-by-side `comparison.md` (default ids `mouse-example-dataset` / `mouse-example-dataset-random`).

### Changed
- `examples/05` and `examples/06` are offline notebooks modeled on `02_train_offline.ipynb` (Hub load → `DataLoader` + `Augmenter` → train → `push_model_to_hub`), not online rollouts. Renamed to `05_train_offline_layerwise_dqn.ipynb` (`LayerwiseDiscreteActionValueHead` / `LayerwiseDqnObjective`) and `06_train_offline_vec_dqn.ipynb` (`VectorActionValueHead` / `VecDqnObjective`).
- `examples/01_collect_dataset.ipynb` uses `NUM_ENVS = 30` and `STEPS_PER_ENV = 50_000` (was 50 × 30,000), matching online notebook `03` while keeping the **1.5M** transition budget.
- `examples/01_collect_dataset.ipynb` adds `ORACLE_PROB_END` (default `1.0`): `oracle_prob` ramps linearly from `0` to that value. Set `0.0` for an always-random dataset to compare offline DQN against online exploration without oracle demos; use a distinct `DATASET_ID` (e.g. `mouse-example-dataset-random`) so the expert dataset is not overwritten.
- Fixed a **GPU memory leak** in `FlexDecodeSession`: `mask_mod` closed over `self`, creating a reference cycle that kept KV buffers alive until cyclic GC ran. Between online rollout and train that often never happened, so each rollout's cache could remain allocated and OOM after a few cycles. `mask_mod` now closes over a dict holder instead. Online notebook `03` uses `NUM_ENVS = 30` and `STEPS_PER_ROLLOUT = 500` (50 OOMs on ~24 GiB fp32) so runs still match offline at **1.5M** transitions / **20k** grads; it drops the Flex session + `torch.cuda.empty_cache()` between phases.
- Aligned with `mouse-gym`: step field `time` is now `step_index` (0-based within the episode; 0 on reset frames), and `episode_index` counts episodes within the current task (resets to 0 at task start). Example notebooks document the new fields; `StepEmbedder` / `Augmenter` default absent/mask value `-1` applies to `step_index` instead of `time`.
- `examples/04_inference.ipynb` replay capture builds **one continuous MP4 per env** (`NUM_ENVS` movies): all frames are concatenated across episodes and tasks so fog-of-war reveals accumulate within a task as the env renders them.
- Example notebooks use `EPISODES_PER_TASK = 20` (20-episode task budget per map).
- `DqnObjective` and `LayerwiseDqnObjective` metrics `q_values_mean`, `q_values_std`, `q_values_min`, and `q_values_max` now report **max online-network Q at the current state** (`max_a Q(s_t, a)`) instead of target-network or taken-action values; `LayerwiseDqnObjective`'s per-layer `layer_{i}_q_mean` uses the same current-state max at each layer. `VecDqnObjective` score metrics now report the parallel quantity for vector heads (`max_a |score(s_t, a)| / π` on valid transitions). The separate `q_values_target` metric was removed from `DqnObjective` as redundant.
- Cached decoding (`Model.forward(..., use_cache=True)`) now runs through **`FlexDecodeSession`** (`mouse_core.models.backbone.flex_decode`): FlexAttention block-sparse attention with a per-sequence KV buffer and per-row RoPE positions, instead of HuggingFace `past_key_values` with left-padded dense attention masks. Training and uncached forwards still use the backbone SDPA path unchanged. On every cached call each row may contribute any number of new steps (including zero) with no lockstep assumption; each row decodes exactly as it would alone (verified in `tests/test_kv_cache.py`). On CUDA the flex kernel compiles automatically inside the session; `model.backbone.model.compile()` now speeds up **training** forwards only, not cached rollout/inference decode.
- `Model.forward` ragged-batch decoding (variable row lengths when `use_cache=True`) is unchanged at the API level: shorter rows are still left-padded for the encoder only; pad tokens never enter the KV cache and are masked out of attention by the flex session.
- `examples/04_inference.ipynb` evaluates with **batched incremental inference**: all envs' new steps are stacked into one `[NUM_ENVS][1]` batch sharing a single batched KV cache, replacing one `B=1` model call per env per timestep (~8× faster model time for 10 envs; greedy actions and Q-values are unchanged).
- Online training notebook `03` collects rollouts with **batched lockstep inference over one `GroupEnv`**: all `NUM_ENVS` (50) envs step together, sharing a single batched KV cache (ragged per-env context prefill, epsilon-greedy catch-up chunks), replacing both the one-env-at-a-time round-robin visits and the intermediate group/cursor rotation machinery (`ENVS_PER_GROUP`, `GROUPS_PER_ROLLOUT`, `STEPS_PER_ENV`, `env_cursor`, and `ENV_STEPS_PER_CYCLE` are gone). Each rollout runs `STEPS_PER_ROLLOUT` (300) lockstep steps followed by `GRADIENT_STEPS_PER_ROLLOUT` (200) optimizer updates; over 100 rollouts the run gathers **1.5M transitions — the same data budget as the offline dataset** from `01_collect_dataset`, making offline and online training directly comparable on both gradient and data budgets. `LEARNING_STARTS` is 15,000 (one rollout) and `EXPLORATION_ENDS` is 1,500,000, so the linear ε decay spans the full run — mirroring the offline dataset's oracle ramp, which anneals random→expert over its whole collection run.
- Example notebooks that run a model (`02`–`06`) compile the transformer backbone in place with `model.backbone.model.compile()` right after model construction/loading. This speeds up **uncached training** forwards after a one-time warmup; cached rollout/inference decode uses the built-in FlexAttention session instead. The encoder and heads stay eager.
- Minimum Python version raised from 3.12 to **3.13** (`requires-python = ">=3.13"`). Development setup (`scripts/install.sh`), CI, publish workflow, and Pyright config now target 3.13; Python 3.12 is no longer supported.
- Environment backend switched from `mouse-env` to [`mouse-gym`](https://github.com/micahr234/mouse-gym) plus [`procedural-frozenlake`](https://github.com/micahr234/procedural-frozenlake). All example notebooks import `EnvConfig` / `make_env` / `make_group_env` from `mouse_gym` and register `Procedural-FrozenLake-v1` via `import procedural_frozenlake`.
- `mouse-gym` step outputs are NumPy-based and expose the Gymnasium `info` dict verbatim: expert Q-values are read from `output["info"]["q_star"]` (enabled with the `emit_q_star=True` env kwarg) instead of the flattened `info_q_star` key; `01_collect_dataset` flattens them back into an `info_q_star` dataset column for `SpObjective` / `SvObjective`. `env.tracker` is now `env.metrics`.
- Expert Q\* from Procedural Frozen Lake is no longer degenerate: value iteration discounts with `q_star_gamma` (default `0.999`), so `argmax(q_star)` breaks ties toward shorter paths instead of collapsing to a fixed action. Live rewards stay unshaped — episode truncation at `max_episode_steps` is what pressures policies to make progress. Movement is controlled by `slippery_success_rate` (default `1/3`); set it to `1.0` for deterministic dynamics in the examples.
- `Datastore.append` now unwraps 0-dim arrays/tensors to plain Python scalars at append time, so NumPy scalar step fields survive the Hugging Face `Dataset` round-trip as scalars instead of 1-element lists.
- Example notebooks restructure the problem around **finite tasks**: every env runs `episodes_per_task=20` with `task_reset_options={"regenerate_map": True}`, so each task is a fresh procedural map (with a fresh id relabeling) and a budget of 20 episodes to figure it out in context. `MAX_EPISODE_STEPS` drops to `30` (was hardcoded `50`, briefly `100`), making the truncation deadline the only within-episode progress pressure.
- Training discounts now mirror the task structure instead of encoding costs: `gamma_step = gamma_episode_terminal = gamma_episode_truncated = 1.0` (undiscounted within a task — Q means "expected remaining points in this task", and hole vs. timeout deaths cost identically: one episode of the budget) and `gamma_task_* = 0.0` (hard value cut at task boundaries, matching the KV-cache reset at inference). This replaces the previous `gamma_step=0.995` + derived boundary-toll scheme in `02`, `03`, and `05`; `05`'s layerwise objective uses the same gammas at every layer (gamma `1.0` has an infinite effective horizon, so there is no finite horizon ladder to interpolate — the layerwise head is pure deep supervision).
- Dataset collection (`01`) and online training (`03`) run a plain step budget with no alignment to episode or task boundaries — the last task of each env is simply cut mid-way, and the TD objectives read boundaries from done codes wherever they land. Since map diversity now comes from per-task regeneration rather than env count, `NUM_ENVS` drops from `1000` to `50` everywhere, and offline and online use identical budgets — 50 envs × 30,000 steps per env = 1.5M transitions and 20,000 optimizer updates — so the two pipelines compare apples to apples (collection exposes this as `STEPS_PER_ENV = 30,000`, replacing `TOTAL_STEPS`).
- `examples/04_inference.ipynb` evaluates a **task budget**: each env runs `TASKS_PER_ENV` (4) held-out tasks and reports points per task (0–20) plus the mean from `env.metrics.task_cum_rewards`. The eval loop keeps per-env current-task row lists and rebuilds the shared batched KV cache with one ragged prefill whenever any env crosses a task boundary (done code 3/4), so context never leaks across maps. Every env uses `render_mode="rgb_array"` and gets one continuous full-length replay MP4 with frames concatenated across the whole eval run.
- Example notebooks that interact with live envs (`01`, `03`, `04`) report **task-level progress** from `mouse-gym`'s `env.metrics.task_cum_rewards` / `task_lengths` (recorded automatically when done codes 3/4 fire) instead of hand-rolling task totals or averaging per-episode rewards.
- Example notebooks enable `permute_obs` / `permute_actions` on every Procedural Frozen Lake env (training, collection, and held-out evaluation): each map carries its own fixed random relabeling of observation and action ids, sampled with the map from `map_seed`. No id has consistent meaning across maps, so models cannot memorize id-level layouts and must infer each map's labeling from context. `info["q_star"]` is reported in external (permuted) action order, so oracle collection is unaffected.

### Fixed
- BFloat16 GPU inference no longer raises `mat1 and mat2 must have the same dtype` after cached decode: pooled hidden states are cast to float32 at the head boundary, and `Model.to(..., dtype=...)` keeps output heads in float32 while encoder/backbone may use bfloat16 or float16 on GPU.
- `FlexDecodeSession` no longer compiles FlexAttention on CUDA float32 (Inductor has no kernel for that path, which broke `examples/04_inference.ipynb` ragged prefills). CUDA cached decode now compiles only for bfloat16/float16 and falls back to the eager flex path if compilation still fails; `04_inference.ipynb` loads the model in bfloat16 on GPU automatically.
- The `examples` extra now includes `matplotlib`, which `04_inference.ipynb` imports for replay animations; a fresh `scripts/install.sh` setup previously failed on that notebook with `ModuleNotFoundError`.
- Example notebooks that construct envs (`01`, `03`, `04`) now use the updated Procedural Frozen Lake constructor: `width`/`height` instead of removed `min_width`/`max_width`/`min_height`/`max_height`, and `slippery_success_rate=1.0` for deterministic movement (the env default is stochastic ice).
- `SpObjective` backward-direction losses (`kl-bwd`, `ce-soft-bwd`) no longer blow up to infinity when the Q targets contain `-inf` padding sentinels. Student logits are now masked to `-inf` at padded actions before `log_softmax`, so the student distribution renormalizes over valid actions in all soft loss types (`js`, `kl-*`, `ce-soft-*`). Label smoothing on soft losses now spreads mass over each row's valid actions only, keeping padded actions at exactly zero teacher probability.
- `VecDqnObjective` no longer mutates `predictions["action_vector_target"]` in place when substituting boundary target vectors; callers that read the target predictions after the objective (or invoke it twice) now see the original values.
- `Augmenter` now raises `ValueError` when direct `scale_*` / `shift_*` parameters are set on a `discrete` modality instead of silently ignoring them (they only apply to `image`).
- Online training notebook `03` previously restarted every rollout at env `0`, so with the default budgets only the first 10 of `NUM_ENVS` environments were ever visited. Rollouts now cycle through envs round-robin via a persistent `env_cursor`, so all configured environments contribute training data.

### Added
- Loading pretrained backbone weights (`Qwen3Backbone(pretrained=...)`, `LlamaBackbone(pretrained=...)`) now warns with the names of any backbone tensors that did not receive checkpoint weights (missing or shape-mismatched), instead of silently leaving them randomly initialized.
- `examples/06_train_offline_vec_dqn.ipynb`: offline Vector-DQN training (`VectorActionValueHead`, `vec_dim=2`, `VecDqnObjective`), same Hub → DataLoader → train flow as `02`.
- `LayerwiseDiscreteActionValueHead` and `action_value_layerwise` model head: one DQN value head per backbone layer, reading pooled hidden states from every transformer block.
- `LayerwiseDqnObjective`: layer `0` uses each `gamma_*_start`; the deepest layer uses the matching deep gamma. Intermediate layers get **linearly increasing effective horizon** `H = 1/(1-γ)` from shallow to deep endpoints (`gamma_l = 1 - 1/H_l`).
- `examples/05_train_offline_layerwise_dqn.ipynb`: offline layerwise DQN training (`LayerwiseDiscreteActionValueHead`, `LayerwiseDqnObjective`), same Hub → DataLoader → train flow as `02`.
- Backbones accept `output_hidden_states=True` and return per-layer hidden states (transformer block outputs) as a third return value.
- `DataLoader.refresh()` re-snapshots underlying stores and drains any prefetched batches, so online replay can pick up newly appended rows without rebuilding the loader.
- `DataLoader` accepts an optional `seed` argument (default `None`) that controls its internal NumPy RNG; when set, each worker's sampling RNG and forked augmenter receive independent child seeds derived via `numpy.random.SeedSequence(seed).spawn(...)`, so multi-worker sampling is deterministic and no two streams (across workers, or between sampling and augmentation) coincide.
- `StepEmbedder` accepts a new `type_embedding_std` parameter to control the initialisation std of the type embedding table independently from the content embedding `std`. **Required when `include_type_token=True`**; raises `ValueError` if omitted to prevent accidental type-to-content signal imbalance.

### Removed
- The `experiments/` folder (online experiment runner, summarizer, stage scripts, and result JSONs). The example notebooks are the single supported path for training and evaluation.
- The `cache_position` parameter on `Model.forward`, `Model.backbone_forward`, and every backbone `forward`. Positions for incremental KV-cache decoding are inferred automatically from the cached sequence length (verified to match full-sequence forwards bit-for-bit in tests).
- `ModernBertBackbone` (bidirectional ModernBERT adapter) and bidirectional teacher masking experiments.

### Changed
- `SpObjective` and `SvObjective` accept `targets_key` (default `"info_q_star"`) to select the per-action Q target column in `objective_data`, parallel to `DqnObjective`'s `action_key` / `reward_key` / `done_key`.

### Fixed
- `examples/03_train_online.ipynb` collect/train loop no longer consumes all env steps in a single rollout; `ENV_STEPS_PER_CYCLE` caps each cycle so DQN updates run repeatedly, and epsilon is recomputed per env transition instead of once per cycle.

### Changed
- `LayerwiseDqnObjective` metrics now include per-layer TD loss (`layer_{i}_loss`) alongside `layer_{i}_q_mean`.
- `LayerwiseDqnObjective` discount schedule is horizon-linear (even spacing in `1/(1-γ)`), replacing the earlier buildup curve.
- Example notebooks share `GRADIENT_STEPS` as the total optimizer-step budget (`TRAIN_STEPS` renamed in `02_train_offline.ipynb`). Online adds `ENV_STEPS_PER_CYCLE`, `STEPS_PER_ENV`, and `GRADIENT_STEPS_PER_CYCLE` to interleave live env interaction with the same number of gradient updates as offline training. `EXPLORATION_FULL_AT_STEP` renamed to `EXPLORATION_ENDS`.
- README quick start now points to the example notebooks instead of an inline training skeleton.
- `DataLoader` with `pack=True` allows empty stores at construction and on `refresh()`; `next_batch()` raises if every store is still empty.
- Example notebooks updated for `mouse-env` 0.5.0: `make_env` now creates a `SingleEnv`, `make_group_env` creates a `GroupEnv`, and `sample_random_input()` replaces `sample_random_inputs()`.
- `examples/03_train_online.ipynb` collection keeps one KV cache per env visit (`STEPS_PER_ENV` transitions), growing it incrementally and discarding it when moving to the next env; the context deque may slide independently.
- `examples/03_train_online.ipynb` collection now processes one env at a time with a single KV cache (discarded after each env's `STEPS_PER_ENV`) instead of windowing envs into `ENV_BATCH_SIZE` parallel caches; step counters now count env transitions directly.
- `examples/03_train_online.ipynb` reorganizes collection and training into separate documented sections with `collect_round` and `train_round` helpers; episode stats print after each collect round instead of a separate evaluation phase.
- `SequenceAugmenter` renamed to `Augmenter` in `mouse_core.data`.
- `load_model` now defaults to `force_download=True`, always pulling the latest checkpoint from the Hub instead of serving a cached copy.
- Online training example notebooks (`03`, `05`, `06`) now use standard ε-greedy linear decay from `1.0` to `0.0` by `EXPLORATION_ENDS`, replacing the inverted 0→1 ramp.
- `StepEmbedder` now uses `modality_fusion="sum"` or `"concat"` instead of the old `concat_modalities` boolean; the example notebooks use explicit sum fusion.
- `examples/02_train_offline.ipynb` and `examples/03_train_online.ipynb` now include final kernel shutdown cells to release all notebook-owned GPU memory after training.
- `DqnObjective`: replaced `use_episodic_reward` with an explicit `reward_key: str = "reward"` parameter; added `done_key: str = "done"` parameter (parallel to `action_key`).
- `DqnObjective`: removed `normalize_reward_mean`, `normalize_reward_std`, `normalize_reward_eps`, `normalize_reward_std_target`, `reward_scale`, and `reward_shift`.
- `DqnObjective`: discount is now computed via a vectorized gamma lookup (`gammas[done_next]`) instead of four boolean masks; the reset-frame bootstrap workaround is removed.
- `DqnObjective`: all S-1 consecutive pairs within a sampled sequence are now trained; the done-based valid-transition mask that excluded transitions from terminal states is removed.
- `StepEmbedder` now raises `ValueError` when `include_type_token=True` and `type_embedding_std` is not provided, forcing intentional configuration of the type embedding scale.
- `examples/02_train_offline.ipynb` and `examples/03_train_online.ipynb` now set `include_type_token=False` on `StepEmbedder`, keeping the summed token purely content-driven.
- `load_stores_from_hub` now snapshots matching dataset parquet shards once and loads local exact file paths in one `datasets.load_dataset("parquet", data_files=...)` call, avoiding one Hub tree/glob request per store.
- `examples/04_inference.ipynb` now separates evaluation env count from replay output count, allowing 100-env evaluation while only embedding the first 10 replay videos.
- FrozenLake training examples now use the full pretrained `Qwen/Qwen3-0.6B` backbone instead of truncating `Qwen3Backbone` with `num_layers=2`.
- `examples/03_train_online.ipynb` now uses the tuned offline baseline's training horizon, context length, batch size, learnable-token scale, and action-value head scale while keeping data collection online.

### Fixed
- `DataLoader` now snapshots both loaded source rows and newly appended rows from each `Datastore`, so replay built from Hub data plus live rollout does not silently ignore recent experience.
- `StepEmbedder` now raises when a required modality is missing from only some rows in a batch instead of silently filling those rows with default values.
- `DqnObjective` and `VecDqnObjective` now skip invalid reset-frame transitions after boundary rows and use the following reset row for boundary bootstrapping when available.
- `SpObjective` now supports `-inf` padded invalid actions in `info_q_star` while still rejecting NaN and `+inf` targets.
- `examples/04_inference.ipynb` now renders only the replay envs created with `render_mode="rgb_array"`, avoiding Gymnasium render-mode warnings from non-video evaluation envs.
- `examples/02_train_offline.ipynb` now passes AdamW beta coefficients with PyTorch's supported `betas` keyword.

## [0.4.0] - 2026-06-25

### Added
- `push_stores_to_hub` now batches named store configs into a single Hub commit.
- `mouse_core.__version__` now exposes the installed `mouse-core` package version from package metadata.
- Added `examples/03_train_online.ipynb`, showing live environment training with epsilon-greedy action selection, `Datastore` replay buffers, `DataLoader` sampling over appended experience, DQN updates, and task-boundary policy-context resets.
- `Augmenter` now provides the public training-time augmentation path for raw `DataLoader` batches, configured by modality specs that use `field` plus an augmentation `type`. A modality `field` may name multiple fields to share the same sampled permutation and per-step mask decision, linear numeric fields use `scale_in_low`, `scale_out_low`, `scale_in_high`, and `scale_out_high` endpoint pairs to derive the reward/value affine transform, and `DataLoader(..., augmenter=augment)` runs augmentation inside the sampling path so worker processes can prepare augmented batches.
- `StepEmbedder` modality specs may now use a tuple/list `field` to encode multiple raw fields with the same modality settings. Learnable scratch modalities no longer need a fake data field; use `{"type": "learnable", "tokens": N}`.
- Notebook 01 now collects expert demonstrations using `q_star_source={"provider": "env_q_star"}` and a curriculum policy (`oracle_prob` 0 → 0.5 → 1.0 across three collection phases).
- `Encoder.forward` now returns a third value `prediction_indices [B, S]` — the absolute flat-sequence position of the prediction token for each step. `Encoder.pool_step_reprs` now takes this tensor instead of `batch_size`, enabling future variable-token-count modalities without any change to the pooling logic.

### Changed
- `push_stores_to_hub(clear=True)` now replaces the full dataset repository contents instead of only clearing dataset shard/card files.
- `DqnObjective` now uses `gamma_step` for running transitions instead of `gamma`; the example DQN configs bootstrap through all done codes by setting episode and task boundary discounts to `0.99`.
- FrozenLake examples now use deterministic 50-step episodes inside one infinite task (`episodes_per_task=0`, `is_slippery=False`, `max_episode_steps=50`) across collection, online training, and inference.
- Example notebooks now flow from dataset collection, to offline training, to online training, to inference. Offline and online training both push to the shared `mouse-example-model` Hub repo, and `examples/04_inference.ipynb` loads whichever checkpoint is currently published there.
- Modality specs now use the public key `type` instead of `embed` in both `StepEmbedder` and `Augmenter` configs.
- Expert Q-values from `mouse_envs` are now read from the environment-produced key `info_q_star`, following the `mouse-env` output contract that forwards `info["q_star"]` as `info_q_star`. `SpObjective`, `SvObjective`, and the collection notebook (`01_collect_dataset`) now use that single field.
- Example notebooks migrated from `FrozenLake-v1` to `Procedural-FrozenLake-v1` with a step penalty (`step_penalty=-0.04`) so the expert Q-table properly distinguishes shorter from longer paths.
- Example notebook 02 trains with `DqnObjective` using separate step, episode-boundary, and task-boundary discount factors. q_star labels from the curriculum are used to guide data collection but not as a direct training signal.
- Notebook 04 now runs a 20-episode success-rate evaluation before the frame-capture animation, and carries the KV-cache across episode boundaries.
- `DqnObjective` now accepts `gamma_task_terminal` and `gamma_task_truncated` parameters to set the bootstrap discount for `done==3` (task terminated) and `done==4` (task truncated) transitions separately from within-task episode boundaries.
- Objectives are now classes (`DqnObjective`, `VecDqnObjective`, `SpObjective`, `SvObjective`) instantiated with their hyperparameters and called with `(objective_data, predictions)`. The separate `*ObjectiveConfig` dataclasses and `*_objective` functions are removed.
- `EnvConfig` now requires `episodes_per_task` (number of episodes per MOUSE task) instead of `max_episode_steps`. Examples updated accordingly.
- `env.step()` and `env.sample_random_inputs()` now return a flat `list[dict]` — one dict per slot across all configs — instead of a nested `list[(list[dict], _)]` per config.
- `done` field now carries 5 codes (`0`=running, `1`=episode terminated, `2`=episode truncated, `3`=task terminated, `4`=task truncated) instead of 3; `StepEmbedder` `done` modality updated to `vocab_size=5`.
- Moved `docs/` reference documentation into the example notebooks; each notebook now explains the relevant concepts inline as it walks through them.
- Moved project logo (`mouse-core.png`) to the repo root.
- Updated README to link to examples instead of standalone doc files.

### Removed
- Removed the old tensor-level `TokenAugmenter` public API in favor of the raw sequence augmentation module.
- `StepEmbedder` removed `num_compute_tokens`; use a `{"type": "learnable", "tokens": N}` modality placed last in the list to get the same dedicated scratch/prediction token.
- `DataLoader` no longer accepts a `sampling` parameter; random windowed sampling is always used.

### Fixed
- `Model.forward` no longer calls `TensorDict.to()` when encoder outputs are already on the target device, avoiding a spurious CUDA availability probe during CPU-only runs.
- `push_stores_to_hub` and `push_to_hub` were writing every config's parquet data to the same path (`data/train-*.parquet`) because `data_dir="data"` was passed explicitly to `DatasetDict.push_to_hub`, overriding the per-config subdirectory behaviour introduced in datasets v5. Each config now writes to its own `data/{config_name}/` directory, so multiple subsets are no longer silently overwritten by the last one pushed.
- `push_stores_to_hub` and `push_to_hub` `clear` parameter now correctly defaults to `True` (the docstring-stated default), ensuring stale parquet shards are wiped before each fresh push.
