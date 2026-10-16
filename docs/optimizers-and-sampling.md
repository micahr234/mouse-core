# Optimizers and action sampling

`from mouse_core import AdamP` exposes the same parameter-group and checkpoint
interface as `AdamW`. Build groups with `model.parameter_groups(lr={...})`.
AdamP trains fp32 parameters and ignores frozen parameters. `fused=False` uses
PyTorch; `fused=True` requires supported contiguous CUDA tensors; the default
automatically chooses CUDA kernels where supported and falls back otherwise.
AdamP checkpoints can move between CPU and CUDA, but are not interchangeable
with AdamW checkpoints.

`mouse_core.sampling` provides:

- `TaskTemperature(low, high, rng)`: draw uniformly once for each task in each
  stream. Inner episode boundaries do not change the temperature.
- `sample_tempered_actions(scores, temperatures, rng)`: sample from temperature
  scaled scores and return actions and their behavior log-probabilities. Zero
  temperature selects the first maximum deterministically with log-probability
  zero. Sampling uses the supplied NumPy generator.
- `tempered_policy_logits(scores, temperatures)`: differentiable logits for the
  same distribution, for policy objectives comparing current and collected
  action probabilities. Zero-temperature rows have zero policy gradient.
- `last_action_scores(scores, valid, num_actions)`: select the final valid
  readout in each decode stream and restrict it to legal actions.

Task identity, online/evaluation schedules, and the choice of temperature bounds
remain the caller's responsibility. For greedy evaluation use zero temperature.
