# Contributing to MOUSE Core

MOUSE is actively developed and contributions are very welcome — whether that's bug reports, new features, experiments, or documentation improvements.

## Ways to contribute

- **Bug reports** — open a GitHub issue with a minimal reproduction and the full error traceback.
- **Feature requests** — open an issue describing the use case. If you have a design idea, sketching it out in the issue first helps align before writing code.
- **Pull requests** — see the workflow below.
- **Experiments and results** — if you run MOUSE on a new environment or task, sharing results (even negative ones) as an issue or discussion is valuable.
- **Documentation** — edits to the README or inline notebook docs are welcome.

## Development setup

```bash
# Clone and create a standard Python 3.15 virtual environment (via uv)
git clone https://github.com/micahr234/mouse-core.git
cd mouse-core
source scripts/install.sh
```

This installs the package in editable mode with the `dev` and `all` extras (`all` bundles every feature extra — `examples` and `kernels`). Activate with `source .venv/bin/activate`. The install uses standard CPython (`3.15`). The pinned PyTorch/Triton pair comes from the nightly index (`cu130` when `nvidia-smi` is present, CPU otherwise), with the Python 3.15 source fix described below. See [Example dependencies](#example-dependencies) for notebook system packages.

### Example dependencies

The `examples` extra installs the environment packages, Jupyter, and plotting tools. The
setup script above already includes it. To add it to an existing repository environment:

```bash
uv pip install -e ".[examples]" --python .venv/bin/python
```

Gymnasium's toy-text environments depend on `pygame-ce`. If no wheel is available for
your platform, building it requires SDL2 and PortMidi development libraries.
On Debian/Ubuntu:

```bash
sudo apt install libsdl2-dev libsdl2-image-dev libsdl2-mixer-dev libsdl2-ttf-dev libportmidi-dev libfreetype6-dev
```

Missing headers cause errors such as `library 'SDL2' not found` or `library 'portmidi' not found`.
The extra also includes `gymnasium[mujoco]` for [example 20](examples/20_train_sequential_actions_dqn.ipynb),
which runs Reacher on CPU without rendering or a display.

### Data preparation workers

Use `DataLoader(..., num_workers=4, prefetch=4)` for four persistent processes
with bounded shared-memory transfer. Standard Python works; no GIL override is
needed. One event/token budget applies to the whole batch, and `next_batch()`
returns `(inputs, objective_data)`. Loss boundaries are available as
`objective_data.to(device)["sequence_ids"]`.

Spawned processes support notebook-local preprocessing through `cloudpickle`.
In Python scripts, construct loaders under `if __name__ == "__main__":`.
Transforms run on CPU and must serialize; keep models and external resources out
of their closures. Worker-local augmenter locks are rebuilt on deserialization.
`refresh()` replaces worker snapshots at the next unconsumed batch. `close()`
reclaims processes and shared memory; previously returned batches stay valid.

The package lives at `src/mouse_core/` (standard src layout) and is imported as `mouse_core`.

Notebooks under [`examples/`](examples/) are committed **without** cell outputs. Clear outputs before committing, e.g. `jupyter nbconvert --clear-output --inplace examples/*.ipynb`.

### Running all examples locally

Use an existing local Qwen3 tokenizer directory to run the notebooks without downloads or publishing:

```bash
.venv/bin/python scripts/run_examples.py \
    --output-dir /tmp/mouse-examples \
    --tokenizer-source /path/to/local/Qwen3-tokenizer \
    --threads 2 --seed 7 --examples all
```

The runner uses explicit short CPU settings and a small randomly initialized Qwen3 model.
It collects real datasets, trains each model, reloads checkpoints, and runs inference and
rendering. It writes per-notebook logs, settings, and results to the output directory.
Publication cells are skipped. This checks execution, not learning quality at full scale.
Tokenizer conditions live in each notebook. Define the same named conditions in an
inference notebook before loading its saved tokenizer.

Repeated-task examples save `repeat_episode_index` only on each inner episode's
reset environment event. Use `RepeatTaskTracker.repeat_episode_start` and its
current index at collection time; do not derive the index while replaying data.
Keep the one-based episode header and two-newline inner-ending separator
consistent across the FrozenLake tokenizer schemas. Only the action-request
newline is a prediction readout.

## Pull request workflow

1. Fork the repository and create a branch from `main`.
2. Make your changes. Keep commits focused — one logical change per commit.
3. Before opening a PR, run:
   ```bash
   pyright src/ tests/
   python scripts/check_example_types.py
   pytest
   ```
   The notebook check extracts code cells and reports type errors by notebook and cell. CI runs source/test type checks, tests, and a wheel/sdist build on every pull request and on pushes to `main`.
4. Open a pull request against `main` with a clear description of what changed and why.

If you add a new feature, include a short usage example in the PR description or in `examples/`. Add tests for observable behavior and failure cases; prefer small regression cases over assertions about source spelling or private implementation details. Notebook regressions should exercise real notebook functions with local models and short environments, without downloads, training runs, or publishing.

When working through a coding agent, complete and review changes locally before explicitly approving any push, release, or publication, as required by [AGENTS.md](AGENTS.md).

## Code style

- Python 3.15+, type-annotated throughout.
- Follow the existing patterns: base classes in `base.py`, public API in `__init__.py`, usage in `examples/`.
- Keep example notebook imports together at the top of the first code cell.
- Avoid silent fallbacks — if a precondition isn't met, raise a clear error.
- Make behavior settings explicit at call sites; avoid defaults that hide configuration. Optional data fields may remain absent.
- Comments should explain *why*, not *what*.

## Releasing to PyPI

Publishing is automated by [`.github/workflows/publish.yml`](.github/workflows/publish.yml) using [PyPI trusted publishing](https://docs.pypi.org/trusted-publishers/) (OIDC).

### Publishing a version

1. Bump `version` in `pyproject.toml` on `main`.
2. Commit, push, and create an annotated tag matching the version (e.g. `v0.1.1` for version `0.1.1`).
3. Push the tag: `git push origin v0.1.1` — the Publish workflow runs on tag push.

You can also run the workflow manually from the Actions tab (**Publish** → **Run workflow**).

## Questions

Open a GitHub Discussion or issue.

## Pinned PyTorch and Triton

The development environment pins PyTorch `2.15.0.dev20261005` and Triton
`3.8.0+gitc01b6774` from the official PyTorch nightly index, using standard
Python 3.15. This pair avoids the nanobind shutdown warning reproduced with
Triton 3.9. The installer and CI apply `scripts/patch_triton.py`, which backports
[Triton PR #10986](https://github.com/triton-lang/triton/pull/10986): the
`AnnAssign.simple` argument required by Python 3.15. Reapply this script after
manually reinstalling Triton. It checks the version and expected source before
editing the installed package.

The official `cp315` wheel still declares Python `<3.15`, so `uv pip check`
reports that upstream metadata mismatch. The metadata is left intact. This
setup depends on the explicit source fix; it was verified in V2B with compiled
CUDA FlexAttention forward/backward and the test suites. Use the nightly CPU
index `https://download.pytorch.org/whl/nightly/cpu` for CPU installations.
