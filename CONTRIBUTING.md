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
source install.sh
```

This installs the package in editable mode with the `dev` and `all` extras (`all` bundles every feature extra — `examples` and `kernels`). Activate with `source .venv/bin/activate`. The install uses standard CPython (`3.15`). The stable PyTorch/Triton pair comes from the release index (`cu132` when `nvidia-smi` is present, CPU otherwise). See [Example dependencies](#example-dependencies) for notebook system packages.

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

### Running examples locally

Open notebooks under `examples/` in Jupyter and configure their explicit model,
tokenizer, and environment settings before running them. Keep publication cells
unexecuted unless you intend to publish their outputs.

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
   pytest
   ```
   CI runs source/test type checks, tests, and a wheel/sdist build on every pull request and on pushes to `main`.
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

The development environment uses stable PyTorch `2.15.0` and Triton `3.9.0`
with standard Python 3.15. Use the official CUDA 13.2 release index
`https://download.pytorch.org/whl/cu132`, or `https://download.pytorch.org/whl/cpu`
for CPU installations. Triton 3.9 includes the Python 3.15 `AnnAssign.simple`
fix and declares Python 3.15 support; no installed-package patch is needed.
The installer upgrades dependencies to the newest releases allowed by the
package requirements. Run `uv pip check` and the test suite after installation.
