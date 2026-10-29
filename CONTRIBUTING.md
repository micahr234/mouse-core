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
# Clone and create a free-threaded Python 3.15t virtual environment (via uv)
git clone https://github.com/micahr234/mouse-core.git
cd mouse-core
source scripts/install.sh
```

This installs the package in editable mode with the `dev` and `all` extras (`all` bundles every feature extra — currently `examples`). Activate with `source .venv/bin/activate`. The install uses free-threaded CPython (`3.15t`). `torch` 2.15 wheels for that interpreter come from the PyTorch test index (`cu132` when `nvidia-smi` is present, CPU otherwise). See [Example dependencies](#example-dependencies) for notebook system packages.

### Example dependencies

The `examples` extra installs the environment packages, Jupyter, and plotting tools. The
setup script above already includes it. To add it to an existing repository environment:

```bash
uv pip install -e ".[examples]" --python .venv/bin/python
```

Gymnasium's toy-text environments depend on `pygame-ce`. If no wheel is available for
free-threaded Python, building it requires SDL2 and PortMidi development libraries.
On Debian/Ubuntu:

```bash
sudo apt install libsdl2-dev libsdl2-image-dev libsdl2-mixer-dev libsdl2-ttf-dev libportmidi-dev libfreetype6-dev
```

Missing headers cause errors such as `library 'SDL2' not found` or `library 'portmidi' not found`.
The extra also includes `gymnasium[mujoco]` for [example 20](examples/20_train_sequential_actions_dqn.ipynb),
which runs Reacher on CPU without rendering or a display.

### Temporary `PYTHON_GIL=0` (remove when possible)

`scripts/install.sh` and CI set `PYTHON_GIL=0` so the free-threaded interpreter **keeps the GIL off** after imports. `tokenizers>=0.23.3` declares `Py_MOD_GIL_NOT_USED`. Triton still does not, so importing `triton._C.libtriton` (pulled in by torch) would otherwise re-enable the GIL.

**Follow-up:** when Triton declares `Py_MOD_GIL_NOT_USED`, drop `PYTHON_GIL=0` from [`scripts/install.sh`](scripts/install.sh) and [`.github/workflows/ci.yml`](.github/workflows/ci.yml).

**Risk:** forcing the GIL off skips CPython’s “this extension did not opt in” safety brake. That is fine for our usual path (read-only HF Dataset slices + numeric / text prepare). Triton’s autotune cache is still racy under `GIL=0` if Python threads launch kernels concurrently.

The package lives at `src/mouse_core/` (standard src layout) and is imported as `mouse_core`.

Notebooks under [`examples/`](examples/) are committed **without** cell outputs. Clear outputs before committing, e.g. `jupyter nbconvert --clear-output --inplace examples/*.ipynb`.

### Running all examples locally

Use an existing local Qwen3 tokenizer directory to run the notebooks without downloads or publishing:

```bash
PYTHON_GIL=0 .venv/bin/python scripts/run_examples.py \
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
   PYTHON_GIL=0 pytest
   ```
   The notebook check extracts code cells and reports type errors by notebook and cell. CI runs source/test type checks, tests, and a wheel/sdist build on every pull request and on pushes to `main`.
4. Open a pull request against `main` with a clear description of what changed and why.

If you add a new feature, include a short usage example in the PR description or in `examples/`. Add tests for observable behavior and failure cases; prefer small regression cases over assertions about source spelling or private implementation details. Notebook regressions should exercise real notebook functions with local models and short environments, without downloads, training runs, or publishing.

When working through a coding agent, complete and review changes locally before explicitly approving any push, release, or publication, as required by [AGENTS.md](AGENTS.md).

## Code style

- Python 3.15+ (free-threaded `3.15t` for multi-worker DataLoader), type-annotated throughout.
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
