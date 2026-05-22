# Contributing to MOUSE

MOUSE is actively developed and contributions are very welcome — whether that's bug reports, new features, experiments, or documentation improvements.

## Ways to contribute

- **Bug reports** — open a GitHub issue with a minimal reproduction and the full error traceback.
- **Feature requests** — open an issue describing the use case. If you have a design idea, sketching it out in the issue first helps align before writing code.
- **Pull requests** — see the workflow below.
- **Experiments and results** — if you run MOUSE on a new environment or task, sharing results (even negative ones) as an issue or discussion is valuable.
- **Documentation** — fixes to typos, clearer explanations, or new examples are all appreciated.

## Development setup

```bash
# Clone and create a virtual environment (Python 3.12, via uv)
git clone https://github.com/micahr234/mouse-core.git
cd mouse-core
source scripts/install.sh
```

This installs the package in editable mode with dev extras. Activate the venv with `source .venv/bin/activate`. Documentation is plain Markdown under `docs/` — edit those files directly; no build step.

## Pull request workflow

1. Fork the repository and create a branch from `main`.
2. Make your changes. Keep commits focused — one logical change per commit.
3. Before opening a PR, run:
   ```bash
   pyright src/ tests/
   pytest
   ```
   `scripts/install.sh` creates a `mouse -> src` symlink so Pyright can resolve `import mouse` (the symlink is gitignored).
4. Open a pull request against `main` with a clear description of what changed and why.

If you add a new feature, include a short usage example in the PR description or in `docs/` / `examples/`.

## Code style

- Python 3.12+, type-annotated throughout.
- Follow the existing patterns for new modules (base classes in `base.py`, public API in `__init__.py`, documentation in `docs/`).
- Avoid silent fallbacks — if a precondition isn't met, raise a clear error.
- Comments should explain *why*, not *what*.

## Questions

Open a GitHub Discussion or issue.
