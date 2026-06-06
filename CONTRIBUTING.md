# Contributing to MOUSE Core

MOUSE is actively developed and contributions are very welcome — whether that's bug reports, new features, experiments, or documentation improvements.

## Ways to contribute

- **Bug reports** — open a GitHub issue with a minimal reproduction and the full error traceback.
- **Feature requests** — open an issue describing the use case. If you have a design idea, sketching it out in the issue first helps align before writing code.
- **Pull requests** — see the workflow below.
- **Experiments and results** — if you run MOUSE on a new environment or task, sharing results (even negative ones) as an issue or discussion is valuable.
- **Documentation** — edits to Markdown under `docs/` or the README are welcome (no doc site build step).

## Development setup

```bash
# Clone and create a virtual environment (Python 3.12, via uv)
git clone https://github.com/micahr234/mouse-core.git
cd mouse-core
source scripts/install.sh
```

This installs the package in editable mode with the `dev` and `all` extras (`all` bundles every feature extra — currently `examples`, which adds Gymnasium for the [`examples/`](examples/) notebooks). Activate with `source .venv/bin/activate`.

The package lives at `src/mouse_core/` (standard src layout) and is imported as `mouse_core`.

Notebooks under [`examples/`](examples/) are committed **without** cell outputs. Clear outputs before committing, e.g. `jupyter nbconvert --clear-output --inplace examples/*.ipynb`.

## Pull request workflow

1. Fork the repository and create a branch from `main`.
2. Make your changes. Keep commits focused — one logical change per commit.
3. Before opening a PR, run:
   ```bash
   pyright src/ tests/
   pytest
   ```
4. Open a pull request against `main` with a clear description of what changed and why.

If you add a new feature, include a short usage example in the PR description or in the relevant `docs/` page / `examples/`.

## Code style

- Python 3.12+, type-annotated throughout.
- Follow the existing patterns: base classes in `base.py`, public API in `__init__.py`, documentation in `docs/`.
- Avoid silent fallbacks — if a precondition isn't met, raise a clear error.
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
