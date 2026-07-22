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
# Clone and create a free-threaded Python 3.14t virtual environment (via uv)
git clone https://github.com/micahr234/mouse-core.git
cd mouse-core
source scripts/install.sh
```

This installs the package in editable mode with the `dev` and `all` extras (`all` bundles every feature extra — currently `examples`). Activate with `source .venv/bin/activate`. The install uses free-threaded CPython (`3.14t`). System packages needed to run the notebooks are documented under **Example dependencies** in the [README](README.md#example-dependencies).

### Temporary `PYTHON_GIL=0` (remove when possible)

`scripts/install.sh` and CI set `PYTHON_GIL=0` so the free-threaded interpreter **keeps the GIL off** after imports. Today `transformers` still pins `tokenizers<=0.23.0`, and those builds re-enable the GIL on import. `tokenizers>=0.23.1` declares free-threading support, but we cannot depend on it until `transformers` allows that version.

**Follow-up:** when `transformers` accepts `tokenizers>=0.23.1`, pin that pair, drop `PYTHON_GIL=0` from [`scripts/install.sh`](scripts/install.sh) and [`.github/workflows/ci.yml`](.github/workflows/ci.yml), and tighten the DataLoader free-threading hint accordingly.

**Risk:** forcing the GIL off skips CPython’s “this extension did not opt in” safety brake. That is fine for our usual path (read-only HF Dataset slices + numeric prepare). It is riskier if multiple worker threads concurrently mutate shared C-extension state — the main case here is a shared Hugging Face `tokenizer` under `TextEmbedder` prepare. Prefer `num_workers=0` for text prepare until the follow-up lands, or treat multi-worker text as experimental.

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

- Python 3.14+ (free-threaded `3.14t` for multi-worker DataLoader), type-annotated throughout.
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
