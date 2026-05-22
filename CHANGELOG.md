# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
while pre-1.0 APIs may change without notice.

## [Unreleased]

### Removed

- Zensical / mkdocstrings docs build; documentation is plain Markdown in `docs/`
- GitHub Pages docs workflow and `scripts/docs.sh`

### Added

- Runnable `examples/` scripts for dataset collection, offline training, and inference.
- `docs/getting-started.md` quick-start guide.
- `tests/` suite with CI (pyright, pytest, strict docs build).
- Public exports from `mouse.data`.
- `py.typed` marker for type checkers.
- GitHub issue and pull request templates.

### Changed

- Expanded README with quick start, naming note, and navigation links.
- Fixed `scripts/install.sh` (removed undefined `setup_git` call).
- Renamed `scripts/test_docs.sh` to `scripts/docs.sh`.

## [0.1.0] - 2025-01-01

Initial alpha release on PyPI as `mouse-core`.
