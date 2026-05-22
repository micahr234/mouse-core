# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
while pre-1.0 APIs may change without notice.

## [Unreleased]

### Changed

- Repo layout aligned with [mouse-env](https://github.com/micahr234/mouse-env): flat `docs/` (guide, architecture, data, losses, examples, mouse_env), README doc table, no doc build.
- Removed `docs/api/`, `docs/index.md`, and separate getting-started page; content consolidated into `docs/guide.md`.

### Removed

- Zensical / mkdocstrings docs build; GitHub Pages docs workflow.

### Added

- `docs/guide.md`, `docs/mouse_env.md`, runnable `examples/`, `tests/`, CI workflow.
- Public exports from `mouse.data`, `py.typed`, PyPI publish version check.

## [0.1.0] - 2025-01-01

Initial alpha release on PyPI as `mouse-core`.
