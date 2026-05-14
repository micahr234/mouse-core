#!/bin/bash
# Publish mouse-core to PyPI.
#
# Usage:
#   ./scripts/publish.sh           # build + upload to PyPI
#   ./scripts/publish.sh --test    # build + upload to TestPyPI (dry run)
#
# Requirements:
#   pip install build twine
#
# Authentication:
#   Set TWINE_USERNAME and TWINE_PASSWORD (or TWINE_TOKEN) as environment
#   variables, or twine will prompt interactively.
#   Recommended: use an API token — set TWINE_USERNAME=__token__ and
#   TWINE_PASSWORD=<your-pypi-api-token>.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

log()   { echo "[INFO] $1"; }
warn()  { echo "[WARN] $1"; }
error() { echo "[ERROR] $1" >&2; }

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------

TEST_PYPI=false
for arg in "$@"; do
    case "$arg" in
        --test) TEST_PYPI=true ;;
        *) error "Unknown argument: $arg"; exit 1 ;;
    esac
done

# ---------------------------------------------------------------------------
# Check dependencies
# ---------------------------------------------------------------------------

# Prefer the project venv, then an already-active venv, then system Python.
if [ -f "$REPO_ROOT/.venv/bin/python" ]; then
    PYTHON="$REPO_ROOT/.venv/bin/python"
    TWINE="$REPO_ROOT/.venv/bin/twine"
elif [ -n "${VIRTUAL_ENV:-}" ] && [ -f "$VIRTUAL_ENV/bin/python" ]; then
    PYTHON="$VIRTUAL_ENV/bin/python"
    TWINE="$VIRTUAL_ENV/bin/twine"
else
    PYTHON="$(command -v python3 || command -v python || true)"
    TWINE="$(command -v twine || true)"
fi

if [ -z "$PYTHON" ] || ! "$PYTHON" -c "" &>/dev/null; then
    error "No Python interpreter found. Run: source scripts/install.sh"
    exit 1
fi

if [ -z "$TWINE" ] || ! "$TWINE" --version &>/dev/null; then
    error "twine not found. Run: source scripts/install.sh"
    exit 1
fi

if ! "$PYTHON" -c "import build" &>/dev/null; then
    error "'build' module not found. Run: source scripts/install.sh"
    exit 1
fi

# ---------------------------------------------------------------------------
# Clean previous builds
# ---------------------------------------------------------------------------

log "Cleaning dist/"
rm -rf dist/

# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

log "Building wheel and sdist..."
"$PYTHON" -m build

log "Built packages:"
ls -lh dist/

# ---------------------------------------------------------------------------
# Check
# ---------------------------------------------------------------------------

log "Running twine check..."
"$TWINE" check dist/*

# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------

if [ -z "${PYPI_TOKEN:-}" ]; then
    error "PYPI_TOKEN is not set. Export it before running this script:"
    error "  export PYPI_TOKEN=pypi-your-token-here"
    exit 1
fi

export TWINE_USERNAME=__token__
export TWINE_PASSWORD="$PYPI_TOKEN"

if [ "$TEST_PYPI" = true ]; then
    log "Uploading to TestPyPI (https://test.pypi.org)..."
    "$TWINE" upload --repository testpypi dist/*
    log ""
    log "Install from TestPyPI with:"
    log "  pip install --index-url https://test.pypi.org/simple/ mouse"
else
    log "Uploading to PyPI..."
    "$TWINE" upload dist/*
    log ""
    log "Install with: pip install mouse"
fi
