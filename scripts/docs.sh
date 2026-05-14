#!/bin/bash
# Build or serve the MkDocs documentation.
#
# Usage:
#   ./scripts/docs.sh          # build static HTML → site/
#   ./scripts/docs.sh serve    # live-reload dev server at http://127.0.0.1:8000
#   ./scripts/docs.sh deploy   # build + push to gh-pages branch


log() {
    echo "[INFO] $1"
}

warn() {
    echo "[WARN] $1"
}

error() {
    echo "[ERROR] $1"
}

success() {
    echo "[SUCCESS] $1"
}

# ---------------------------------------------------------------------------
# Resolve mkdocs binary
# ---------------------------------------------------------------------------

find_zensical() {
    # Prefer the venv binary; fall back to whatever is on PATH.
    if [ -f ".venv/bin/zensical" ]; then
        echo ".venv/bin/zensical"
    elif command -v zensical >/dev/null 2>&1; then
        echo "zensical"
    else
        echo ""
    fi
}

# Sets ZENSICAL_BIN or exits with an error message.
resolve_zensical() {
    ZENSICAL_BIN=$(find_zensical)
    if [ -z "$ZENSICAL_BIN" ]; then
        error "zensical not found."
        log  "Install the docs extras first:"
        log  "  uv pip install -e '.[docs]' --python .venv/bin/python"
        exit 1
    fi
}

# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

cmd_build() {
    resolve_zensical
    log "Building docs → site/ ..."
    if "$ZENSICAL_BIN" build --strict; then
        success "Docs built successfully. Open site/index.html to preview."
    else
        error "zensical build failed."
        exit 1
    fi
}

cmd_serve() {
    resolve_zensical
    log "Starting live-reload server at http://127.0.0.1:8000 (Ctrl-C to stop) ..."
    "$ZENSICAL_BIN" serve
}

cmd_deploy() {
    resolve_zensical
    log "Deploying docs to gh-pages branch ..."
    if "$ZENSICAL_BIN" gh-deploy --force; then
        success "Docs deployed to gh-pages."
    else
        error "zensical gh-deploy failed."
        exit 1
    fi
}

# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

# Ensure we're always running from the repo root regardless of where the
# script was invoked from.
cd "$(dirname "$0")/.." || exit 1

case "${1:-build}" in
    build)   cmd_build  ;;
    serve)   cmd_serve  ;;
    deploy)  cmd_deploy ;;
    *)
        error "Unknown command: '$1'"
        echo "Usage: $0 [build|serve|deploy]"
        exit 1
        ;;
esac
