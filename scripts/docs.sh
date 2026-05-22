#!/bin/bash
# Serve the Zensical documentation with live-reload.
#
# Usage:
#   ./scripts/docs.sh    # live-reload dev server at http://127.0.0.1:8000

log() {
    echo "[INFO] $1"
}

error() {
    echo "[ERROR] $1"
}

# ---------------------------------------------------------------------------
# Resolve zensical binary
# ---------------------------------------------------------------------------

find_zensical() {
    if [ -f ".venv/bin/zensical" ]; then
        echo ".venv/bin/zensical"
    elif command -v zensical >/dev/null 2>&1; then
        echo "zensical"
    else
        echo ""
    fi
}

# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

cd "$(dirname "$0")/.." || exit 1

ZENSICAL_BIN=$(find_zensical)
if [ -z "$ZENSICAL_BIN" ]; then
    error "zensical not found."
    log  "Install the docs extras first:"
    log  "  source scripts/install.sh"
    exit 1
fi

log "Starting live-reload server at http://127.0.0.1:8000 (Ctrl-C to stop) ..."
"$ZENSICAL_BIN" serve
