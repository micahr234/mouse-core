#!/bin/bash
# Install dependencies and set up the dev environment.
# Run with: source scripts/install.sh
#
# Uses `return` throughout (never `exit`) because this script is sourced —
# `exit` would terminate the user's shell.

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

# Install uv package manager
install_uv() {
    log "Installing uv package manager..."

    if command -v uv >/dev/null 2>&1; then
        success "uv is already installed: $(uv --version)"
        return
    fi

    if ! curl -LsSf https://astral.sh/uv/install.sh | sh; then
        error "Failed to install uv."
        return 1
    fi

    export PATH="$HOME/.local/bin:$PATH"

    if ! command -v uv >/dev/null 2>&1; then
        error "Failed to install uv."
        return 1
    fi

    success "uv installed successfully: $(uv --version)"
}

# Create and setup virtual environment
setup_venv() {
    log "Creating virtual environment..."

    if [ -d ".venv" ]; then
        warn "Removing existing virtual environment..."
        rm -rf .venv
    fi

    # Free-threaded 3.14t so DataLoader(num_workers>0) can use real thread parallelism.
    log "Ensuring free-threaded Python 3.14t is available..."
    if ! uv python install 3.14t; then
        error "Failed to install Python 3.14t"
        return 1
    fi

    if ! uv venv --python 3.14t; then
        error "Failed to create virtual environment"
        return 1
    fi

    success "Virtual environment created"

    # TEMPORARY: Triton still re-enables the GIL on import (no Py_mod_gil slot).
    # tokenizers>=0.23.2 is free-thread-safe. Drop PYTHON_GIL=0 when Triton
    # declares Py_MOD_GIL_NOT_USED — see CONTRIBUTING.md.
    if ! grep -qxF 'export PYTHON_GIL=0' .venv/bin/activate 2>/dev/null; then
        echo 'export PYTHON_GIL=0' >> .venv/bin/activate
    fi

    # Install project dependencies (core + all optional extras).
    log "Installing project dependencies..."
    # --refresh bypasses uv's cache so branch-pinned git extras (mouse-gym@main,
    # procedural-frozenlake@main) re-resolve to the current head.
    if ! uv pip install -e ".[dev,all]" --python .venv/bin/python --index-strategy unsafe-best-match --refresh; then
        error "Failed to install project dependencies"
        return 1
    fi

    success "Project dependencies installed"
}

# Main installation process: uv, venv, project dependencies
main() {
    cd "$(dirname "${BASH_SOURCE[0]}")/.." || return 1

    echo "Starting Installation"
    echo "=================================="

    log "Installing packages..."
    install_uv || return 1
    setup_venv || return 1

    echo ""
    echo "Installation complete!"
    echo ""
    log "Activate the virtual environment:"
    echo "  source .venv/bin/activate"
}

main "$@"
