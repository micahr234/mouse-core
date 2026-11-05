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

    # Standard Python; DataLoader uses spawned process workers.
    log "Ensuring standard Python 3.15 is available..."
    if ! uv python install 3.15; then
        error "Failed to install Python 3.15"
        return 1
    fi

    if ! uv venv --python 3.15; then
        error "Failed to create virtual environment"
        return 1
    fi

    success "Virtual environment created"

    # Install project dependencies (core + all optional extras).
    log "Installing project dependencies..."
    # Install the pinned Python 3.15 PyTorch/Triton nightly pair.
    if command -v nvidia-smi >/dev/null 2>&1; then
        TORCH_INDEX="https://download.pytorch.org/whl/nightly/cu130"
    else
        TORCH_INDEX="https://download.pytorch.org/whl/nightly/cpu"
    fi
    # --refresh bypasses uv's cache so branch-pinned git extras (mouse-gym@main,
    # procedural-frozenlake@main) re-resolve to the current head.
    if ! uv pip install -e ".[dev,all]" --python .venv/bin/python --index-strategy unsafe-best-match --extra-index-url "$TORCH_INDEX" --refresh; then
        error "Failed to install project dependencies"
        return 1
    fi

    .venv/bin/python scripts/patch_triton.py || return 1

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
