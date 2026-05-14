#!/bin/bash
# Install dependencies and set up the dev environment.
# Run with: source scripts/install.sh

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

# Setup git configuration
setup_git() {
    log "Setting up git configuration..."

    # Resolve the effective identity (local → global → system fallback).
    effective_email="$(git config user.email 2>/dev/null || true)"
    effective_name="$(git config user.name 2>/dev/null || true)"

    if [ -z "$effective_email" ]; then
        git config --local user.email "user@example.com"
        warn "No git email configured — set repo-local placeholder. Update with: git config user.email 'you@example.com'"
    fi

    if [ -z "$effective_name" ]; then
        git config --local user.name "User"
        warn "No git name configured — set repo-local placeholder. Update with: git config user.name 'Your Name'"
    fi

    success "Git configured: $(git config user.name) <$(git config user.email)>"
}

# Install uv package manager
install_uv() {
    log "Installing uv package manager..."
    
    # Check if uv is already installed
    if command -v uv >/dev/null 2>&1; then
        success "uv is already installed: $(uv --version)"
        return
    fi
    
    # Download and install uv
    if ! curl -LsSf https://astral.sh/uv/install.sh | sh; then
        error "Failed to install uv."
        exit 1
    fi
    
    # Add to PATH for current session
    export PATH="$HOME/.local/bin:$PATH"
    
    # Verify installation
    if ! command -v uv >/dev/null 2>&1; then
        error "Failed to install uv."
        exit 1
    fi
    
    success "uv installed successfully: $(uv --version)"
}

# Create and setup virtual environment
setup_venv() {
    log "Creating virtual environment..."
    
    # Remove existing venv if it exists
    if [ -d ".venv" ]; then
        warn "Removing existing virtual environment..."
        rm -rf .venv
    fi
    
    # Create new virtual environment
    if ! uv venv --python python3.12; then
        error "Failed to create virtual environment"
        exit 1
    fi
    
    success "Virtual environment created"
    
    # Install project dependencies (core + all optional extras).
    log "Installing project dependencies..."
    if ! uv pip install -e ".[dev,docs]" --python .venv/bin/python --index-strategy unsafe-best-match; then
        error "Failed to install project dependencies"
        exit 1
    fi

    success "Project dependencies installed"
}

# Main installation process: git, uv, venv, project dependencies
main() {
    echo "Starting Installation"
    echo "=================================="
    
    log "Installing packages..."
    setup_git
    install_uv
    setup_venv
    
    echo ""
    echo "Installation complete!"
}

# Run main function
main "$@"
