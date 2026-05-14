#!/bin/bash
# Installation script for project dependencies
# Run with: source install.sh


# Logging function
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

    if [ -z "$(git config --global user.email)" ]; then
        git config --global user.email "user@example.com"
        log "Set default git email (none was configured)"
    fi

    if [ -z "$(git config --global user.name)" ]; then
        git config --global user.name "User"
        log "Set default git name (none was configured)"
    fi

    success "Git configured: $(git config --global user.name) <$(git config --global user.email)>"
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
