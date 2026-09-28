#!/bin/bash
# Start a Cursor self-hosted worker for the Mouse checkouts inside tmux.
#
# Usage:
#   scripts/worker.sh            # name: mouse
#   scripts/worker.sh <name>     # override the worker name
#
# Expects sibling checkouts next to this repo:
#   <parent>/mouse-core
#   <parent>/mouse-experiment
#   <parent>/mouse-gym
#
# Run on the home server. The worker runs in a tmux session named
# mouse-worker so you can detach and reattach. Stop it with Ctrl-C inside
# the session. The worker opens an outbound HTTPS connection only.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PARENT="$(dirname "$HERE")"
SESSION="mouse-worker"
# Primary assignment identity first; then the other Mouse Project repos.
REPOS=(mouse-core mouse-experiment mouse-gym)

log() {
    echo "[INFO] $1"
}

error() {
    echo "[ERROR] $1" >&2
}

usage() {
    cat <<'EOF'
Usage:
  scripts/worker.sh            start a worker named mouse
  scripts/worker.sh <name>     start a worker with that name

Run on the home server. Clone mouse-core, mouse-experiment, and mouse-gym
as siblings under the same parent directory. The worker registers all three
checkouts (--worker-dir once each) and runs in the foreground inside a tmux
session named mouse-worker. Attach later with: tmux attach -t mouse-worker
Stop it with Ctrl-C inside the session. Outbound HTTPS only.

If mouse-worker already exists, the script attaches to it.
EOF
}

require_tmux() {
    if command -v tmux >/dev/null 2>&1; then
        return
    fi
    error "tmux is not installed."
    exit 1
}

require_agent() {
    if command -v agent >/dev/null 2>&1; then
        return
    fi
    error "agent CLI is not installed."
    echo "curl https://cursor.com/install -fsS | bash"
    exit 1
}

require_login() {
    local status lower rc
    set +e
    status="$(agent status 2>&1)"
    rc=$?
    set -e
    lower="$(printf '%s' "$status" | tr '[:upper:]' '[:lower:]')"
    if [[ "$lower" == *"not logged in"* || "$lower" == *"not authenticated"* ]]; then
        error "agent is not logged in. Run agent login once, then retry."
        exit 1
    fi
    if (( rc != 0 )); then
        error "agent is not logged in. Run agent login once, then retry."
        exit 1
    fi
}

resolve_worker_dirs() {
    local repo path
    WORKER_DIRS=()
    for repo in "${REPOS[@]}"; do
        path="$PARENT/$repo"
        if [[ ! -d "$path" ]]; then
            error "Missing sibling checkout: $path"
            error "Clone mouse-core, mouse-experiment, and mouse-gym as siblings under $PARENT"
            exit 1
        fi
        WORKER_DIRS+=("$path")
    done
}

session_exists() {
    tmux has-session -t "=$SESSION" 2>/dev/null
}

attach_session() {
    if [[ ! -t 0 || ! -t 1 ]]; then
        error "Session '$SESSION' is already running. Attach from a terminal: tmux attach -t $SESSION"
        exit 1
    fi
    log "Attaching to existing session '$SESSION'"
    exec tmux attach-session -t "=$SESSION"
}

start_session() {
    local name="$1"
    local args=()
    local dir
    for dir in "${WORKER_DIRS[@]}"; do
        args+=(--worker-dir "$dir")
    done
    log "Starting worker '$name' in tmux session '$SESSION' (${WORKER_DIRS[*]})"
    # new-session without -d creates the session, runs the worker in the
    # foreground pane, and attaches. Ctrl-C stops the worker; the session
    # ends when the worker exits. First --worker-dir is the assignment identity.
    exec tmux new-session -s "$SESSION" -c "${WORKER_DIRS[0]}" -- \
        agent worker start --name "$name" "${args[@]}"
}

main() {
    local name
    case "${1:-}" in
        -h|--help)
            usage
            exit 0
            ;;
    esac
    if [[ $# -gt 1 ]]; then
        error "usage: scripts/worker.sh [name]"
        exit 1
    fi
    name="${1:-mouse}"
    if [[ -z "$name" ]]; then
        name="mouse"
    fi
    require_tmux
    if session_exists; then
        attach_session
    fi
    resolve_worker_dirs
    require_agent
    require_login
    start_session "$name"
}

main "$@"
