#!/usr/bin/env bash
# ==============================================================================
# PixelMemory — Launcher for macOS & Linux
# ==============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Ensure Cargo is in PATH if installed
if [ -d "$HOME/.cargo/bin" ] && [[ ":$PATH:" != *":$HOME/.cargo/bin:"* ]]; then
    export PATH="$HOME/.cargo/bin:$PATH"
fi

# Auto-run setup if virtual environment is missing
if [ ! -f ".venv/bin/python" ]; then
    echo "[!] Virtual environment (.venv) not found. Running initial setup..."
    bash "$SCRIPT_DIR/setup.sh"
fi

PYTHON_EXE=".venv/bin/python"
if [ ! -f "$PYTHON_EXE" ]; then
    PYTHON_EXE="python3"
fi

exec "$PYTHON_EXE" "$SCRIPT_DIR/launch.py" "$@"
