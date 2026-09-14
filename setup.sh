#!/usr/bin/env bash
# ==============================================================================
# PixelMemory — Automated Setup Script (macOS & Linux)
# ==============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "====================================================================="
echo "  PixelMemory Platform Setup (macOS / Linux)"
echo "  Local Semantic Photo Search · Privacy Preserving · Zero Cloud"
echo "====================================================================="
echo ""

# 1. Detect Python 3.10 - 3.12
PYTHON_BIN=""
for candidate in python3.12 python3.11 python3.10 python3; do
    if command -v "$candidate" >/dev/null 2>&1; then
        VER=$("$candidate" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
        MAJOR=$(echo "$VER" | cut -d. -f1)
        MINOR=$(echo "$VER" | cut -d. -f2)
        if [ "$MAJOR" -eq 3 ] && [ "$MINOR" -ge 10 ]; then
            PYTHON_BIN="$candidate"
            break
        fi
    fi
done

if [ -z "$PYTHON_BIN" ]; then
    echo "[!] Python 3.10-3.12 was not found on your system."
    if [[ "$OSTYPE" == "darwin"* ]]; then
        echo "On macOS, install Python via Homebrew:"
        echo "    brew install python@3.12"
    else
        echo "On Linux (Ubuntu/Debian):"
        echo "    sudo apt update && sudo apt install -y python3 python3-venv python3-pip"
    fi
    exit 1
fi

echo "[OK] Found Python: $($PYTHON_BIN --version) ($PYTHON_BIN)"

# 2. Setup Virtual Environment
if [ ! -f ".venv/bin/python" ]; then
    echo ""
    echo "[*] Creating isolated virtual environment in .venv..."
    "$PYTHON_BIN" -m venv .venv
    echo "[OK] Virtual environment created."
else
    echo "[OK] Existing virtual environment found."
fi
VENV_PY=".venv/bin/python"

# 3. Install Python Dependencies
echo ""
echo "[*] Installing / upgrading Python dependencies (requirements.txt)..."
"$VENV_PY" -m pip install --upgrade pip --quiet
"$VENV_PY" -m pip install -r requirements.txt --quiet
echo "[OK] All Python dependencies installed successfully."

# 4. Check Ollama Installation & Service
echo ""
echo "── Ollama Local AI Check ──────────────────────────────────────────"
if ! command -v ollama >/dev/null 2>&1; then
    echo "[!] Ollama was not detected on your system."
    echo "    Ollama is used for high-speed local GPU/Apple Neural Engine vision"
    echo "    (Moondream2) and travel story generation (Gemma 4)."
    echo ""
    if [[ "$OSTYPE" == "darwin"* ]]; then
        echo "To install Ollama on macOS via Homebrew, run:"
        echo "    brew install --cask ollama"
        echo "Or download directly from: https://ollama.com/download/mac"
    else
        echo "To install Ollama on Linux, run:"
        echo "    curl -fsSL https://ollama.com/install.sh | sh"
    fi
else
    echo "[OK] Ollama is installed."
    # Ensure Ollama daemon is active
    if ! curl -s "http://localhost:11434/api/tags" >/dev/null 2>&1; then
        echo "[*] Starting Ollama background service..."
        if [[ "$OSTYPE" == "darwin"* ]]; then
            open -a Ollama || ollama serve >/dev/null 2>&1 &
        else
            ollama serve >/dev/null 2>&1 &
        fi
        sleep 3
    fi

    echo "[*] Ensuring Ollama models are pulled:"
    echo "    • Vision Model:   moondream (Moondream2)"
    echo "    • Story Model:    gemma4:e2b (Gemma 4 2B)"
    echo ""
    echo "Pulling 'moondream' (vision VLM)..."
    ollama pull moondream || true
    echo ""
    echo "Pulling 'gemma4:e2b' (story LLM)..."
    ollama pull gemma4:e2b || true
    echo "[OK] Ollama models ready."
fi

# 5. Check Desktop Build Prerequisites (Tauri)
echo ""
echo "── Native Desktop (Tauri) Prerequisites ───────────────────────────"
export PATH="$HOME/.cargo/bin:$PATH"
if command -v cargo >/dev/null 2>&1; then
    echo "[OK] Rust/Cargo detected ($(cargo --version))."
else
    echo "[i] Rust/Cargo not found. Desktop mode requires Rust (https://rustup.rs)."
fi

if command -v npm >/dev/null 2>&1; then
    if [ ! -d "node_modules" ]; then
        echo "[*] Installing Tauri frontend dependencies..."
        npm install --silent
    fi
    echo "[OK] Tauri npm dependencies installed."
fi

echo ""
echo "====================================================================="
echo "  [OK] PixelMemory Setup Complete!"
echo "====================================================================="
echo ""
echo "You can now launch PixelMemory anytime:"
echo ""
echo "  • Launch Native Desktop App:   ./launch.sh --desktop"
echo "  • Launch Web Browser Mode:     ./launch.sh"
echo "  • System Diagnostics:          ./launch.sh --status"
echo ""
