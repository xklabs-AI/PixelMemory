#!/usr/bin/env python3
"""
PixelMemory Platform Launcher
Unified launcher for starting the search server, exploring demo data, ingesting photos,
and checking system diagnostics.
"""

import os
import sys
import time
import socket
import argparse
import subprocess
import webbrowser
import shutil
import threading
from pathlib import Path
from urllib.request import urlopen
from urllib.error import URLError

WORKSPACE_ROOT = Path(__file__).parent.resolve()
VENV_PYTHON = WORKSPACE_ROOT / ".venv" / "Scripts" / "python.exe"
if not VENV_PYTHON.exists():
    VENV_PYTHON = WORKSPACE_ROOT / ".venv" / "bin" / "python"

# ── Self-relaunch into virtual environment or Python 3.12 if needed ──────────
script_path = str(Path(__file__).resolve())
if VENV_PYTHON.exists() and Path(sys.executable).resolve() != VENV_PYTHON.resolve():
    code = subprocess.run([str(VENV_PYTHON), script_path] + sys.argv[1:]).returncode
    sys.exit(code)
elif not VENV_PYTHON.exists():
    py312 = Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Python" / "Python312" / "python.exe"
    if py312.exists() and Path(sys.executable).resolve() != py312.resolve():
        code = subprocess.run([str(py312), script_path] + sys.argv[1:]).returncode
        sys.exit(code)


# ── Ensure Cargo bin is in PATH for Tauri ────────────────────────────────────
CARGO_BIN = Path.home() / ".cargo" / "bin"
if CARGO_BIN.exists() and str(CARGO_BIN) not in os.environ.get("PATH", ""):
    os.environ["PATH"] = f"{CARGO_BIN}{os.pathsep}{os.environ.get('PATH', '')}"

# Add workspace to Python path so `backend` imports resolve cleanly
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from backend.config import DATA_DIR, DB_PATH, THUMB_DIR, ZVEC_DIR, PORT, HOST
from backend.db import init_db, get_conn, get_stats


# ── Color & Styling Helpers ──────────────────────────────────────────────────
IS_WIN = sys.platform == "win32"
if IS_WIN:
    # Enable ANSI escape sequences and UTF-8 console output on Windows
    os.system("")
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

C_RESET = "\033[0m"
C_BOLD = "\033[1m"
C_DIM = "\033[2m"
C_CYAN = "\033[36m"
C_GREEN = "\033[32m"
C_YELLOW = "\033[33m"
C_MAGENTA = "\033[35m"
C_RED = "\033[31m"
C_PURPLE = "\033[38;5;141m"


def banner():
    print(rf"""
{C_PURPLE}{C_BOLD}  ___ _         _ __  __                           
 | _ (_)_ _____| |  \/  |___ _ __  ___ _ _ _  _    
 |  _/ \ \ / -_) | |\/| / -_) '  \/ _ \ '_| || |   
 |_| |_/_\_\___|_|_|  |_\___|_|_|_\___/_|  \_, |   
                                           |__/    {C_RESET}
  {C_DIM}Local Semantic Photo Search · Privacy Preserving · Zero Cloud Dependencies{C_RESET}
""")


def is_port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0


def wait_for_server(url: str, timeout: float = 12.0) -> bool:
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            with urlopen(url, timeout=0.8) as resp:
                if resp.status == 200:
                    return True
        except (URLError, TimeoutError, ConnectionRefusedError, OSError):
            time.sleep(0.3)
    return False


def get_hardware_info() -> dict:
    info = {"gpu_available": False, "gpu_name": "None (CPU only)", "vram_gb": 0.0}
    try:
        import torch
        if torch.cuda.is_available():
            info["gpu_available"] = True
            info["gpu_name"] = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            info["vram_gb"] = round(props.total_memory / (1024 ** 3), 1)
    except Exception:
        pass
    return info


def print_system_status():
    init_db()
    hw = get_hardware_info()

    with get_conn() as conn:
        stats = get_stats(conn)

    vector_count = 0
    try:
        from backend.search import SemanticSearch
        vector_count = SemanticSearch().count
    except Exception:
        pass

    thumb_count = 0
    thumb_size_mb = 0.0
    if THUMB_DIR.exists():
        thumbs = list(THUMB_DIR.glob("*.jpg"))
        thumb_count = len(thumbs)
        thumb_size_mb = round(sum(f.stat().st_size for f in thumbs) / (1024 * 1024), 2)

    db_size_mb = round(DB_PATH.stat().st_size / (1024 * 1024), 2) if DB_PATH.exists() else 0.0

    print(f"\n{C_BOLD}── Hardware & Environment ──────────────────────────────────────────{C_RESET}")
    print(f"  Python:       {C_CYAN}{sys.version.split()[0]}{C_RESET} ({sys.executable})")
    if hw["gpu_available"]:
        print(f"  GPU Compute:  {C_GREEN}✔ Enabled{C_RESET} ({hw['gpu_name']} - {hw['vram_gb']} GB VRAM)")
    else:
        print(f"  GPU Compute:  {C_YELLOW}⚠ Disabled{C_RESET} (Running in CPU mode)")

    from backend.describer import get_installed_ollama_models
    from backend.config import OLLAMA_HOST, DEFAULT_VLM_MODEL, STORY_LLM_MODEL
    installed_models = get_installed_ollama_models()
    has_moondream = any("moondream" in m.lower() for m in installed_models)
    has_gemma = any("gemma4:e2b" in m.lower() or "gemma4" in m.lower() for m in installed_models)

    if installed_models:
        vlm_s = f"{C_GREEN}✔ moondream ready{C_RESET}" if has_moondream else f"{C_YELLOW}⚠ missing (run: ollama pull moondream){C_RESET}"
        story_s = f"{C_GREEN}✔ gemma4:e2b ready{C_RESET}" if has_gemma else f"{C_YELLOW}⚠ missing (run: ollama pull gemma4:e2b){C_RESET}"
        print(f"  Ollama AI:    {C_GREEN}✔ Connected{C_RESET} ({OLLAMA_HOST})")
        print(f"    • Vision:   {vlm_s}")
        print(f"    • Stories:  {story_s}")
    else:
        print(f"  Ollama AI:    {C_YELLOW}⚠ Offline or Not Running{C_RESET} (Fallback: local HuggingFace PyTorch)")
        print(f"                Tip: Start Ollama or install from https://ollama.com")

    print(f"\n{C_BOLD}── Storage & Library Stats ─────────────────────────────────────────{C_RESET}")
    print(f"  Data Root:    {DATA_DIR}")
    print(f"  Database:     {stats.get('total', 0)} total images (DB: {db_size_mb} MB)")
    print(f"  Processed:    {stats.get('metadata_done', 0)} with metadata | {stats.get('described', 0)} described by VLM")
    print(f"  Searchable:   {C_GREEN}{vector_count}{C_RESET} vectors in Zvec")
    print(f"  Thumbnails:   {thumb_count} cached images ({thumb_size_mb} MB)")
    print(f"{C_BOLD}────────────────────────────────────────────────────────────────────{C_RESET}\n")


def start_server(port: int = PORT, host: str = HOST, auto_open: bool = True):
    print(f"\n{C_PURPLE}{C_BOLD}Starting PixelMemory Platform...{C_RESET}")
    init_db()

    target_url = f"http://localhost:{port}"

    if is_port_in_use(port, "127.0.0.1"):
        print(f"{C_YELLOW}Port {port} is already active.{C_RESET}")
        if auto_open:
            print(f"Opening browser at {C_CYAN}{target_url}{C_RESET}...")
            webbrowser.open(target_url)
        return

    # Check if database is empty to inform the user
    with get_conn() as conn:
        stats = get_stats(conn)
    total_imgs = stats.get("total", 0)
    if total_imgs == 0:
        print(f"{C_YELLOW}ℹ Note: Archive currently has 0 indexed photos.{C_RESET}")
        print(f"  You can click {C_BOLD}'✨ Load Demo Archive'{C_RESET} in the web UI to test with sample memories immediately.\n")
    else:
        print(f"{C_GREEN}✔ Found {total_imgs} indexed photos ready for search.{C_RESET}\n")

    print(f"Serving web application at: {C_BOLD}{C_CYAN}{target_url}{C_RESET}")
    print(f"{C_DIM}Press Ctrl+C in this terminal to shut down the server.{C_RESET}\n")

    # Start browser opener in background thread once server starts responding
    if auto_open:
        import threading
        def _opener():
            if wait_for_server(f"{target_url}/api/stats"):
                time.sleep(0.3)
                webbrowser.open(target_url)
        threading.Thread(target=_opener, daemon=True).start()

    # Launch uvicorn
    import uvicorn
    from backend.server import app
    try:
        uvicorn.run(app, host=host, port=port, log_level="info")
    except KeyboardInterrupt:
        print(f"\n{C_YELLOW}PixelMemory server stopped.{C_RESET}")


def seed_demo(auto_open: bool = True, port: int = PORT):
    print(f"\n{C_PURPLE}{C_BOLD}Seeding Quick-Start Demo Archive...{C_RESET}")
    from backend.demo import seed_demo_archive
    res = seed_demo_archive(clear_existing=False)
    print(f"{C_GREEN}✔ Successfully seeded {res['seeded']} sample memories!{C_RESET}")
    print(f"  Total searchable vectors: {C_BOLD}{res['total_vectors']}{C_RESET}")
    print(f"  Demo photos directory:    {res['demo_dir']}\n")

    start_server(port=port, auto_open=auto_open)


def run_ingest(directory: str, skip_describe: bool = False):
    p = Path(directory)
    if not p.is_dir():
        print(f"{C_RED}Error: '{directory}' is not a valid directory.{C_RESET}")
        return

    print(f"\n{C_PURPLE}{C_BOLD}Ingesting Photo Directory: {p.resolve()}{C_RESET}")
    from backend.ingest import main as ingest_main
    old_argv = sys.argv
    try:
        cmd_args = ["ingest.py", str(p)]
        if skip_describe:
            cmd_args.append("--skip-describe")
        sys.argv = cmd_args
        ingest_main()
    finally:
        sys.argv = old_argv


def clear_database():
    confirm = input(f"{C_RED}{C_BOLD}Are you sure you want to clear all indexed photos from the database? (y/N): {C_RESET}").strip().lower()
    if confirm == "y":
        with get_conn() as conn:
            conn.execute("DELETE FROM images")
        try:
            from backend.search import SemanticSearch
            s = SemanticSearch()
            s._client.delete_collection("image_descriptions")
            s._collection = s._client.get_or_create_collection(
                name="image_descriptions",
                metadata={"hnsw:space": "cosine"},
            )
        except Exception:
            pass
        print(f"{C_GREEN}Database and vector index cleared.{C_RESET}\n")
    else:
        print("Cancelled.")


def check_desktop_prerequisites() -> bool:
    """Verify that cargo and build prerequisites are available."""
    cargo_bin = Path.home() / ".cargo" / "bin"
    if cargo_bin.exists() and str(cargo_bin) not in os.environ.get("PATH", ""):
        os.environ["PATH"] = f"{cargo_bin}{os.pathsep}{os.environ.get('PATH', '')}"

    if shutil.which("cargo") is None:
        print(f"\n{C_RED}Error: Cargo was not found in PATH.{C_RESET}")
        print(f"Please restart your terminal to reload environment variables, or ensure Rust is installed.\n")
        return False

    has_msvc = shutil.which("link") is not None
    vswhere = Path(r"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe")
    if not has_msvc and vswhere.exists():
        try:
            out = subprocess.check_output([str(vswhere), "-latest", "-requires", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64"], text=True)
            if out.strip():
                has_msvc = True
        except Exception:
            pass

    release_exe = WORKSPACE_ROOT / "src-tauri" / "target" / "release" / "PixelMemory.exe"
    debug_exe = WORKSPACE_ROOT / "src-tauri" / "target" / "debug" / "PixelMemory.exe"
    if not has_msvc and not release_exe.exists() and not debug_exe.exists():
        print(f"\n{C_YELLOW}⚠ Microsoft C++ Build Tools (MSVC link.exe) not detected.{C_RESET}")
        print(f"  Tauri requires C++ Build Tools to compile native desktop binaries.")
        print(f"  To install in an elevated (Admin) terminal, run:")
        print(f"    {C_CYAN}winget install Microsoft.VisualStudio.2022.BuildTools --override \"--passive --wait --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended\"{C_RESET}")
        print(f"  Or download from: {C_CYAN}https://aka.ms/vs/17/release/vs_BuildTools.exe{C_RESET}\n")
        print(f"  {C_BOLD}[1]{C_RESET} Launch in Web Browser Mode (FastAPI)")
        print(f"  {C_BOLD}[2]{C_RESET} Try compiling anyway with Tauri")
        print(f"  {C_BOLD}[0]{C_RESET} Exit\n")
        choice = input(f"{C_BOLD}Select an option [1, 2, or 0, default 1]: {C_RESET}").strip()
        if choice == "2":
            return True
        elif choice == "0":
            return False
        else:
            start_server(auto_open=True)
            return False

    return True


def start_desktop_app(port: int = PORT, host: str = "127.0.0.1"):
    """
    Launch PixelMemory Native Desktop Application (Tauri).
    Ensures backend server is active, waits for health check, and launches Tauri.
    """
    if not check_desktop_prerequisites():
        return

    print(f"\n{C_PURPLE}{C_BOLD}Starting PixelMemory Desktop App (Tauri)...{C_RESET}")
    init_db()

    target_url = f"http://127.0.0.1:{port}"

    if not is_port_in_use(port, "127.0.0.1"):
        print(f"Starting local AI backend server on {target_url}...")
        import uvicorn
        from backend.server import app

        server_err = None

        def _run_server():
            nonlocal server_err
            try:
                config = uvicorn.Config(app=app, host="127.0.0.1", port=port, log_level="warning")
                server = uvicorn.Server(config)
                server.run()
            except BaseException as e:
                server_err = e

        server_thread = threading.Thread(target=_run_server, daemon=True)
        server_thread.start()

        # Wait with visible progress indicator and fail-fast on server errors
        start_wait = time.time()
        connected = False
        print("Waiting for backend server to become ready", end="", flush=True)
        while time.time() - start_wait < 45.0:
            if server_err:
                print(f"\n{C_RED}Backend server encountered an error: {server_err}{C_RESET}")
                return
            if is_port_in_use(port, "127.0.0.1"):
                try:
                    with urlopen(f"{target_url}/api/stats", timeout=1.0) as resp:
                        if resp.status == 200:
                            connected = True
                            print(f" {C_GREEN}✔ ready!{C_RESET}")
                            break
                except Exception:
                    pass
            print(".", end="", flush=True)
            time.sleep(0.6)

        if not connected:
            print(f"\n{C_RED}Failed to connect to backend server at {target_url}{C_RESET}")
            return
        print(f"{C_GREEN}✔ Backend server initialized and ready.{C_RESET}")
    else:
        print(f"{C_GREEN}✔ Backend server already running on port {port}.{C_RESET}")

    # Check for compiled Tauri executable or run dev via npm
    candidates = [
        WORKSPACE_ROOT / "src-tauri" / "target" / "release" / "PixelMemory.exe",
        WORKSPACE_ROOT / "src-tauri" / "target" / "release" / "app.exe",
        WORKSPACE_ROOT / "src-tauri" / "target" / "debug" / "PixelMemory.exe",
        WORKSPACE_ROOT / "src-tauri" / "target" / "debug" / "app.exe",
    ]
    binary_to_run = next((p for p in candidates if p.exists()), None)

    try:
        if binary_to_run:
            print(f"Launching desktop binary: {binary_to_run.name} ({binary_to_run.parent.name} build)...")
            subprocess.run([str(binary_to_run)], cwd=str(WORKSPACE_ROOT))
        else:
            print(f"Launching Tauri in development mode ({C_CYAN}npm run tauri:dev{C_RESET})...")
            subprocess.run("npm run tauri:dev", shell=True, cwd=str(WORKSPACE_ROOT))
    except KeyboardInterrupt:
        print(f"\n{C_YELLOW}Desktop app closed.{C_RESET}")
    finally:
        print(f"{C_GREEN}✔ Desktop session finished.{C_RESET}")


def check_or_pull_ollama_models():
    """Verify Ollama service and prompt to pull default models if missing."""
    print(f"\n{C_PURPLE}{C_BOLD}── Ollama Local AI Diagnostics & Setup ─────────────────────────────{C_RESET}")
    from backend.describer import get_installed_ollama_models
    from backend.config import OLLAMA_HOST
    installed = get_installed_ollama_models()
    if not installed:
        print(f"{C_YELLOW}⚠ Ollama service is not responding at {OLLAMA_HOST}.{C_RESET}")
        print("  Please make sure Ollama is installed and running:")
        if IS_WIN:
            print(f"    • Install via winget:   {C_CYAN}winget install Ollama.Ollama{C_RESET}")
            print(f"    • Download installer:   {C_CYAN}https://ollama.com/download/windows{C_RESET}")
        else:
            print(f"    • Install via Homebrew: {C_CYAN}brew install --cask ollama{C_RESET}")
            print(f"    • Download installer:   {C_CYAN}https://ollama.com/download/mac{C_RESET}")
        return

    print(f"{C_GREEN}✔ Ollama service is active at {OLLAMA_HOST}.{C_RESET}")
    has_moondream = any("moondream" in m.lower() for m in installed)
    has_gemma = any("gemma4:e2b" in m.lower() or "gemma4" in m.lower() for m in installed)

    print(f"  • Vision Model (Moondream2):  " + (f"{C_GREEN}✔ Installed{C_RESET}" if has_moondream else f"{C_YELLOW}⚠ Not found{C_RESET}"))
    print(f"  • Story Model (Gemma 4 2B):   " + (f"{C_GREEN}✔ Installed{C_RESET}" if has_gemma else f"{C_YELLOW}⚠ Not found{C_RESET}"))

    missing = []
    if not has_moondream:
        missing.append("moondream")
    if not has_gemma:
        missing.append("gemma4:e2b")

    if missing:
        pull = input(f"\n{C_BOLD}Would you like to pull missing models ({', '.join(missing)}) now? (Y/n): {C_RESET}").strip().lower()
        if pull not in ("n", "no"):
            for m in missing:
                print(f"\n{C_CYAN}Pulling {m}...{C_RESET}")
                subprocess.run(["ollama", "pull", m])
            print(f"\n{C_GREEN}✔ All models updated successfully!{C_RESET}")
    else:
        print(f"\n{C_GREEN}✔ All default models (moondream, gemma4:e2b) are installed and ready!{C_RESET}")


def interactive_menu():
    while True:
        banner()
        hw = get_hardware_info()
        gpu_str = f"{C_GREEN}{hw['gpu_name']}{C_RESET}" if hw["gpu_available"] else f"{C_YELLOW}CPU Mode{C_RESET}"
        print(f"  Hardware: {gpu_str}  ·  Server: {C_CYAN}http://localhost:{PORT}{C_RESET}\n")

        print(f"  {C_BOLD}[1]{C_RESET} {C_GREEN}▶  Start Web Platform & Open Browser{C_RESET}")
        print(f"  {C_BOLD}[2]{C_RESET} {C_PURPLE}✨ Launch with Quick Demo Archive (12 Sample Photos){C_RESET}")
        print(f"  {C_BOLD}[3]{C_RESET} 📁 Ingest a Local Photo Directory")
        print(f"  {C_BOLD}[4]{C_RESET} 📊 System Health & Library Diagnostics")
        print(f"  {C_BOLD}[5]{C_RESET} 🗑️  Clear / Reset Database")
        print(f"  {C_BOLD}[6]{C_RESET} {C_CYAN}🖥️  Launch Native Desktop App (Tauri){C_RESET}")
        print(f"  {C_BOLD}[7]{C_RESET} 🦙 Ollama AI Models & Setup Check")
        print(f"  {C_BOLD}[0]{C_RESET} 🚪 Exit\n")

        choice = input(f"{C_BOLD}Select an option [1-7, or Enter for 1]: {C_RESET}").strip()
        if choice in ("", "1"):
            start_server(auto_open=True)
            break
        elif choice == "2":
            seed_demo(auto_open=True)
            break
        elif choice == "3":
            dir_path = input(f"\n{C_BOLD}Enter path to photo directory (or drag & drop folder): {C_RESET}").strip().strip('"\'')
            if dir_path:
                skip = input(f"Fast mode (skip VLM GPU description)? (y/N): ").strip().lower() == "y"
                run_ingest(dir_path, skip_describe=skip)
                input(f"\n{C_DIM}Press Enter to return to menu...{C_RESET}")
        elif choice == "4":
            print_system_status()
            input(f"{C_DIM}Press Enter to return to menu...{C_RESET}")
        elif choice == "5":
            clear_database()
            input(f"{C_DIM}Press Enter to return to menu...{C_RESET}")
        elif choice == "6":
            start_desktop_app()
            break
        elif choice == "7":
            check_or_pull_ollama_models()
            input(f"\n{C_DIM}Press Enter to return to menu...{C_RESET}")
        elif choice == "0":
            print("Goodbye!")
            break


def main():
    parser = argparse.ArgumentParser(description="PixelMemory Platform Launcher")
    parser.add_argument("--serve", action="store_true", help="Start FastAPI web platform and open browser")
    parser.add_argument("--demo", action="store_true", help="Seed demo photos and launch web platform")
    parser.add_argument("--ingest", metavar="DIR", help="Ingest a photo directory")
    parser.add_argument("--skip-describe", action="store_true", help="Skip VLM descriptions during ingest")
    parser.add_argument("--status", action="store_true", help="Display system and library diagnostics and exit")
    parser.add_argument("--desktop", "--tauri", dest="desktop", action="store_true", help="Launch Native Desktop Application (Tauri)")
    parser.add_argument("--doctor", "--setup", dest="doctor", action="store_true", help="Check and setup local Ollama AI models")
    parser.add_argument("--port", type=int, default=PORT, help=f"Server port (default: {PORT})")
    parser.add_argument("--host", default=HOST, help=f"Server host (default: {HOST})")
    parser.add_argument("--no-browser", action="store_true", help="Do not automatically open web browser")
    args = parser.parse_args()

    if args.status:
        banner()
        print_system_status()
    elif args.doctor:
        banner()
        check_or_pull_ollama_models()
    elif args.desktop:
        start_desktop_app(port=args.port, host=args.host)
    elif args.demo:
        seed_demo(auto_open=not args.no_browser, port=args.port)
    elif args.ingest:
        run_ingest(args.ingest, skip_describe=args.skip_describe)
    elif args.serve:
        start_server(port=args.port, host=args.host, auto_open=not args.no_browser)
    else:
        interactive_menu()


if __name__ == "__main__":
    main()
