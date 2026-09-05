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
from pathlib import Path
from urllib.request import urlopen
from urllib.error import URLError

WORKSPACE_ROOT = Path(__file__).parent.resolve()
VENV_PYTHON = WORKSPACE_ROOT / ".venv" / "Scripts" / "python.exe"
if not VENV_PYTHON.exists():
    VENV_PYTHON = WORKSPACE_ROOT / ".venv" / "bin" / "python"

# ── Self-relaunch into virtual environment if needed ──────────────────────────
if VENV_PYTHON.exists() and Path(sys.executable).resolve() != VENV_PYTHON.resolve():
    os.execv(str(VENV_PYTHON), [str(VENV_PYTHON)] + sys.argv)

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
    print(f"""
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

    from backend.describer import is_ollama_ready, get_active_vlm_model, get_available_vlm_models
    from backend.config import OLLAMA_HOST
    if is_ollama_ready():
        active_vlm = get_active_vlm_model()
        available = [m["id"] for m in get_available_vlm_models() if m.get("installed")]
        models_str = ", ".join(available) if available else active_vlm
        print(f"  Ollama VLM:   {C_GREEN}✔ Connected{C_RESET} ({OLLAMA_HOST} · Active: {C_BOLD}{active_vlm}{C_RESET} | Models: {models_str})")
    else:
        print(f"  Ollama VLM:   {C_YELLOW}⚠ Offline{C_RESET} (Fallback: local HuggingFace PyTorch)")

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
        print(f"  {C_BOLD}[0]{C_RESET} 🚪 Exit\n")

        choice = input(f"{C_BOLD}Select an option [1-5, or Enter for 1]: {C_RESET}").strip()
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
    parser.add_argument("--port", type=int, default=PORT, help=f"Server port (default: {PORT})")
    parser.add_argument("--host", default=HOST, help=f"Server host (default: {HOST})")
    parser.add_argument("--no-browser", action="store_true", help="Do not automatically open web browser")
    args = parser.parse_args()

    if args.status:
        banner()
        print_system_status()
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
