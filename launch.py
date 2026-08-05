#!/usr/bin/env python3
"""Launch the Dendritron Transplant Console on localhost."""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
import threading
import time
import webbrowser

from demo_server.engine import DemoEngine
from demo_server.server import DemoHTTPServer


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--threads", type=int, default=2, help="PyTorch CPU threads")
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--skip-verify", action="store_true", help="Skip startup asset hashing")
    return parser.parse_args()


def main() -> None:
    args = arguments()
    root = Path(__file__).resolve().parent
    if not args.skip_verify:
        subprocess.run(
            [sys.executable, str(root / "scripts" / "verify_assets.py")],
            cwd=root,
            check=True,
        )
    print("Loading sparse recipient, VIVERE cards, and frozen definition bank...")
    engine = DemoEngine(root, threads=args.threads)
    server = DemoHTTPServer((args.host, args.port), root, engine)
    url = f"http://{args.host}:{args.port}"
    print(f"Dendritron Transplant Console: {url}")
    print("The Qwen donor is not loaded; only the CPU recipient and frozen assets are live.")
    if not args.no_browser:
        def open_browser() -> None:
            time.sleep(0.7)
            webbrowser.open(url)
        threading.Thread(target=open_browser, daemon=True).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping demo.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
