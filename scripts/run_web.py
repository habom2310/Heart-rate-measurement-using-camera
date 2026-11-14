#!/usr/bin/env python3
"""Convenience launcher for the rPPG Web Application.

Checks if `web/dist` exists; if not, builds it or offers to launch `npm run dev`.
Serves the production build using Python's built-in HTTP server or forwards
to Vite development server.
"""

from __future__ import annotations

import argparse
import functools
import http.server
import shutil
import socketserver
import subprocess
import sys
import webbrowser
from pathlib import Path


def find_npm() -> str | None:
    """Find npm executable on PATH, handling Windows .cmd extension."""
    for cmd in ["npm.cmd", "npm"]:
        path = shutil.which(cmd)
        if path:
            return path
    return None


def run_npm_dev(web_dir: Path) -> None:
    """Launch Vite development server with npm run dev."""
    npm_path = find_npm()
    if not npm_path:
        print("[ERROR] 'npm' was not found on your system PATH.")
        print("Please install Node.js (https://nodejs.org) to run development mode.")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("  Starting rPPG Web Development Server (Vite)")
    print("  Local URL: http://localhost:5173")
    print("  Press Ctrl+C to terminate.")
    print("=" * 60 + "\n")

    try:
        subprocess.run([npm_path, "run", "dev"], cwd=web_dir)
    except KeyboardInterrupt:
        print("\n[INFO] Vite development server stopped.")


def build_web(web_dir: Path) -> bool:
    """Run npm run build in web directory."""
    npm_path = find_npm()
    if not npm_path:
        print("[ERROR] 'npm' was not found on your system PATH.")
        print("Cannot build web frontend without Node.js and npm.")
        return False

    print("[INFO] Building frontend assets (npm run build)...")
    try:
        ret = subprocess.run([npm_path, "run", "build"], cwd=web_dir)
        return ret.returncode == 0
    except Exception as e:
        print(f"[ERROR] Build failed: {e}")
        return False


def serve_dist(dist_dir: Path, port: int = 8000, open_browser: bool = True) -> None:
    """Serve web/dist directory using Python http.server."""
    handler_class = functools.partial(
        http.server.SimpleHTTPRequestHandler,
        directory=str(dist_dir),
    )

    # Allow port reuse immediately after restart
    socketserver.TCPServer.allow_reuse_address = True

    try:
        with socketserver.TCPServer(("", port), handler_class) as httpd:
            url = f"http://localhost:{port}"
            print("\n" + "=" * 60)
            print("  rPPG Web Application Server")
            print("=" * 60)
            print(f"  Serving build from : {dist_dir}")
            print(f"  Local URL          : {url}")
            print("  Press Ctrl+C to stop.")
            print("=" * 60 + "\n")

            if open_browser:
                try:
                    webbrowser.open(url)
                except Exception:
                    pass

            httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[INFO] Web server stopped.")
    except OSError as e:
        print(f"\n[ERROR] Could not start server on port {port}: {e}")
        print("Tip: Specify a different port using --port <port>")
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Launch or serve the rPPG Web Application."
    )
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Run Vite development server (npm run dev) at http://localhost:5173",
    )
    parser.add_argument(
        "--build",
        action="store_true",
        help="Force rebuild frontend assets before serving",
    )
    parser.add_argument(
        "--port",
        "-p",
        type=int,
        default=8000,
        help="Port to serve the production build on (default: 8000)",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not automatically open the web browser",
    )
    args = parser.parse_args()

    # Paths relative to script location
    repo_root = Path(__file__).resolve().parent.parent
    web_dir = repo_root / "web"
    dist_dir = web_dir / "dist"

    if not web_dir.is_dir():
        print(f"[ERROR] 'web' directory not found at {web_dir}")
        sys.exit(1)

    if args.dev:
        run_npm_dev(web_dir)
        return

    # Check dist directory
    index_html = dist_dir / "index.html"
    if args.build or not index_html.is_file():
        print(f"[INFO] Production build not found or rebuild requested at {dist_dir}")
        success = build_web(web_dir)
        if not success or not index_html.is_file():
            print("[WARN] Build unsuccessful. Falling back to dev server...")
            run_npm_dev(web_dir)
            return

    serve_dist(dist_dir, port=args.port, open_browser=not args.no_browser)


if __name__ == "__main__":
    main()
