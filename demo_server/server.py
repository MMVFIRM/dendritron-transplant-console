"""Dependency-light local HTTP server for the demo console."""
from __future__ import annotations

from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
import traceback
from typing import Any
from urllib.parse import urlparse

from .engine import DemoEngine


class DemoHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], root: Path, engine: DemoEngine) -> None:
        super().__init__(address, DemoRequestHandler)
        self.root = root
        self.web_root = root / "web"
        self.engine = engine


class DemoRequestHandler(BaseHTTPRequestHandler):
    server: DemoHTTPServer
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: Any) -> None:
        print(f"[demo] {self.address_string()} - {format % args}")

    def _json(self, value: Any, status: int = HTTPStatus.OK) -> None:
        payload = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self.send_response(int(status))
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(payload)

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length < 0 or length > 1_000_000:
            raise ValueError("request body is too large")
        raw = self.rfile.read(length)
        if not raw:
            return {}
        value = json.loads(raw.decode("utf-8"))
        if not isinstance(value, dict):
            raise ValueError("JSON request must be an object")
        return value

    def _serve_static(self, path: str) -> None:
        requested = "index.html" if path in {"", "/"} else path.lstrip("/")
        candidate = (self.server.web_root / requested).resolve()
        if self.server.web_root.resolve() not in candidate.parents and candidate != self.server.web_root.resolve():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        if not candidate.is_file():
            candidate = self.server.web_root / "index.html"
        content = candidate.read_bytes()
        content_type = mimetypes.guess_type(candidate.name)[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", f"{content_type}; charset=utf-8" if content_type.startswith("text/") else content_type)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'self'; style-src 'self' 'unsafe-inline'; script-src 'self'; img-src 'self' data:; connect-src 'self'")
        self.end_headers()
        self.wfile.write(content)


    def do_HEAD(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path in {"/api/health", "/api/status", "/api/benchmark"}:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", "0")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            return
        requested = "index.html" if path in {"", "/"} else path.lstrip("/")
        candidate = (self.server.web_root / requested).resolve()
        if self.server.web_root.resolve() not in candidate.parents and candidate != self.server.web_root.resolve():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        if not candidate.is_file():
            candidate = self.server.web_root / "index.html"
        content_type = mimetypes.guess_type(candidate.name)[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", f"{content_type}; charset=utf-8" if content_type.startswith("text/") else content_type)
        self.send_header("Content-Length", str(candidate.stat().st_size))
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'self'; style-src 'self' 'unsafe-inline'; script-src 'self'; img-src 'self' data:; connect-src 'self'")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        try:
            if path == "/api/status":
                self._json(self.server.engine.status())
            elif path == "/api/benchmark":
                self._json(self.server.engine.benchmark_report())
            elif path == "/api/health":
                self._json({"ok": True, "engine": "live"})
            else:
                self._serve_static(path)
        except Exception as error:  # pragma: no cover - defensive server boundary
            traceback.print_exc()
            self._json({"error": type(error).__name__, "message": str(error)}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        try:
            request = self._read_json()
            if path == "/api/analyze":
                self._json(self.server.engine.analyze(request))
            elif path == "/api/compare":
                self._json(self.server.engine.compare(request))
            else:
                self._json({"error": "not_found"}, HTTPStatus.NOT_FOUND)
        except (ValueError, KeyError, json.JSONDecodeError) as error:
            self._json({"error": type(error).__name__, "message": str(error)}, HTTPStatus.BAD_REQUEST)
        except Exception as error:  # pragma: no cover - defensive server boundary
            traceback.print_exc()
            self._json({"error": type(error).__name__, "message": str(error)}, HTTPStatus.INTERNAL_SERVER_ERROR)
