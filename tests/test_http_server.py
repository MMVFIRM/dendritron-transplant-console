from __future__ import annotations

import json
import threading
from urllib.request import Request, urlopen

from demo_server.engine import DemoEngine
from demo_server.server import DemoHTTPServer


def request_json(url: str, payload: dict | None = None) -> dict:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="GET" if data is None else "POST",
    )
    with urlopen(request, timeout=30) as response:  # noqa: S310 - local test server
        return json.loads(response.read().decode("utf-8"))


def test_http_api_and_static_frontend(root, engine: DemoEngine) -> None:
    server = DemoHTTPServer(("127.0.0.1", 0), root, engine)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    base = f"http://{host}:{port}"
    try:
        health = request_json(base + "/api/health")
        assert health == {"ok": True, "engine": "live"}
        status = request_json(base + "/api/status")
        assert status["ready"] is True
        analysis = request_json(
            base + "/api/analyze",
            {"example_id": "true_if_the", "mode": "correct", "max_new_tokens": 1},
        )
        assert analysis["steps"][0]["target"]["rank"] == 1
        with urlopen(base + "/", timeout=30) as response:  # noqa: S310
            html = response.read().decode("utf-8")
            assert "Dendritron Transplant Console" in html
            assert response.headers["Content-Security-Policy"]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=10)
