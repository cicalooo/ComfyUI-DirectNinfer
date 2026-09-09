from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading
import time

import pytest

from ninfer.client import (
    NInferClient,
    NInferClientError,
    NInferHTTPError,
    NInferProtocolError,
    NInferTimeoutError,
    build_chat_request,
    is_connection_reset_error,
    parse_chat_response,
)


class _ClientHandler(BaseHTTPRequestHandler):
    response_mode = "ok"
    received: dict = {}

    def log_message(self, *_args):
        return

    def do_GET(self):  # noqa: N802
        if self.path == "/health":
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"ok")
            return
        self.send_response(404)
        self.end_headers()

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length)
        if self.path == "/slow":
            time.sleep(0.25)
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"{}")
            return
        _ClientHandler.received = json.loads(raw.decode("utf-8"))
        if _ClientHandler.response_mode == "error":
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b'{"error":"bad request"}')
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(
            b'{"choices":[{"message":{"content":"answer","reasoning_content":"hidden"}}]}'
        )


def _server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _ClientHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def test_build_multimodal_request_and_reasoning_mapping():
    request = build_chat_request(
        model_id="qwen3.8-27b",
        system_prompt="Improve it.",
        user_prompt="A red fox",
        image_urls=["data:image/png;base64,AA==", "data:image/png;base64,BB=="],
        reasoning_mode="low",
        max_tokens=128,
        seed=7,
        repetition_penalty=1.1,
        frequency_penalty=0.2,
    )
    payload = request.payload()
    assert payload["model"] == "qwen3.8-27b"
    assert payload["reasoning_effort"] == "low"
    assert payload["seed"] == 7
    assert payload["repetition_penalty"] == pytest.approx(1.1)
    assert payload["messages"][1]["content"][0]["image_url"]["url"].startswith("data:")
    assert payload["messages"][1]["content"][-1]["text"] == "A red fox"


def test_disabled_reasoning_is_sent_as_none_and_content_excludes_reasoning():
    request = build_chat_request(
        model_id="qwen3.8-27b",
        system_prompt="",
        user_prompt="hello",
        reasoning_mode="disabled",
    )
    assert request.payload()["reasoning_effort"] == "none"
    response = parse_chat_response(
        {
            "choices": [
                {
                    "message": {
                        "content": "final only",
                        "reasoning_content": "private reasoning",
                    }
                }
            ]
        }
    )
    assert response.content == "final only"
    assert response.reasoning_content == "private reasoning"


def test_client_completion_and_http_error():
    server, thread = _server()
    try:
        client = NInferClient(f"http://127.0.0.1:{server.server_port}")
        request = build_chat_request(
            model_id="qwen3.8-27b",
            system_prompt="",
            user_prompt="hello",
        )
        response = client.complete(request)
        assert response.content == "answer"
        assert _ClientHandler.received["messages"][0]["content"] == "hello"

        _ClientHandler.response_mode = "error"
        with pytest.raises(NInferHTTPError) as error:
            client.complete(request)
        assert error.value.status == 400
    finally:
        _ClientHandler.response_mode = "ok"
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

def test_malformed_response_and_timeout():
    with pytest.raises(NInferProtocolError):
        parse_chat_response({"choices": []})

    server, thread = _server()
    try:
        client = NInferClient(f"http://127.0.0.1:{server.server_port}")
        with pytest.raises(NInferTimeoutError):
            client.request("POST", "/slow", body={}, timeout_s=0.03)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_connection_reset_detection_and_retry(monkeypatch):
    assert is_connection_reset_error(
        ConnectionResetError(10054, "An existing connection was forcibly closed")
    )
    client = NInferClient("http://127.0.0.1:9")
    calls = {"n": 0}

    class _Boom:
        def open(self, *_args, **_kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                raise ConnectionResetError(
                    10054, "An existing connection was forcibly closed by the remote host"
                )

            class _Resp:
                status = 200
                headers = {}

                def read(self):
                    return b'{"choices":[{"message":{"content":"recovered"}}]}'

                def __enter__(self):
                    return self

                def __exit__(self, *_exc):
                    return False

            return _Resp()

    client._opener = _Boom()
    monkeypatch.setattr("ninfer.client.time.sleep", lambda *_args, **_kwargs: None)
    request = build_chat_request(
        model_id="qwen3.8-27b",
        system_prompt="",
        user_prompt="hello",
    )
    response = client.complete(request, retries=1, retry_backoff_s=0.0)
    assert response.content == "recovered"
    assert calls["n"] == 2

    calls["n"] = 0

    class _AlwaysBoom:
        def open(self, *_args, **_kwargs):
            calls["n"] += 1
            raise ConnectionResetError(10054, "forcibly closed")

    client._opener = _AlwaysBoom()
    with pytest.raises(NInferClientError, match="connection failed"):
        client.complete(request, retries=1, retry_backoff_s=0.0)
    assert calls["n"] == 2
