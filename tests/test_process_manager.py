from __future__ import annotations

import socket
import sys
import textwrap

import pytest

import ninfer.process_manager as pm
from ninfer.client import ChatRequest
from ninfer.process_manager import (
    LifecycleTimeouts,
    NInferConfigurationError,
    ServerConfig,
    build_command,
    complete,
    parse_launch_flags,
    start_server,
    stop_server,
    validate_server_config,
    wait_until_ready,
)


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _fake_server(tmp_path, *, ignore_term: bool = False):
    script = tmp_path / ("fake_ignore.py" if ignore_term else "fake_server.py")
    ignore = "signal.SIG_IGN" if ignore_term else "stop"
    script.write_text(
        textwrap.dedent(
            f"""
            import http.server
            import json
            import signal
            import sys

            def stop(*args):
                raise SystemExit(0)

            signal.signal(signal.SIGTERM, {ignore})
            if hasattr(signal, "SIGBREAK"):
                signal.signal(signal.SIGBREAK, {ignore})

            class Handler(http.server.BaseHTTPRequestHandler):
                def log_message(self, *args):
                    pass
                def do_GET(self):
                    if self.path == "/health":
                        self.send_response(200)
                        self.end_headers()
                        self.wfile.write(b"ok")
                    elif self.path == "/v1/models":
                        raw = b'{{"data":[{{"id":"qwen3.8-27b"}}]}}'
                        self.send_response(200)
                        self.send_header("Content-Type", "application/json")
                        self.send_header("Content-Length", str(len(raw)))
                        self.end_headers()
                        self.wfile.write(raw)
                    else:
                        self.send_response(404)
                        self.end_headers()
                def do_POST(self):
                    size = int(self.headers.get("Content-Length", "0"))
                    self.rfile.read(size)
                    body = {{"choices": [{{"message": {{"content": "fake answer"}}}}]}}
                    raw = json.dumps(body).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(raw)))
                    self.end_headers()
                    self.wfile.write(raw)

            server = http.server.ThreadingHTTPServer(("127.0.0.1", int(sys.argv[sys.argv.index("--port") + 1])), Handler)
            server.serve_forever()
            """
        ),
        encoding="utf-8",
    )
    return script


def test_command_construction_and_launch_flag_parser():
    config = ServerConfig(
        executable="ninfer-serve.exe",
        model_artifact="model.ninfer",
        spec_backend="mtp",
        draft_tokens=3,
        extra_flags=("--no-cuda-graph",),
    )
    command = build_command(config)
    assert command[:2] == ["ninfer-serve.exe", "model.ninfer"]
    assert "--max-context" in command
    assert ["--spec", "mtp"] == command[command.index("--spec") : command.index("--spec") + 2]
    assert command[-1] == "--no-cuda-graph"
    assert parse_launch_flags('--log-stats-interval-ms 0 --request-log-jsonl "a b.jsonl"') == (
        "--log-stats-interval-ms",
        "0",
        "--request-log-jsonl",
        "a b.jsonl",
    )


def test_command_clamps_kv_capacity_to_ninfer_usable_range():
    oversized = ServerConfig(
        executable="ninfer-serve.exe",
        model_artifact="model.ninfer",
        max_context=4096,
        kv_capacity=16384,
        max_concurrency=1,
        spec_backend=None,
        draft_tokens=None,
        lm_head_draft=False,
    )
    command = build_command(oversized)
    assert command[command.index("--kv-capacity") + 1] == "4096"

    concurrent = ServerConfig(
        executable="ninfer-serve.exe",
        model_artifact="model.ninfer",
        max_context=4096,
        kv_capacity=32768,
        max_concurrency=2,
        spec_backend=None,
        draft_tokens=None,
        lm_head_draft=False,
    )
    command = build_command(concurrent)
    assert command[command.index("--kv-capacity") + 1] == "8192"

    exact = ServerConfig(
        executable="ninfer-serve.exe",
        model_artifact="model.ninfer",
        max_context=4096,
        kv_capacity=4096,
        max_concurrency=1,
        spec_backend=None,
        draft_tokens=None,
        lm_head_draft=False,
    )
    command = build_command(exact)
    assert command[command.index("--kv-capacity") + 1] == "4096"

    automatic = ServerConfig(
        executable="ninfer-serve.exe",
        model_artifact="model.ninfer",
        max_context=4096,
        kv_capacity="auto",
        max_concurrency=1,
        spec_backend=None,
        draft_tokens=None,
        lm_head_draft=False,
    )
    command = build_command(automatic)
    assert command[command.index("--kv-capacity") + 1] == "auto"


def test_validation_rejects_unsupported_flags(tmp_path, monkeypatch):
    artifact = tmp_path / "model.ninfer"
    artifact.write_bytes(b"test")
    config = ServerConfig(
        executable=sys.executable,
        model_artifact=str(artifact),
        spec_backend=None,
        draft_tokens=None,
        lm_head_draft=False,
        vision=False,
        verify_model_id=False,
        validate_flags=True,
    )
    all_flags = {
        "--host",
        "--port",
        "--model-id",
        "--device",
        "--max-context",
        "--kv-capacity",
        "--max-concurrency",
        "--max-pending-requests",
        "--prefill-chunk",
        "--kv-dtype",
        "--default-max-tokens",
    }
    monkeypatch.setattr(pm, "probe_supported_flags", lambda *_args, **_kwargs: all_flags)
    assert validate_server_config(config)[1] == artifact.resolve()
    monkeypatch.setattr(pm, "probe_supported_flags", lambda *_args, **_kwargs: {"--host"})
    with pytest.raises(NInferConfigurationError, match="does not advertise"):
        validate_server_config(config)


def test_validation_requires_kv_capacity_for_context(tmp_path):
    artifact = tmp_path / "model.ninfer"
    artifact.write_bytes(b"test")
    config = ServerConfig(
        executable=sys.executable,
        model_artifact=str(artifact),
        max_context=16384,
        kv_capacity=4096,
        spec_backend=None,
        draft_tokens=None,
        lm_head_draft=False,
        validate_flags=False,
    )
    with pytest.raises(NInferConfigurationError, match="kv_capacity must be at least max_context"):
        validate_server_config(config)


@pytest.mark.parametrize("host", ["0.0.0.0", "192.168.1.10", "example.com"])
def test_validation_rejects_non_loopback_hosts(tmp_path, host):
    artifact = tmp_path / "model.ninfer"
    artifact.write_bytes(b"test")
    config = ServerConfig(
        executable=sys.executable,
        model_artifact=str(artifact),
        host=host,
        spec_backend=None,
        draft_tokens=None,
        lm_head_draft=False,
        validate_flags=False,
    )

    with pytest.raises(NInferConfigurationError, match="loopback"):
        validate_server_config(config)


def test_health_completion_and_graceful_teardown(tmp_path):
    script = _fake_server(tmp_path)
    artifact = tmp_path / "model.ninfer"
    artifact.write_bytes(b"fake")
    config = ServerConfig(
        executable=sys.executable,
        model_artifact=str(script),
        port=_free_port(),
        spec_backend=None,
        draft_tokens=None,
        lm_head_draft=False,
        validate_flags=False,
    )
    handle = start_server(config)
    try:
        wait_until_ready(handle, 5.0, poll_interval=0.05)
        response = complete(
            handle,
            ChatRequest(model=config.model_id, messages=[{"role": "user", "content": "hi"}]),
        )
        assert response.content == "fake answer"
    finally:
        report = stop_server(
            handle,
            LifecycleTimeouts(
                graceful_shutdown_s=2.0,
                force_kill_s=2.0,
                vram_reclaim_s=0.2,
            ),
        )
    assert report.process_exited
    assert report.descendants_gone is not False


def test_ipv6_loopback_base_url(tmp_path, monkeypatch):
    script = _fake_server(tmp_path)
    config = ServerConfig(
        executable=sys.executable,
        model_artifact=str(script),
        host="::1",
        port=8080,
        spec_backend=None,
        draft_tokens=None,
        lm_head_draft=False,
        validate_flags=False,
    )
    monkeypatch.setattr(pm, "validate_server_config", lambda _config: (script, script))
    monkeypatch.setattr(pm.subprocess, "Popen", lambda *_args, **_kwargs: type(
        "Process",
        (),
        {"pid": 123, "stderr": None},
    )())
    monkeypatch.setattr(pm, "_initial_descendant_pids", lambda _pid: set())
    monkeypatch.setattr(pm.threading.Thread, "start", lambda _self: None)

    handle = start_server(config)

    assert handle.base_url == "http://[::1]:8080"


def test_zero_port_selects_available_loopback_port(tmp_path):
    script = _fake_server(tmp_path)
    config = ServerConfig(
        executable=sys.executable,
        model_artifact=str(script),
        port=0,
        spec_backend=None,
        draft_tokens=None,
        lm_head_draft=False,
        validate_flags=False,
    )
    handle = start_server(config)
    try:
        assert handle.config.port > 0
        assert handle.base_url.endswith(f":{handle.config.port}")
        wait_until_ready(handle, 5.0, poll_interval=0.05)
    finally:
        stop_server(
            handle,
            LifecycleTimeouts(
                graceful_shutdown_s=2.0,
                force_kill_s=2.0,
                vram_reclaim_s=0.2,
            ),
        )


def test_force_kill_fallback(tmp_path):
    script = _fake_server(tmp_path, ignore_term=True)
    artifact = tmp_path / "model.ninfer"
    artifact.write_bytes(b"fake")
    config = ServerConfig(
        executable=sys.executable,
        model_artifact=str(script),
        port=_free_port(),
        spec_backend=None,
        draft_tokens=None,
        lm_head_draft=False,
        validate_flags=False,
    )
    handle = start_server(config)
    try:
        wait_until_ready(handle, 5.0, poll_interval=0.05)
        report = stop_server(
            handle,
            LifecycleTimeouts(
                graceful_shutdown_s=0.1,
                force_kill_s=2.0,
                vram_reclaim_s=0.2,
            ),
        )
        assert report.forced
        assert report.process_exited
    finally:
        if handle.process.poll() is None:
            handle.process.kill()
            handle.process.wait(timeout=2)
