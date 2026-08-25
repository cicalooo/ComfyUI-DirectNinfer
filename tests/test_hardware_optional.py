"""Opt-in smoke test for a real local NInfer/Qwen installation.

The normal suite never launches a model. Set NINFER_HARDWARE_TESTS=1 together
with NINFER_EXECUTABLE and NINFER_MODEL_ARTIFACT to run this on the target GPU.
"""

from __future__ import annotations

import os

import pytest

from ninfer.client import ChatRequest
from ninfer.process_manager import (
    LifecycleTimeouts,
    ServerConfig,
    complete,
    start_server,
    stop_server,
    wait_until_ready,
)
from ninfer.vram import snapshot_vram


@pytest.mark.skipif(
    os.environ.get("NINFER_HARDWARE_TESTS") != "1",
    reason="set NINFER_HARDWARE_TESTS=1 to run the real NInfer smoke test",
)
def test_real_ninfer_one_request_and_reclaim():
    executable = os.environ.get("NINFER_EXECUTABLE")
    artifact = os.environ.get("NINFER_MODEL_ARTIFACT")
    if not executable or not artifact:
        pytest.fail("NINFER_EXECUTABLE and NINFER_MODEL_ARTIFACT are required")
    config = ServerConfig(
        executable=executable,
        model_artifact=artifact,
        port=8080,
    )
    # Keep the public process-manager test contract clear: the optional test
    # records the baseline before launch and passes it to the handle.
    baseline = snapshot_vram(config.device)
    handle = start_server(config, baseline_vram=baseline)
    try:
        wait_until_ready(handle, 300.0)
        response = complete(
            handle,
            ChatRequest(
                model=config.model_id,
                messages=[
                    {
                        "role": "user",
                        "content": "Reply with exactly: hardware smoke test passed",
                    }
                ],
                max_tokens=32,
                reasoning_effort="none",
                timeout_s=300.0,
            ),
        )
        assert response.content.strip()
    finally:
        report = stop_server(
            handle,
            LifecycleTimeouts(
                startup_s=300.0,
                request_s=300.0,
                graceful_shutdown_s=20.0,
                force_kill_s=20.0,
                vram_reclaim_s=60.0,
            ),
        )
    assert report.process_exited
    assert report.descendants_gone is not False
