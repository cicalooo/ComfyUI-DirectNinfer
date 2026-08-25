from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import nodes.ninfer_qwen_node as node_module
from nodes.ninfer_qwen_node import NInferQwenNode, deterministic_input_hash


def _kwargs(tmp_path, **overrides):
    values = {
        "system_prompt": "Return only the improved prompt.",
        "user_prompt": "A mountain at sunrise",
        "style_prompt": "cinematic",
        "negative_prompt": "blurry",
        "asset_description": "",
        "context_size": 4096,
        "kv_capacity": 4096,
        "reasoning_mode": "disabled",
        "max_output_tokens": 64,
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "repetition_penalty": 1.0,
        "frequency_penalty": 0.0,
        "seed": -1,
        "ninfer_executable": "ninfer-serve.exe",
        "model_artifact": str(tmp_path / "model.ninfer"),
        "host": "127.0.0.1",
        "port": 8080,
        "model_id": "qwen3.8-27b",
        "device": 0,
        "speculative_backend": "mtp",
        "draft_tokens": 3,
        "lm_head_draft": True,
        "no_cuda_graph": True,
        "server_launch_flags": "",
        "vision": False,
        "startup_timeout": 2.0,
        "request_timeout": 2.0,
        "shutdown_timeout": 1.0,
        "force_kill_timeout": 1.0,
        "vram_reclaim_timeout": 1.0,
        "force_execute": True,
        "unload_comfyui_before_launch": True,
        "unload_after_request": True,
        "image": None,
    }
    values.update(overrides)
    return values


def test_cache_hash_is_deterministic_but_force_execute_is_explicit():
    node = NInferQwenNode()
    stable_a = node.IS_CHANGED(user_prompt="hello", image=None, force_execute=False)
    stable_b = node.IS_CHANGED(user_prompt="hello", image=None, force_execute=False)
    assert stable_a == stable_b
    assert NInferQwenNode.IS_CHANGED(user_prompt="hello", image=None, force_execute=False) == stable_a
    fixed_a = node.IS_CHANGED(
        user_prompt="hello",
        seed=123,
        force_execute=True,
        unload_comfyui_before_launch=True,
        unload_after_request=True,
        no_cuda_graph=True,
    )
    fixed_b = node.IS_CHANGED(
        user_prompt="hello",
        seed=123,
        force_execute=True,
        unload_comfyui_before_launch=False,
        unload_after_request=False,
        no_cuda_graph=False,
        request_timeout=900.0,
    )
    assert fixed_a == fixed_b
    forced_a = node.IS_CHANGED(user_prompt="hello", image=None, force_execute=True)
    forced_b = node.IS_CHANGED(user_prompt="hello", image=None, force_execute=True)
    assert forced_a != forced_b
    assert deterministic_input_hash({"a": 1, "b": [2, 3]}) == deterministic_input_hash(
        {"b": [2, 3], "a": 1}
    )


def test_fixed_seed_response_cache_ignores_lifecycle_controls(tmp_path, monkeypatch):
    node = NInferQwenNode()
    calls = []

    monkeypatch.setattr(node_module, "release_comfyui_memory", lambda: ())
    monkeypatch.setattr(node_module, "snapshot_vram", lambda _device: None)
    monkeypatch.setattr(node_module, "start_server", lambda *args, **kwargs: calls.append("start"))
    monkeypatch.setattr(node_module, "wait_until_ready", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        node_module,
        "complete",
        lambda *args, **kwargs: SimpleNamespace(content="cached prompt"),
    )
    monkeypatch.setattr(
        node_module,
        "stop_server",
        lambda *args, **kwargs: SimpleNamespace(vram_reclaimed=True, notes=()),
    )

    values = _kwargs(tmp_path, seed=123)
    assert node.enhance_prompt(**values) == ("cached prompt",)
    assert node.enhance_prompt(
        **{**values, "unload_comfyui_before_launch": False, "unload_after_request": False}
    ) == ("cached prompt",)
    assert calls == ["start"]


def test_exception_still_tears_down_started_server(tmp_path, monkeypatch):
    artifact = tmp_path / "model.ninfer"
    artifact.write_bytes(b"fake")
    node = NInferQwenNode()
    fake_handle = SimpleNamespace()
    calls: list[str] = []

    monkeypatch.setattr(node_module, "release_comfyui_memory", lambda: ())
    monkeypatch.setattr(node_module, "snapshot_vram", lambda _device: None)
    monkeypatch.setattr(node_module, "start_server", lambda *args, **kwargs: fake_handle)
    monkeypatch.setattr(node_module, "wait_until_ready", lambda *args, **kwargs: None)

    def fail_complete(*args, **kwargs):
        raise RuntimeError("request failed")

    monkeypatch.setattr(node_module, "complete", fail_complete)

    def fake_stop(handle, timeouts):
        assert handle is fake_handle
        calls.append("stop")
        return SimpleNamespace(vram_reclaimed=True, notes=())

    monkeypatch.setattr(node_module, "stop_server", fake_stop)
    with pytest.raises(RuntimeError, match="request failed"):
        node.enhance_prompt(**_kwargs(tmp_path))
    assert calls == ["stop"]


def test_image_requires_explicit_vision_enable(tmp_path, monkeypatch):
    node = NInferQwenNode()
    monkeypatch.setattr(node_module, "release_comfyui_memory", lambda: ())
    with pytest.raises(ValueError, match="vision is disabled"):
        node.enhance_prompt(
            **_kwargs(
                tmp_path,
                image=np.zeros((1, 2, 2, 3), dtype=np.float32),
                vision=False,
            )
        )
