from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import nodes.ninfer_qwen_node as node_module
from nodes.ninfer_advanced_node import AmpNInferAdvancedNode, merge_advanced
from nodes.ninfer_qwen_node import NInferQwenNode, deterministic_input_hash
import ninfer.models as models_module
from ninfer.models import DEFAULT_MODELS_DIR, derive_model_id, resolve_model_artifact, scan_ninfer_models


def _kwargs(tmp_path, **overrides):
    artifact = tmp_path / "model.ninfer"
    if not artifact.exists():
        artifact.write_bytes(b"fake")
    values = {
        "system_prompt": "Return only the improved prompt.",
        "user_prompt": "A mountain at sunrise",
        "context_size": 16384,
        "kv_capacity": 16384,
        "max_output_tokens": 64,
        "temperature": 0.6,
        "seed": -1,
        "ninfer_executable": "ninfer-serve.exe",
        "models_dir": str(tmp_path),
        "model_artifact": "model.ninfer",
        "vision": False,
        "startup_timeout": 2.0,
        "request_timeout": 2.0,
        "shutdown_timeout": 1.0,
        "force_kill_timeout": 1.0,
        "vram_reclaim_timeout": 1.0,
        "force_execute": True,
        "unload_comfyui_before_launch": True,
        "unload_after_request": True,
        "no_cuda_graph": True,
    }
    values.update(overrides)
    return values


def test_cache_hash_is_deterministic_but_force_execute_is_explicit():
    node = NInferQwenNode()
    stable_a = node.IS_CHANGED(user_prompt="hello", image_1=None, force_execute=False)
    stable_b = node.IS_CHANGED(user_prompt="hello", image_1=None, force_execute=False)
    assert stable_a == stable_b
    assert NInferQwenNode.IS_CHANGED(user_prompt="hello", image_1=None, force_execute=False) == stable_a
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
        advanced={"request_timeout": 900.0},
    )
    assert fixed_a == fixed_b
    forced_a = node.IS_CHANGED(user_prompt="hello", image_1=None, force_execute=True)
    forced_b = node.IS_CHANGED(user_prompt="hello", image_1=None, force_execute=True)
    assert forced_a != forced_b
    assert deterministic_input_hash({"a": 1, "b": [2, 3]}) == deterministic_input_hash(
        {"b": [2, 3], "a": 1}
    )


def test_fixed_seed_response_cache_ignores_lifecycle_controls(tmp_path, monkeypatch):
    node = NInferQwenNode()
    calls = []

    monkeypatch.setattr(node_module, "release_comfyui_memory", lambda: ())
    monkeypatch.setattr(node_module, "snapshot_vram", lambda _device: None)
    monkeypatch.setattr(node_module, "start_server", lambda *args, **kwargs: calls.append("start") or SimpleNamespace(advertised_model_id="qwen3.8-27b"))
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
    node = NInferQwenNode()
    fake_handle = SimpleNamespace(advertised_model_id="qwen3.8-27b")
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


def test_is_changed_unwraps_input_is_list_widgets():
    node = NInferQwenNode()
    listed = node.IS_CHANGED(
        user_prompt=["hello"],
        seed=[123, 123],
        force_execute=[False, False],
        unload_comfyui_before_launch=[True],
        unload_after_request=[True],
        no_cuda_graph=[True],
    )
    scalar = node.IS_CHANGED(
        user_prompt="hello",
        seed=123,
        force_execute=False,
        unload_comfyui_before_launch=True,
        unload_after_request=True,
        no_cuda_graph=True,
    )
    assert listed == scalar


def test_image_list_and_expanding_slots_are_collected(tmp_path, monkeypatch):
    node = NInferQwenNode()
    captured: dict[str, object] = {}

    monkeypatch.setattr(node_module, "release_comfyui_memory", lambda: ())
    monkeypatch.setattr(node_module, "snapshot_vram", lambda _device: None)
    monkeypatch.setattr(
        node_module,
        "start_server",
        lambda *args, **kwargs: SimpleNamespace(advertised_model_id="qwen3.8-27b"),
    )
    monkeypatch.setattr(node_module, "wait_until_ready", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        node_module,
        "stop_server",
        lambda *args, **kwargs: SimpleNamespace(vram_reclaimed=True, notes=()),
    )
    monkeypatch.setattr(
        node_module,
        "complete",
        lambda *args, **kwargs: SimpleNamespace(content="ok"),
    )

    original_build = node_module.build_chat_request

    def capture_build(*args, **kwargs):
        request = original_build(*args, **kwargs)
        urls = []
        for message in request.messages:
            content = message.get("content")
            if isinstance(content, list):
                urls.extend(
                    part["image_url"]["url"]
                    for part in content
                    if isinstance(part, dict) and part.get("type") == "image_url"
                )
        captured["image_urls"] = urls
        return request

    monkeypatch.setattr(node_module, "build_chat_request", capture_build)

    first = np.zeros((1, 2, 2, 3), dtype=np.float32)
    second = np.ones((1, 2, 2, 3), dtype=np.float32)
    third = np.full((1, 2, 2, 3), 0.5, dtype=np.float32)
    node.enhance_prompt(
        **_kwargs(
            tmp_path,
            vision=True,
            image_list=[first, second],
            image_1=third,
        )
    )
    assert captured["image_urls"] is not None
    assert len(list(captured["image_urls"])) == 3


def test_image_requires_explicit_vision_enable(tmp_path, monkeypatch):
    node = NInferQwenNode()
    monkeypatch.setattr(node_module, "release_comfyui_memory", lambda: ())
    with pytest.raises(ValueError, match="vision is disabled"):
        node.enhance_prompt(
            **_kwargs(
                tmp_path,
                image_1=np.zeros((1, 2, 2, 3), dtype=np.float32),
                vision=False,
            )
        )


def test_advanced_node_defaults_and_merge():
    bundled, = AmpNInferAdvancedNode().build(
        reasoning_mode="low",
        top_p=0.5,
        top_k=10,
        repetition_penalty=1.1,
        frequency_penalty=0.1,
        host="127.0.0.1",
        port=9090,
        device=1,
        speculative_backend="off",
        draft_tokens=2,
        lm_head_draft=False,
        server_launch_flags="--log-stats-interval-ms 0",
        startup_timeout=30.0,
        request_timeout=40.0,
        shutdown_timeout=1.0,
        force_kill_timeout=1.0,
        vram_reclaim_timeout=1.0,
    )
    assert bundled["reasoning_mode"] == "low"
    assert bundled["port"] == 9090
    merged = merge_advanced(None)
    assert merged["reasoning_mode"] == "disabled"
    assert merged["top_p"] == pytest.approx(0.9)


def test_scan_and_resolve_model_artifact(tmp_path):
    nested = tmp_path / "sub"
    nested.mkdir()
    (nested / "qwen.ninfer").write_bytes(b"x")
    models = scan_ninfer_models(str(tmp_path))
    assert models == ["sub/qwen.ninfer"]
    resolved = resolve_model_artifact(str(tmp_path), "sub/qwen.ninfer")
    assert resolved.endswith("qwen.ninfer")
    assert derive_model_id("qwen3_8_27b.ninfer") == "qwen3.8-27b"
    assert derive_model_id("Qwen3.8-27B-Uncensored.ninfer") == "Qwen3.8-27B-Uncensored"


def test_default_models_directory_discovers_local_artifacts():
    assert DEFAULT_MODELS_DIR.endswith("artifacts")
    assert "Qwen3.8-27B-Uncensored.ninfer" in scan_ninfer_models(DEFAULT_MODELS_DIR)


def test_legacy_windows_models_dir_falls_back_to_default(tmp_path, monkeypatch):
    fallback_dir = tmp_path / "artifacts"
    fallback_dir.mkdir()
    (fallback_dir / "model.ninfer").write_bytes(b"fake")
    monkeypatch.setattr(models_module, "DEFAULT_MODELS_DIR", str(fallback_dir))

    resolved = resolve_model_artifact(r"C:\models", "model.ninfer")

    assert resolved == str(fallback_dir / "model.ninfer")
