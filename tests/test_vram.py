from __future__ import annotations

import sys
import types

from ninfer import vram


def test_unknown_vram_snapshot_honors_explicit_device_zero(monkeypatch):
    baseline = vram.VramSnapshot(
        device=1,
        total_bytes=None,
        free_bytes=None,
        used_bytes=None,
        torch_allocated_bytes=None,
        torch_reserved_bytes=None,
        source="unavailable",
        timestamp=0.0,
    )
    seen: list[int] = []

    monkeypatch.setattr(
        vram,
        "snapshot_vram",
        lambda device: seen.append(device) or baseline,
    )

    result = vram.wait_for_vram_reclaim(baseline, timeout_s=1.0, device=0)

    assert result.reclaimed is None
    assert seen == [0]


def test_unload_uses_public_model_manager_operations(monkeypatch):
    calls: list[str] = []
    model_management = types.ModuleType("comfy.model_management")
    model_management.unload_all_models = lambda: calls.append("unload_all_models")
    model_management.cleanup_models = lambda: calls.append("cleanup_models")
    model_management.soft_empty_cache = lambda: calls.append("soft_empty_cache")
    comfy = types.ModuleType("comfy")
    comfy.model_management = model_management
    monkeypatch.setitem(sys.modules, "comfy", comfy)
    monkeypatch.setitem(sys.modules, "comfy.model_management", model_management)

    assert vram.unload_comfyui_models() == ()
    assert calls == ["unload_all_models", "soft_empty_cache"]
