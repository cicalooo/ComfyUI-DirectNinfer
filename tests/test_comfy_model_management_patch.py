from __future__ import annotations

import sys
import types
import weakref

from ninfer.comfy_model_management_patch import install_comfy_model_management_patch


class _RefTarget:
    pass


class _FakeLoadedModel:
    def __init__(self):
        self.real_model = None
        self.model = object()
        self.model_finalizer = None

    def is_dead(self):
        return self.real_model() is not None and self.model is None

    def model_unload(self, memory_to_free=None, unpatch_weights=True):
        self.model.detach()
        self.model_finalizer.detach()
        self.model_finalizer = None
        self.real_model = None
        return True


def _orig_cleanup_models(mm):
    def cleanup_models():
        to_delete = []
        for i in range(len(mm.current_loaded_models)):
            if mm.current_loaded_models[i].real_model() is None:
                to_delete = [i] + to_delete
        for i in to_delete:
            x = mm.current_loaded_models.pop(i)
            del x

    return cleanup_models


def _install_fake_mm(monkeypatch):
    class FakeLoadedModel(_FakeLoadedModel):
        pass

    mm = types.ModuleType("comfy.model_management")
    mm.LoadedModel = FakeLoadedModel
    mm.current_loaded_models = []
    mm.cleanup_models = _orig_cleanup_models(mm)
    comfy = types.ModuleType("comfy")
    comfy.model_management = mm
    monkeypatch.setitem(sys.modules, "comfy", comfy)
    monkeypatch.setitem(sys.modules, "comfy.model_management", mm)
    return mm


def test_is_dead_none_real_model(monkeypatch):
    mm = _install_fake_mm(monkeypatch)
    assert install_comfy_model_management_patch() is True
    wrapper = mm.LoadedModel()
    wrapper.real_model = None
    assert wrapper.is_dead() is False


def test_is_dead_true_when_patcher_gone(monkeypatch):
    mm = _install_fake_mm(monkeypatch)
    assert install_comfy_model_management_patch() is True
    wrapper = mm.LoadedModel()
    live = _RefTarget()
    wrapper.real_model = weakref.ref(live)
    wrapper.model = None
    assert wrapper.is_dead() is True


def test_is_dead_delegates_to_orig_when_ref_present(monkeypatch):
    mm = _install_fake_mm(monkeypatch)
    install_comfy_model_management_patch()
    wrapper = mm.LoadedModel()
    live = _RefTarget()
    wrapper.real_model = weakref.ref(live)
    wrapper.model = object()
    assert wrapper.is_dead() is False


def test_cleanup_models_mixed_entries(monkeypatch):
    mm = _install_fake_mm(monkeypatch)
    install_comfy_model_management_patch()
    live_target = _RefTarget()
    live = mm.LoadedModel()
    live.real_model = weakref.ref(live_target)
    doomed = _RefTarget()
    dead_ref = mm.LoadedModel()
    dead_ref.real_model = weakref.ref(doomed)
    del doomed
    unloaded = mm.LoadedModel()
    unloaded.real_model = None
    mm.current_loaded_models[:] = [live, dead_ref, unloaded]
    mm.cleanup_models()
    assert live in mm.current_loaded_models
    assert dead_ref not in mm.current_loaded_models
    assert unloaded not in mm.current_loaded_models


def test_cleanup_calls_original_after_filtering(monkeypatch):
    mm = _install_fake_mm(monkeypatch)
    calls = {"n": 0}
    orig = mm.cleanup_models

    def counting_cleanup():
        calls["n"] += 1
        orig()

    mm.cleanup_models = counting_cleanup
    install_comfy_model_management_patch()
    live_target = _RefTarget()
    live = mm.LoadedModel()
    live.real_model = weakref.ref(live_target)
    unloaded = mm.LoadedModel()
    unloaded.real_model = None
    mm.current_loaded_models[:] = [live, unloaded]
    mm.cleanup_models()
    assert calls["n"] == 1
    assert live in mm.current_loaded_models
    assert unloaded not in mm.current_loaded_models


def test_install_is_idempotent(monkeypatch):
    mm = _install_fake_mm(monkeypatch)
    assert install_comfy_model_management_patch() is True
    first = mm.LoadedModel.is_dead
    assert install_comfy_model_management_patch() is True
    assert mm.LoadedModel.is_dead is first


def test_install_returns_false_without_comfy(monkeypatch):
    dummy = types.ModuleType("comfy")
    monkeypatch.setitem(sys.modules, "comfy", dummy)
    monkeypatch.delitem(sys.modules, "comfy.model_management", raising=False)
    assert install_comfy_model_management_patch() is False


def test_model_unload_without_finalizer(monkeypatch):
    mm = _install_fake_mm(monkeypatch)
    install_comfy_model_management_patch()
    wrapper = mm.LoadedModel()

    class _Model:
        def detach(self, unpatch_weights=True):
            return None

    wrapper.model = _Model()
    wrapper.model_finalizer = None
    assert wrapper.model_unload() is True
    assert wrapper.real_model is None


def test_model_unload_without_model(monkeypatch):
    mm = _install_fake_mm(monkeypatch)
    install_comfy_model_management_patch()
    wrapper = mm.LoadedModel()
    wrapper.model = None
    wrapper.model_finalizer = None
    wrapper.real_model = object()
    assert wrapper.model_unload() is True
    assert wrapper.real_model is None
