"""Monkeypatch ComfyUI LoadedModel None-real_model crashes without editing core files."""

from __future__ import annotations

import logging
from typing import Any, Callable

LOGGER = logging.getLogger(__name__)

_SENTINEL = "_amp_ninfer_model_management_patched"


class _NoopFinalizer:
    def detach(self) -> None:
        return None


_NOOP_FINALIZER = _NoopFinalizer()


def _safe_is_dead(orig: Callable[..., bool]) -> Callable[..., bool]:
    def is_dead(self: Any) -> bool:
        ref = getattr(self, "real_model", None)
        if ref is None or not callable(ref):
            return False
        return orig(self)

    return is_dead


def _safe_model_unload(orig: Callable[..., Any]) -> Callable[..., Any]:
    def model_unload(self: Any, memory_to_free: Any = None, unpatch_weights: bool = True) -> Any:
        model = getattr(self, "model", None)
        if model is None:
            finalizer = getattr(self, "model_finalizer", None)
            if finalizer is not None:
                detach = getattr(finalizer, "detach", None)
                if callable(detach):
                    detach()
            self.model_finalizer = None
            self.real_model = None
            return True
        if getattr(self, "model_finalizer", None) is None:
            self.model_finalizer = _NOOP_FINALIZER
        return orig(self, memory_to_free=memory_to_free, unpatch_weights=unpatch_weights)

    return model_unload


def _safe_cleanup_models(orig: Callable[..., Any], mm: Any) -> Callable[..., Any]:
    def cleanup_models() -> Any:
        models = mm.current_loaded_models
        to_delete: list[int] = []
        for i in range(len(models)):
            ref = getattr(models[i], "real_model", None)
            if ref is None or not callable(ref):
                to_delete = [i] + to_delete
        for i in to_delete:
            x = models.pop(i)
            del x
        return orig()

    return cleanup_models


def install_comfy_model_management_patch() -> bool:
    """Patch ComfyUI model_management if present. Idempotent. Returns True if applied."""

    try:
        import comfy.model_management as mm  # type: ignore
    except ImportError:
        return False

    loaded_model = getattr(mm, "LoadedModel", None)
    if loaded_model is None:
        return False
    if getattr(loaded_model, _SENTINEL, False):
        return True

    orig_is_dead = loaded_model.is_dead
    orig_unload = loaded_model.model_unload
    orig_cleanup = getattr(mm, "cleanup_models", None)
    loaded_model.is_dead = _safe_is_dead(orig_is_dead)
    loaded_model.model_unload = _safe_model_unload(orig_unload)
    loaded_model._amp_ninfer_orig_is_dead = orig_is_dead
    loaded_model._amp_ninfer_orig_model_unload = orig_unload
    setattr(loaded_model, _SENTINEL, True)

    if callable(orig_cleanup):
        mm.cleanup_models = _safe_cleanup_models(orig_cleanup, mm)
        mm._amp_ninfer_orig_cleanup_models = orig_cleanup
        # Finalizers created before this install still hold orig_cleanup.
        # New model_load calls register mm.cleanup_models (the wrapper).

    LOGGER.info(
        "amp-ninfer: patched LoadedModel.is_dead, LoadedModel.model_unload, and cleanup_models"
    )
    return True


__all__ = ["install_comfy_model_management_patch"]
