"""Best-effort global VRAM measurement and ComfyUI cache cleanup."""

from __future__ import annotations

from dataclasses import dataclass
import gc
import logging
import time


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class VramSnapshot:
    device: int
    total_bytes: int | None
    free_bytes: int | None
    used_bytes: int | None
    torch_allocated_bytes: int | None
    torch_reserved_bytes: int | None
    source: str
    timestamp: float


@dataclass(frozen=True)
class VramWaitResult:
    reclaimed: bool | None
    snapshot: VramSnapshot | None


def _nvml_snapshot(device: int) -> VramSnapshot | None:
    try:
        import pynvml  # type: ignore
    except ImportError:
        return None
    try:
        pynvml.nvmlInit()
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(device)
            memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
            return VramSnapshot(
                device=device,
                total_bytes=int(memory.total),
                free_bytes=int(memory.free),
                used_bytes=int(memory.used),
                torch_allocated_bytes=None,
                torch_reserved_bytes=None,
                source="nvml",
                timestamp=time.monotonic(),
            )
        finally:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
    except Exception as exc:  # NVML is optional and drivers vary by machine.
        LOGGER.debug("NVML VRAM snapshot unavailable: %s", exc)
        return None


def _torch_snapshot(device: int) -> VramSnapshot | None:
    try:
        import torch  # type: ignore
    except ImportError:
        return None
    try:
        if not torch.cuda.is_available():
            return None
        with torch.cuda.device(device):
            free_bytes, total_bytes = torch.cuda.mem_get_info(device)
            allocated = torch.cuda.memory_allocated(device)
            reserved = torch.cuda.memory_reserved(device)
        return VramSnapshot(
            device=device,
            total_bytes=int(total_bytes),
            free_bytes=int(free_bytes),
            used_bytes=int(total_bytes - free_bytes),
            torch_allocated_bytes=int(allocated),
            torch_reserved_bytes=int(reserved),
            source="torch",
            timestamp=time.monotonic(),
        )
    except Exception as exc:
        LOGGER.debug("Torch VRAM snapshot unavailable: %s", exc)
        return None


def snapshot_vram(device: int = 0) -> VramSnapshot:
    """Return a global snapshot, or an explicitly unknown snapshot."""

    snapshot = _nvml_snapshot(device) or _torch_snapshot(device)
    if snapshot is not None:
        return snapshot
    return VramSnapshot(
        device=device,
        total_bytes=None,
        free_bytes=None,
        used_bytes=None,
        torch_allocated_bytes=None,
        torch_reserved_bytes=None,
        source="unavailable",
        timestamp=time.monotonic(),
    )


def unload_comfyui_models() -> tuple[str, ...]:
    """Ask ComfyUI's model manager to release resident models if available."""

    try:
        import comfy.model_management as model_management  # type: ignore
    except ImportError:
        return ()
    errors: list[str] = []
    # Names differ slightly between ComfyUI revisions.  Call only functions
    # that exist and keep the node usable when an older revision lacks one.
    for name in ("unload_all_models", "cleanup_models", "soft_empty_cache"):
        function = getattr(model_management, name, None)
        if not callable(function):
            continue
        try:
            function()
        except Exception as exc:
            errors.append(f"{name}: {exc}")
    return tuple(errors)


def clear_python_cuda_caches() -> tuple[str, ...]:
    """Collect Python objects and clear PyTorch's allocator caches."""

    errors: list[str] = []
    try:
        gc.collect()
    except Exception as exc:
        errors.append(f"gc.collect: {exc}")
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception as exc:
                errors.append(f"torch.cuda.empty_cache: {exc}")
            ipc_collect = getattr(torch.cuda, "ipc_collect", None)
            if callable(ipc_collect):
                try:
                    ipc_collect()
                except Exception as exc:
                    errors.append(f"torch.cuda.ipc_collect: {exc}")
    except ImportError:
        pass
    except Exception as exc:
        errors.append(f"torch cache cleanup: {exc}")
    return tuple(errors)


def release_comfyui_memory() -> tuple[str, ...]:
    """Unload ComfyUI models and clear Python/CUDA caches."""

    return unload_comfyui_models() + clear_python_cuda_caches()


def _is_within_tolerance(
    baseline: VramSnapshot,
    current: VramSnapshot,
    tolerance_bytes: int,
) -> bool:
    if baseline.used_bytes is not None and current.used_bytes is not None:
        return current.used_bytes <= baseline.used_bytes + tolerance_bytes
    if baseline.free_bytes is not None and current.free_bytes is not None:
        return current.free_bytes + tolerance_bytes >= baseline.free_bytes
    return False


def wait_for_vram_reclaim(
    baseline: VramSnapshot,
    *,
    timeout_s: float,
    tolerance_bytes: int = 256 * 1024 * 1024,
    device: int | None = None,
    poll_interval_s: float = 0.25,
) -> VramWaitResult:
    """Wait until global used VRAM returns close to the pre-launch baseline."""

    if timeout_s <= 0:
        raise ValueError("timeout_s must be positive")
    if baseline.used_bytes is None and baseline.free_bytes is None:
        return VramWaitResult(reclaimed=None, snapshot=snapshot_vram(device or baseline.device))
    target_device = baseline.device if device is None else device
    deadline = time.monotonic() + timeout_s
    latest = snapshot_vram(target_device)
    while True:
        if _is_within_tolerance(baseline, latest, max(0, tolerance_bytes)):
            return VramWaitResult(reclaimed=True, snapshot=latest)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return VramWaitResult(reclaimed=False, snapshot=latest)
        time.sleep(min(poll_interval_s, remaining))
        latest = snapshot_vram(target_device)


__all__ = [
    "VramSnapshot",
    "VramWaitResult",
    "clear_python_cuda_caches",
    "release_comfyui_memory",
    "snapshot_vram",
    "unload_comfyui_models",
    "wait_for_vram_reclaim",
]
