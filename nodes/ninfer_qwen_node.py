"""ComfyUI node that leases a Qwen NInfer process for prompt enhancement."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
import hashlib
import logging
import os
from pathlib import Path
import threading
from typing import Any
import uuid

try:  # Package import when ComfyUI loads this directory as a custom node.
    from ..ninfer.client import ChatRequest, build_chat_request
    from ..ninfer.models import (
        DEFAULT_MODELS_DIR,
        derive_model_id,
        native_model_names,
        native_model_path,
        register_model_folder,
        resolve_model_artifact,
        scan_ninfer_models,
    )
    from ..ninfer.multimodal import DEFAULT_VISION_MAX_SIDE, images_to_data_urls
    from ..ninfer.process_manager import (
        LifecycleTimeouts,
        NInferConfigurationError,
        NInferStartupError,
        ServerConfig,
        complete,
        fit_kv_capacity,
        is_runtime_capacity_failure,
        parse_launch_flags,
        parse_runtime_capacity_error,
        start_server,
        stop_server,
        suggest_reduced_max_context,
        wait_until_ready,
    )
    from ..ninfer.vram import release_comfyui_memory, snapshot_vram
    from .ninfer_advanced_node import ADVANCED_CACHE_KEYS, merge_advanced
except ImportError:  # Direct test/import from the repository root.
    from ninfer.client import ChatRequest, build_chat_request
    from ninfer.models import (
        DEFAULT_MODELS_DIR,
        derive_model_id,
        native_model_names,
        native_model_path,
        register_model_folder,
        resolve_model_artifact,
        scan_ninfer_models,
    )
    from ninfer.multimodal import DEFAULT_VISION_MAX_SIDE, images_to_data_urls
    from ninfer.process_manager import (
        LifecycleTimeouts,
        NInferConfigurationError,
        NInferStartupError,
        ServerConfig,
        complete,
        fit_kv_capacity,
        is_runtime_capacity_failure,
        parse_launch_flags,
        parse_runtime_capacity_error,
        start_server,
        stop_server,
        suggest_reduced_max_context,
        wait_until_ready,
    )
    from ninfer.vram import release_comfyui_memory, snapshot_vram
    from nodes.ninfer_advanced_node import ADVANCED_CACHE_KEYS, merge_advanced


LOGGER = logging.getLogger(__name__)
_NINFER_PROCESS_LOCK = threading.RLock()
# After heavy ComfyUI models leave residual VRAM, ninfer-serve can load weights
# then fail Engine runtime reservation. Retry with a smaller max_context.
_MAX_STARTUP_CAPACITY_RETRIES = 5
_MIN_CONTEXT_ON_CAPACITY_RETRY = 1024
MAX_IMAGE_INPUTS = 20
DEFAULT_SYSTEM_PROMPT = (
    "You improve image-generation prompts. Keep the user's intent, add concrete "
    "visual detail (subject, lighting, materials, composition, camera), and reply "
    "with only the final prompt. No preamble, markdown, or analysis."
)
_CACHE_IGNORED_INPUTS = frozenset(
    {
        "force_execute",
        "unload_comfyui_before_launch",
        "unload_after_request",
        "startup_timeout",
        "request_timeout",
        "shutdown_timeout",
        "force_kill_timeout",
        "vram_reclaim_timeout",
        "no_cuda_graph",
        "advanced",
    }
)
_FIXED_SEED_RESPONSE_CACHE: OrderedDict[str, str] = OrderedDict()
_FIXED_SEED_RESPONSE_CACHE_LIMIT = 32

register_model_folder()


def _default_ninfer_executable() -> str:
    """Return a usable default for the host platform and local install."""

    configured = os.environ.get("NINFER_EXECUTABLE", "").strip()
    if configured:
        return configured
    if os.name == "nt":
        for candidate in (
            r"C:\ninfer\ninfer-serve.exe",
            r"C:\ninfer\ninfer-rtx3090-windows-x64-0.6.1-rtx3090\ninfer-serve.exe",
            r"C:\ninfer\ninfer-rtx3090-windows-x64-0.6.0-rtx3090\ninfer-serve.exe",
        ):
            if Path(candidate).is_file():
                return candidate
        return r"C:\ninfer\ninfer-serve.exe"
    installed = Path("/opt/ninfer-3090/current/ninfer-serve")
    return str(installed) if installed.is_file() else "ninfer-serve"


def _hash_value(digest: "hashlib._Hash", value: Any) -> None:
    """Add a stable representation of common ComfyUI input values."""

    if value is None:
        digest.update(b"none;")
        return
    if isinstance(value, bool):
        digest.update(f"bool:{value};".encode("utf-8"))
        return
    if isinstance(value, (int, float, str)):
        digest.update(
            f"{type(value).__name__}:{value!r};".encode("utf-8", errors="surrogatepass")
        )
        return
    if isinstance(value, bytes):
        digest.update(b"bytes:")
        digest.update(len(value).to_bytes(8, "big"))
        digest.update(value)
        return
    if isinstance(value, Mapping):
        digest.update(b"mapping{")
        for key in sorted(value, key=lambda item: repr(item)):
            _hash_value(digest, key)
            _hash_value(digest, value[key])
        digest.update(b"};")
        return
    if isinstance(value, (list, tuple)):
        digest.update(f"{type(value).__name__}[".encode())
        for item in value:
            _hash_value(digest, item)
        digest.update(b"];")
        return
    if isinstance(value, Path):
        _hash_value(digest, str(value))
        return
    try:
        import torch  # type: ignore

        if isinstance(value, torch.Tensor):
            cpu = value.detach().to(device="cpu").contiguous()
            digest.update(b"torch:")
            _hash_value(digest, str(cpu.dtype))
            _hash_value(digest, tuple(cpu.shape))
            digest.update(cpu.numpy().tobytes())
            return
    except ImportError:
        pass
    except Exception as exc:
        digest.update(f"tensor-error:{exc!r};".encode("utf-8"))
        return
    try:
        import numpy as np  # type: ignore

        if isinstance(value, np.ndarray):
            digest.update(b"numpy:")
            _hash_value(digest, str(value.dtype))
            _hash_value(digest, tuple(value.shape))
            digest.update(value.tobytes(order="C"))
            return
    except ImportError:
        pass
    digest.update(f"repr:{type(value).__name__}:{value!r};".encode("utf-8"))


def deterministic_input_hash(inputs: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    _hash_value(digest, inputs)
    return digest.hexdigest()


def _generation_cache_key(inputs: Mapping[str, Any]) -> str | None:
    if int(inputs.get("seed", -1)) < 0:
        return None
    cache_inputs = {
        key: value for key, value in inputs.items() if key not in _CACHE_IGNORED_INPUTS
    }
    return deterministic_input_hash(cache_inputs)


def _is_image_key(key: str) -> bool:
    if key in {"image", "image_list", "Image List"}:
        return True
    return key.startswith("image_") and key[6:].isdigit()


def _unwrap_input(value: Any) -> Any:
    """ComfyUI INPUT_IS_LIST delivers every widget as a list (padded to the longest)."""

    while isinstance(value, list):
        if not value:
            return None
        value = value[0]
    return value


def _flatten_image_value(value: Any) -> list[Any]:
    """Turn a tensor, batch, or ComfyUI IMAGE list into per-connection values."""

    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        images: list[Any] = []
        for item in value:
            images.extend(_flatten_image_value(item))
        return images
    return [value]


def _collect_images(kwargs: Mapping[str, Any]) -> list[Any]:
    images: list[Any] = []
    for key in ("image", "image_list", "Image List"):
        images.extend(_flatten_image_value(kwargs.get(key)))
    for index in range(1, MAX_IMAGE_INPUTS + 1):
        images.extend(_flatten_image_value(kwargs.get(f"image_{index}")))
    return images


class NInferQwenNode:
    """Enhance a prompt with a local NInfer model and return a STRING."""

    CATEGORY = "DirectNinfer"
    FUNCTION = "enhance_prompt"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("enhanced_prompt",)
    OUTPUT_NODE = False
    # Receive a full ComfyUI IMAGE list in one call instead of re-running NInfer
    # once per list item. Scalar widgets still work when tests pass them unwrapped.
    INPUT_IS_LIST = True

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        model_list = native_model_names() or scan_ninfer_models(DEFAULT_MODELS_DIR)
        optional: dict[str, Any] = {
            "advanced": ("AMP_NINFER_ADV",),
            "no_cuda_graph": (
                "BOOLEAN",
                {
                    "default": True,
                    "tooltip": "Disable CUDA Graph capture on 24 GB cards.",
                },
            ),
            "image_list": (
                "IMAGE",
                {
                    "tooltip": (
                        "Optional ComfyUI Image list (OUTPUT_IS_LIST). All items "
                        "are sent in one request. Expanding image_1..image_20 still work."
                    ),
                },
            ),
            "vision_max_side": (
                "INT",
                {
                    "default": DEFAULT_VISION_MAX_SIDE,
                    "min": 0,
                    "max": 4096,
                    "step": 28,
                    "tooltip": (
                        "Maximum encoded image side for vision requests. "
                        "Values are Qwen-aligned (multiples of 28). Default "
                        "336 is the safe 24 GB profile. 0 keeps source size "
                        "within the shared pixel budget."
                    ),
                },
            ),
            "vision_format": (
                ["png", "jpeg", "auto"],
                {
                    "default": "png",
                    "tooltip": (
                        "Wire format after on-the-fly conversion. Any ComfyUI "
                        "IMAGE (RGB/RGBA/L/CHW/BHWC) is normalized to RGB; "
                        "JPEG/auto are rewritten to PNG because ninfer-serve "
                        "JPEG/swscaler can crash on some Windows runtimes."
                    ),
                },
            ),
            "vision_jpeg_quality": (
                "INT",
                {
                    "default": 85,
                    "min": 1,
                    "max": 95,
                    "step": 1,
                    "tooltip": "Kept for compatibility; PNG wire format ignores JPEG quality.",
                },
            ),
            "image_1": (
                "IMAGE",
                {
                    "tooltip": "Optional reference image. Connecting one adds another slot (max 20).",
                },
            ),
        }
        return {
            "required": {
                "system_prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": DEFAULT_SYSTEM_PROMPT,
                        "tooltip": "Instructions for how the model rewrites the user input.",
                    },
                ),
                "user_prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": "User Input: the prompt to enhance.",
                    },
                ),
                "context_size": (
                    "INT",
                    {
                        "default": 16384,
                        "min": 1024,
                        "max": 131072,
                        "step": 1024,
                        "tooltip": "NInfer --max-context. Steps of 1024.",
                    },
                ),
                "kv_capacity": (
                    "INT",
                    {
                        "default": 16384,
                        "min": 1024,
                        "max": 262144,
                        "step": 1024,
                        "tooltip": "Shared KV tokens. With concurrency 1 this cannot exceed context; extra is clamped.",
                    },
                ),
                "max_output_tokens": (
                    "INT",
                    {
                        "default": 1024,
                        "min": 16,
                        "max": 8192,
                        "step": 16,
                        "tooltip": "Max tokens in the enhanced prompt.",
                    },
                ),
                "temperature": (
                    "FLOAT",
                    {
                        "default": 0.6,
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.01,
                        "tooltip": "Sampling temperature. Lower is more deterministic.",
                    },
                ),
                "seed": (
                    "INT",
                    {
                        "default": -1,
                        "min": -1,
                        "max": 2147483647,
                        "step": 1,
                        "tooltip": "-1 uses a fresh server seed. Fixed seeds cache the result.",
                    },
                ),
                "ninfer_executable": (
                    "STRING",
                    {
                        "default": _default_ninfer_executable(),
                        "multiline": False,
                        "tooltip": "Full path to ninfer-serve.",
                    },
                ),
                "models_dir": (
                    "STRING",
                    {
                        "default": DEFAULT_MODELS_DIR,
                        "multiline": False,
                        "tooltip": "Folder scanned for .ninfer files.",
                    },
                ),
                "model_artifact": (
                    model_list,
                    {
                        "tooltip": "Selected .ninfer; Refresh rescans models_dir.",
                    },
                ),
                "vision": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Load vision weights. Required when any IMAGE is connected.",
                    },
                ),
                "force_execute": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "With seed=-1, rerun each queue. Fixed seeds stay cached.",
                    },
                ),
                "unload_comfyui_before_launch": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Free ComfyUI CUDA memory before starting NInfer.",
                    },
                ),
                "unload_after_request": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "Deprecated compatibility input. NInfer process exit releases "
                            "its own VRAM; DirectNinfer does not unload ComfyUI models afterward."
                        ),
                    },
                ),
            },
            "optional": optional,
        }

    @classmethod
    def VALIDATE_INPUTS(
        cls,
        model_artifact: str = "",
        models_dir: str = "",
    ) -> bool | str:
        """Validate model_artifact dynamically against ComfyUI registry and models_dir."""
        artifact_str = str(_unwrap_input(model_artifact) or "").strip()
        dir_str = str(_unwrap_input(models_dir) or "").strip()
        if not artifact_str or artifact_str.startswith("(no .ninfer"):
            return "No .ninfer model artifact selected. Set models_dir and click Refresh."
        path = native_model_path(artifact_str)
        if path is not None:
            return True
        try:
            resolve_model_artifact(dir_str or DEFAULT_MODELS_DIR, artifact_str)
            return True
        except Exception as exc:
            return f"Model artifact '{artifact_str}' could not be resolved: {exc}"

    @classmethod
    def IS_CHANGED(cls, **kwargs: Any) -> str:
        """Cache deterministic generations while allowing fresh random runs."""

        unwrapped = {
            key: value if _is_image_key(key) else _unwrap_input(value)
            for key, value in kwargs.items()
        }
        force_execute = bool(unwrapped.get("force_execute", True))
        cache_inputs = {
            key: value
            for key, value in unwrapped.items()
            if key not in _CACHE_IGNORED_INPUTS
        }
        advanced = merge_advanced(unwrapped.get("advanced"))
        cache_inputs.update({key: advanced[key] for key in ADVANCED_CACHE_KEYS})
        digest = deterministic_input_hash(cache_inputs)
        if force_execute and int(unwrapped.get("seed", -1)) < 0:
            return f"{digest}:force:{uuid.uuid4().hex}"
        return digest

    @staticmethod
    def _cleanup(label: str) -> None:
        errors = release_comfyui_memory()
        for error in errors:
            LOGGER.warning("NInfer %s cleanup warning: %s", label, error)

    @classmethod
    def _prepare_gpu_for_launch(cls, label: str, device: int) -> None:
        """Ask ComfyUI to unload resident models once before NInfer starts."""

        cls._cleanup(label)
        snapshot = snapshot_vram(int(device))
        free_bytes = getattr(snapshot, "free_bytes", None) if snapshot is not None else None
        if free_bytes is not None:
            LOGGER.info(
                "NInfer pre-launch VRAM free≈%.0f MiB (source=%s)",
                free_bytes / (1024 * 1024),
                getattr(snapshot, "source", "unknown"),
            )

    @staticmethod
    def _safe_stop_handle(handle: Any, timeouts: LifecycleTimeouts) -> None:
        if handle is None:
            return
        try:
            stop_server(handle, timeouts)
        except BaseException as teardown_error:
            LOGGER.warning(
                "NInfer stop during capacity retry failed: %s", teardown_error
            )

    @staticmethod
    def _processing_interrupted() -> bool:
        try:
            import comfy.model_management as model_management  # type: ignore
        except ImportError:
            return False
        checker = getattr(model_management, "processing_interrupted", None)
        return bool(checker()) if callable(checker) else False

    def enhance_prompt(
        self,
        system_prompt: str,
        user_prompt: str,
        context_size: int,
        kv_capacity: int,
        max_output_tokens: int,
        temperature: float,
        seed: int,
        ninfer_executable: str,
        models_dir: str,
        model_artifact: str,
        vision: bool,
        force_execute: bool,
        unload_comfyui_before_launch: bool,
        unload_after_request: bool,
        no_cuda_graph: bool = True,
        vision_max_side: int = DEFAULT_VISION_MAX_SIDE,
        vision_format: str = "png",
        vision_jpeg_quality: int = 85,
        advanced: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> tuple[str]:
        system_prompt = _unwrap_input(system_prompt)
        user_prompt = _unwrap_input(user_prompt)
        context_size = _unwrap_input(context_size)
        kv_capacity = _unwrap_input(kv_capacity)
        max_output_tokens = _unwrap_input(max_output_tokens)
        temperature = _unwrap_input(temperature)
        seed = _unwrap_input(seed)
        ninfer_executable = _unwrap_input(ninfer_executable)
        models_dir = _unwrap_input(models_dir)
        model_artifact = _unwrap_input(model_artifact)
        vision = _unwrap_input(vision)
        force_execute = _unwrap_input(force_execute)
        unload_comfyui_before_launch = _unwrap_input(unload_comfyui_before_launch)
        unload_after_request = _unwrap_input(unload_after_request)
        no_cuda_graph = _unwrap_input(no_cuda_graph)
        raw_vision_max_side = _unwrap_input(vision_max_side)
        vision_max_side = int(
            DEFAULT_VISION_MAX_SIDE
            if raw_vision_max_side is None
            else raw_vision_max_side
        )
        vision_format = str(_unwrap_input(vision_format) or "png").lower()
        vision_jpeg_quality = int(_unwrap_input(vision_jpeg_quality) or 85)
        advanced = _unwrap_input(advanced)
        settings = merge_advanced(advanced)
        timeouts = LifecycleTimeouts(
            startup_s=float(settings["startup_timeout"]),
            request_s=float(settings["request_timeout"]),
            graceful_shutdown_s=float(settings["shutdown_timeout"]),
            force_kill_s=float(settings["force_kill_timeout"]),
            vram_reclaim_s=float(settings["vram_reclaim_timeout"]),
        )
        timeouts.validate()

        images = _collect_images(kwargs)
        image_urls: list[str] = []
        if images:
            if not vision:
                raise NInferConfigurationError(
                    "An IMAGE input was provided, but vision is disabled. Enable vision "
                    "to launch NInfer with its media weights."
                )
            image_urls.extend(
                images_to_data_urls(
                    images,
                    max_side=vision_max_side,
                    output_format=vision_format,
                    jpeg_quality=vision_jpeg_quality,
                )
            )

        artifact_path = native_model_path(str(model_artifact))
        if artifact_path is None:
            artifact_path = resolve_model_artifact(models_dir, model_artifact)
        derived_model_id = derive_model_id(artifact_path)
        cache_key = _generation_cache_key(
            {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "context_size": context_size,
                "kv_capacity": kv_capacity,
                "max_output_tokens": max_output_tokens,
                "temperature": temperature,
                "seed": seed,
                "ninfer_executable": ninfer_executable,
                "models_dir": models_dir,
                "model_artifact": artifact_path,
                "vision": vision,
                "force_execute": force_execute,
                "unload_comfyui_before_launch": unload_comfyui_before_launch,
                "unload_after_request": unload_after_request,
                "no_cuda_graph": no_cuda_graph,
                "image_urls": image_urls,
                **{key: settings[key] for key in ADVANCED_CACHE_KEYS},
            }
        )
        spec = None if settings["speculative_backend"] == "off" else settings["speculative_backend"]
        launch_max_context = int(context_size)
        launch_kv_capacity = int(kv_capacity)
        extra_flags = parse_launch_flags(str(settings["server_launch_flags"]))
        device = int(settings["device"])

        def _make_config(max_context: int, kv_tokens: int) -> ServerConfig:
            fitted_kv = fit_kv_capacity(max_context, kv_tokens, 1)
            return ServerConfig(
                executable=ninfer_executable,
                model_artifact=artifact_path,
                host=str(settings["host"]),
                port=int(settings["port"]),
                model_id=derived_model_id,
                device=device,
                max_context=max_context,
                kv_capacity=fitted_kv,
                spec_backend=spec,
                draft_tokens=int(settings["draft_tokens"]) if spec else None,
                lm_head_draft=bool(settings["lm_head_draft"]) if spec else False,
                vision=bool(vision),
                no_cuda_graph=bool(no_cuda_graph),
                extra_flags=extra_flags,
            )

        result: str | None = None
        primary_error: BaseException | None = None
        handle: Any | None = None
        with _NINFER_PROCESS_LOCK:
            if cache_key is not None:
                cached_result = _FIXED_SEED_RESPONSE_CACHE.get(cache_key)
                if cached_result is not None:
                    _FIXED_SEED_RESPONSE_CACHE.move_to_end(cache_key)
                    return (cached_result,)
            try:
                if unload_comfyui_before_launch:
                    self._prepare_gpu_for_launch("before launch", device)
                baseline = snapshot_vram(device)
                last_startup_error: BaseException | None = None
                for attempt in range(_MAX_STARTUP_CAPACITY_RETRIES):
                    config = _make_config(launch_max_context, launch_kv_capacity)
                    try:
                        handle = start_server(config, baseline_vram=baseline)
                        wait_until_ready(handle, timeouts.startup_s)
                        last_startup_error = None
                        break
                    except NInferStartupError as exc:
                        last_startup_error = exc
                        diagnostic = ""
                        if handle is not None:
                            diagnostic = handle.diagnostic_tail()
                            self._safe_stop_handle(handle, timeouts)
                            handle = None
                        combined = f"{exc}\n{diagnostic}"
                        if (
                            attempt >= _MAX_STARTUP_CAPACITY_RETRIES - 1
                            or not is_runtime_capacity_failure(combined)
                        ):
                            raise
                        requested_available = parse_runtime_capacity_error(combined)
                        requested_b = (
                            requested_available[0] if requested_available else None
                        )
                        available_b = (
                            requested_available[1] if requested_available else None
                        )
                        reduced = suggest_reduced_max_context(
                            launch_max_context,
                            requested_bytes=requested_b,
                            available_bytes=available_b,
                            min_context=_MIN_CONTEXT_ON_CAPACITY_RETRY,
                        )
                        if reduced is None:
                            raise
                        LOGGER.warning(
                            "NInfer Engine runtime capacity shortfall "
                            "(requested=%s available=%s); retrying with "
                            "max_context %s → %s (attempt %s/%s)",
                            requested_b,
                            available_b,
                            launch_max_context,
                            reduced,
                            attempt + 2,
                            _MAX_STARTUP_CAPACITY_RETRIES,
                        )
                        launch_max_context = reduced
                        launch_kv_capacity = min(launch_kv_capacity, reduced)
                if last_startup_error is not None:
                    raise last_startup_error
                assert handle is not None
                model_id = handle.advertised_model_id or derived_model_id
                request: ChatRequest = build_chat_request(
                    model_id=model_id,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    image_urls=image_urls,
                    reasoning_mode=str(settings["reasoning_mode"]),
                    max_tokens=int(max_output_tokens),
                    temperature=float(temperature),
                    top_p=float(settings["top_p"]),
                    top_k=int(settings["top_k"]),
                    repetition_penalty=float(settings["repetition_penalty"]),
                    frequency_penalty=float(settings["frequency_penalty"]),
                    seed=int(seed),
                    timeout_s=timeouts.request_s,
                )
                response = complete(
                    handle, request, cancel_check=self._processing_interrupted
                )
                result = response.content
            except BaseException as exc:
                primary_error = exc
            finally:
                if handle is not None:
                    try:
                        report = stop_server(handle, timeouts)
                        if report.vram_reclaimed is False:
                            LOGGER.warning(
                                "NInfer process exited, but VRAM did not return to the baseline: %s",
                                "; ".join(report.notes),
                            )
                    except BaseException as teardown_error:
                        if primary_error is None:
                            primary_error = teardown_error
                        else:
                            LOGGER.error(
                                "NInfer teardown also failed after the primary error: %s",
                                teardown_error,
                                exc_info=True,
                            )
            if primary_error is not None:
                raise primary_error
            if result is None:
                raise RuntimeError("NInfer returned no text")
            if cache_key is not None:
                _FIXED_SEED_RESPONSE_CACHE[cache_key] = result
                _FIXED_SEED_RESPONSE_CACHE.move_to_end(cache_key)
                while len(_FIXED_SEED_RESPONSE_CACHE) > _FIXED_SEED_RESPONSE_CACHE_LIMIT:
                    _FIXED_SEED_RESPONSE_CACHE.popitem(last=False)
            return (result,)


__all__ = ["NInferQwenNode", "deterministic_input_hash"]
