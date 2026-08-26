"""ComfyUI node that leases a Qwen NInfer process for prompt enhancement."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
import hashlib
import logging
from pathlib import Path
import threading
from typing import Any
import uuid

try:  # Package import when ComfyUI loads this directory as a custom node.
    from ..ninfer.client import ChatRequest, build_chat_request
    from ..ninfer.models import (
        DEFAULT_MODELS_DIR,
        derive_model_id,
        resolve_model_artifact,
        scan_ninfer_models,
    )
    from ..ninfer.multimodal import image_batch_to_data_urls
    from ..ninfer.process_manager import (
        LifecycleTimeouts,
        NInferConfigurationError,
        ServerConfig,
        complete,
        parse_launch_flags,
        start_server,
        stop_server,
        wait_until_ready,
    )
    from ..ninfer.vram import release_comfyui_memory, snapshot_vram
    from .ninfer_advanced_node import ADVANCED_CACHE_KEYS, merge_advanced
except ImportError:  # Direct test/import from the repository root.
    from ninfer.client import ChatRequest, build_chat_request
    from ninfer.models import (
        DEFAULT_MODELS_DIR,
        derive_model_id,
        resolve_model_artifact,
        scan_ninfer_models,
    )
    from ninfer.multimodal import image_batch_to_data_urls
    from ninfer.process_manager import (
        LifecycleTimeouts,
        NInferConfigurationError,
        ServerConfig,
        complete,
        parse_launch_flags,
        start_server,
        stop_server,
        wait_until_ready,
    )
    from ninfer.vram import release_comfyui_memory, snapshot_vram
    from nodes.ninfer_advanced_node import ADVANCED_CACHE_KEYS, merge_advanced


LOGGER = logging.getLogger(__name__)
_NINFER_PROCESS_LOCK = threading.RLock()
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


def _collect_images(kwargs: Mapping[str, Any]) -> list[Any]:
    images: list[Any] = []
    legacy = kwargs.get("image")
    if legacy is not None:
        images.append(legacy)
    for index in range(1, MAX_IMAGE_INPUTS + 1):
        value = kwargs.get(f"image_{index}")
        if value is not None:
            images.append(value)
    return images


class NInferQwenNode:
    """Enhance a prompt with a local NInfer model and return a STRING."""

    CATEGORY = "Amp NInfer"
    FUNCTION = "enhance_prompt"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("enhanced_prompt",)
    OUTPUT_NODE = False

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        model_list = scan_ninfer_models(DEFAULT_MODELS_DIR)
        optional: dict[str, Any] = {
            "advanced": ("AMP_NINFER_ADV",),
            "no_cuda_graph": (
                "BOOLEAN",
                {
                    "default": True,
                    "tooltip": "Disable CUDA Graph capture on 24 GB cards.",
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
                        "tooltip": "KV cache size; must be ≥ context. Steps of 1024.",
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
                        "default": (
                            r"C:\ninfer\ninfer-rtx3090-windows-x64-0.6.0-rtx3090"
                            r"\ninfer-serve.exe"
                        ),
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
                        "default": True,
                        "tooltip": "Free ComfyUI CUDA memory after NInfer exits.",
                    },
                ),
            },
            "optional": optional,
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs: Any) -> str:
        """Cache deterministic generations while allowing fresh random runs."""

        force_execute = bool(kwargs.get("force_execute", True))
        cache_inputs = {
            key: value for key, value in kwargs.items() if key not in _CACHE_IGNORED_INPUTS
        }
        advanced = merge_advanced(kwargs.get("advanced"))
        cache_inputs.update({key: advanced[key] for key in ADVANCED_CACHE_KEYS})
        digest = deterministic_input_hash(cache_inputs)
        if force_execute and int(kwargs.get("seed", -1)) < 0:
            return f"{digest}:force:{uuid.uuid4().hex}"
        return digest

    @staticmethod
    def _cleanup(label: str) -> None:
        errors = release_comfyui_memory()
        for error in errors:
            LOGGER.warning("NInfer %s cleanup warning: %s", label, error)

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
        advanced: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> tuple[str]:
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
            for image in images:
                image_urls.extend(
                    image_batch_to_data_urls(
                        image, max_side=1024, output_format="jpeg", jpeg_quality=90
                    )
                )

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
                "images": images,
                **{key: settings[key] for key in ADVANCED_CACHE_KEYS},
            }
        )
        spec = None if settings["speculative_backend"] == "off" else settings["speculative_backend"]
        config = ServerConfig(
            executable=ninfer_executable,
            model_artifact=artifact_path,
            host=str(settings["host"]),
            port=int(settings["port"]),
            model_id=derived_model_id,
            device=int(settings["device"]),
            max_context=int(context_size),
            kv_capacity=int(kv_capacity),
            spec_backend=spec,
            draft_tokens=int(settings["draft_tokens"]) if spec else None,
            lm_head_draft=bool(settings["lm_head_draft"]) if spec else False,
            vision=bool(vision),
            no_cuda_graph=bool(no_cuda_graph),
            extra_flags=parse_launch_flags(str(settings["server_launch_flags"])),
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
                    self._cleanup("before launch")
                baseline = snapshot_vram(int(settings["device"]))
                handle = start_server(config, baseline_vram=baseline)
                wait_until_ready(handle, timeouts.startup_s)
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
                response = complete(handle, request)
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
                if unload_after_request:
                    try:
                        self._cleanup("after request")
                    except BaseException as cleanup_error:
                        if primary_error is None:
                            primary_error = cleanup_error
                        else:
                            LOGGER.error(
                                "ComfyUI cleanup also failed after the primary error: %s",
                                cleanup_error,
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
