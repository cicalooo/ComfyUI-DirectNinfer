"""ComfyUI node that leases Qwen3.8-27B from a short-lived NInfer process."""

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
except ImportError:  # Direct test/import from the repository root.
    from ninfer.client import ChatRequest, build_chat_request
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


LOGGER = logging.getLogger(__name__)
_NINFER_PROCESS_LOCK = threading.RLock()
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
    # IMAGE tensors are normally torch.Tensor.  Keep the hashing operation on
    # CPU so it does not leave an extra CUDA allocation behind.
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


class NInferQwenNode:
    """Enhance a prompt with Qwen3.8-27B and return a STRING."""

    CATEGORY = "NInfer/Prompt"
    FUNCTION = "enhance_prompt"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("enhanced_prompt",)
    OUTPUT_NODE = False

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "system_prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": (
                            "You improve image-generation prompts. Preserve the user's "
                            "intent, add useful concrete visual detail, and return only "
                            "the final prompt without analysis or markdown."
                        ),
                    },
                ),
                "user_prompt": (
                    "STRING",
                    {"multiline": True, "default": "Describe the image I should create."},
                ),
                "style_prompt": (
                    "STRING",
                    {"multiline": True, "default": "", "label_on": "style"},
                ),
                "negative_prompt": (
                    "STRING",
                    {"multiline": True, "default": "", "label_on": "negative"},
                ),
                "asset_description": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": "Optional text describing the supplied IMAGE asset.",
                    },
                ),
                "context_size": (
                    "INT",
                    {"default": 4096, "min": 256, "max": 131072, "step": 256},
                ),
                "kv_capacity": (
                    "INT",
                    {
                        "default": 4096,
                        "min": 256,
                        "max": 262144,
                        "step": 256,
                        "tooltip": "Must be at least context_size. Keep explicit on a 24 GB card.",
                    },
                ),
                "reasoning_mode": (
                    ["disabled", "low", "medium"],
                    {"default": "disabled"},
                ),
                "max_output_tokens": (
                    "INT",
                    {"default": 192, "min": 16, "max": 8192, "step": 16},
                ),
                "temperature": (
                    "FLOAT",
                    {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.01},
                ),
                "top_p": (
                    "FLOAT",
                    {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "top_k": (
                    "INT",
                    {"default": 20, "min": 0, "max": 256, "step": 1},
                ),
                "repetition_penalty": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.5,
                        "max": 2.0,
                        "step": 0.01,
                        "tooltip": "Sent only when non-neutral; support depends on NInfer build.",
                    },
                ),
                "frequency_penalty": (
                    "FLOAT",
                    {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.01},
                ),
                "seed": (
                    "INT",
                    {
                        "default": -1,
                        "min": -1,
                        "max": 2147483647,
                        "step": 1,
                        "tooltip": "-1 uses a fresh server seed.",
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
                    },
                ),
                "model_artifact": (
                    "STRING",
                    {
                        "default": r"C:\models\qwen3_8_27b.ninfer",
                        "multiline": False,
                        "tooltip": "Full path to the .ninfer Qwen3.8-27B artifact.",
                    },
                ),
                "host": ("STRING", {"default": "127.0.0.1", "multiline": False}),
                "port": (
                    "INT",
                    {"default": 8080, "min": 1, "max": 65535, "step": 1},
                ),
                "model_id": (
                    "STRING",
                    {"default": "qwen3.8-27b", "multiline": False},
                ),
                "device": ("INT", {"default": 0, "min": 0, "max": 31, "step": 1}),
                "speculative_backend": (
                    ["mtp", "off", "dflash"],
                    {"default": "mtp"},
                ),
                "draft_tokens": (
                    "INT",
                    {"default": 3, "min": 1, "max": 15, "step": 1},
                ),
                "lm_head_draft": ("BOOLEAN", {"default": True}),
                "server_launch_flags": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "Additional flags, parsed without a shell; typed flags must not be repeated.",
                    },
                ),
                "vision": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Loads NInfer vision weights. Required when IMAGE is supplied.",
                    },
                ),
                "startup_timeout": (
                    "FLOAT",
                    {"default": 120.0, "min": 1.0, "max": 1800.0, "step": 1.0},
                ),
                "request_timeout": (
                    "FLOAT",
                    {"default": 180.0, "min": 1.0, "max": 3600.0, "step": 1.0},
                ),
                "shutdown_timeout": (
                    "FLOAT",
                    {"default": 10.0, "min": 0.1, "max": 300.0, "step": 0.5},
                ),
                "force_kill_timeout": (
                    "FLOAT",
                    {"default": 10.0, "min": 0.1, "max": 300.0, "step": 0.5},
                ),
                "vram_reclaim_timeout": (
                    "FLOAT",
                    {"default": 20.0, "min": 0.1, "max": 600.0, "step": 0.5},
                ),
                "force_execute": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "With seed=-1, rerun each queue execution; fixed seeds stay cached.",
                    },
                ),
                "unload_comfyui_before_launch": ("BOOLEAN", {"default": True}),
                "unload_after_request": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "no_cuda_graph": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Disable CUDA Graph capture for reliable 16K startup on 24 GB cards.",
                    },
                ),
                "image": ("IMAGE",),
            },
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs: Any) -> str:
        """Cache deterministic generations while allowing fresh random runs."""

        force_execute = bool(kwargs.get("force_execute", True))
        cache_inputs = {
            key: value for key, value in kwargs.items() if key not in _CACHE_IGNORED_INPUTS
        }
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
        style_prompt: str,
        negative_prompt: str,
        asset_description: str,
        context_size: int,
        kv_capacity: int,
        reasoning_mode: str,
        max_output_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        repetition_penalty: float,
        frequency_penalty: float,
        seed: int,
        ninfer_executable: str,
        model_artifact: str,
        host: str,
        port: int,
        model_id: str,
        device: int,
        speculative_backend: str,
        draft_tokens: int,
        lm_head_draft: bool,
        server_launch_flags: str,
        vision: bool,
        startup_timeout: float,
        request_timeout: float,
        shutdown_timeout: float,
        force_kill_timeout: float,
        vram_reclaim_timeout: float,
        force_execute: bool,
        unload_comfyui_before_launch: bool,
        unload_after_request: bool,
        no_cuda_graph: bool = True,
        image: Any | None = None,
    ) -> tuple[str]:
        timeouts = LifecycleTimeouts(
            startup_s=float(startup_timeout),
            request_s=float(request_timeout),
            graceful_shutdown_s=float(shutdown_timeout),
            force_kill_s=float(force_kill_timeout),
            vram_reclaim_s=float(vram_reclaim_timeout),
        )
        timeouts.validate()

        image_urls: list[str] = []
        if image is not None:
            if not vision:
                raise NInferConfigurationError(
                    "An IMAGE input was provided, but vision is disabled. Enable vision "
                    "to launch NInfer with its media weights."
                )
            image_urls = image_batch_to_data_urls(
                image, max_side=1024, output_format="jpeg", jpeg_quality=90
            )

        request: ChatRequest = build_chat_request(
            model_id=model_id,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            negative_prompt=negative_prompt,
            style_prompt=style_prompt,
            asset_description=asset_description,
            image_urls=image_urls,
            reasoning_mode=reasoning_mode,
            max_tokens=int(max_output_tokens),
            temperature=float(temperature),
            top_p=float(top_p),
            top_k=int(top_k),
            repetition_penalty=float(repetition_penalty),
            frequency_penalty=float(frequency_penalty),
            seed=int(seed),
            timeout_s=timeouts.request_s,
        )
        cache_key = _generation_cache_key(
            {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "style_prompt": style_prompt,
                "negative_prompt": negative_prompt,
                "asset_description": asset_description,
                "context_size": context_size,
                "kv_capacity": kv_capacity,
                "reasoning_mode": reasoning_mode,
                "max_output_tokens": max_output_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "repetition_penalty": repetition_penalty,
                "frequency_penalty": frequency_penalty,
                "seed": seed,
                "ninfer_executable": ninfer_executable,
                "model_artifact": model_artifact,
                "host": host,
                "port": port,
                "model_id": model_id,
                "device": device,
                "speculative_backend": speculative_backend,
                "draft_tokens": draft_tokens,
                "lm_head_draft": lm_head_draft,
                "server_launch_flags": server_launch_flags,
                "vision": vision,
                "startup_timeout": startup_timeout,
                "request_timeout": request_timeout,
                "shutdown_timeout": shutdown_timeout,
                "force_kill_timeout": force_kill_timeout,
                "vram_reclaim_timeout": vram_reclaim_timeout,
                "force_execute": force_execute,
                "unload_comfyui_before_launch": unload_comfyui_before_launch,
                "unload_after_request": unload_after_request,
                "no_cuda_graph": no_cuda_graph,
                "image": image,
            }
        )
        spec = None if speculative_backend == "off" else speculative_backend
        config = ServerConfig(
            executable=ninfer_executable,
            model_artifact=model_artifact,
            host=host,
            port=int(port),
            model_id=model_id,
            device=int(device),
            max_context=int(context_size),
            kv_capacity=int(kv_capacity),
            spec_backend=spec,
            draft_tokens=int(draft_tokens) if spec else None,
            lm_head_draft=bool(lm_head_draft) if spec else False,
            vision=bool(vision),
            no_cuda_graph=bool(no_cuda_graph),
            extra_flags=parse_launch_flags(server_launch_flags),
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
                baseline = snapshot_vram(int(device))
                handle = start_server(config, baseline_vram=baseline)
                wait_until_ready(handle, timeouts.startup_s)
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
