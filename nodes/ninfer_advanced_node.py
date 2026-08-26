"""Optional advanced settings for Amp NInfer."""

from __future__ import annotations

from typing import Any


ADVANCED_DEFAULTS: dict[str, Any] = {
    "reasoning_mode": "disabled",
    "top_p": 0.9,
    "top_k": 40,
    "repetition_penalty": 1.05,
    "frequency_penalty": 0.0,
    "host": "127.0.0.1",
    "port": 8080,
    "device": 0,
    "speculative_backend": "mtp",
    "draft_tokens": 3,
    "lm_head_draft": True,
    "server_launch_flags": "",
    "startup_timeout": 120.0,
    "request_timeout": 180.0,
    "shutdown_timeout": 10.0,
    "force_kill_timeout": 10.0,
    "vram_reclaim_timeout": 20.0,
}

ADVANCED_CACHE_KEYS = (
    "reasoning_mode",
    "top_p",
    "top_k",
    "repetition_penalty",
    "frequency_penalty",
    "host",
    "port",
    "device",
    "speculative_backend",
    "draft_tokens",
    "lm_head_draft",
    "server_launch_flags",
)


def merge_advanced(advanced: dict[str, Any] | None) -> dict[str, Any]:
    merged = dict(ADVANCED_DEFAULTS)
    if isinstance(advanced, dict):
        for key in ADVANCED_DEFAULTS:
            if key in advanced and advanced[key] is not None:
                merged[key] = advanced[key]
    return merged


class AmpNInferAdvancedNode:
    """Bundle optional NInfer sampling and server settings."""

    CATEGORY = "Amp NInfer"
    FUNCTION = "build"
    RETURN_TYPES = ("AMP_NINFER_ADV",)
    RETURN_NAMES = ("advanced",)
    OUTPUT_NODE = False

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "reasoning_mode": (
                    ["disabled", "low", "medium"],
                    {
                        "default": ADVANCED_DEFAULTS["reasoning_mode"],
                        "tooltip": "Qwen reasoning effort. Off by default.",
                    },
                ),
                "top_p": (
                    "FLOAT",
                    {
                        "default": ADVANCED_DEFAULTS["top_p"],
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Nucleus sampling cutoff.",
                    },
                ),
                "top_k": (
                    "INT",
                    {
                        "default": ADVANCED_DEFAULTS["top_k"],
                        "min": 0,
                        "max": 256,
                        "step": 1,
                        "tooltip": "Keep the top K tokens. 0 disables.",
                    },
                ),
                "repetition_penalty": (
                    "FLOAT",
                    {
                        "default": ADVANCED_DEFAULTS["repetition_penalty"],
                        "min": 0.5,
                        "max": 2.0,
                        "step": 0.01,
                        "tooltip": "Penalize repeated tokens. Sent only if not 1.0.",
                    },
                ),
                "frequency_penalty": (
                    "FLOAT",
                    {
                        "default": ADVANCED_DEFAULTS["frequency_penalty"],
                        "min": -2.0,
                        "max": 2.0,
                        "step": 0.01,
                        "tooltip": "Frequency penalty passed to NInfer.",
                    },
                ),
                "host": (
                    "STRING",
                    {
                        "default": ADVANCED_DEFAULTS["host"],
                        "multiline": False,
                        "tooltip": "Bind address for ninfer-serve.",
                    },
                ),
                "port": (
                    "INT",
                    {
                        "default": ADVANCED_DEFAULTS["port"],
                        "min": 1,
                        "max": 65535,
                        "step": 1,
                        "tooltip": "TCP port for ninfer-serve.",
                    },
                ),
                "device": (
                    "INT",
                    {
                        "default": ADVANCED_DEFAULTS["device"],
                        "min": 0,
                        "max": 31,
                        "step": 1,
                        "tooltip": "CUDA device index.",
                    },
                ),
                "speculative_backend": (
                    ["mtp", "off", "dflash"],
                    {
                        "default": ADVANCED_DEFAULTS["speculative_backend"],
                        "tooltip": "Speculative decoding backend.",
                    },
                ),
                "draft_tokens": (
                    "INT",
                    {
                        "default": ADVANCED_DEFAULTS["draft_tokens"],
                        "min": 1,
                        "max": 15,
                        "step": 1,
                        "tooltip": "Draft tokens when spec is not off.",
                    },
                ),
                "lm_head_draft": (
                    "BOOLEAN",
                    {
                        "default": ADVANCED_DEFAULTS["lm_head_draft"],
                        "tooltip": "Use LM-head draft when spec is enabled.",
                    },
                ),
                "server_launch_flags": (
                    "STRING",
                    {
                        "default": ADVANCED_DEFAULTS["server_launch_flags"],
                        "multiline": True,
                        "tooltip": "Extra ninfer-serve flags. Do not repeat typed options.",
                    },
                ),
                "startup_timeout": (
                    "FLOAT",
                    {
                        "default": ADVANCED_DEFAULTS["startup_timeout"],
                        "min": 1.0,
                        "max": 1800.0,
                        "step": 1.0,
                        "tooltip": "Seconds to wait for /health.",
                    },
                ),
                "request_timeout": (
                    "FLOAT",
                    {
                        "default": ADVANCED_DEFAULTS["request_timeout"],
                        "min": 1.0,
                        "max": 3600.0,
                        "step": 1.0,
                        "tooltip": "Seconds to wait for the chat response.",
                    },
                ),
                "shutdown_timeout": (
                    "FLOAT",
                    {
                        "default": ADVANCED_DEFAULTS["shutdown_timeout"],
                        "min": 0.1,
                        "max": 300.0,
                        "step": 0.5,
                        "tooltip": "Seconds for graceful process exit.",
                    },
                ),
                "force_kill_timeout": (
                    "FLOAT",
                    {
                        "default": ADVANCED_DEFAULTS["force_kill_timeout"],
                        "min": 0.1,
                        "max": 300.0,
                        "step": 0.5,
                        "tooltip": "Seconds to wait after a forced kill.",
                    },
                ),
                "vram_reclaim_timeout": (
                    "FLOAT",
                    {
                        "default": ADVANCED_DEFAULTS["vram_reclaim_timeout"],
                        "min": 0.1,
                        "max": 600.0,
                        "step": 0.5,
                        "tooltip": "Seconds to wait for VRAM to return.",
                    },
                ),
            }
        }

    def build(self, **kwargs: Any) -> tuple[dict[str, Any]]:
        return (merge_advanced(kwargs),)


__all__ = [
    "ADVANCED_CACHE_KEYS",
    "ADVANCED_DEFAULTS",
    "AmpNInferAdvancedNode",
    "merge_advanced",
]
