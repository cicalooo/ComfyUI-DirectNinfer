"""ComfyUI custom-node package for short-lived NInfer prompt enhancement."""

__version__ = "0.1.0"

try:
    from .ninfer.models import (
        DEFAULT_MODELS_DIR,
        EMPTY_MODEL_PLACEHOLDER,
        native_model_names,
        scan_ninfer_models,
    )
    from .nodes.ninfer_advanced_node import AmpNInferAdvancedNode
    from .nodes.ninfer_qwen_node import NInferQwenNode
except ImportError:  # Allows a direct loader/test import of this file.
    from ninfer.models import (
        DEFAULT_MODELS_DIR,
        EMPTY_MODEL_PLACEHOLDER,
        native_model_names,
        scan_ninfer_models,
    )
    from nodes.ninfer_advanced_node import AmpNInferAdvancedNode
    from nodes.ninfer_qwen_node import NInferQwenNode

NODE_CLASS_MAPPINGS = {
    "NInferQwenNode": NInferQwenNode,
    "AmpNInferAdvanced": AmpNInferAdvancedNode,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "NInferQwenNode": "DirectNinfer",
    "AmpNInferAdvanced": "DirectNinfer Advanced",
}
WEB_DIRECTORY = "./web"

try:
    from aiohttp import web
    from server import PromptServer
except ImportError:
    PromptServer = None
    web = None

_prompt_server = getattr(PromptServer, "instance", None) if PromptServer is not None else None
if _prompt_server is not None and web is not None:

    @_prompt_server.routes.post("/amp_ninfer/scan_models")
    async def _amp_ninfer_scan_models(request):
        try:
            data = await request.json()
        except Exception:
            data = {}
        if not isinstance(data, dict):
            data = {}
        directory = data.get("dir")
        if not isinstance(directory, str) or not directory.strip():
            directory = DEFAULT_MODELS_DIR
        models = scan_ninfer_models(directory)
        if models == [EMPTY_MODEL_PLACEHOLDER]:
            native = native_model_names()
            if native and native != [EMPTY_MODEL_PLACEHOLDER]:
                models = native
        payload = {"models": models}
        if models == [EMPTY_MODEL_PLACEHOLDER]:
            payload["error"] = f"No .ninfer files found under {directory}"
        return web.json_response(payload)


__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "WEB_DIRECTORY",
    "__version__",
]
