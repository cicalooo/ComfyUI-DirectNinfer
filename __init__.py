"""ComfyUI custom-node package for short-lived NInfer Qwen3.8-27B prompts."""

try:
    from .nodes.ninfer_qwen_node import NInferQwenNode
except ImportError:  # Allows a direct loader/test import of this file.
    from nodes.ninfer_qwen_node import NInferQwenNode

NODE_CLASS_MAPPINGS = {"NInferQwenNode": NInferQwenNode}
NODE_DISPLAY_NAME_MAPPINGS = {"NInferQwenNode": "NInfer Qwen3.8-27B"}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
