"""Conversion of native ComfyUI IMAGE tensors to data URLs.

ComfyUI represents images as float tensors shaped ``[batch, height, width,
channels]`` with values normally in ``[0, 1]``.  NInfer accepts OpenAI-style
``image_url`` parts, so this module performs the only conversion needed by the
node and never accepts arbitrary URLs or filesystem paths.
"""

from __future__ import annotations

from io import BytesIO
import base64
from typing import Any, Sequence


class MultimodalInputError(ValueError):
    """The supplied ComfyUI image tensor cannot be encoded."""


def _as_numpy(image: Any) -> Any:
    try:
        import torch  # type: ignore

        if isinstance(image, torch.Tensor):
            return image.detach().to(device="cpu").contiguous().numpy()
    except ImportError:
        pass
    except Exception as exc:
        raise MultimodalInputError(f"could not move IMAGE tensor to CPU: {exc}") from exc
    try:
        import numpy as np  # type: ignore

        return np.asarray(image)
    except ImportError as exc:
        raise MultimodalInputError(
            "IMAGE conversion requires PyTorch or NumPy in the ComfyUI environment"
        ) from exc
    except Exception as exc:
        raise MultimodalInputError(f"could not interpret IMAGE input: {exc}") from exc


def _to_uint8(array: Any) -> Any:
    import numpy as np  # type: ignore

    if array.dtype.kind in {"f", "c"}:
        array = np.nan_to_num(array, nan=0.0, posinf=1.0, neginf=0.0)
        # ComfyUI IMAGE tensors are [0, 1].  Accommodate already-scaled test
        # or plugin tensors without silently wrapping values.
        maximum = float(np.max(array)) if array.size else 0.0
        if maximum <= 1.0:
            array = array * 255.0
    array = np.clip(array, 0, 255).astype(np.uint8)
    return array


def _batch_array(image: Any) -> Any:
    array = _as_numpy(image)
    if getattr(array, "ndim", None) == 3:
        array = array[None, ...]
    if getattr(array, "ndim", None) != 4:
        raise MultimodalInputError(
            "IMAGE must have shape [B,H,W,C] or [H,W,C]"
        )
    if array.shape[0] < 1 or array.shape[1] < 1 or array.shape[2] < 1:
        raise MultimodalInputError("IMAGE batch and dimensions must be non-empty")
    if array.shape[3] not in {1, 3, 4}:
        raise MultimodalInputError("IMAGE must have 1, 3, or 4 channels")
    return _to_uint8(array)


def _encode_one(
    array: Any,
    *,
    max_side: int,
    output_format: str,
    jpeg_quality: int,
) -> str:
    try:
        from PIL import Image  # type: ignore
    except ImportError as exc:
        raise MultimodalInputError("Pillow is required for native IMAGE inputs") from exc

    channels = int(array.shape[2])
    mode = {1: "L", 3: "RGB", 4: "RGBA"}[channels]
    pil_image = Image.fromarray(array, mode=mode)
    if max_side > 0:
        width, height = pil_image.size
        largest = max(width, height)
        if largest > max_side:
            scale = max_side / largest
            resized = (
                max(1, round(width * scale)),
                max(1, round(height * scale)),
            )
            pil_image = pil_image.resize(resized, Image.Resampling.LANCZOS)

    normalized_format = output_format.lower().lstrip(".")
    if normalized_format in {"jpg", "jpeg"}:
        normalized_format = "jpeg"
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")
        mime = "image/jpeg"
        save_kwargs = {"quality": max(1, min(95, jpeg_quality)), "optimize": True}
    elif normalized_format == "png":
        mime = "image/png"
        save_kwargs = {"optimize": True}
    else:
        raise MultimodalInputError("output_format must be png or jpeg")

    output = BytesIO()
    pil_image.save(output, format=normalized_format.upper(), **save_kwargs)
    encoded = base64.b64encode(output.getvalue()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def image_batch_to_data_urls(
    image: Any,
    *,
    max_side: int = 1024,
    output_format: str = "png",
    jpeg_quality: int = 90,
) -> list[str]:
    """Encode every image in a ComfyUI IMAGE batch in order."""

    if max_side < 0:
        raise ValueError("max_side must not be negative")
    array = _batch_array(image)
    return [
        _encode_one(
            array[index],
            max_side=max_side,
            output_format=output_format,
            jpeg_quality=jpeg_quality,
        )
        for index in range(array.shape[0])
    ]


def image_tensor_to_data_url(
    image: Any,
    *,
    max_side: int = 1024,
    output_format: str = "png",
    jpeg_quality: int = 90,
) -> str:
    """Encode one IMAGE tensor; reject accidental multi-image batches."""

    urls = image_batch_to_data_urls(
        image,
        max_side=max_side,
        output_format=output_format,
        jpeg_quality=jpeg_quality,
    )
    if len(urls) != 1:
        raise MultimodalInputError(
            "image_tensor_to_data_url received a batch; use image_batch_to_data_urls"
        )
    return urls[0]


def build_image_content(image_urls: Sequence[str]) -> list[dict[str, Any]]:
    """Build ordered OpenAI image parts after conversion."""

    for url in image_urls:
        if not isinstance(url, str) or not url.startswith("data:"):
            raise MultimodalInputError("only local data URLs are accepted")
    return [
        {"type": "image_url", "image_url": {"url": url}}
        for url in image_urls
    ]


__all__ = [
    "MultimodalInputError",
    "build_image_content",
    "image_batch_to_data_urls",
    "image_tensor_to_data_url",
]
