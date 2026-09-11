"""Conversion of native ComfyUI IMAGE tensors to data URLs.

ComfyUI represents images as float tensors shaped ``[batch, height, width,
channels]`` with values normally in ``[0, 1]``.  NInfer accepts OpenAI-style
``image_url`` parts, so this module performs the only conversion needed by the
node and never accepts arbitrary URLs or filesystem paths.
"""

from __future__ import annotations

from io import BytesIO
import base64
import math
from typing import Any, Sequence

# Qwen2-VL / Qwen2.5-VL / Qwen3-VL merge 2×2 patches of size 14, so both
# spatial dims must be multiples of 28. Sending 1024 (not divisible by 28)
# or many unaligned Image-list frames can crash ninfer-serve (connection reset).
IMAGE_PATCH_FACTOR = 28
MIN_PIXELS = 4 * 28 * 28
# Keep vision frames small enough for 24 GB cards that already spent ~17 GiB
# on weights plus KV. Oversized media decode in ninfer-serve can abort the
# HTTP socket after /health succeeds.
MAX_PIXELS = 336 * 28 * 28
MAX_ASPECT_RATIO = 200.0
DEFAULT_VISION_MAX_SIDE = 336
# ninfer-serve on Windows routes JPEG through FFmpeg/swscaler and can heap-
# corrupt (exit 0xC0000374). Always ship RGB PNG on the wire.
SAFE_VISION_WIRE_FORMAT = "png"


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


def _normalize_hwc(array: Any) -> Any:
    """Accept common ComfyUI / torch layouts and return HWC with 1/3/4 channels."""

    import numpy as np  # type: ignore

    array = np.asarray(array)
    if array.ndim == 2:
        array = array[..., None]
    elif array.ndim == 3:
        if array.shape[0] in {1, 3, 4} and array.shape[-1] not in {1, 3, 4}:
            # CHW -> HWC
            array = np.transpose(array, (1, 2, 0))
    elif array.ndim == 4:
        raise MultimodalInputError("internal batch frames must be HWC")
    else:
        raise MultimodalInputError(
            "IMAGE must have shape [H,W], [H,W,C], [C,H,W], [B,H,W,C], or [B,C,H,W]"
        )
    if array.ndim != 3:
        raise MultimodalInputError("IMAGE frame must be HWC after normalization")
    if array.shape[0] < 1 or array.shape[1] < 1:
        raise MultimodalInputError("IMAGE dimensions must be non-empty")
    channels = int(array.shape[2])
    if channels == 2:
        # LA-style: keep L, drop weak alpha-ish second channel by repeating L.
        array = array[..., :1]
        channels = 1
    if channels not in {1, 3, 4}:
        if channels > 4:
            array = array[..., :3]
            channels = 3
        else:
            raise MultimodalInputError("IMAGE must have 1, 3, or 4 channels")
    if channels == 1:
        array = np.repeat(array, 3, axis=2)
    return array


def _batch_array(image: Any) -> Any:
    array = _as_numpy(image)
    if getattr(array, "ndim", None) == 4:
        # BCHW when channel axis is clearly not last.
        if array.shape[1] in {1, 3, 4} and array.shape[-1] not in {1, 3, 4}:
            array = array.transpose(0, 2, 3, 1)
    elif getattr(array, "ndim", None) in {2, 3}:
        array = array[None, ...]
    else:
        raise MultimodalInputError(
            "IMAGE must have shape [H,W], [H,W,C], [C,H,W], [B,H,W,C], or [B,C,H,W]"
        )
    if getattr(array, "ndim", None) != 4:
        raise MultimodalInputError("IMAGE batch must be 4D after normalization")
    if array.shape[0] < 1:
        raise MultimodalInputError("IMAGE batch and dimensions must be non-empty")
    frames = [_normalize_hwc(array[index]) for index in range(array.shape[0])]
    import numpy as np  # type: ignore

    stacked = np.stack(frames, axis=0)
    return _to_uint8(stacked)


def _round_by_factor(number: float, factor: int) -> int:
    return round(number / factor) * factor


def _ceil_by_factor(number: float, factor: int) -> int:
    return math.ceil(number / factor) * factor


def _floor_by_factor(number: float, factor: int) -> int:
    return math.floor(number / factor) * factor


def smart_resize(
    height: int,
    width: int,
    *,
    factor: int = IMAGE_PATCH_FACTOR,
    min_pixels: int = MIN_PIXELS,
    max_pixels: int = MAX_PIXELS,
) -> tuple[int, int]:
    """Qwen-VL smart_resize: factor-aligned dims within a pixel budget."""

    if factor < 1:
        raise ValueError("factor must be >= 1")
    if height < 1 or width < 1:
        raise MultimodalInputError("IMAGE dimensions must be positive")
    if min_pixels < 1 or max_pixels < 1:
        raise ValueError("pixel budget must be positive")

    if max(height, width) / min(height, width) > MAX_ASPECT_RATIO:
        if height > width:
            height = max(1, int(width * MAX_ASPECT_RATIO))
        else:
            width = max(1, int(height * MAX_ASPECT_RATIO))

    h_bar = max(factor, _round_by_factor(height, factor))
    w_bar = max(factor, _round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, _floor_by_factor(height / beta, factor))
        w_bar = max(factor, _floor_by_factor(width / beta, factor))
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = _ceil_by_factor(height * beta, factor)
        w_bar = _ceil_by_factor(width * beta, factor)
    return h_bar, w_bar


def _pixel_budget(
    *,
    image_count: int,
    max_side: int,
    max_pixels: int | None,
) -> tuple[int, int]:
    if max_pixels is None:
        if max_side > 0:
            # Align the side budget to the patch factor so 512/448-style
            # widgets do not silently expand past the intended VRAM envelope.
            aligned_side = max(IMAGE_PATCH_FACTOR, _floor_by_factor(max_side, IMAGE_PATCH_FACTOR))
            max_pixels = min(MAX_PIXELS, aligned_side * aligned_side)
        else:
            max_pixels = MAX_PIXELS
    per_image = max(MIN_PIXELS, max_pixels // max(1, image_count))
    return MIN_PIXELS, per_image


def _encode_one(
    array: Any,
    *,
    max_side: int,
    output_format: str,
    jpeg_quality: int,
    patch_factor: int = IMAGE_PATCH_FACTOR,
    min_pixels: int = MIN_PIXELS,
    max_pixels: int = MAX_PIXELS,
) -> str:
    try:
        from PIL import Image  # type: ignore
    except ImportError as exc:
        raise MultimodalInputError("Pillow is required for native IMAGE inputs") from exc

    channels = int(array.shape[2])
    if channels == 4:
        mode = "RGBA"
    elif channels == 3:
        mode = "RGB"
    else:
        mode = "L"
    pil_image = Image.fromarray(array, mode=mode)
    # Flatten alpha onto white and force 8-bit RGB. Any non-RGB wire format
    # (JPEG/YUV especially) can trip ninfer-serve's FFmpeg path.
    if pil_image.mode == "RGBA":
        background = Image.new("RGB", pil_image.size, (255, 255, 255))
        background.paste(pil_image, mask=pil_image.split()[3])
        pil_image = background
    elif pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")
    width, height = pil_image.size
    if patch_factor > 1:
        new_height, new_width = smart_resize(
            height,
            width,
            factor=patch_factor,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
        if (new_width, new_height) != (width, height):
            pil_image = pil_image.resize(
                (new_width, new_height), Image.Resampling.BILINEAR
            )
        width, height = pil_image.size
    elif max_side > 0:
        largest = max(width, height)
        if largest > max_side:
            scale = max_side / largest
            resized = (
                max(1, round(width * scale)),
                max(1, round(height * scale)),
            )
            pil_image = pil_image.resize(resized, Image.Resampling.BILINEAR)

    # Always emit RGB PNG on the wire. The vision_format widget is accepted for
    # compatibility but JPEG is rewritten because ninfer-serve can heap-corrupt
    # while swscaling JPEG frames (Windows exit 3221226356 / 0xC0000374).
    requested = output_format.lower().lstrip(".")
    if requested not in {"jpg", "jpeg", "png", "auto", ""}:
        raise MultimodalInputError("output_format must be png, jpeg, or auto")
    _ = jpeg_quality  # retained for API compatibility; unused for PNG wire format
    mime = "image/png"
    save_kwargs = {"optimize": True}

    output = BytesIO()
    pil_image.save(output, format="PNG", **save_kwargs)
    encoded = base64.b64encode(output.getvalue()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def image_batch_to_data_urls(
    image: Any,
    *,
    max_side: int = DEFAULT_VISION_MAX_SIDE,
    output_format: str = SAFE_VISION_WIRE_FORMAT,
    jpeg_quality: int = 85,
    patch_factor: int = IMAGE_PATCH_FACTOR,
    min_pixels: int | None = None,
    max_pixels: int | None = None,
) -> list[str]:
    """Encode every image in a ComfyUI IMAGE batch in order."""

    return images_to_data_urls(
        [image],
        max_side=max_side,
        output_format=output_format,
        jpeg_quality=jpeg_quality,
        patch_factor=patch_factor,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )


def images_to_data_urls(
    images: Sequence[Any],
    *,
    max_side: int = DEFAULT_VISION_MAX_SIDE,
    output_format: str = SAFE_VISION_WIRE_FORMAT,
    jpeg_quality: int = 85,
    patch_factor: int = IMAGE_PATCH_FACTOR,
    min_pixels: int | None = None,
    max_pixels: int | None = None,
) -> list[str]:
    """Encode one or more IMAGE tensors / batches, sharing a pixel budget.

    When several Image-list items are sent together, each frame is resized to a
    Qwen-aligned size whose pixels fit ``max_pixels / n`` so the vision tower
    does not OOM or abort the HTTP connection.
    """

    if max_side < 0:
        raise ValueError("max_side must not be negative")
    batches = [_batch_array(image) for image in images]
    count = sum(int(batch.shape[0]) for batch in batches)
    if count < 1:
        return []
    budget_min, budget_max = _pixel_budget(
        image_count=count, max_side=max_side, max_pixels=max_pixels
    )
    if min_pixels is not None:
        budget_min = min_pixels
    urls: list[str] = []
    for batch in batches:
        for index in range(batch.shape[0]):
            urls.append(
                _encode_one(
                    batch[index],
                    max_side=max_side,
                    output_format=output_format,
                    jpeg_quality=jpeg_quality,
                    patch_factor=patch_factor,
                    min_pixels=budget_min,
                    max_pixels=budget_max,
                )
            )
    return urls


def image_tensor_to_data_url(
    image: Any,
    *,
    max_side: int = DEFAULT_VISION_MAX_SIDE,
    output_format: str = SAFE_VISION_WIRE_FORMAT,
    jpeg_quality: int = 85,
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
    "DEFAULT_VISION_MAX_SIDE",
    "IMAGE_PATCH_FACTOR",
    "MAX_PIXELS",
    "MIN_PIXELS",
    "SAFE_VISION_WIRE_FORMAT",
    "MultimodalInputError",
    "build_image_content",
    "image_batch_to_data_urls",
    "image_tensor_to_data_url",
    "images_to_data_urls",
    "smart_resize",
]

