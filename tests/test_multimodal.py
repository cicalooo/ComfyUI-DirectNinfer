from __future__ import annotations

import base64
from io import BytesIO

import numpy as np
import pytest
from PIL import Image

from ninfer.multimodal import (
    IMAGE_PATCH_FACTOR,
    MultimodalInputError,
    image_batch_to_data_urls,
    image_tensor_to_data_url,
    images_to_data_urls,
    smart_resize,
)


def _decode(url: str) -> Image.Image:
    encoded = url.split(",", 1)[1]
    return Image.open(BytesIO(base64.b64decode(encoded)))


def test_image_batch_preserves_order_and_resizes():
    first = np.zeros((4, 8, 3), dtype=np.float32)
    first[..., 0] = 1.0
    second = np.zeros((4, 8, 3), dtype=np.float32)
    second[..., 1] = 1.0
    urls = image_batch_to_data_urls(
        np.stack([first, second]),
        max_side=4,
        patch_factor=1,
        output_format="png",
    )
    assert len(urls) == 2
    assert all(url.startswith("data:image/png;base64,") for url in urls)
    assert _decode(urls[0]).size == (4, 2)
    assert _decode(urls[1]).getpixel((0, 0))[1] == 255


def test_single_image_helper_rejects_batch():
    image = np.zeros((2, 2, 3), dtype=np.float32)
    assert image_tensor_to_data_url(image, output_format="png").startswith(
        "data:image/png;base64,"
    )
    with pytest.raises(MultimodalInputError):
        image_tensor_to_data_url(np.zeros((2, 2, 2, 3), dtype=np.float32))


def test_channels_and_formats_are_validated():
    # 2-channel frames are normalized on the fly (treated as L -> RGB).
    urls = image_batch_to_data_urls(
        np.zeros((1, 2, 2, 2), dtype=np.float32), patch_factor=1
    )
    assert urls[0].startswith("data:image/png;base64,")
    with pytest.raises(MultimodalInputError, match="output_format"):
        image_batch_to_data_urls(
            np.zeros((1, 2, 2, 3), dtype=np.float32), output_format="webp"
        )
    # JPEG requests are rewritten to RGB PNG to avoid ninfer swscaler crashes.
    rewritten = image_batch_to_data_urls(
        np.zeros((1, 2, 2, 4), dtype=np.float32),
        output_format="jpeg",
        patch_factor=1,
    )[0]
    assert rewritten.startswith("data:image/png;base64,")


def test_smart_resize_aligns_to_qwen_patch_factor():
    from ninfer.multimodal import MAX_PIXELS

    height, width = smart_resize(1024, 1024)
    assert height % IMAGE_PATCH_FACTOR == 0
    assert width % IMAGE_PATCH_FACTOR == 0
    assert height * width <= MAX_PIXELS
    small_h, small_w = smart_resize(10, 20)
    assert small_h >= IMAGE_PATCH_FACTOR
    assert small_w >= IMAGE_PATCH_FACTOR


def test_image_list_shares_pixel_budget_and_aligns():
    from ninfer.multimodal import MAX_PIXELS

    frames = [np.ones((1, 1024, 768, 3), dtype=np.float32) for _ in range(4)]
    urls = images_to_data_urls(frames, max_side=1024, output_format="png")
    assert len(urls) == 4
    sizes = [_decode(url).size for url in urls]
    for width, height in sizes:
        assert width % IMAGE_PATCH_FACTOR == 0
        assert height % IMAGE_PATCH_FACTOR == 0
        assert width * height <= MAX_PIXELS // 4 + IMAGE_PATCH_FACTOR**2
        assert _decode(urls[0]).mode == "RGB"


def test_rgba_and_chw_are_converted_to_rgb_png():
    rgba = np.zeros((1, 56, 56, 4), dtype=np.float32)
    rgba[..., 0] = 1.0
    rgba[..., 3] = 0.5
    url = images_to_data_urls(rgba, max_side=56, output_format="jpeg")[0]
    assert url.startswith("data:image/png;base64,")
    decoded = _decode(url)
    assert decoded.mode == "RGB"
    assert decoded.size[0] % IMAGE_PATCH_FACTOR == 0
    assert decoded.size[1] % IMAGE_PATCH_FACTOR == 0

    chw = np.zeros((3, 40, 40), dtype=np.float32)
    chw[1, ...] = 1.0
    chw_url = images_to_data_urls([chw], max_side=40, output_format="auto")[0]
    assert chw_url.startswith("data:image/png;base64,")
    assert _decode(chw_url).mode == "RGB"
