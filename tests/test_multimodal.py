from __future__ import annotations

import base64
from io import BytesIO

import numpy as np
import pytest
from PIL import Image

from ninfer.multimodal import (
    MultimodalInputError,
    image_batch_to_data_urls,
    image_tensor_to_data_url,
)


def _decode(url: str) -> Image.Image:
    encoded = url.split(",", 1)[1]
    return Image.open(BytesIO(base64.b64decode(encoded)))


def test_image_batch_preserves_order_and_resizes():
    first = np.zeros((4, 8, 3), dtype=np.float32)
    first[..., 0] = 1.0
    second = np.zeros((4, 8, 3), dtype=np.float32)
    second[..., 1] = 1.0
    urls = image_batch_to_data_urls(np.stack([first, second]), max_side=4)
    assert len(urls) == 2
    assert all(url.startswith("data:image/png;base64,") for url in urls)
    assert _decode(urls[0]).size == (4, 2)
    assert _decode(urls[1]).getpixel((0, 0))[1] == 255


def test_single_image_helper_rejects_batch():
    image = np.zeros((2, 2, 3), dtype=np.float32)
    assert image_tensor_to_data_url(image).startswith("data:image/png;base64,")
    with pytest.raises(MultimodalInputError):
        image_tensor_to_data_url(np.zeros((2, 2, 2, 3), dtype=np.float32))


def test_channels_and_formats_are_validated():
    with pytest.raises(MultimodalInputError, match="channels"):
        image_batch_to_data_urls(np.zeros((1, 2, 2, 2), dtype=np.float32))
    with pytest.raises(MultimodalInputError, match="output_format"):
        image_batch_to_data_urls(
            np.zeros((1, 2, 2, 3), dtype=np.float32), output_format="webp"
        )
    jpeg = image_batch_to_data_urls(
        np.zeros((1, 2, 2, 4), dtype=np.float32), output_format="jpeg"
    )[0]
    assert jpeg.startswith("data:image/jpeg;base64,")
