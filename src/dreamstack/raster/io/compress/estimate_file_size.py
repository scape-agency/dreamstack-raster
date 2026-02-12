"""
estimate_file_size
==================

Estimate compressed file size without saving.

"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

from .compress_image import compress_image

if TYPE_CHECKING:
    from numpy.typing import NDArray


CompressionFormat = Literal["jpeg", "webp", "png"]


def estimate_file_size(
    image: NDArray[np.uint8],
    quality: int = 85,
    *,
    format: CompressionFormat = "jpeg",
) -> float:
    """Estimate file size without saving.

    Args:
        image: Input image.
        quality: Compression quality.
        format: Output format.

    Returns:
        Estimated file size in kilobytes.
    """
    data = compress_image(image, quality, format=format)
    return len(data) / 1024
