"""
optimize_for_web
================

Optimize image for web delivery.

"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

from .compress_to_size import compress_to_size
from .compression_result import CompressionResult

if TYPE_CHECKING:
    from numpy.typing import NDArray


CompressionFormat = Literal["jpeg", "webp", "png"]


def optimize_for_web(
    image: NDArray[np.uint8],
    max_dimension: int = 1920,
    max_size_kb: int = 500,
    *,
    output_format: CompressionFormat = "webp",
) -> CompressionResult:
    """Optimize image for web delivery.

    Resizes if needed and compresses to target size.
    Defaults to WebP for best compression/quality ratio.

    Args:
        image: Input image.
        max_dimension: Maximum width or height.
        max_size_kb: Maximum file size.
        output_format: Output format (webp recommended).

    Returns:
        CompressionResult with optimized image.

    Example:
        >>> result = optimize_for_web(large_image, max_dimension=1200)
        >>> result.save("web_ready.webp")
    """
    import cv2  # pylint: disable=import-outside-toplevel

    h, w = image.shape[:2]

    # Resize if needed
    if max(w, h) > max_dimension:
        scale = max_dimension / max(w, h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        # pylint: disable=line-too-long
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)  # type: ignore[assignment]

    return compress_to_size(image, max_size_kb, format=output_format)
