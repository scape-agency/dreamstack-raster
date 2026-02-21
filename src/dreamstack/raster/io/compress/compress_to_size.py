# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - compress_to_size
================

Compress image to fit within a target file size.

"""


# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

from io import BytesIO
from typing import TYPE_CHECKING, Literal

import numpy as np

from .compression_result import CompressionResult

if TYPE_CHECKING:
    # pylint: disable=import-outside-toplevel
    from numpy.typing import NDArray


CompressionFormat = Literal["jpeg", "webp", "png"]


def compress_to_size(
    image: NDArray[np.uint8],
    max_size_kb: int = 500,
    *,
    output_format: CompressionFormat = "jpeg",
    min_quality: int = 20,
    max_quality: int = 95,
) -> CompressionResult:
    """Compress image to fit within a target file size.

    Iteratively reduces quality until the image fits within
    the specified size limit.

    Args:
        image: Input image (BGR, 3 channels).
        max_size_kb: Maximum file size in kilobytes.
        output_format: Output format (jpeg, webp, png).
        min_quality: Minimum acceptable quality.
        max_quality: Starting quality.

    Returns:
        CompressionResult with compressed data and metadata.

    Example:
        >>> from dreamstack.raster.io import compress_to_size
        >>> result = compress_to_size(image, max_size_kb=200)
        >>> result.save("optimized.jpg")
        >>> print(f"Final size: {result.size_kb:.1f}KB at quality {result.quality}")
    """
    import cv2

    # pylint: disable=import-outside-toplevel
    from PIL import Image

    # Convert BGR to RGB for PIL
    if image.ndim == 3 and image.shape[2] == 3:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif image.ndim == 3 and image.shape[2] == 4:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
    else:
        rgb = image

    pil_image = Image.fromarray(rgb)

    # Determine format-specific settings
    if output_format == "jpeg":
        pil_format = "JPEG"
        quality_key = "quality"
    elif output_format == "webp":
        pil_format = "WEBP"
        quality_key = "quality"
    else:
        # PNG uses compression level, return immediately
        buffer = BytesIO()
        pil_image.save(buffer, format="PNG", compress_level=9)
        data = buffer.getvalue()
        return CompressionResult(
            data=data,
            size_kb=len(data) / 1024,
            quality=100,
            format="png",
            iterations=1,
        )

    # Binary search for optimal quality
    quality = max_quality
    iterations = 0
    best_data = None
    best_quality = max_quality

    max_iterations = (max_quality - min_quality) // 5 + 5

    while quality >= min_quality and iterations < max_iterations:
        iterations += 1

        buffer = BytesIO()

        # Handle RGBA for JPEG
        if pil_format == "JPEG" and pil_image.mode == "RGBA":
            # Create white background
            bg = Image.new("RGB", pil_image.size, (255, 255, 255))
            bg.paste(pil_image, mask=pil_image.split()[3])
            bg.save(buffer, format=pil_format, **{quality_key: quality})
        else:
            pil_image.save(buffer, format=pil_format, **{quality_key: quality})

        data = buffer.getvalue()
        size_kb = len(data) / 1024

        if size_kb <= max_size_kb:
            # Found acceptable size
            best_data = data
            best_quality = quality
            break

        # Reduce quality
        quality -= 5
        best_data = data
        best_quality = quality + 5

    # Use last successful compression
    if best_data is None:
        # Even at min quality, use whatever we got
        buffer = BytesIO()
        if pil_format == "JPEG" and pil_image.mode == "RGBA":
            bg = Image.new("RGB", pil_image.size, (255, 255, 255))
            bg.paste(pil_image, mask=pil_image.split()[3])
            bg.save(buffer, format=pil_format, **{quality_key: min_quality})
        else:
            pil_image.save(
                buffer, format=pil_format, **{quality_key: min_quality}
            )
        best_data = buffer.getvalue()
        best_quality = min_quality

    return CompressionResult(
        data=best_data,
        size_kb=len(best_data) / 1024,
        quality=best_quality,
        format=output_format,
        iterations=iterations,
    )
