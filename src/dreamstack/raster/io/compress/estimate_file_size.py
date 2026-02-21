# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - estimate_file_size
==================

Estimate compressed file size without saving.

"""


# =============================================================================
# Imports
# =============================================================================

# Import | Future
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
    output_format: CompressionFormat = "jpeg",
) -> float:
    """
    Estimate file size without saving.

    Args:
        image: Input image.
        quality: Compression quality.
        output_format: Output format.

    Returns:
        Estimated file size in kilobytes.
    """
    data = compress_image(
        image,
        quality,
        output_format=output_format,
    )
    return len(data) / 1024
