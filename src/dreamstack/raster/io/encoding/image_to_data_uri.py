# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Image to Data URI
=================

Encode image to data URI for HTML embedding.

"""


# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

# =============================================================================
# Type Checking Imports
# =============================================================================

if TYPE_CHECKING:
    # pylint: disable=import-outside-toplevel
    from numpy.typing import NDArray

from dreamstack.raster.io.encoding.image_to_base64 import image_to_base64

ImageFormat = Literal["png", "jpeg", "webp", "gif"]


def image_to_data_uri(
    image: NDArray[np.uint8],
    image_format: ImageFormat = "png",
    *,
    quality: int = 85,
) -> str:
    """Encode image to data URI for HTML embedding.

    Returns a complete data URI that can be used directly in HTML img src.

    Args:
        image: Input image.
        image_format: Output format.
        quality: Compression quality for JPEG/WebP.

    Returns:
        Data URI string (data:image/format;base64,...).

    Example:
        >>> uri = image_to_data_uri(image, image_format="jpeg", quality=80)
        >>> html = f'<img src="{uri}" />'
    """
    b64 = image_to_base64(image, image_format, quality=quality)
    mime_type = f"image/{image_format}"
    return f"data:{mime_type};base64,{b64}"
