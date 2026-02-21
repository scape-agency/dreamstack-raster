# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Image to Base64
===============

Encode image array to base64 string.

"""


# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

import base64
from io import BytesIO
from typing import TYPE_CHECKING, Literal

import numpy as np

# =============================================================================
# Type Checking Imports
# =============================================================================

if TYPE_CHECKING:
    # pylint: disable=import-outside-toplevel
    from numpy.typing import NDArray


ImageFormat = Literal["png", "jpeg", "webp", "gif"]


def image_to_base64(
    image: NDArray[np.uint8],
    image_format: ImageFormat = "png",
    *,
    quality: int = 85,
) -> str:
    """Encode image array to base64 string.

    Args:
        image: Input image (BGR or RGB, 3-4 channels).
        image_format: Output format.
        quality: Compression quality for JPEG/WebP.

    Returns:
        Base64 encoded string.

    Example:
        >>> b64 = image_to_base64(image, format="jpeg")
        >>> # Use for embedding in HTML/JSON
    """
    import cv2  # pylint: disable=import-outside-toplevel

    # pylint: disable=import-outside-toplevel
    from PIL import Image

    # Convert BGR to RGB for PIL
    if image.ndim == 3:
        if image.shape[2] == 3:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif image.shape[2] == 4:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
        else:
            rgb = image
    else:
        rgb = image

    pil_image = Image.fromarray(rgb)
    buffer = BytesIO()

    # Save with appropriate format
    format_upper = image_format.upper()
    if format_upper == "JPEG":
        if pil_image.mode == "RGBA":
            bg = Image.new("RGB", pil_image.size, (255, 255, 255))
            bg.paste(pil_image, mask=pil_image.split()[3])
            pil_image = bg
        pil_image.save(buffer, format="JPEG", quality=quality)
    elif format_upper == "WEBP":
        pil_image.save(buffer, format="WEBP", quality=quality)
    elif format_upper == "GIF":
        pil_image.save(buffer, format="GIF")
    else:
        pil_image.save(buffer, format="PNG")

    return base64.b64encode(buffer.getvalue()).decode("utf-8")
