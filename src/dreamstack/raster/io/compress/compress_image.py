# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - compress_image
==============

Compress image at a specific quality level.

"""


# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

from io import BytesIO
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    # pylint: disable=import-outside-toplevel
    from numpy.typing import NDArray


CompressionFormat = Literal["jpeg", "webp", "png"]


def compress_image(
    image: NDArray[np.uint8],
    quality: int = 85,
    *,
    output_format: CompressionFormat = "jpeg",
) -> bytes:
    """Compress image at a specific quality level.

    Simple compression without file size targeting.

    Args:
        image: Input image (BGR, 3 channels).
        quality: Compression quality (1-100).
        output_format: Output format.

    Returns:
        Compressed image bytes.

    Example:
        >>> data = compress_image(image, quality=80)
        >>> with open("output.jpg", "wb") as f:
        ...     f.write(data)
    """
    import cv2  # pylint: disable=import-outside-toplevel

    # pylint: disable=import-outside-toplevel
    from PIL import Image  # pylint: disable=import-outside-toplevel

    # Convert BGR to RGB
    if image.ndim == 3 and image.shape[2] == 3:
        rgb = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB,
        )
    else:
        rgb = image

    pil_image = Image.fromarray(rgb)
    buffer = BytesIO()

    if output_format == "jpeg":
        if pil_image.mode == "RGBA":
            bg = Image.new(
                "RGB",
                pil_image.size,
                (255, 255, 255),
            )
            bg.paste(pil_image, mask=pil_image.split()[3])
            pil_image = bg
        pil_image.save(buffer, format="JPEG", quality=quality)
    elif output_format == "webp":
        pil_image.save(buffer, format="WEBP", quality=quality)
    else:
        pil_image.save(buffer, format="PNG", compress_level=9)

    return buffer.getvalue()
