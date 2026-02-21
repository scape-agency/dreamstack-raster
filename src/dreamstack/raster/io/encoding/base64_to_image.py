# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Base64 to Image
===============

Decode base64 string to image array.

"""


# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

import base64
from io import BytesIO
from typing import TYPE_CHECKING

import numpy as np

# =============================================================================
# Type Checking Imports
# =============================================================================

if TYPE_CHECKING:
    # pylint: disable=import-outside-toplevel
    from numpy.typing import NDArray


def base64_to_image(b64_string: str) -> NDArray[np.uint8]:
    """Decode base64 string to image array.

    Handles both raw base64 and data URIs.

    Args:
        b64_string: Base64 encoded string or data URI.

    Returns:
        Image as numpy array (RGB).

    Example:
        >>> image = base64_to_image(b64_data)
        >>> # Or with data URI
        >>> image = base64_to_image("data:image/png;base64,...")
    """
    # pylint: disable=import-outside-toplevel
    from PIL import Image

    # Handle data URI
    if b64_string.startswith("data:"):
        # Extract base64 part after the comma
        _, b64_string = b64_string.split(",", 1)

    # Decode
    data = base64.b64decode(b64_string)
    buffer = BytesIO(data)
    pil_image = Image.open(buffer)

    return np.array(pil_image)
