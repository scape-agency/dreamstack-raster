"""
Data URI to Image
=================

Decode data URI to image array.

"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

from dreamstack.raster.io.encoding.base64_to_image import base64_to_image


def data_uri_to_image(data_uri: str) -> NDArray[np.uint8]:
    """Decode data URI to image array.

    Alias for base64_to_image that explicitly expects a data URI.

    Args:
        data_uri: Data URI string.

    Returns:
        Image as numpy array (RGB).
    """
    return base64_to_image(data_uri)
