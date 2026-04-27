"""Translate by percentage operation."""

# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .translate import translate


def translate_percentage(
    image: NDArray[np.uint8],
    tx_percent: float = 0.0,
    ty_percent: float = 0.0,
    *,
    border_mode: str = "constant",
    border_value: int | tuple[int, int, int] = 0,
) -> NDArray[np.uint8]:
    """Translate image by percentage of dimensions.

    Shifts image by a fraction of its width/height.
    Useful for consistent transformations across different image sizes.

    Parameters
    ----------
    image : NDArray[np.uint8]
        Input image.
    tx_percent : float, optional
        Translation as fraction of width (-1.0 to 1.0). Default is 0.0.
    ty_percent : float, optional
        Translation as fraction of height (-1.0 to 1.0). Default is 0.0.
    border_mode : str, optional
        Border handling mode. Default is "constant".
    border_value : int or tuple, optional
        Fill value for constant border. Default is 0.

    Returns
    -------
    NDArray[np.uint8]
        Translated image.

    Examples
    --------
    >>> # Shift image by 25% of width and 12.5% of height
    >>> translated = translate_percentage(img, tx_percent=0.25, ty_percent=0.125)
    """
    h, w = image.shape[:2]
    tx = int(w * tx_percent)
    ty = int(h * ty_percent)
    return translate(image, tx, ty, border_mode=border_mode, border_value=border_value)
