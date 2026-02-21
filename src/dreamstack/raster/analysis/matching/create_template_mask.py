"""
Create Template Mask Function
=============================

Create mask from template for transparent matching.

"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def create_template_mask(
    template: NDArray[np.uint8],
    transparent_color: tuple[int, int, int] | None = None,
    threshold: int = 10,
) -> NDArray[np.uint8]:
    """Create mask from template for transparent matching.

    Parameters
    ----------
    template : NDArray[np.uint8]
        Template image.
    transparent_color : tuple[int, int, int], optional
        Color to treat as transparent. If None, uses top-left pixel.
    threshold : int, optional
        Color matching threshold. Default is 10.

    Returns
    -------
    NDArray[np.uint8]
        Binary mask (255 = use, 0 = ignore).
    """
    if transparent_color is None:
        # Use top-left corner pixel as transparent color
        if template.ndim == 3:
            transparent_color = tuple(template[0, 0])
        else:
            transparent_color = template[0, 0]

    if template.ndim == 2:
        # Grayscale
        assert transparent_color is not None
        diff = np.abs(template.astype(np.int32) - transparent_color)
        mask = (diff > threshold).astype(np.uint8) * 255
    else:
        # Color
        diff = np.sqrt(
            np.sum(
                (template.astype(np.float32) - np.array(transparent_color))
                ** 2,
                axis=2,
            )
        )
        mask = (diff > threshold).astype(np.uint8) * 255

    return mask
