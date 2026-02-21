# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Layer stacking operations."""


# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    # pylint: disable=import-outside-toplevel
    from numpy.typing import NDArray


def stack_layers(
    layers: Sequence[NDArray[np.uint8]],
    *,
    resize_to_base: bool = True,
) -> NDArray[np.uint8]:
    """Stack multiple image layers using alpha compositing.

    Layers are composited from bottom to top (first layer is the base).
    All layers should be RGBA for proper alpha blending.

    Args:
        layers: Sequence of RGBA images (4 channels).
        resize_to_base: Resize all layers to match the first layer's size.

    Returns:
        Composited RGBA image.

    Example:
        >>> from dreamstack.raster.compositing import stack_layers
        >>> result = stack_layers([background, midground, foreground])
    """
    import cv2

    if not layers:
        raise ValueError("At least one layer required")

    # Start with base layer
    base = layers[0].copy()

    # Ensure RGBA
    if base.ndim == 2:
        base = cv2.cvtColor(base, cv2.COLOR_GRAY2RGBA)
    elif base.shape[2] == 3:
        base = cv2.cvtColor(base, cv2.COLOR_BGR2BGRA)

    base = base.astype(np.float32)
    h, w = base.shape[:2]

    # Composite each layer
    for layer in layers[1:]:
        overlay = layer.copy()

        # Ensure RGBA
        if overlay.ndim == 2:
            overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2RGBA)
        elif overlay.shape[2] == 3:
            overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2BGRA)

        # Resize if needed
        if resize_to_base and overlay.shape[:2] != (h, w):
            overlay = cv2.resize(
                overlay, (w, h), interpolation=cv2.INTER_LINEAR
            )

        overlay = overlay.astype(np.float32)

        # Alpha compositing (Porter-Duff over)
        src_alpha = overlay[:, :, 3:4] / 255.0
        dst_alpha = base[:, :, 3:4] / 255.0

        out_alpha = src_alpha + dst_alpha * (1 - src_alpha)

        # Avoid division by zero
        safe_alpha = np.where(out_alpha > 0, out_alpha, 1)

        rgb = (
            overlay[:, :, :3] * src_alpha
            + base[:, :, :3] * dst_alpha * (1 - src_alpha)
        ) / safe_alpha

        base[:, :, :3] = rgb
        base[:, :, 3:4] = out_alpha * 255

    return base.astype(np.uint8)
