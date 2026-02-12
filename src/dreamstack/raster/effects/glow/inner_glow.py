"""
Inner Glow
==========

Create inner glow effects.

"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import cv2
import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


def inner_glow(
    image: NDArray[np.uint8],
    *,
    color: tuple[int, int, int] = (255, 255, 200),
    blur: float = 5.0,
    opacity: float = 0.5,
    choke: int = 0,
    source: Literal["edge", "center"] = "edge",
) -> NDArray[np.uint8]:
    """Add an inner glow effect to an image.

    Creates a glow inside the image content edges or from center.

    Args:
        image: Input image with alpha channel.
        color: Glow color (RGB).
        blur: Glow blur radius.
        opacity: Glow opacity (0.0 to 1.0).
        choke: Glow choke in pixels.
        source: Glow source ("edge" or "center").

    Returns:
        Image with inner glow applied.

    Example:
        >>> result = inner_glow(image, color=(255, 255, 0), blur=10)
    """
    # Ensure BGRA
    if image.ndim == 2:
        img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGRA)
    elif image.shape[2] == 3:
        img = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    else:
        img = image.copy()

    h, w = img.shape[:2]
    result = img.astype(np.float32)

    # Get alpha mask
    alpha = img[:, :, 3].astype(np.float32) / 255.0

    if source == "edge":
        # Edge source - glow from edges inward
        eroded = cv2.erode(alpha, None, iterations=1)
        glow = alpha - eroded
    else:
        # Center source - glow from center outward
        glow = alpha.copy()

    # Apply choke
    if choke > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (choke * 2 + 1, choke * 2 + 1)
        )
        glow = cv2.erode(glow, kernel)

    # Apply blur
    if blur > 0:
        glow = cv2.GaussianBlur(glow, (0, 0), blur)

    # Mask to content area
    glow = glow * alpha * opacity

    # Apply glow color using screen blend
    glow_color = np.array([color[2], color[1], color[0]], dtype=np.float32)

    for c in range(3):
        # Screen blend: 1 - (1 - a) * (1 - b)
        base = result[:, :, c] / 255.0
        blend = glow_color[c] / 255.0
        screened = 1 - (1 - base) * (1 - blend * glow)
        result[:, :, c] = screened * 255

    return np.clip(result, 0, 255).astype(np.uint8)
