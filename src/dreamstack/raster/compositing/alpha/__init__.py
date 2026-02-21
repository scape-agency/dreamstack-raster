# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Alpha Compositing Operations
============================

Alpha channel operations for compositing layers.

"""


# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray


def alpha_composite(
    foreground: NDArray[np.uint8],
    background: NDArray[np.uint8],
    *,
    position: tuple[int, int] = (0, 0),
) -> NDArray[np.uint8]:
    """Composite foreground over background using alpha.

    Parameters
    ----------
    foreground : NDArray[np.uint8]
        Foreground image with alpha (BGRA).
    background : NDArray[np.uint8]
        Background image (BGR or BGRA).
    position : tuple[int, int], optional
        Position to place foreground (x, y). Default is (0, 0).

    Returns
    -------
    NDArray[np.uint8]
        Composited image.
    """
    x, y = position

    # Ensure both have alpha
    if foreground.shape[2] == 3:
        fg: NDArray[np.uint8] = cv2.cvtColor(  # type: ignore[assignment]
            foreground, cv2.COLOR_BGR2BGRA
        )
    else:
        fg = foreground

    if background.shape[2] == 3:
        result: NDArray[np.uint8] = cv2.cvtColor(  # type: ignore[assignment]
            background, cv2.COLOR_BGR2BGRA
        )
    else:
        result = background.copy()

    h, w = fg.shape[:2]
    bg_h, bg_w = result.shape[:2]

    # Calculate overlap region
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(bg_w, x + w)
    y2 = min(bg_h, y + h)

    if x1 >= x2 or y1 >= y2:
        return result

    # Source region in foreground
    fx1 = x1 - x
    fy1 = y1 - y
    fx2 = fx1 + (x2 - x1)
    fy2 = fy1 + (y2 - y1)

    # Get regions
    fg_region = fg[fy1:fy2, fx1:fx2]
    bg_region = result[y1:y2, x1:x2]

    # Extract alpha
    fg_alpha = fg_region[:, :, 3:4].astype(np.float32) / 255.0
    bg_alpha = bg_region[:, :, 3:4].astype(np.float32) / 255.0

    # Alpha compositing formula
    out_alpha = fg_alpha + bg_alpha * (1 - fg_alpha)
    out_alpha_safe = np.maximum(out_alpha, 1e-6)

    # Composite RGB
    fg_rgb = fg_region[:, :, :3].astype(np.float32)
    bg_rgb = bg_region[:, :, :3].astype(np.float32)

    out_rgb = (
        fg_rgb * fg_alpha + bg_rgb * bg_alpha * (1 - fg_alpha)
    ) / out_alpha_safe

    # Combine
    result[y1:y2, x1:x2, :3] = np.clip(out_rgb, 0, 255).astype(np.uint8)
    result[y1:y2, x1:x2, 3:4] = np.clip(out_alpha * 255, 0, 255).astype(
        np.uint8
    )

    return result


def extract_alpha(image: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Extract alpha channel from image.

    Parameters
    ----------
    image : NDArray[np.uint8]
        Input image (BGRA).

    Returns
    -------
    NDArray[np.uint8]
        Alpha channel.
    """
    if image.ndim == 2:
        return np.full(image.shape, 255, dtype=np.uint8)

    if image.shape[2] == 4:
        return image[:, :, 3]

    return np.full(image.shape[:2], 255, dtype=np.uint8)


def set_alpha(
    image: NDArray[np.uint8],
    alpha: NDArray[np.uint8],
) -> NDArray[np.uint8]:
    """Set alpha channel on an image.

    Parameters
    ----------
    image : NDArray[np.uint8]
        Input image (BGR or BGRA).
    alpha : NDArray[np.uint8]
        Alpha channel.

    Returns
    -------
    NDArray[np.uint8]
        Image with alpha (BGRA).
    """
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGRA)  # type: ignore[assignment]
    elif image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)  # type: ignore[assignment]
    else:
        image = image.copy()

    image[:, :, 3] = alpha
    return image


def premultiply_alpha(image: NDArray[np.uint8]) -> NDArray[np.float32]:
    """Premultiply RGB by alpha.

    Parameters
    ----------
    image : NDArray[np.uint8]
        BGRA image.

    Returns
    -------
    NDArray[np.float32]
        Premultiplied float image.
    """
    result = image.astype(np.float32) / 255.0

    if image.shape[2] == 4:
        alpha = result[:, :, 3:4]
        result[:, :, :3] *= alpha

    return result


def unpremultiply_alpha(image: NDArray[np.float32]) -> NDArray[np.uint8]:
    """Unpremultiply RGB by alpha.

    Parameters
    ----------
    image : NDArray[np.float32]
        Premultiplied float image.

    Returns
    -------
    NDArray[np.uint8]
        BGRA image.
    """
    result = image.copy()

    if image.shape[2] == 4:
        alpha = result[:, :, 3:4]
        alpha_safe = np.maximum(alpha, 1e-6)
        result[:, :, :3] /= alpha_safe

    return np.clip(result * 255, 0, 255).astype(np.uint8)
