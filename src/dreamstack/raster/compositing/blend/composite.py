"""
Composite Function
==================

Composite layers with blend mode and alpha.

"""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

from dreamstack.raster.compositing.blend.blend import blend
from dreamstack.raster.compositing.blend.blend_mode import BlendMode


def composite(
    base: NDArray[np.uint8],
    overlay: NDArray[np.uint8],
    *,
    mode: BlendMode = BlendMode.NORMAL,
    opacity: float = 1.0,
    position: tuple[int, int] = (0, 0),
) -> NDArray[np.uint8]:
    """Composite overlay onto base with blend mode and alpha.

    Full compositing with blend mode, opacity, and positioning.
    Respects alpha channels in both images.

    Args:
        base: Base image (bottom layer).
        overlay: Overlay image (top layer).
        mode: Blend mode to apply.
        opacity: Overlay opacity (0.0-1.0).
        position: Position to place overlay (x, y).

    Returns:
        Composited image.

    Example:
        >>> result = composite(
        ...     background,
        ...     foreground,
        ...     mode=BlendMode.OVERLAY,
        ...     opacity=0.8,
        ...     position=(100, 100),
        ... )
    """
    h, w = base.shape[:2]
    x, y = position
    oh, ow = overlay.shape[:2]

    # Ensure RGBA
    if base.ndim == 2:
        result: NDArray[np.uint8] = cv2.cvtColor(base, cv2.COLOR_GRAY2BGRA)  # type: ignore[assignment]
    elif base.shape[2] == 3:
        result = cv2.cvtColor(base, cv2.COLOR_BGR2BGRA)  # type: ignore[assignment]
    else:
        result = base.copy()

    if overlay.ndim == 2:
        over: NDArray[np.uint8] = cv2.cvtColor(overlay, cv2.COLOR_GRAY2BGRA)  # type: ignore[assignment]
    elif overlay.shape[2] == 3:
        over = cv2.cvtColor(overlay, cv2.COLOR_BGR2BGRA)
    else:
        over = overlay.copy()

    # Calculate overlap region
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(w, x + ow)
    y2 = min(h, y + oh)

    if x1 >= x2 or y1 >= y2:
        return result

    # Source region in overlay
    ox1 = x1 - x
    oy1 = y1 - y
    ox2 = ox1 + (x2 - x1)
    oy2 = oy1 + (y2 - y1)

    # Extract regions
    base_region = result[y1:y2, x1:x2]
    over_region = over[oy1:oy2, ox1:ox2]

    # Apply blend mode on RGB
    blended_rgb = blend(
        base_region[:, :, :3].copy(),  # type: ignore[arg-type]
        over_region[:, :, :3].copy(),  # type: ignore[arg-type]
        mode,
        opacity=1.0,  # We'll handle opacity with alpha
    )

    # Handle alpha compositing
    base_alpha = base_region[:, :, 3:4].astype(np.float32) / 255.0
    over_alpha = over_region[:, :, 3:4].astype(np.float32) / 255.0 * opacity

    # Porter-Duff over
    out_alpha = over_alpha + base_alpha * (1 - over_alpha)
    safe_alpha = np.where(out_alpha > 0, out_alpha, 1)

    blended_f = blended_rgb.astype(np.float32)
    base_f = base_region[:, :, :3].astype(np.float32)

    out_rgb = (
        blended_f * over_alpha + base_f * base_alpha * (1 - over_alpha)
    ) / safe_alpha

    result[y1:y2, x1:x2, :3] = np.clip(out_rgb, 0, 255).astype(np.uint8)
    result[y1:y2, x1:x2, 3:4] = np.clip(out_alpha * 255, 0, 255).astype(
        np.uint8
    )

    return result
