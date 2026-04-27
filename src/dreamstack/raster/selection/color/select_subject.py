"""
Select Subject
==============

AI-powered subject selection using saliency detection.

"""

# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np

from dreamstack.raster.selection.shapes.selection import Selection

# =============================================================================
# Type Checking Imports
# =============================================================================

if TYPE_CHECKING:
    # pylint: disable=import-outside-toplevel
    from numpy.typing import NDArray


def select_subject(
    image: NDArray[np.uint8],
    *,
    refine_edges: bool = True,
    sensitivity: float = 0.5,
) -> Selection:
    """Automatically select the main subject in an image.

    Uses saliency detection and edge-aware algorithms to
    identify and select the primary subject.

    Args:
        image: Input image (BGR or BGRA).
        refine_edges: Apply edge refinement for better boundaries.
        sensitivity: Detection sensitivity (0.0-1.0).

    Returns:
        Selection of the detected subject.

    Example:
        >>> sel = select_subject(portrait_image)
        >>> cutout = sel.apply_to_image(portrait_image)
    """
    # Work with BGR
    if image.ndim == 2:
        src = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        src = image[:, :, :3]
    else:
        src = image

    h, w = src.shape[:2]

    # Use spectral residual saliency
    saliency = cv2.saliency.StaticSaliencySpectralResidual_create()  # type: ignore[attr-defined]
    success, saliency_map = saliency.computeSaliency(src)

    if not success:
        # Fallback to fine-grained saliency
        saliency = cv2.saliency.StaticSaliencyFineGrained_create()  # type: ignore[attr-defined]
        success, saliency_map = saliency.computeSaliency(src)

    if not success:
        # Return empty selection if saliency fails
        return Selection(mask=np.zeros((h, w), dtype=np.uint8))

    # Normalize saliency map
    saliency_map = (saliency_map * 255).astype(np.uint8)

    # Apply threshold based on sensitivity
    threshold = int((1.0 - sensitivity) * 128)
    _, mask = cv2.threshold(
        saliency_map,
        threshold,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    # Morphological operations to clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # Fill holes
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(mask, contours, -1, 255, -1)

    if refine_edges:
        # Edge-aware refinement using guided filter
        try:
            gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
            mask = cv2.ximgproc.guidedFilter(  # type: ignore[attr-defined]
                gray, mask.astype(np.float32), 8, 0.01
            ).astype(np.uint8)
        except AttributeError:
            # Fall back to bilateral filter if ximgproc not available
            mask = cv2.bilateralFilter(mask, 9, 75, 75)

    return Selection(mask=mask)
