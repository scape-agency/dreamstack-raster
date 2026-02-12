"""
Refine Edge
===========

Advanced edge refinement for selections.

"""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

from dreamstack.raster.selection.shapes.selection import Selection


def refine_edge(
    selection: Selection,
    image: NDArray[np.uint8],
    *,
    radius: int = 5,
    smooth: float = 1.0,
    feather: float = 0.0,
    contrast: float = 0.0,
    shift_edge: int = 0,
    decontaminate: bool = False,
) -> Selection:
    """Refine selection edges with advanced controls.

    Provides precise control over edge quality similar to
    Photoshop's Refine Edge/Select and Mask.

    Args:
        selection: Input selection.
        image: Source image for edge detection.
        radius: Edge detection radius.
        smooth: Edge smoothing amount (0-10).
        feather: Feather amount (0-100).
        contrast: Edge contrast increase (0-100).
        shift_edge: Shift edge inward (-100) or outward (+100).
        decontaminate: Remove color fringing.

    Returns:
        Refined selection.

    Example:
        >>> refined = refine_edge(selection, image, radius=3, feather=1)
    """
    h, w = selection.mask.shape
    mask = selection.mask.astype(np.float32) / 255.0

    # Get source image
    if image.ndim == 2:
        src = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        src = image[:, :, :3]
    else:
        src = image

    # Edge detection on image
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    # Find edges near selection boundary
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1)
    )

    # Create edge zone around selection
    dilated = cv2.dilate((mask * 255).astype(np.uint8), kernel)
    eroded = cv2.erode((mask * 255).astype(np.uint8), kernel)
    edge_zone = cv2.subtract(dilated, eroded)

    # Combine with image edges
    edge_combined = cv2.bitwise_and(edge_zone, edges)

    # Apply edge to refine mask
    if np.any(edge_combined > 0):
        # Use image edges to refine selection boundary
        try:
            refined_mask = cv2.ximgproc.guidedFilter(
                gray,
                mask,
                radius,
                0.01,
            )
        except AttributeError:
            # Fallback without ximgproc
            refined_mask = (
                cv2.bilateralFilter(
                    (mask * 255).astype(np.uint8),
                    radius * 2 + 1,
                    75,
                    75,
                ).astype(np.float32)
                / 255.0
            )
    else:
        refined_mask = mask

    # Smooth edges
    if smooth > 0:
        blur_size = int(smooth * 2 + 1)
        if blur_size % 2 == 0:
            blur_size += 1
        refined_mask = cv2.GaussianBlur(refined_mask, (blur_size, blur_size), smooth)

    # Feather
    if feather > 0:
        refined_mask = cv2.GaussianBlur(refined_mask, (0, 0), feather)

    # Contrast
    if contrast > 0:
        # Increase edge contrast
        factor = 1 + (contrast / 50.0)
        refined_mask = np.clip((refined_mask - 0.5) * factor + 0.5, 0, 1)

    # Shift edge
    if shift_edge != 0:
        if shift_edge > 0:
            # Expand
            kernel_size = abs(shift_edge) * 2 + 1
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
            )
            refined_mask = cv2.dilate(refined_mask, kernel)
        else:
            # Contract
            kernel_size = abs(shift_edge) * 2 + 1
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
            )
            refined_mask = cv2.erode(refined_mask, kernel)

    # Decontaminate colors
    if decontaminate:
        # This would require color replacement, for now just sharpen edges
        # to reduce fringing visibility
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        refined_mask = cv2.filter2D(refined_mask, -1, kernel)
        refined_mask = np.clip(refined_mask, 0, 1)

    # Convert back to uint8
    result_mask = (refined_mask * 255).astype(np.uint8)

    return Selection(mask=result_mask)
