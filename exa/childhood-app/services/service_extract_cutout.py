"""
Extract Cutout Service
======================

Extract and scale bounding box cutouts from images.
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image


def extract_cutout(
    image: np.ndarray,
    bbox: tuple[int, int, int, int],
    margin: int = 50,
    max_size: int = 1200,
    segment_size: tuple[int, int] = (250, 250),
    segment_align: bool = True,
) -> tuple[Image.Image, dict]:
    """Extract and scale bounding box cutout with smart sizing.

    The largest dimension is scaled to max_size.
    The smallest dimension is rounded up to a multiple of segment size.
    Bounding box is expanded equally on both sides, filling with
    transparent pixels where the source image has no data.

    Parameters
    ----------
    image : np.ndarray
        Source image (BGR).
    bbox : tuple[int, int, int, int]
        Bounding box (x, y, width, height).
    margin : int
        Margin around bbox (also max random offset). Default 50.
    max_size : int
        Maximum size for largest dimension. Default 1200.
    segment_size : tuple[int, int]
        Segment size for alignment. Default (250, 250).
    segment_align : bool
        Align smallest dimension to segment multiple. Default True.

    Returns
    -------
    tuple[Image.Image, dict]
        Scaled cutout with alpha channel and metadata about sizing.
    """
    img_h, img_w = image.shape[:2]
    x, y, bw, bh = bbox

    # Add margin to bounding box
    x1_desired = x - margin
    y1_desired = y - margin
    x2_desired = x + bw + margin
    y2_desired = y + bh + margin

    crop_w = x2_desired - x1_desired
    crop_h = y2_desired - y1_desired

    # Determine target dimensions
    # Largest dimension -> max_size
    if crop_w >= crop_h:
        # Width is largest
        scale = max_size / crop_w
        target_w = max_size
        target_h = int(crop_h * scale)
    else:
        # Height is largest
        scale = max_size / crop_h
        target_h = max_size
        target_w = int(crop_w * scale)

    # Align smallest dimension to segment multiple
    if segment_align:
        seg_w, seg_h = segment_size
        if target_w <= target_h:
            # Width is smallest, align to seg_w
            target_w = ((target_w + seg_w - 1) // seg_w) * seg_w
        else:
            # Height is smallest, align to seg_h
            target_h = ((target_h + seg_h - 1) // seg_h) * seg_h

    # Recalculate crop dimensions to match target aspect ratio
    target_aspect = target_w / target_h
    crop_aspect = crop_w / crop_h

    if target_aspect > crop_aspect:
        # Need wider crop
        new_crop_w = int(crop_h * target_aspect)
        expand = (new_crop_w - crop_w) // 2
        x1_desired -= expand
        x2_desired += expand
        crop_w = new_crop_w
    elif target_aspect < crop_aspect:
        # Need taller crop
        new_crop_h = int(crop_w / target_aspect)
        expand = (new_crop_h - crop_h) // 2
        y1_desired -= expand
        y2_desired += expand
        crop_h = new_crop_h

    # Calculate actual crop bounds (clamped to image)
    x1_actual = max(0, x1_desired)
    y1_actual = max(0, y1_desired)
    x2_actual = min(img_w, x2_desired)
    y2_actual = min(img_h, y2_desired)

    # Crop from source image
    crop = image[y1_actual:y2_actual, x1_actual:x2_actual]

    # Convert BGR to RGB
    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

    # Convert to PIL with alpha
    pil_crop = Image.fromarray(crop_rgb).convert("RGBA")

    # Create full-size canvas with transparency
    full_crop = Image.new("RGBA", (crop_w, crop_h), (0, 0, 0, 0))

    # Calculate paste position for the actual crop
    paste_x = x1_actual - x1_desired
    paste_y = y1_actual - y1_desired
    full_crop.paste(pil_crop, (paste_x, paste_y))

    # Scale to target size
    result = full_crop.resize((target_w, target_h), Image.Resampling.LANCZOS)

    # Metadata about the cutout
    metadata = {
        "original_bbox": [x, y, bw, bh],
        "margin": margin,
        "crop_bounds": [x1_desired, y1_desired, x2_desired, y2_desired],
        "actual_bounds": [x1_actual, y1_actual, x2_actual, y2_actual],
        "target_size": [target_w, target_h],
        "has_padding": (
            x1_desired < 0
            or y1_desired < 0
            or x2_desired > img_w
            or y2_desired > img_h
        ),
    }

    return result, metadata
