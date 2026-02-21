"""Internal interpolation helper."""

# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

from typing import Literal

Interpolation = Literal["nearest", "linear", "cubic", "lanczos", "area"]


def _get_cv2_interpolation(method: Interpolation) -> int:
    """Map interpolation string to OpenCV constant."""
    import cv2

    mapping = {
        "nearest": cv2.INTER_NEAREST,
        "linear": cv2.INTER_LINEAR,
        "cubic": cv2.INTER_CUBIC,
        "lanczos": cv2.INTER_LANCZOS4,
        "area": cv2.INTER_AREA,
    }
    return mapping.get(method, cv2.INTER_LINEAR)
