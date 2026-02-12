"""
Dreamstack Raster - Face Analysis Module
========================================

Face detection, landmark detection, and alignment utilities.
Uses MediaPipe for face mesh detection when available.

"""

from __future__ import annotations

from dreamstack.raster.analysis.face.alignment import (
    AlignmentResult,
    align_eyes,
    apply_transform,
    compute_inverse_transform,
    normalize_face_scale,
)
from dreamstack.raster.analysis.face.detection import (
    FaceBbox,
    FaceLandmarks,
    crop_face,
    detect_face,
    detect_faces,
    detect_landmarks,
)

__all__: list[str] = [
    # Alignment
    "AlignmentResult",
    "align_eyes",
    "normalize_face_scale",
    "compute_inverse_transform",
    "apply_transform",
    # Detection
    "FaceBbox",
    "FaceLandmarks",
    "detect_face",
    "detect_faces",
    "detect_landmarks",
    "crop_face",
]
