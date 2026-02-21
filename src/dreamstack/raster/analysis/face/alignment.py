# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Face Alignment Operations
=========================

Face alignment utilities for consistent face positioning.
Includes eye alignment and scale normalization.

"""


# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass
class AlignmentResult:
    """Result of face alignment operation.

    Attributes:
        image: Aligned image.
        transform_matrix: 3x3 transformation matrix applied.
        eye_positions: Detected eye positions (left, right).
        scale_factor: Scale factor applied.
        rotation_angle: Rotation angle in degrees.
    """

    image: NDArray[np.uint8]
    transform_matrix: NDArray[np.float64]
    eye_positions: tuple[tuple[int, int], tuple[int, int]] | None = None
    scale_factor: float = 1.0
    rotation_angle: float = 0.0


def align_eyes(
    image: NDArray[np.uint8],
    landmarks: list[tuple[int, int]] | None = None,
    target_eye_distance: float | None = None,
) -> AlignmentResult:
    """Align eyes horizontally in the image.

    Rotates the image so that eyes are on the same horizontal line.
    Optionally scales to achieve a target inter-eye distance.

    Args:
        image: Input image (BGR, 3 channels).
        landmarks: Optional eye landmark positions [(left_eye), (right_eye)].
        target_eye_distance: Optional target distance between eyes in pixels.

    Returns:
        AlignmentResult with aligned image and transformation info.

    Example:
        >>> from dreamstack.raster.analysis.face import align_eyes
        >>> result = align_eyes(image)
        >>> aligned_image = result.image
        >>> print(f"Rotated {result.rotation_angle:.1f} degrees")
    """
    import cv2  # pylint: disable=import-outside-toplevel

    # Detect landmarks if not provided
    if landmarks is None:
        landmarks = _detect_eye_landmarks(image)

    if landmarks is None or len(landmarks) < 2:
        # Cannot align without eye positions
        return AlignmentResult(
            image=image.copy(),
            transform_matrix=np.eye(3),
            eye_positions=None,
        )

    # Extract eye positions
    left_eye = np.array(landmarks[0])
    right_eye = np.array(landmarks[1])

    # Calculate rotation angle to align eyes horizontally
    delta = right_eye - left_eye
    angle = np.degrees(np.arctan2(delta[1], delta[0]))

    # Get center between eyes
    center = ((left_eye + right_eye) / 2).astype(int)

    # Compute scale if target distance specified
    current_distance = np.linalg.norm(delta)
    if target_eye_distance is not None and current_distance > 0:
        scale = target_eye_distance / current_distance
    else:
        scale = 1.0

    # Create transformation matrix
    M = cv2.getRotationMatrix2D(  # pylint: disable=invalid-name
        tuple(center), float(angle), float(scale)
    )

    # Apply transformation
    h, w = image.shape[:2]
    aligned = np.asarray(
        cv2.warpAffine(
            image,
            M,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        ),
        dtype=np.uint8,
    )

    # Convert to 3x3 matrix for completeness
    transform_3x3 = np.vstack([M, [0, 0, 1]])

    return AlignmentResult(
        image=aligned,
        transform_matrix=transform_3x3,
        eye_positions=(tuple(left_eye), tuple(right_eye)),
        scale_factor=float(scale),
        rotation_angle=float(angle),
    )


def _detect_eye_landmarks(
    image: NDArray[np.uint8],
) -> list[tuple[int, int]] | None:
    """Detect eye landmarks using MediaPipe.

    Internal function to detect eye positions.
    """
    try:
        # pylint: disable=import-outside-toplevel
        import mediapipe as mp  # type: ignore[import-not-found]
    except ImportError:
        return None

    mp_face_mesh = mp.solutions.face_mesh

    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        min_detection_confidence=0.5,
    ) as face_mesh:
        # MediaPipe expects RGB
        import cv2  # pylint: disable=import-outside-toplevel

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return None

        landmarks = results.multi_face_landmarks[0]
        h, w = image.shape[:2]

        # Eye landmark indices for MediaPipe Face Mesh
        # pylint: disable=invalid-name
        LEFT_EYE_CENTER = 468
        RIGHT_EYE_CENTER = 473
        LEFT_EYE_INNER = 133
        RIGHT_EYE_INNER = 362
        # pylint: enable=invalid-name

        try:
            left = landmarks.landmark[LEFT_EYE_CENTER]
            right = landmarks.landmark[RIGHT_EYE_CENTER]
        except IndexError:
            left = landmarks.landmark[LEFT_EYE_INNER]
            right = landmarks.landmark[RIGHT_EYE_INNER]

        return [
            (int(left.x * w), int(left.y * h)),
            (int(right.x * w), int(right.y * h)),
        ]


def normalize_face_scale(
    image: NDArray[np.uint8],
    target_size: tuple[int, int] = (512, 512),
    face_bbox: tuple[int, int, int, int] | None = None,
    face_ratio: float = 0.6,
) -> AlignmentResult:
    """Normalize scale of face in image.

    Centers and scales the face to occupy a consistent portion of the frame.

    Args:
        image: Input image.
        target_size: Target output size (width, height).
        face_bbox: Optional face bounding box (x1, y1, x2, y2).
        face_ratio: Target ratio of face size to image size.

    Returns:
        AlignmentResult with normalized image.

    Example:
        >>> # Create 512x512 image with face centered and filling 60%
        >>> result = normalize_face_scale(image, (512, 512), face_ratio=0.6)
    """
    import cv2  # pylint: disable=import-outside-toplevel

    h, w = image.shape[:2]
    target_w, target_h = target_size

    # Detect face if bbox not provided
    if face_bbox is None:
        face_bbox = _detect_face_bbox(image)

    if face_bbox is not None:
        x1, y1, x2, y2 = face_bbox
        face_size = max(x2 - x1, y2 - y1)

        # Calculate scale to achieve target face ratio
        target_face_size = min(target_w, target_h) * face_ratio
        scale = target_face_size / face_size if face_size > 0 else 1.0

        # Calculate face center
        face_center = ((x1 + x2) / 2, (y1 + y2) / 2)
    else:
        # No face detected, use center crop
        scale = min(target_w / w, target_h / h)
        face_center = (w / 2, h / 2)

    # Create transformation matrix
    M = np.array(  # pylint: disable=invalid-name
        [
            [scale, 0, target_w / 2 - scale * face_center[0]],
            [0, scale, target_h / 2 - scale * face_center[1]],
        ],
        dtype=np.float32,
    )

    # Apply transformation
    normalized = np.asarray(
        cv2.warpAffine(
            image,
            M,
            target_size,
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        ),
        dtype=np.uint8,
    )

    transform_3x3 = np.vstack([M, [0, 0, 1]])

    return AlignmentResult(
        image=normalized,
        transform_matrix=transform_3x3,
        scale_factor=scale,
    )


def _detect_face_bbox(
    image: NDArray[np.uint8],
) -> tuple[int, int, int, int] | None:
    """Detect face bounding box.

    Internal function using MediaPipe Face Detection.
    """
    try:
        # pylint: disable=import-outside-toplevel
        import mediapipe as mp  # type: ignore[import-not-found]
    except ImportError:
        return None

    mp_face_detection = mp.solutions.face_detection

    with mp_face_detection.FaceDetection(
        model_selection=1,
        min_detection_confidence=0.5,
    ) as face_detection:
        import cv2  # pylint: disable=import-outside-toplevel

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_detection.process(rgb)

        if not results.detections:
            return None

        # Get first face
        detection = results.detections[0]
        bbox = detection.location_data.relative_bounding_box

        h, w = image.shape[:2]
        x1 = int(bbox.xmin * w)
        y1 = int(bbox.ymin * h)
        x2 = int((bbox.xmin + bbox.width) * w)
        y2 = int((bbox.ymin + bbox.height) * h)

        return (x1, y1, x2, y2)


def compute_inverse_transform(
    transform: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute inverse of transformation matrix.

    Useful for mapping points from aligned space back to original.

    Args:
        transform: 3x3 transformation matrix.

    Returns:
        Inverse transformation matrix.
    """
    return np.asarray(np.linalg.inv(transform), dtype=np.float64)


def apply_transform(
    image: NDArray[np.uint8],
    transform: NDArray[np.float64],
    output_size: tuple[int, int],
) -> NDArray[np.uint8]:
    """Apply transformation matrix to image.

    Args:
        image: Input image.
        transform: 3x3 transformation matrix.
        output_size: Output (width, height).

    Returns:
        Transformed image.
    """
    import cv2  # pylint: disable=import-outside-toplevel

    # Extract 2x3 affine matrix from 3x3
    M = transform[:2, :]  # pylint: disable=invalid-name

    return np.asarray(
        cv2.warpAffine(
            image,
            M,
            output_size,
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        ),
        dtype=np.uint8,
    )
