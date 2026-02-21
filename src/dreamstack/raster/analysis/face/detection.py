# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Face Detection Operations
=========================

Face and landmark detection using MediaPipe.

"""


# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

# =============================================================================
# Type Checking Imports
# =============================================================================

if TYPE_CHECKING:
    # pylint: disable=import-outside-toplevel
    from numpy.typing import NDArray


@dataclass
class FaceBbox:
    """Face bounding box.

    Attributes:
        x1: Left edge.
        y1: Top edge.
        x2: Right edge.
        y2: Bottom edge.
        confidence: Detection confidence.
    """

    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float = 1.0

    @property
    def width(self) -> int:
        """Bounding box width."""
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        """Bounding box height."""
        return self.y2 - self.y1

    @property
    def center(self) -> tuple[int, int]:
        """Center point of bounding box."""
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)

    @property
    def area(self) -> int:
        """Area of bounding box."""
        return self.width * self.height

    def as_tuple(self) -> tuple[int, int, int, int]:
        """Return as (x1, y1, x2, y2) tuple."""
        return (self.x1, self.y1, self.x2, self.y2)

    def expand(self, factor: float = 1.2) -> FaceBbox:
        """Expand bounding box by a factor.

        Args:
            factor: Expansion factor (1.0 = no change).

        Returns:
            New expanded FaceBbox.
        """
        cx, cy = self.center
        half_w = int(self.width * factor / 2)
        half_h = int(self.height * factor / 2)

        return FaceBbox(
            x1=cx - half_w,
            y1=cy - half_h,
            x2=cx + half_w,
            y2=cy + half_h,
            confidence=self.confidence,
        )


def detect_face(
    image: NDArray[np.uint8],
    min_confidence: float = 0.5,
) -> FaceBbox | None:
    """Detect the primary face in an image.

    Returns the largest/most confident face detection.

    Args:
        image: Input image (BGR, 3 channels).
        min_confidence: Minimum detection confidence.

    Returns:
        FaceBbox or None if no face detected.

    Example:
        >>> from dreamstack.raster.analysis.face import detect_face
        >>> face = detect_face(image)
        >>> if face:
        ...     print(f"Face at {face.center}")
    """
    faces = detect_faces(image, min_confidence)

    if not faces:
        return None

    # Return face with highest confidence
    return max(faces, key=lambda f: f.confidence)


def detect_faces(
    image: NDArray[np.uint8],
    min_confidence: float = 0.5,
    max_faces: int = 10,
) -> list[FaceBbox]:
    """Detect all faces in an image.

    Args:
        image: Input image (BGR, 3 channels).
        min_confidence: Minimum detection confidence.
        max_faces: Maximum number of faces to return.

    Returns:
        List of FaceBbox objects.

    Example:
        >>> faces = detect_faces(group_photo)
        >>> print(f"Found {len(faces)} faces")
    """
    try:
        # pylint: disable=import-outside-toplevel
        import mediapipe as mp  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError(
            "mediapipe is required for face detection. "
            "Install with: pip install mediapipe"
        ) from exc

    import cv2  # pylint: disable=import-outside-toplevel

    mp_face_detection = mp.solutions.face_detection

    with mp_face_detection.FaceDetection(
        model_selection=1,
        min_detection_confidence=min_confidence,
    ) as face_detection:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_detection.process(rgb)

        if not results.detections:
            return []

        h, w = image.shape[:2]
        faces = []

        for detection in results.detections[:max_faces]:
            bbox = detection.location_data.relative_bounding_box
            confidence = detection.score[0] if detection.score else 1.0

            x1 = int(max(0, bbox.xmin * w))
            y1 = int(max(0, bbox.ymin * h))
            x2 = int(min(w, (bbox.xmin + bbox.width) * w))
            y2 = int(min(h, (bbox.ymin + bbox.height) * h))

            faces.append(
                FaceBbox(
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    confidence=confidence,
                )
            )

        return faces


@dataclass
class FaceLandmarks:
    """Face landmark positions.

    Attributes:
        points: List of (x, y) landmark positions.
        left_eye: Left eye center position.
        right_eye: Right eye center position.
        nose_tip: Nose tip position.
        mouth_center: Mouth center position.
    """

    points: list[tuple[int, int]]
    left_eye: tuple[int, int] | None = None
    right_eye: tuple[int, int] | None = None
    nose_tip: tuple[int, int] | None = None
    mouth_center: tuple[int, int] | None = None


def detect_landmarks(
    image: NDArray[np.uint8],
    min_confidence: float = 0.5,
) -> FaceLandmarks | None:
    """Detect face landmarks using MediaPipe Face Mesh.

    Returns 468 facial landmarks plus iris landmarks.

    Args:
        image: Input image (BGR, 3 channels).
        min_confidence: Minimum detection confidence.

    Returns:
        FaceLandmarks or None if no face detected.

    Example:
        >>> landmarks = detect_landmarks(image)
        >>> if landmarks and landmarks.left_eye:
        ...     print(f"Left eye at {landmarks.left_eye}")
    """
    try:
        # pylint: disable=import-outside-toplevel
        import mediapipe as mp  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError(
            "mediapipe is required for landmark detection. "
            "Install with: pip install mediapipe"
        ) from exc

    import cv2  # pylint: disable=import-outside-toplevel

    mp_face_mesh = mp.solutions.face_mesh

    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=min_confidence,
    ) as face_mesh:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return None

        landmarks = results.multi_face_landmarks[0]
        h, w = image.shape[:2]

        # Convert all landmarks to pixel coordinates
        points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks.landmark]

        # Key landmark indices
        # pylint: disable=invalid-name
        LEFT_EYE_CENTER = 468
        RIGHT_EYE_CENTER = 473
        NOSE_TIP = 4
        MOUTH_CENTER = 13

        # Fallback indices
        LEFT_EYE_INNER = 133
        RIGHT_EYE_INNER = 362
        # pylint: enable=invalid-name

        # Extract key landmarks
        try:
            left_eye = points[LEFT_EYE_CENTER]
            right_eye = points[RIGHT_EYE_CENTER]
        except IndexError:
            left_eye = (
                points[LEFT_EYE_INNER]
                if len(points) > LEFT_EYE_INNER
                else None
            )
            right_eye = (
                points[RIGHT_EYE_INNER]
                if len(points) > RIGHT_EYE_INNER
                else None
            )

        nose_tip = points[NOSE_TIP] if len(points) > NOSE_TIP else None
        mouth_center = (
            points[MOUTH_CENTER] if len(points) > MOUTH_CENTER else None
        )

        return FaceLandmarks(
            points=points,
            left_eye=left_eye,
            right_eye=right_eye,
            nose_tip=nose_tip,
            mouth_center=mouth_center,
        )


def crop_face(
    image: NDArray[np.uint8],
    face_bbox: FaceBbox | None = None,
    expand_factor: float = 1.3,
) -> NDArray[np.uint8] | None:
    """Crop face region from image.

    Args:
        image: Input image.
        face_bbox: Optional face bounding box (will detect if not provided).
        expand_factor: Expand bbox by this factor to include more context.

    Returns:
        Cropped face image or None if no face detected.

    Example:
        >>> face_crop = crop_face(image, expand_factor=1.5)
    """
    if face_bbox is None:
        face_bbox = detect_face(image)

    if face_bbox is None:
        return None

    h, w = image.shape[:2]
    bbox = face_bbox.expand(expand_factor)

    # Clamp to image bounds
    x1 = max(0, bbox.x1)
    y1 = max(0, bbox.y1)
    x2 = min(w, bbox.x2)
    y2 = min(h, bbox.y2)

    return image[y1:y2, x1:x2].copy()
