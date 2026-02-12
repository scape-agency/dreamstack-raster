"""
Magnetic Lasso Selection
========================

Edge-snapping lasso selection tool.

"""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

from dreamstack.raster.selection.shapes.selection import Selection


def magnetic_lasso(
    image: NDArray[np.uint8],
    seed_points: list[tuple[int, int]],
    *,
    edge_sensitivity: float = 0.5,
    frequency: int = 10,
    width: int = 20,
    feather: int = 0,
) -> Selection:
    """Create an edge-snapping lasso selection.

    The magnetic lasso snaps to nearby edges, making it easier
    to trace complex boundaries.

    Args:
        image: Input image (BGR or BGRA).
        seed_points: Initial guide points (sparse).
        edge_sensitivity: Edge detection sensitivity (0.0-1.0).
        frequency: Point frequency along path.
        width: Search width for edge snapping (pixels).
        feather: Feather radius for soft edges.

    Returns:
        Selection snapped to edges.

    Example:
        >>> # Rough outline that snaps to edges
        >>> seeds = [(100, 100), (200, 50), (300, 100)]
        >>> sel = magnetic_lasso(image, seeds)
    """
    if len(seed_points) < 2:
        h, w = image.shape[:2]
        return Selection(mask=np.zeros((h, w), dtype=np.uint8))

    h, w = image.shape[:2]

    # Convert to grayscale for edge detection
    if image.ndim == 2:
        gray = image
    else:
        gray = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2GRAY)

    # Compute edge map
    edge_thresh = int((1.0 - edge_sensitivity) * 100)
    edges = cv2.Canny(gray, edge_thresh, edge_thresh * 2)

    # Distance transform for edge proximity
    _, edge_dist = cv2.threshold(edges, 1, 255, cv2.THRESH_BINARY_INV)
    dist_transform = cv2.distanceTransform(edge_dist, cv2.DIST_L2, 5)

    # Interpolate and snap points
    snapped_points: list[tuple[int, int]] = []

    for i in range(len(seed_points) - 1):
        start = seed_points[i]
        end = seed_points[i + 1]

        # Interpolate between seed points
        num_samples = max(
            2, int(np.linalg.norm(np.array(end) - np.array(start)) / frequency)
        )

        for t in np.linspace(0, 1, num_samples):
            # Initial point
            px = int(start[0] + t * (end[0] - start[0]))
            py = int(start[1] + t * (end[1] - start[1]))

            # Search for nearby edge
            best_x, best_y = px, py
            best_dist = float("inf")

            # Search in neighborhood
            half_width = width // 2
            for dx in range(-half_width, half_width + 1):
                for dy in range(-half_width, half_width + 1):
                    nx, ny = px + dx, py + dy
                    if 0 <= nx < w and 0 <= ny < h:
                        d = dist_transform[ny, nx]
                        if d < best_dist:
                            best_dist = d
                            best_x, best_y = nx, ny

            snapped_points.append((best_x, best_y))

    # Add final point
    snapped_points.append(seed_points[-1])

    # Close the path if needed
    if len(snapped_points) >= 3:
        snapped_points.append(snapped_points[0])

    # Create mask from snapped points
    mask = np.zeros((h, w), dtype=np.uint8)
    pts = np.array(snapped_points, dtype=np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(mask, [pts], 255, cv2.LINE_AA)

    if feather > 0:
        mask = cv2.GaussianBlur(mask, (0, 0), feather)

    return Selection(mask=mask)
