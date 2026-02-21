# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Active Contour (Snake) Operations
=================================

Active contour models describe the boundaries of shapes in images.
Uses energy-minimizing splines guided by image features for
precise image segmentation.

"""


# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray


@dataclass
class ActiveContourConfig:
    """Configuration for active contour detection.

    Attributes
    ----------
    alpha : float
        Snake length shape parameter. Higher values make snake contract faster.
    beta : float
        Snake smoothness shape parameter. Higher values make snake smoother.
    gamma : float
        Time stepping parameter.
    w_line : float
        Controls attraction to brightness.
    w_edge : float
        Controls attraction to edges.
    max_iterations : int
        Maximum number of iterations.
    convergence_threshold : float
        Stop when movement is below this threshold.
    gaussian_sigma : float
        Sigma for Gaussian pre-smoothing.
    """

    alpha: float = 0.015
    beta: float = 10.0
    gamma: float = 0.001
    w_line: float = 0.0
    w_edge: float = 1.0
    max_iterations: int = 2500
    convergence_threshold: float = 0.1
    gaussian_sigma: float = 3.0


@dataclass
class ActiveContourResult:
    """Result from active contour detection.

    Attributes
    ----------
    contour : NDArray[np.float64]
        Final contour points (N, 2).
    initial_contour : NDArray[np.float64]
        Initial contour points.
    iterations : int
        Number of iterations performed.
    converged : bool
        Whether the algorithm converged.
    """

    contour: NDArray[np.float64]
    initial_contour: NDArray[np.float64]
    iterations: int = 0
    converged: bool = False


def create_circular_contour(
    center: tuple[float, float],
    radius: float,
    num_points: int = 400,
) -> NDArray[np.float64]:
    """Create a circular initial contour.

    Parameters
    ----------
    center : tuple[float, float]
        Center of circle (x, y).
    radius : float
        Radius of the circle.
    num_points : int, optional
        Number of points along contour. Default is 400.

    Returns
    -------
    NDArray[np.float64]
        Contour points array (N, 2).

    Examples
    --------
    >>> init = create_circular_contour((220, 100), radius=100)
    >>> result = active_contour(img, init)
    """
    s = np.linspace(0, 2 * np.pi, num_points)
    x = center[0] + radius * np.cos(s)
    y = center[1] + radius * np.sin(s)
    return np.array([x, y]).T


def create_elliptical_contour(
    center: tuple[float, float],
    axes: tuple[float, float],
    num_points: int = 400,
    angle: float = 0.0,
) -> NDArray[np.float64]:
    """Create an elliptical initial contour.

    Parameters
    ----------
    center : tuple[float, float]
        Center of ellipse (x, y).
    axes : tuple[float, float]
        Semi-axes lengths (a, b).
    num_points : int, optional
        Number of points. Default is 400.
    angle : float, optional
        Rotation angle in degrees. Default is 0.

    Returns
    -------
    NDArray[np.float64]
        Contour points array (N, 2).
    """
    s = np.linspace(0, 2 * np.pi, num_points)
    a, b = axes

    # Generate ellipse
    x = a * np.cos(s)
    y = b * np.sin(s)

    # Rotate if needed
    if angle != 0:
        rad = np.deg2rad(angle)
        cos_val = np.cos(rad)
        sin_val = np.sin(rad)
        x_rot = x * cos_val - y * sin_val
        y_rot = x * sin_val + y * cos_val
        x, y = x_rot, y_rot

    # Translate to center
    x += center[0]
    y += center[1]

    return np.array([x, y]).T


def create_rectangular_contour(
    bounds: tuple[float, float, float, float],
    num_points: int = 400,
) -> NDArray[np.float64]:
    """Create a rectangular initial contour.

    Parameters
    ----------
    bounds : tuple[float, float, float, float]
        Rectangle as (x, y, width, height).
    num_points : int, optional
        Number of points. Default is 400.

    Returns
    -------
    NDArray[np.float64]
        Contour points array (N, 2).
    """
    x, y, w, h = bounds
    points_per_side = num_points // 4

    # Create points along each side
    top = np.column_stack(
        [np.linspace(x, x + w, points_per_side), np.full(points_per_side, y)]
    )
    right = np.column_stack(
        [
            np.full(points_per_side, x + w),
            np.linspace(y, y + h, points_per_side),
        ]
    )
    bottom = np.column_stack(
        [
            np.linspace(x + w, x, points_per_side),
            np.full(points_per_side, y + h),
        ]
    )
    left = np.column_stack(
        [np.full(points_per_side, x), np.linspace(y + h, y, points_per_side)]
    )

    return np.vstack([top, right, bottom, left])


def active_contour(
    image: NDArray[np.uint8],
    initial_contour: NDArray[np.float64],
    *,
    config: ActiveContourConfig | None = None,
    preprocess: bool = True,
) -> ActiveContourResult:
    """Apply active contour model (snakes) for image segmentation.

    Uses energy-minimizing splines that are guided by external
    image forces (edges) and internal forces (smoothness).

    Parameters
    ----------
    image : NDArray[np.uint8]
        Input image (grayscale or color).
    initial_contour : NDArray[np.float64]
        Initial contour points (N, 2).
    config : ActiveContourConfig, optional
        Algorithm configuration.
    preprocess : bool, optional
        Apply Gaussian smoothing. Default is True.

    Returns
    -------
    ActiveContourResult
        Result containing final contour and metadata.

    Examples
    --------
    >>> from skimage.segmentation import active_contour as sk_active_contour
    >>>
    >>> # Create circular initial contour
    >>> init = create_circular_contour((220, 100), radius=100)
    >>>
    >>> # Run active contour
    >>> result = active_contour(img, init)
    >>> final_contour = result.contour

    Notes
    -----
    This function requires scikit-image to be installed.
    It wraps skimage.segmentation.active_contour.
    """
    try:
        # pylint: disable=import-outside-toplevel
        from skimage.filters import gaussian

        # pylint: disable=import-outside-toplevel
        from skimage.segmentation import active_contour as sk_active_contour
    except ImportError as exc:
        raise ImportError(
            "scikit-image is required for active_contour. "
            "Install with: pip install scikit-image"
        ) from exc

    if config is None:
        config = ActiveContourConfig()

    # Convert to grayscale if needed
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Normalize to float
    img_float = gray.astype(np.float64) / 255.0

    # Apply Gaussian smoothing
    if preprocess:
        img_float = gaussian(img_float, config.gaussian_sigma)

    # Run active contour
    contour = sk_active_contour(
        img_float,
        initial_contour,
        alpha=config.alpha,
        beta=config.beta,
        gamma=config.gamma,
        w_line=config.w_line,
        w_edge=int(config.w_edge),
        max_num_iter=config.max_iterations,
        convergence=config.convergence_threshold,
    )

    return ActiveContourResult(
        contour=contour,
        initial_contour=initial_contour,
        iterations=config.max_iterations,
        converged=True,
    )


def draw_contour(
    image: NDArray[np.uint8],
    contour: NDArray,
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
    closed: bool = True,
) -> NDArray[np.uint8]:
    """Draw a contour on an image.

    Parameters
    ----------
    image : NDArray[np.uint8]
        Image to draw on (will be copied).
    contour : NDArray
        Contour points (N, 2).
    color : tuple[int, int, int], optional
        BGR color. Default is green.
    thickness : int, optional
        Line thickness. Default is 2.
    closed : bool, optional
        Close the contour. Default is True.

    Returns
    -------
    NDArray[np.uint8]
        Image with contour drawn.
    """
    result = image.copy()

    # Ensure contour is in correct format
    pts = contour.astype(np.int32).reshape((-1, 1, 2))

    cv2.polylines(result, [pts], closed, color, thickness)

    return result


def contour_to_mask(
    contour: NDArray,
    image_shape: tuple[int, int],
) -> NDArray[np.uint8]:
    """Convert contour to binary mask.

    Parameters
    ----------
    contour : NDArray
        Contour points (N, 2).
    image_shape : tuple[int, int]
        Output mask shape (height, width).

    Returns
    -------
    NDArray[np.uint8]
        Binary mask where inside is 255.
    """
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    pts = contour.astype(np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(mask, [pts], 255)
    return mask


def extract_contour_region(
    image: NDArray[np.uint8],
    contour: NDArray,
    *,
    background: int | tuple[int, int, int] = 0,
) -> NDArray[np.uint8]:
    """Extract region inside contour.

    Parameters
    ----------
    image : NDArray[np.uint8]
        Input image.
    contour : NDArray
        Contour defining the region.
    background : int or tuple, optional
        Background color for outside region. Default is 0.

    Returns
    -------
    NDArray[np.uint8]
        Image with only contour region visible.
    """
    mask = contour_to_mask(contour, image.shape[:2])

    if image.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        if isinstance(background, int):
            background = (background, background, background)

    result = np.where(mask > 0, image, background)
    return result.astype(np.uint8)


def contour_area(contour: NDArray) -> float:
    """Calculate area enclosed by contour.

    Parameters
    ----------
    contour : NDArray
        Contour points (N, 2).

    Returns
    -------
    float
        Area in pixels.
    """
    pts = contour.astype(np.int32).reshape((-1, 1, 2))
    return cv2.contourArea(pts)


def contour_perimeter(contour: NDArray, closed: bool = True) -> float:
    """Calculate contour perimeter.

    Parameters
    ----------
    contour : NDArray
        Contour points (N, 2).
    closed : bool, optional
        Whether contour is closed. Default is True.

    Returns
    -------
    float
        Perimeter length in pixels.
    """
    pts = contour.astype(np.int32).reshape((-1, 1, 2))
    return cv2.arcLength(pts, closed)


def contour_centroid(contour: NDArray) -> tuple[float, float]:
    """Calculate contour centroid.

    Parameters
    ----------
    contour : NDArray
        Contour points (N, 2).

    Returns
    -------
    tuple[float, float]
        Centroid coordinates (x, y).
    """
    pts = contour.astype(np.int32).reshape((-1, 1, 2))
    moments = cv2.moments(pts)

    if moments["m00"] == 0:
        return (0.0, 0.0)

    cx = moments["m10"] / moments["m00"]
    cy = moments["m01"] / moments["m00"]

    return (cx, cy)
