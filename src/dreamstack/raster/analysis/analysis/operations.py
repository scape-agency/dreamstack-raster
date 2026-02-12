# -*- coding: utf-8 -*-

"""
Color Analysis Operations
=========================

Functional API for color analysis operations.
Provides functions for dominant color detection and background analysis.
"""

from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np
from numpy.typing import NDArray


def get_dominant_color(
    image: NDArray[np.uint8],
    k: int = 1,
    max_iterations: int = 10,
    epsilon: float = 1.0,
) -> NDArray[np.int32]:
    """Find the dominant color in an image using K-means clustering.

    Parameters
    ----------
    image : NDArray[np.uint8]
        Input image in BGR format.
    k : int, optional
        Number of color clusters. Default 1 (single dominant color).
    max_iterations : int, optional
        Maximum K-means iterations. Default 10.
    epsilon : float, optional
        K-means convergence epsilon. Default 1.0.

    Returns
    -------
    NDArray[np.int32]
        Dominant color as BGR array [B, G, R].

    Examples
    --------
    >>> dominant = get_dominant_color(image)
    >>> print(f"Dominant color: BGR({dominant[0]}, {dominant[1]}, {dominant[2]})")
    """
    # Reshape image to list of pixels
    data = image.reshape((-1, 3)).astype(np.float32)

    # K-means criteria
    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        max_iterations,
        epsilon,
    )

    # Run K-means
    _, labels, centers = cv2.kmeans(
        data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
    )

    if k == 1:
        return centers[0].astype(np.int32)

    # Find the most common cluster
    _, counts = np.unique(labels, return_counts=True)
    dominant_idx = np.argmax(counts)
    return centers[dominant_idx].astype(np.int32)


def get_dominant_colors(
    image: NDArray[np.uint8],
    k: int = 5,
    max_iterations: int = 10,
    epsilon: float = 1.0,
) -> List[Tuple[NDArray[np.int32], float]]:
    """Find multiple dominant colors with their proportions.

    Parameters
    ----------
    image : NDArray[np.uint8]
        Input image in BGR format.
    k : int, optional
        Number of color clusters. Default 5.
    max_iterations : int, optional
        Maximum K-means iterations. Default 10.
    epsilon : float, optional
        K-means convergence epsilon. Default 1.0.

    Returns
    -------
    list[tuple[NDArray[np.int32], float]]
        List of (color, proportion) tuples, sorted by proportion descending.
        Colors are BGR arrays, proportions are 0-1 floats.

    Examples
    --------
    >>> colors = get_dominant_colors(image, k=3)
    >>> for color, prop in colors:
    ...     print(f"Color: {color}, Proportion: {prop:.1%}")
    """
    # Reshape image to list of pixels
    data = image.reshape((-1, 3)).astype(np.float32)
    total_pixels = len(data)

    # K-means criteria
    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        max_iterations,
        epsilon,
    )

    # Run K-means
    _, labels, centers = cv2.kmeans(
        data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
    )

    # Count pixels in each cluster
    unique_labels, counts = np.unique(labels, return_counts=True)

    # Build result list
    results = []
    for label, count in zip(unique_labels, counts):
        color = centers[label].astype(np.int32)
        proportion = count / total_pixels
        results.append((color, proportion))

    # Sort by proportion descending
    results.sort(key=lambda x: x[1], reverse=True)
    return results


def get_most_common_color(
    image: NDArray[np.uint8],
) -> NDArray[np.uint8]:
    """Find the most common exact color in an image.

    Unlike get_dominant_color which uses clustering,
    this finds the exact pixel value that appears most frequently.
    Useful for images with flat colors or uniform backgrounds.

    Parameters
    ----------
    image : NDArray[np.uint8]
        Input image in BGR format.

    Returns
    -------
    NDArray[np.uint8]
        Most common color as BGR array.

    Examples
    --------
    >>> common = get_most_common_color(image)
    >>> print(f"Most common: BGR({common[0]}, {common[1]}, {common[2]})")
    """
    # Reshape to list of pixels
    pixels = image.reshape(-1, 3)

    # Find unique colors and their counts
    unique, counts = np.unique(pixels, axis=0, return_counts=True)

    # Return most common
    most_common_idx = np.argmax(counts)
    return unique[most_common_idx]


def find_background_color(
    image: NDArray[np.uint8],
    edge_sample_percent: float = 0.1,
    method: str = "edge_dominant",
) -> NDArray[np.int32]:
    """Detect the background color of an image.

    Parameters
    ----------
    image : NDArray[np.uint8]
        Input image in BGR format.
    edge_sample_percent : float, optional
        Percentage of image edges to sample. Default 0.1 (10%).
    method : str, optional
        Detection method:
        - "edge_dominant": K-means on edge pixels (default)
        - "edge_common": Most common color on edges
        - "corners": Average corner pixel colors
        - "dominant": Overall dominant color

    Returns
    -------
    NDArray[np.int32]
        Estimated background color as BGR array.

    Examples
    --------
    >>> bg_color = find_background_color(image)
    >>> # Use for background replacement
    >>> mask = np.all(np.abs(image - bg_color) < 10, axis=2)
    """
    h, w = image.shape[:2]

    if method == "dominant":
        return get_dominant_color(image)

    elif method == "corners":
        # Sample corners
        corner_size = max(1, int(min(h, w) * edge_sample_percent))
        corners = [
            image[:corner_size, :corner_size],  # Top-left
            image[:corner_size, -corner_size:],  # Top-right
            image[-corner_size:, :corner_size],  # Bottom-left
            image[-corner_size:, -corner_size:],  # Bottom-right
        ]
        all_corners = np.vstack([c.reshape(-1, 3) for c in corners])
        return np.mean(all_corners, axis=0).astype(np.int32)

    else:  # edge_dominant or edge_common
        # Sample edges
        edge_width = max(1, int(min(h, w) * edge_sample_percent))
        edges = [
            image[:edge_width, :],  # Top
            image[-edge_width:, :],  # Bottom
            image[:, :edge_width],  # Left
            image[:, -edge_width:],  # Right
        ]
        all_edges = np.vstack([e.reshape(-1, 3) for e in edges])

        if method == "edge_common":
            unique, counts = np.unique(all_edges, axis=0, return_counts=True)
            return unique[np.argmax(counts)].astype(np.int32)
        else:  # edge_dominant
            return get_dominant_color(
                all_edges.reshape(1, -1, 3).astype(np.uint8)
            )


def adjust_background_color(
    color: NDArray,
    min_brightness: int = 25,
    max_brightness: int = 240,
    darken_factor: float = 0.9,
    brighten_factor: float = 1.1,
) -> NDArray[np.int32]:
    """Adjust background color to avoid extreme values.

    Ensures the background color has good contrast with objects
    by adjusting colors that are too bright or too dark.

    Parameters
    ----------
    color : NDArray
        Input BGR color.
    min_brightness : int, optional
        Minimum allowed channel value. Default 25.
    max_brightness : int, optional
        Maximum allowed channel value. Default 240.
    darken_factor : float, optional
        Multiplier for bright colors. Default 0.9.
    brighten_factor : float, optional
        Multiplier for dark colors. Default 1.1.

    Returns
    -------
    NDArray[np.int32]
        Adjusted BGR color.

    Examples
    --------
    >>> bright = np.array([250, 250, 250])
    >>> adjusted = adjust_background_color(bright)
    >>> # adjusted will be darker
    """
    color = color.astype(np.float32)

    # Check if too bright
    if np.all(color > max_brightness):
        color = color * darken_factor

    # Check if too dark
    if np.all(color < min_brightness):
        color = color * brighten_factor

    # Clamp to valid range
    color = np.clip(color, 0, 255)

    return color.astype(np.int32)


def create_gradient_background(
    size: Tuple[int, int],
    center_color: NDArray,
    darken_factor: float = 0.8,
    gradient_type: str = "radial",
) -> NDArray[np.uint8]:
    """Create a gradient background image.

    Parameters
    ----------
    size : tuple[int, int]
        Output image size as (width, height).
    center_color : NDArray
        Color at the center/start (BGR).
    darken_factor : float, optional
        Factor to darken the edge/end color. Default 0.8.
    gradient_type : str, optional
        Gradient type:
        - "radial": Circular gradient from center (default)
        - "linear_h": Horizontal left-to-right
        - "linear_v": Vertical top-to-bottom

    Returns
    -------
    NDArray[np.uint8]
        Gradient background image.

    Examples
    --------
    >>> bg = create_gradient_background((800, 600), np.array([200, 200, 200]))
    >>> cv2.imwrite("background.jpg", bg)
    """
    width, height = size
    center_color = np.array(center_color, dtype=np.float32)
    edge_color = center_color * darken_factor

    gradient = np.zeros((height, width, 3), dtype=np.uint8)

    if gradient_type == "radial":
        # Radial gradient from center
        y, x = np.ogrid[:height, :width]
        center_x, center_y = width / 2, height / 2
        distance = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
        max_distance = np.sqrt(center_x ** 2 + center_y ** 2)
        normalized = distance / max_distance

        for i in range(3):
            gradient[:, :, i] = (
                edge_color[i] * normalized
                + center_color[i] * (1 - normalized)
            ).astype(np.uint8)

    elif gradient_type == "linear_h":
        # Horizontal gradient
        for x in range(width):
            t = x / (width - 1) if width > 1 else 0
            color = center_color * (1 - t) + edge_color * t
            gradient[:, x, :] = color.astype(np.uint8)

    elif gradient_type == "linear_v":
        # Vertical gradient
        for y in range(height):
            t = y / (height - 1) if height > 1 else 0
            color = center_color * (1 - t) + edge_color * t
            gradient[y, :, :] = color.astype(np.uint8)

    else:
        raise ValueError(f"Unknown gradient type: {gradient_type}")

    return gradient
