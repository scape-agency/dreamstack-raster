# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Color Analyzer
==============

Analyzes colors in images for background detection,
dominant color extraction, and color palette generation.
"""


# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from dreamstack.raster.analysis.analysis.operations import (
    adjust_background_color,
    create_gradient_background,
    find_background_color,
    get_dominant_color,
    get_dominant_colors,
    get_most_common_color,
)


class ColorAnalyzer:
    """Analyzes colors in images for extraction and masking.

    Provides methods for finding dominant colors, detecting backgrounds,
    and generating replacement backgrounds.

    Examples
    --------
    >>> from dreamstack.raster.analysis.coloranalysis import ColorAnalyzer
    >>> analyzer = ColorAnalyzer()
    >>>
    >>> # Find dominant color
    >>> dominant = analyzer.dominant_color(image)
    >>>
    >>> # Get color palette
    >>> palette = analyzer.color_palette(image, num_colors=5)
    >>>
    >>> # Detect background
    >>> bg = analyzer.background_color(image)

    Notes
    -----
    For images with flat backgrounds, use `most_common_color()`.
    For photographs or complex images, use `dominant_color()` with K-means.
    """

    def dominant_color(
        self,
        image: NDArray[np.uint8],
        num_clusters: int = 1,
    ) -> NDArray[np.int32]:
        """Find the dominant color in an image.

        Uses K-means clustering to find the most representative color.

        Parameters
        ----------
        image : NDArray[np.uint8]
            Input image in BGR format.
        num_clusters : int, optional
            Number of clusters for K-means. Default 1.

        Returns
        -------
        NDArray[np.int32]
            Dominant color as BGR array.
        """
        return get_dominant_color(image, k=num_clusters)

    def color_palette(
        self,
        image: NDArray[np.uint8],
        num_colors: int = 5,
    ) -> list[tuple[NDArray[np.int32], float]]:
        """Extract a color palette from an image.

        Parameters
        ----------
        image : NDArray[np.uint8]
            Input image in BGR format.
        num_colors : int, optional
            Number of colors in the palette. Default 5.

        Returns
        -------
        list[tuple[NDArray[np.int32], float]]
            List of (color, proportion) tuples.
        """
        return get_dominant_colors(image, k=num_colors)

    def most_common_color(
        self,
        image: NDArray[np.uint8],
    ) -> NDArray[np.uint8]:
        """Find the exact most common pixel color.

        Parameters
        ----------
        image : NDArray[np.uint8]
            Input image in BGR format.

        Returns
        -------
        NDArray[np.uint8]
            Most common color as BGR array.
        """
        return get_most_common_color(image)

    def background_color(
        self,
        image: NDArray[np.uint8],
        method: str = "edge_dominant",
        edge_percent: float = 0.1,
    ) -> NDArray[np.int32]:
        """Detect the background color of an image.

        Parameters
        ----------
        image : NDArray[np.uint8]
            Input image in BGR format.
        method : str, optional
            Detection method. Default "edge_dominant".
        edge_percent : float, optional
            Edge sampling percentage. Default 0.1.

        Returns
        -------
        NDArray[np.int32]
            Estimated background color.
        """
        return find_background_color(
            image,
            edge_sample_percent=edge_percent,
            method=method,
        )

    def adjusted_background(
        self,
        image: NDArray[np.uint8],
        **kwargs,
    ) -> NDArray[np.int32]:
        """Get background color adjusted for good contrast.

        Parameters
        ----------
        image : NDArray[np.uint8]
            Input image in BGR format.
        **kwargs
            Arguments passed to adjust_background_color().

        Returns
        -------
        NDArray[np.int32]
            Adjusted background color.
        """
        bg = self.background_color(image)
        return adjust_background_color(bg, **kwargs)

    def create_background(
        self,
        size: tuple[int, int],
        color: NDArray | None = None,
        gradient: bool = True,
        source_image: NDArray[np.uint8] | None = None,
    ) -> NDArray[np.uint8]:
        """Create a background image.

        Parameters
        ----------
        size : tuple[int, int]
            Size as (width, height).
        color : NDArray | None, optional
            Background color. If None and source_image provided,
            uses detected background color.
        gradient : bool, optional
            Create a gradient or solid color. Default True.
        source_image : NDArray[np.uint8] | None, optional
            Source image to detect background color from.

        Returns
        -------
        NDArray[np.uint8]
            Background image.
        """
        if color is None:
            if source_image is not None:
                color = self.background_color(source_image)
            else:
                color = np.array([255, 255, 255], dtype=np.int32)  # White

        if gradient:
            return create_gradient_background(size, color)
        else:
            # Solid color
            width, height = size
            bg = np.full((height, width, 3), color, dtype=np.uint8)
            return bg

    @staticmethod
    def bgr_to_rgb(color: NDArray) -> tuple[int, int, int]:
        """Convert BGR color to RGB tuple.

        Parameters
        ----------
        color : NDArray
            BGR color array.

        Returns
        -------
        tuple[int, int, int]
            RGB tuple.
        """
        return (int(color[2]), int(color[1]), int(color[0]))

    @staticmethod
    def rgb_to_bgr(r: int, g: int, b: int) -> NDArray[np.uint8]:
        """Convert RGB values to BGR array.

        Parameters
        ----------
        r, g, b : int
            RGB values.

        Returns
        -------
        NDArray[np.uint8]
            BGR array.
        """
        return np.array([b, g, r], dtype=np.uint8)

    @staticmethod
    def to_hex(color: NDArray) -> str:
        """Convert BGR color to hex string.

        Parameters
        ----------
        color : NDArray
            BGR color array.

        Returns
        -------
        str
            Hex color string (e.g., "#FF0000").
        """
        r, g, b = int(color[2]), int(color[1]), int(color[0])
        return f"#{r:02X}{g:02X}{b:02X}"

    @staticmethod
    def from_hex(hex_color: str) -> NDArray[np.uint8]:
        """Convert hex string to BGR array.

        Parameters
        ----------
        hex_color : str
            Hex color string (with or without #).

        Returns
        -------
        NDArray[np.uint8]
            BGR array.
        """
        hex_color = hex_color.lstrip("#")
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        return np.array([b, g, r], dtype=np.uint8)
