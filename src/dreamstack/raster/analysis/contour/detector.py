"""
Contour Detector
================

Configurable contour detection and analysis class.
Provides a stateful interface with customizable detection parameters.

"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from numpy.typing import NDArray

from dreamstack.raster.analysis.contour.info import ContourInfo
from dreamstack.raster.analysis.contour.operations import (
    analyze_contours,
    approximate_contour,
    filter_by_area,
    filter_by_aspect_ratio,
    filter_by_circularity,
    find_contours,
    get_bounding_boxes,
    scale_contour,
)


@dataclass
class DetectionConfig:
    """Configuration for contour detection.

    Parameters
    ----------
    min_area_ratio : float
        Minimum contour area as ratio of image area. Default 0.0002.
    max_area_ratio : float
        Maximum contour area as ratio of image area. Default 0.95.
    approximation_epsilon : float
        Epsilon for contour approximation (percentage). Default 0.02.
    margin : int
        Margin around detected regions in pixels. Default 25.
    mode : int
        Contour retrieval mode. Default cv2.RETR_EXTERNAL.
    method : int
        Contour approximation method. Default cv2.CHAIN_APPROX_SIMPLE.
    min_circularity : float
        Minimum circularity filter (0-1). Default 0.0 (no filter).
    max_circularity : float
        Maximum circularity filter (0-1). Default 1.0 (no filter).
    min_aspect_ratio : float
        Minimum aspect ratio filter. Default 0.0 (no filter).
    max_aspect_ratio : float
        Maximum aspect ratio filter. Default infinity (no filter).

    Examples
    --------
    >>> config = DetectionConfig(min_area_ratio=0.001, margin=50)
    >>> detector = ContourDetector(config)
    """

    min_area_ratio: float = 0.0002
    max_area_ratio: float = 0.95
    approximation_epsilon: float = 0.02
    margin: int = 25
    mode: int = field(default_factory=lambda: cv2.RETR_EXTERNAL)
    method: int = field(default_factory=lambda: cv2.CHAIN_APPROX_SIMPLE)
    min_circularity: float = 0.0
    max_circularity: float = 1.0
    min_aspect_ratio: float = 0.0
    max_aspect_ratio: float = float("inf")


class ContourDetector:
    """Detects and analyzes contours in images.

    Provides a configurable interface for finding contours, filtering
    by various geometric properties, and extracting bounding regions.

    Attributes
    ----------
    config : DetectionConfig
        Detection configuration parameters.

    Examples
    --------
    >>> from dreamstack.raster.analysis.contour import ContourDetector
    >>> detector = ContourDetector()
    >>>
    >>> # Find and filter contours
    >>> contours = detector.detect(binary_image)
    >>> print(f"Found {len(contours)} objects")
    >>>
    >>> # Get bounding boxes with margin
    >>> boxes = detector.get_boxes(contours)

    Notes
    -----
    The detector is designed to be reusable across multiple images
    with consistent configuration. For one-off operations, use the
    functional API in `operations.py`.
    """

    def __init__(
        self,
        config: DetectionConfig | None = None,
    ) -> None:
        """Initialize the contour detector.

        Parameters
        ----------
        config : DetectionConfig | None, optional
            Detection configuration. Uses defaults if None.
        """
        self.config = config or DetectionConfig()

    def detect(
        self,
        binary_image: NDArray[np.uint8],
        apply_filters: bool = True,
    ) -> list[ContourInfo]:
        """Detect and analyze contours in a binary image.

        Parameters
        ----------
        binary_image : NDArray[np.uint8]
            Binary (thresholded) image.
        apply_filters : bool, optional
            Whether to apply configured filters. Default True.

        Returns
        -------
        list[ContourInfo]
            List of detected and optionally filtered contours,
            sorted by area (largest first).

        Examples
        --------
        >>> _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        >>> contours = detector.detect(binary)
        """
        # Find raw contours
        raw_contours = find_contours(
            binary_image,
            mode=self.config.mode,
            method=self.config.method,
        )

        # Analyze
        analyzed = analyze_contours(raw_contours, sort_by="area", descending=True)

        if not apply_filters:
            return analyzed

        # Apply configured filters
        return self.filter(analyzed, binary_image.shape[:2])

    def filter(
        self,
        contours: list[ContourInfo],
        image_size: tuple[int, int],
    ) -> list[ContourInfo]:
        """Apply all configured filters to contours.

        Parameters
        ----------
        contours : list[ContourInfo]
            Contours to filter.
        image_size : tuple[int, int]
            Image dimensions as (height, width).

        Returns
        -------
        list[ContourInfo]
            Filtered contours.
        """
        height, width = image_size
        image_area = height * width

        # Area filter
        filtered = filter_by_area(
            contours,
            image_area,
            min_ratio=self.config.min_area_ratio,
            max_ratio=self.config.max_area_ratio,
        )

        # Circularity filter
        filtered = filter_by_circularity(
            filtered,
            min_circularity=self.config.min_circularity,
            max_circularity=self.config.max_circularity,
        )

        # Aspect ratio filter
        filtered = filter_by_aspect_ratio(
            filtered,
            min_ratio=self.config.min_aspect_ratio,
            max_ratio=self.config.max_aspect_ratio,
        )

        return filtered

    def get_boxes(
        self,
        contours: list[ContourInfo],
        margin: int | None = None,
        image_size: tuple[int, int] | None = None,
    ) -> list[tuple[int, int, int, int]]:
        """Get bounding boxes for contours.

        Parameters
        ----------
        contours : list[ContourInfo]
            List of contours.
        margin : int | None, optional
            Margin to add. Uses config value if None.
        image_size : tuple[int, int] | None, optional
            Image dimensions to clamp boxes.

        Returns
        -------
        list[tuple[int, int, int, int]]
            Bounding boxes as (x, y, width, height).
        """
        margin = margin if margin is not None else self.config.margin
        return get_bounding_boxes(contours, margin=margin, image_size=image_size)

    def approximate(
        self,
        contour: NDArray[np.int32] | ContourInfo,
        epsilon: float | None = None,
    ) -> NDArray[np.int32]:
        """Approximate a contour with fewer points.

        Parameters
        ----------
        contour : NDArray[np.int32] | ContourInfo
            Input contour or ContourInfo.
        epsilon : float | None, optional
            Approximation epsilon. Uses config percentage if None.

        Returns
        -------
        NDArray[np.int32]
            Approximated contour.
        """
        raw_contour = contour.contour if isinstance(contour, ContourInfo) else contour
        return approximate_contour(
            raw_contour,
            epsilon_percent=epsilon or self.config.approximation_epsilon,
        )

    def find_largest(
        self,
        binary_image: NDArray[np.uint8],
    ) -> ContourInfo | None:
        """Find the largest contour by area.

        Parameters
        ----------
        binary_image : NDArray[np.uint8]
            Binary image.

        Returns
        -------
        ContourInfo | None
            Largest contour or None if no contours found.
        """
        contours = self.detect(binary_image, apply_filters=False)
        return contours[0] if contours else None

    def scale(
        self,
        contour: NDArray[np.int32] | ContourInfo,
        scale: float,
        offset: tuple[int, int] = (0, 0),
    ) -> NDArray[np.int32]:
        """Scale and offset a contour.

        Parameters
        ----------
        contour : NDArray[np.int32] | ContourInfo
            Input contour.
        scale : float
            Scale factor.
        offset : tuple[int, int], optional
            Offset to subtract.

        Returns
        -------
        NDArray[np.int32]
            Transformed contour.
        """
        raw_contour = contour.contour if isinstance(contour, ContourInfo) else contour
        return scale_contour(raw_contour, scale, offset)

    def with_config(self, **kwargs) -> ContourDetector:
        """Create a new detector with modified configuration.

        Parameters
        ----------
        **kwargs
            Configuration parameters to override.

        Returns
        -------
        ContourDetector
            New detector instance with updated config.

        Examples
        --------
        >>> detector = ContourDetector()
        >>> strict_detector = detector.with_config(min_area_ratio=0.01)
        """
        from dataclasses import replace

        new_config = replace(self.config, **kwargs)
        return ContourDetector(new_config)
