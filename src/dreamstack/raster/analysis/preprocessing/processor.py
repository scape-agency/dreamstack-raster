"""
Image Preprocessor
==================

Configurable image preprocessing for detection workflows.
Provides a stateful interface with customizable preprocessing parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from dreamstack.raster.analysis.preprocessing.operations import (
    apply_clahe,
    binarize,
    detect_edges,
    to_grayscale,
)


@dataclass
class PreprocessingConfig:
    """Configuration for image preprocessing.

    Parameters
    ----------
    blur_kernel_size : tuple[int, int]
        Gaussian blur kernel size. Default (11, 11).
    clahe_clip_limit : float
        CLAHE clip limit for contrast enhancement. Default 3.0.
    clahe_tile_size : tuple[int, int]
        CLAHE tile grid size. Default (8, 8).
    threshold_method : str
        Binarization method. Default "otsu".
    threshold_value : int
        Fixed threshold value (for "simple" method). Default 127.
    edge_method : str
        Edge detection method. Default "canny".
    canny_low : int
        Canny low threshold. Default 50.
    canny_high : int
        Canny high threshold. Default 150.
    morph_kernel_size : int
        Morphological operation kernel size. Default 5.
    morph_dilate_iter : int
        Dilation iterations for mask cleanup. Default 5.
    morph_erode_iter : int
        Erosion iterations for mask cleanup. Default 3.

    Examples
    --------
    >>> config = PreprocessingConfig(blur_kernel_size=(7, 7), clahe_clip_limit=2.0)
    >>> processor = ImagePreprocessor(config)
    """

    blur_kernel_size: tuple[int, int] = (11, 11)
    clahe_clip_limit: float = 3.0
    clahe_tile_size: tuple[int, int] = (8, 8)
    threshold_method: str = "otsu"
    threshold_value: int = 127
    edge_method: str = "canny"
    canny_low: int = 50
    canny_high: int = 150
    morph_kernel_size: int = 5
    morph_dilate_iter: int = 5
    morph_erode_iter: int = 3


class ImagePreprocessor:
    """Handles image preprocessing operations for object detection.

    Provides a configurable pipeline for preparing images
    for contour detection and object extraction.

    Attributes
    ----------
    config : PreprocessingConfig
        Preprocessing configuration parameters.

    Examples
    --------
    >>> from dreamstack.raster.analysis.preprocessing import ImagePreprocessor
    >>> processor = ImagePreprocessor()
    >>>
    >>> # Load and preprocess
    >>> image = processor.load("input.jpg")
    >>> results = processor.preprocess(image)
    >>> binary = results["threshold"]

    Notes
    -----
    The preprocessor is designed to be reusable across multiple images.
    For one-off operations, use the functional API in `operations.py`.
    """

    def __init__(self, config: PreprocessingConfig | None = None) -> None:
        """Initialize the image preprocessor.

        Parameters
        ----------
        config : PreprocessingConfig | None, optional
            Preprocessing configuration. Uses defaults if None.
        """
        self.config = config or PreprocessingConfig()

    def load(self, path: str | Path) -> NDArray[np.uint8]:
        """Load an image from disk.

        Parameters
        ----------
        path : str | Path
            Path to the image file.

        Returns
        -------
        NDArray[np.uint8]
            Loaded image in BGR format.

        Raises
        ------
        FileNotFoundError
            If the image file doesn't exist.
        ValueError
            If the image cannot be loaded.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")

        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Failed to load image: {path}")

        return image  # type: ignore[return-value]

    def blur(self, image: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """Apply Gaussian blur to reduce noise.

        Parameters
        ----------
        image : NDArray[np.uint8]
            Input image.

        Returns
        -------
        NDArray[np.uint8]
            Blurred image.
        """
        return cv2.GaussianBlur(  # type: ignore[return-value]
            image, self.config.blur_kernel_size, 0
        )

    def enhance_contrast(self, image: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """Enhance image contrast using CLAHE.

        Parameters
        ----------
        image : NDArray[np.uint8]
            Input image.

        Returns
        -------
        NDArray[np.uint8]
            Contrast-enhanced image.
        """
        return apply_clahe(
            image,
            clip_limit=self.config.clahe_clip_limit,
            tile_size=self.config.clahe_tile_size,
        )

    def to_grayscale(self, image: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """Convert image to grayscale.

        Parameters
        ----------
        image : NDArray[np.uint8]
            Input image.

        Returns
        -------
        NDArray[np.uint8]
            Grayscale image.
        """
        return to_grayscale(image)

    def binarize(self, image: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """Convert to binary using configured method.

        Parameters
        ----------
        image : NDArray[np.uint8]
            Input grayscale image.

        Returns
        -------
        NDArray[np.uint8]
            Binary image.
        """
        return binarize(
            image,
            method=self.config.threshold_method,
            threshold=self.config.threshold_value,
        )

    def detect_edges(self, image: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """Detect edges using configured method.

        Parameters
        ----------
        image : NDArray[np.uint8]
            Input grayscale image.

        Returns
        -------
        NDArray[np.uint8]
            Edge image.
        """
        return detect_edges(
            image,
            method=self.config.edge_method,
            low_threshold=self.config.canny_low,
            high_threshold=self.config.canny_high,
        )

    def clean_mask(self, mask: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """Clean a binary mask using morphological operations.

        Applies closing (fill holes) then opening (remove noise).

        Parameters
        ----------
        mask : NDArray[np.uint8]
            Input binary mask.

        Returns
        -------
        NDArray[np.uint8]
            Cleaned mask.
        """
        # Dilate to connect nearby regions
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(
            mask, kernel, iterations=self.config.morph_dilate_iter
        )
        # Erode to restore size
        eroded = cv2.erode(
            dilated, kernel, iterations=self.config.morph_erode_iter
        )
        return eroded  # type: ignore[return-value]

    def preprocess(
        self,
        image: NDArray[np.uint8],
        return_all: bool = True,
    ) -> dict[str, NDArray[np.uint8]] | NDArray[np.uint8]:
        """Run full preprocessing pipeline.

        Parameters
        ----------
        image : NDArray[np.uint8]
            Input image in BGR format.
        return_all : bool, optional
            If True, return dict with all intermediate results.
            If False, return only the binary threshold. Default True.

        Returns
        -------
        dict[str, NDArray[np.uint8]] | NDArray[np.uint8]
            If return_all is True:
                Dictionary containing:
                - "original": Original image copy
                - "blurred": After Gaussian blur
                - "contrast": After CLAHE
                - "grayscale": Grayscale conversion
                - "threshold": Binary threshold
                - "edges": Edge detection result
            If return_all is False:
                Binary threshold image only.

        Examples
        --------
        >>> results = processor.preprocess(image)
        >>> binary = results["threshold"]
        >>>
        >>> # Or get only binary
        >>> binary = processor.preprocess(image, return_all=False)
        """
        # Pipeline stages
        blurred = self.blur(image)
        contrast = self.enhance_contrast(blurred)
        grayscale = self.to_grayscale(contrast)
        threshold = self.binarize(grayscale)
        edges = self.detect_edges(grayscale)

        if not return_all:
            return threshold

        return {
            "original": image.copy(),
            "blurred": blurred,
            "contrast": contrast,
            "grayscale": grayscale,
            "threshold": threshold,
            "edges": edges,
        }

    def preprocess_from_file(
        self,
        path: str | Path,
        return_all: bool = True,
    ) -> dict[str, NDArray[np.uint8]] | NDArray[np.uint8]:
        """Load and preprocess an image file.

        Parameters
        ----------
        path : str | Path
            Path to the image file.
        return_all : bool, optional
            Return all intermediate results. Default True.

        Returns
        -------
        dict | NDArray
            Preprocessing results.
        """
        image = self.load(path)
        return self.preprocess(image, return_all=return_all)

    def with_config(self, **kwargs) -> ImagePreprocessor:
        """Create a new preprocessor with modified configuration.

        Parameters
        ----------
        **kwargs
            Configuration parameters to override.

        Returns
        -------
        ImagePreprocessor
            New preprocessor instance with updated config.

        Examples
        --------
        >>> processor = ImagePreprocessor()
        >>> smooth_processor = processor.with_config(blur_kernel_size=(21, 21))
        """
        from dataclasses import (
            replace,
        )  # pylint: disable=import-outside-toplevel

        new_config = replace(self.config, **kwargs)
        return ImagePreprocessor(new_config)
