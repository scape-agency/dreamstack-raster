# -*- coding: utf-8 -*-

"""
Preprocessing Operations
========================

Functional API for image preprocessing operations.
Provides stateless functions for preparing images for contour detection.
"""

from __future__ import annotations

from typing import Dict, Tuple

import cv2
import numpy as np
from numpy.typing import NDArray


def to_grayscale(
    image: NDArray[np.uint8],
    method: str = "luminosity",
) -> NDArray[np.uint8]:
    """Convert image to grayscale.

    Parameters
    ----------
    image : NDArray[np.uint8]
        Input image in BGR format.
    method : str, optional
        Conversion method:
        - "luminosity": Use human perception weights (default)
        - "average": Equal weights for all channels
        - "lightness": (max + min) / 2

    Returns
    -------
    NDArray[np.uint8]
        Grayscale image.

    Examples
    --------
    >>> gray = to_grayscale(image)
    >>> gray = to_grayscale(image, method="average")
    """
    if len(image.shape) == 2:
        return image  # Already grayscale

    if method == "luminosity":
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif method == "average":
        return np.mean(image, axis=2).astype(np.uint8)
    elif method == "lightness":
        max_val = np.max(image, axis=2)
        min_val = np.min(image, axis=2)
        return ((max_val.astype(np.float32) + min_val) / 2).astype(np.uint8)
    else:
        raise ValueError(f"Unknown method: {method}")


def apply_clahe(
    image: NDArray[np.uint8],
    clip_limit: float = 3.0,
    tile_size: Tuple[int, int] = (8, 8),
) -> NDArray[np.uint8]:
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).

    Enhances local contrast while limiting noise amplification.
    Works on grayscale or the L-channel of LAB color space.

    Parameters
    ----------
    image : NDArray[np.uint8]
        Input image (grayscale or BGR).
    clip_limit : float, optional
        Threshold for contrast limiting. Default 3.0.
    tile_size : tuple[int, int], optional
        Size of grid tiles for histogram equalization. Default (8, 8).

    Returns
    -------
    NDArray[np.uint8]
        Contrast-enhanced image.

    Examples
    --------
    >>> enhanced = apply_clahe(image, clip_limit=2.0)
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)

    if len(image.shape) == 2:
        # Grayscale
        return clahe.apply(image)
    else:
        # Color - apply to L channel in LAB space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        l_enhanced = clahe.apply(l_channel)
        lab_enhanced = cv2.merge((l_enhanced, a_channel, b_channel))
        return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)


def binarize(
    image: NDArray[np.uint8],
    method: str = "otsu",
    threshold: int = 127,
    invert: bool = True,
    adaptive_block_size: int = 11,
    adaptive_c: int = 2,
) -> NDArray[np.uint8]:
    """Convert image to binary (black and white).

    Parameters
    ----------
    image : NDArray[np.uint8]
        Input grayscale image.
    method : str, optional
        Thresholding method:
        - "otsu": Automatic threshold using Otsu's method (default)
        - "simple": Fixed threshold value
        - "adaptive_mean": Adaptive using local mean
        - "adaptive_gaussian": Adaptive using Gaussian weighted mean
    threshold : int, optional
        Threshold value for "simple" method. Default 127.
    invert : bool, optional
        If True, objects are white on black background. Default True.
    adaptive_block_size : int, optional
        Block size for adaptive methods (must be odd). Default 11.
    adaptive_c : int, optional
        Constant subtracted from mean in adaptive methods. Default 2.

    Returns
    -------
    NDArray[np.uint8]
        Binary image (0 and 255 values only).

    Examples
    --------
    >>> binary = binarize(gray)  # Otsu's method
    >>> binary = binarize(gray, method="adaptive_gaussian")
    """
    # Ensure grayscale
    if len(image.shape) > 2:
        image = to_grayscale(image)

    # Determine threshold flags
    if invert:
        thresh_type = cv2.THRESH_BINARY_INV
        adaptive_type = cv2.THRESH_BINARY_INV
    else:
        thresh_type = cv2.THRESH_BINARY
        adaptive_type = cv2.THRESH_BINARY

    if method == "otsu":
        _, binary = cv2.threshold(
            image, 0, 255, thresh_type + cv2.THRESH_OTSU
        )
    elif method == "simple":
        _, binary = cv2.threshold(image, threshold, 255, thresh_type)
    elif method == "adaptive_mean":
        binary = cv2.adaptiveThreshold(
            image,
            255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            adaptive_type,
            adaptive_block_size,
            adaptive_c,
        )
    elif method == "adaptive_gaussian":
        binary = cv2.adaptiveThreshold(
            image,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            adaptive_type,
            adaptive_block_size,
            adaptive_c,
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    return binary


def detect_edges(
    image: NDArray[np.uint8],
    method: str = "canny",
    low_threshold: int = 50,
    high_threshold: int = 150,
    kernel_size: int = 3,
    dilate: bool = True,
    erode: bool = True,
) -> NDArray[np.uint8]:
    """Detect edges in an image.

    Parameters
    ----------
    image : NDArray[np.uint8]
        Input image (grayscale or BGR).
    method : str, optional
        Edge detection method:
        - "canny": Canny edge detection (default)
        - "sobel": Sobel gradient magnitude
        - "laplacian": Laplacian of Gaussian
    low_threshold : int, optional
        Low threshold for Canny. Default 50.
    high_threshold : int, optional
        High threshold for Canny. Default 150.
    kernel_size : int, optional
        Kernel size for Sobel/Laplacian. Default 3.
    dilate : bool, optional
        Apply dilation to close gaps. Default True.
    erode : bool, optional
        Apply erosion to thin edges. Default True.

    Returns
    -------
    NDArray[np.uint8]
        Edge image.

    Examples
    --------
    >>> edges = detect_edges(gray)
    >>> edges = detect_edges(gray, method="sobel")
    """
    # Ensure grayscale
    if len(image.shape) > 2:
        gray = to_grayscale(image)
    else:
        gray = image

    if method == "canny":
        edges = cv2.Canny(gray, low_threshold, high_threshold)
    elif method == "sobel":
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=kernel_size)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=kernel_size)
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        edges = np.uint8(np.clip(magnitude, 0, 255))
    elif method == "laplacian":
        laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=kernel_size)
        edges = np.uint8(np.abs(laplacian))
    else:
        raise ValueError(f"Unknown method: {method}")

    # Morphological operations
    if dilate:
        edges = cv2.dilate(edges, None)
    if erode:
        edges = cv2.erode(edges, None)

    return edges


def morphological_open(
    image: NDArray[np.uint8],
    kernel_size: int = 5,
    iterations: int = 1,
) -> NDArray[np.uint8]:
    """Apply morphological opening (erosion then dilation).

    Removes small objects and noise while preserving larger shapes.

    Parameters
    ----------
    image : NDArray[np.uint8]
        Input binary image.
    kernel_size : int, optional
        Size of structuring element. Default 5.
    iterations : int, optional
        Number of iterations. Default 1.

    Returns
    -------
    NDArray[np.uint8]
        Processed image.
    """
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
    )
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=iterations)


def morphological_close(
    image: NDArray[np.uint8],
    kernel_size: int = 5,
    iterations: int = 1,
) -> NDArray[np.uint8]:
    """Apply morphological closing (dilation then erosion).

    Fills small holes while preserving object boundaries.

    Parameters
    ----------
    image : NDArray[np.uint8]
        Input binary image.
    kernel_size : int, optional
        Size of structuring element. Default 5.
    iterations : int, optional
        Number of iterations. Default 1.

    Returns
    -------
    NDArray[np.uint8]
        Processed image.
    """
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
    )
    return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=iterations)


def preprocess_for_contours(
    image: NDArray[np.uint8],
    blur_kernel: Tuple[int, int] = (11, 11),
    clahe_clip: float = 3.0,
    threshold_method: str = "otsu",
) -> Dict[str, NDArray[np.uint8]]:
    """Full preprocessing pipeline for contour detection.

    Applies blur, contrast enhancement, grayscale conversion,
    thresholding, and edge detection.

    Parameters
    ----------
    image : NDArray[np.uint8]
        Input image in BGR format.
    blur_kernel : tuple[int, int], optional
        Gaussian blur kernel size. Default (11, 11).
    clahe_clip : float, optional
        CLAHE clip limit. Default 3.0.
    threshold_method : str, optional
        Binarization method. Default "otsu".

    Returns
    -------
    dict[str, NDArray[np.uint8]]
        Dictionary containing all intermediate results:
        - "original": Original image copy
        - "blurred": After Gaussian blur
        - "contrast": After CLAHE
        - "grayscale": Grayscale conversion
        - "threshold": Binary threshold
        - "edges": Canny edge detection

    Examples
    --------
    >>> results = preprocess_for_contours(image)
    >>> binary = results["threshold"]
    >>> contours = find_contours(binary)
    """
    # Blur to reduce noise
    blurred = cv2.GaussianBlur(image, blur_kernel, 0)

    # Enhance contrast
    contrast = apply_clahe(blurred, clip_limit=clahe_clip)

    # Convert to grayscale
    grayscale = to_grayscale(contrast)

    # Binarize
    threshold = binarize(grayscale, method=threshold_method)

    # Edge detection
    edges = detect_edges(grayscale)

    return {
        "original": image.copy(),
        "blurred": blurred,
        "contrast": contrast,
        "grayscale": grayscale,
        "threshold": threshold,
        "edges": edges,
    }
