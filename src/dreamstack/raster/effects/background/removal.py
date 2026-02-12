# -*- coding: utf-8 -*-

"""
Background Removal Operations
============================

AI-based background removal using rembg with fallback options.
Includes mask extraction, refinement, and compositing utilities.

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


# Available rembg models
ModelName = Literal[
    "u2net",
    "u2netp",
    "u2net_human_seg",
    "u2net_cloth_seg",
    "silueta",
    "isnet-general-use",
    "isnet-anime",
    "sam",
]


@dataclass
class RemovalConfig:
    """Configuration for background removal.
    
    Attributes:
        model_name: The rembg model to use for segmentation.
        alpha_matting: Enable alpha matting for refined edges.
        alpha_matting_foreground_threshold: Foreground threshold for matting.
        alpha_matting_background_threshold: Background threshold for matting.
        alpha_matting_erode_size: Erosion size for matting refinement.
        post_process_mask: Apply post-processing to the mask.
    """
    
    model_name: ModelName = "u2net"
    alpha_matting: bool = False
    alpha_matting_foreground_threshold: int = 240
    alpha_matting_background_threshold: int = 10
    alpha_matting_erode_size: int = 10
    post_process_mask: bool = False


@dataclass
class MaskRefinementConfig:
    """Configuration for mask refinement operations.
    
    Attributes:
        dilate_iterations: Number of dilation iterations.
        erode_iterations: Number of erosion iterations.
        blur_size: Gaussian blur kernel size (0 for no blur).
        feather_amount: Edge feathering amount.
        threshold: Threshold value for binary mask (None for no thresholding).
    """
    
    dilate_iterations: int = 0
    erode_iterations: int = 0
    blur_size: int = 0
    feather_amount: int = 0
    threshold: int | None = None


def _check_rembg_available() -> bool:
    """Check if rembg is available."""
    try:
        import rembg  # noqa: F401
        return True
    except ImportError:
        return False


def remove_background(
    image: NDArray[np.uint8],
    config: RemovalConfig | None = None,
    *,
    model_name: ModelName | None = None,
    alpha_matting: bool | None = None,
) -> NDArray[np.uint8]:
    """Remove background from an image using AI segmentation.
    
    Uses rembg library for AI-based background removal. Returns RGBA image
    with transparent background.
    
    Args:
        image: Input image as numpy array (BGR or RGB, 3 channels).
        config: Optional RemovalConfig for detailed settings.
        model_name: Override model name (u2net, isnet-general-use, etc.).
        alpha_matting: Override alpha matting setting.
    
    Returns:
        RGBA image with background removed (4 channels).
    
    Raises:
        ImportError: If rembg is not installed.
        ValueError: If image format is invalid.
    
    Example:
        >>> from dreamstack.raster.effects.background import remove_background
        >>> result = remove_background(image, model_name="u2net")
        >>> # result is RGBA with transparent background
    """
    if not _check_rembg_available():
        raise ImportError(
            "rembg is required for background removal. "
            "Install with: pip install rembg"
        )
    
    import rembg
    
    # Use config or defaults
    cfg = config or RemovalConfig()
    
    # Override with keyword arguments
    if model_name is not None:
        cfg = RemovalConfig(
            model_name=model_name,
            alpha_matting=cfg.alpha_matting if alpha_matting is None else alpha_matting,
            alpha_matting_foreground_threshold=cfg.alpha_matting_foreground_threshold,
            alpha_matting_background_threshold=cfg.alpha_matting_background_threshold,
            alpha_matting_erode_size=cfg.alpha_matting_erode_size,
            post_process_mask=cfg.post_process_mask,
        )
    elif alpha_matting is not None:
        cfg = RemovalConfig(
            model_name=cfg.model_name,
            alpha_matting=alpha_matting,
            alpha_matting_foreground_threshold=cfg.alpha_matting_foreground_threshold,
            alpha_matting_background_threshold=cfg.alpha_matting_background_threshold,
            alpha_matting_erode_size=cfg.alpha_matting_erode_size,
            post_process_mask=cfg.post_process_mask,
        )
    
    # Validate input
    if image.ndim != 3:
        raise ValueError(f"Expected 3-channel image, got {image.ndim} dimensions")
    
    # Convert to RGBA for rembg
    from PIL import Image
    
    if image.shape[2] == 3:
        # Assume BGR from OpenCV, convert to RGB
        import cv2
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif image.shape[2] == 4:
        import cv2
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)[:, :, :3]
    else:
        raise ValueError(f"Expected 3 or 4 channels, got {image.shape[2]}")
    
    pil_image = Image.fromarray(rgb_image)
    
    # Remove background
    result = rembg.remove(
        pil_image,
        session=rembg.new_session(cfg.model_name),
        alpha_matting=cfg.alpha_matting,
        alpha_matting_foreground_threshold=cfg.alpha_matting_foreground_threshold,
        alpha_matting_background_threshold=cfg.alpha_matting_background_threshold,
        alpha_matting_erode_size=cfg.alpha_matting_erode_size,
        post_process_mask=cfg.post_process_mask,
    )
    
    # Convert back to numpy RGBA
    return np.array(result)


def extract_alpha_mask(
    image: NDArray[np.uint8],
    config: RemovalConfig | None = None,
    *,
    model_name: ModelName | None = None,
    threshold: int | None = None,
) -> NDArray[np.uint8]:
    """Extract alpha mask from an image without removing background.
    
    Returns a single-channel grayscale mask where white (255) represents
    foreground and black (0) represents background.
    
    Args:
        image: Input image as numpy array (BGR, 3 channels).
        config: Optional RemovalConfig for segmentation settings.
        model_name: Override model name.
        threshold: Apply binary threshold to mask (0-255). None for soft mask.
    
    Returns:
        Grayscale alpha mask (single channel, 0-255).
    
    Example:
        >>> mask = extract_alpha_mask(image, model_name="u2net")
        >>> binary_mask = extract_alpha_mask(image, threshold=128)
    """
    # Get RGBA result
    rgba = remove_background(image, config, model_name=model_name)
    
    # Extract alpha channel
    mask = rgba[:, :, 3]
    
    # Apply threshold if specified
    if threshold is not None:
        mask = np.where(mask >= threshold, 255, 0).astype(np.uint8)
    
    return mask


def refine_mask(
    mask: NDArray[np.uint8],
    config: MaskRefinementConfig | None = None,
    *,
    dilate_iterations: int | None = None,
    erode_iterations: int | None = None,
    blur_size: int | None = None,
    feather_amount: int | None = None,
) -> NDArray[np.uint8]:
    """Refine a mask with morphological operations and blurring.
    
    Apply dilation, erosion, blurring, and feathering to improve mask edges.
    
    Args:
        mask: Input grayscale mask (single channel).
        config: Optional MaskRefinementConfig for settings.
        dilate_iterations: Number of dilation iterations (expands mask).
        erode_iterations: Number of erosion iterations (shrinks mask).
        blur_size: Gaussian blur kernel size (must be odd, 0 for no blur).
        feather_amount: Edge feathering in pixels.
    
    Returns:
        Refined grayscale mask.
    
    Example:
        >>> refined = refine_mask(mask, dilate_iterations=2, blur_size=5)
    """
    import cv2
    
    # Use config or defaults
    cfg = config or MaskRefinementConfig()
    
    # Override with keyword arguments
    d_iter = dilate_iterations if dilate_iterations is not None else cfg.dilate_iterations
    e_iter = erode_iterations if erode_iterations is not None else cfg.erode_iterations
    b_size = blur_size if blur_size is not None else cfg.blur_size
    f_amount = feather_amount if feather_amount is not None else cfg.feather_amount
    
    result = mask.copy()
    
    # Create kernel for morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    
    # Apply dilation (expands white regions)
    if d_iter > 0:
        result = cv2.dilate(result, kernel, iterations=d_iter)
    
    # Apply erosion (shrinks white regions)
    if e_iter > 0:
        result = cv2.erode(result, kernel, iterations=e_iter)
    
    # Apply Gaussian blur for smoothing
    if b_size > 0:
        # Ensure kernel size is odd
        if b_size % 2 == 0:
            b_size += 1
        result = cv2.GaussianBlur(result, (b_size, b_size), 0)
    
    # Apply feathering (soft edge transition)
    if f_amount > 0:
        # Use distance transform for feathering
        dist = cv2.distanceTransform(result, cv2.DIST_L2, 5)
        dist = np.clip(dist / f_amount, 0, 1)
        result = (dist * 255).astype(np.uint8)
    
    return result


def composite_on_background(
    rgba_image: NDArray[np.uint8],
    background_color: tuple[int, int, int] = (255, 255, 255),
) -> NDArray[np.uint8]:
    """Composite an RGBA image onto a solid color background.
    
    Blend the RGBA image with the alpha channel onto a solid background.
    
    Args:
        rgba_image: RGBA image with alpha channel (4 channels).
        background_color: RGB tuple for background (default white).
    
    Returns:
        RGB image composited on background (3 channels).
    
    Example:
        >>> # Composite on white background
        >>> rgb = composite_on_background(rgba_image)
        >>> # Composite on black background
        >>> rgb = composite_on_background(rgba_image, (0, 0, 0))
    """
    if rgba_image.ndim != 3 or rgba_image.shape[2] != 4:
        raise ValueError("Expected RGBA image with 4 channels")
    
    # Extract channels
    rgb = rgba_image[:, :, :3].astype(np.float32)
    alpha = rgba_image[:, :, 3:4].astype(np.float32) / 255.0
    
    # Create background
    bg = np.full_like(rgb, background_color, dtype=np.float32)
    
    # Alpha blend
    result = rgb * alpha + bg * (1 - alpha)
    
    return result.astype(np.uint8)


def create_color_background(
    size: tuple[int, int],
    color: tuple[int, int, int] = (255, 255, 255),
) -> NDArray[np.uint8]:
    """Create a solid color background image.
    
    Args:
        size: (width, height) of the background.
        color: RGB color tuple.
    
    Returns:
        RGB background image.
    """
    height, width = size[1], size[0]
    bg = np.zeros((height, width, 3), dtype=np.uint8)
    bg[:, :] = color
    return bg
