# -*- coding: utf-8 -*-

"""
Depth Estimation Operations
===========================

Functional interface for depth estimation and depth map utilities.

"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np

from dreamstack.raster.analysis.depth.estimator import (
    DepthConfig,
    DepthEstimator,
    DepthResult,
    ModelName,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray


def estimate_depth(
    image: NDArray[np.uint8],
    model_name: ModelName = "LiheYoung/depth-anything-base-hf",
    *,
    invert: bool = False,
    return_normalized: bool = True,
) -> NDArray[np.float32]:
    """Estimate depth from a single image.
    
    Functional interface for depth estimation. Returns the depth map
    as a float32 array.
    
    Args:
        image: Input image (BGR or RGB, 3 channels).
        model_name: HuggingFace model identifier.
        invert: Invert depth (closer = brighter).
        return_normalized: Return normalized (0-1) depth map.
    
    Returns:
        Depth map as float32 array.
    
    Example:
        >>> from dreamstack.raster.analysis.depth import estimate_depth
        >>> depth = estimate_depth(image)
        >>> depth_uint8 = (depth * 255).astype(np.uint8)
    """
    config = DepthConfig(
        model_name=model_name,
        invert=invert,
    )
    estimator = DepthEstimator(config)
    result = estimator.estimate(image)
    
    return result.depth_normalized if return_normalized else result.depth_map


def estimate_depth_batch(
    images: list[NDArray[np.uint8]],
    model_name: ModelName = "LiheYoung/depth-anything-base-hf",
    *,
    invert: bool = False,
) -> list[DepthResult]:
    """Estimate depth for multiple images.
    
    More efficient than calling estimate_depth multiple times
    as the model is loaded once.
    
    Args:
        images: List of input images.
        model_name: HuggingFace model identifier.
        invert: Invert depth values.
    
    Returns:
        List of DepthResult objects.
    
    Example:
        >>> results = estimate_depth_batch([img1, img2, img3])
        >>> depths = [r.to_uint8() for r in results]
    """
    config = DepthConfig(
        model_name=model_name,
        invert=invert,
    )
    estimator = DepthEstimator(config)
    return estimator.estimate_batch(images)


def normalize_depth(
    depth: NDArray[np.float32],
    min_val: float | None = None,
    max_val: float | None = None,
    invert: bool = False,
) -> NDArray[np.float32]:
    """Normalize depth map to 0-1 range.
    
    Optionally specify custom min/max for consistent normalization
    across multiple depth maps.
    
    Args:
        depth: Raw depth map.
        min_val: Minimum value for normalization (default: auto).
        max_val: Maximum value for normalization (default: auto).
        invert: Invert the normalized values.
    
    Returns:
        Normalized depth map (0-1 range).
    
    Example:
        >>> normalized = normalize_depth(raw_depth, min_val=0, max_val=10)
    """
    d_min = min_val if min_val is not None else float(depth.min())
    d_max = max_val if max_val is not None else float(depth.max())
    
    if d_max > d_min:
        normalized = (depth - d_min) / (d_max - d_min)
    else:
        normalized = np.zeros_like(depth)
    
    if invert:
        normalized = 1.0 - normalized
    
    return normalized.astype(np.float32)


def colorize_depth(
    depth: NDArray[np.float32],
    colormap: str = "inferno",
    normalize: bool = True,
) -> NDArray[np.uint8]:
    """Convert depth map to colored visualization.
    
    Uses matplotlib colormaps for visualization.
    
    Args:
        depth: Depth map (float32).
        colormap: Matplotlib colormap name (inferno, viridis, plasma, magma).
        normalize: Normalize input to 0-1 range.
    
    Returns:
        RGB image with depth colorized (3 channels, uint8).
    
    Example:
        >>> colored = colorize_depth(depth, colormap="plasma")
        >>> cv2.imwrite("depth_colored.png", cv2.cvtColor(colored, cv2.COLOR_RGB2BGR))
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError(
            "matplotlib required for colormaps. "
            "Install with: pip install matplotlib"
        )
    
    # Normalize if needed
    if normalize and (depth.max() > 1.0 or depth.min() < 0.0):
        depth = normalize_depth(depth)
    
    # Apply colormap
    cmap = plt.get_cmap(colormap)
    colored = cmap(depth)[:, :, :3]
    
    return (colored * 255).astype(np.uint8)


def save_depth_image(
    depth: NDArray[np.float32],
    output_path: str | Path,
    *,
    colormap: str | None = None,
    as_16bit: bool = False,
) -> Path:
    """Save depth map as an image file.
    
    Can save as grayscale, 16-bit, or colorized image.
    
    Args:
        depth: Depth map (float32, should be normalized 0-1).
        output_path: Output file path.
        colormap: If specified, save colorized version.
        as_16bit: Save as 16-bit PNG for higher precision.
    
    Returns:
        Path to saved file.
    
    Example:
        >>> save_depth_image(depth, "depth.png")
        >>> save_depth_image(depth, "depth_color.png", colormap="inferno")
        >>> save_depth_image(depth, "depth_16bit.png", as_16bit=True)
    """
    import cv2
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if colormap is not None:
        # Save colorized
        colored = colorize_depth(depth, colormap)
        cv2.imwrite(str(output_path), cv2.cvtColor(colored, cv2.COLOR_RGB2BGR))
    elif as_16bit:
        # Save as 16-bit for precision
        depth_16 = (depth * 65535).astype(np.uint16)
        cv2.imwrite(str(output_path), depth_16)
    else:
        # Save as 8-bit grayscale
        depth_8 = (depth * 255).astype(np.uint8)
        cv2.imwrite(str(output_path), depth_8)
    
    return output_path


def normalize_depth_advanced(
    depth: NDArray[np.float32],
    method: Literal["minmax", "percentile", "log"] = "minmax",
    output_range: tuple[float, float] = (0.0, 1.0),
    clip_percentiles: tuple[float, float] = (2, 98),
) -> NDArray[np.float32]:
    """Normalize depth map using various methods.
    
    Provides multiple normalization strategies for different use cases.
    
    Args:
        depth: Raw depth map (float32).
        method: Normalization strategy:
            - "minmax": Scale between min and max values.
            - "percentile": Robust scaling between percentiles.
            - "log": Logarithmic transformation for high dynamic range.
        output_range: Target output range (min, max).
        clip_percentiles: Percentiles for "percentile" method.
    
    Returns:
        Normalized depth map.
    
    Example:
        >>> # Robust normalization ignoring outliers
        >>> normed = normalize_depth_advanced(depth, method="percentile")
        >>> 
        >>> # Scale to 0-100 range
        >>> scaled = normalize_depth_advanced(depth, output_range=(0, 100))
    """
    d_min, d_max = output_range
    
    if method == "minmax":
        min_val, max_val = float(depth.min()), float(depth.max())
        if max_val > min_val:
            normalized = (depth - min_val) / (max_val - min_val)
        else:
            normalized = np.zeros_like(depth)
    
    elif method == "percentile":
        lo, hi = np.percentile(depth, clip_percentiles)
        clipped = np.clip(depth, lo, hi)
        if hi > lo:
            normalized = (clipped - lo) / (hi - lo)
        else:
            normalized = np.zeros_like(depth)
    
    elif method == "log":
        # Shift to positive values
        shifted = depth - depth.min() + 1
        log_depth = np.log(shifted)
        log_min, log_max = log_depth.min(), log_depth.max()
        if log_max > log_min:
            normalized = (log_depth - log_min) / (log_max - log_min)
        else:
            normalized = np.zeros_like(depth)
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Scale to output range
    normalized = normalized * (d_max - d_min) + d_min
    
    return normalized.astype(np.float32)


def save_depth_ply(
    depth: NDArray[np.float32],
    image: NDArray[np.uint8],
    output_path: str | Path,
    *,
    spacing: float = 0.01,
    z_multiplier: float = 0.01,
) -> Path:
    """Save depth map as PLY point cloud with colors.
    
    Creates a 3D point cloud where each pixel becomes a point,
    positioned by depth and colored by the original image.
    
    Args:
        depth: Depth map (normalized 0-1 or absolute values).
        image: Color image (BGR, same dimensions as depth).
        output_path: Output .ply file path.
        spacing: Horizontal spacing between points.
        z_multiplier: Scale factor for depth (z) values.
    
    Returns:
        Path to saved PLY file.
    
    Example:
        >>> from dreamstack.raster.analysis.depth import (
        ...     estimate_depth, save_depth_ply
        ... )
        >>> depth = estimate_depth(image)
        >>> save_depth_ply(depth, image, "scene.ply", z_multiplier=0.5)
    """
    import cv2
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Ensure image matches depth dimensions
    if image.shape[:2] != depth.shape:
        image = cv2.resize(image, (depth.shape[1], depth.shape[0]))
    
    # Convert BGR to RGB
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    h, w = depth.shape
    points = []
    
    for y in range(h):
        for x in range(w):
            z = float(depth[y, x]) * z_multiplier
            px = (x - w / 2) * spacing
            py = (y - h / 2) * spacing
            r, g, b = rgb[y, x]
            points.append(f"{px:.4f} {py:.4f} {z:.4f} {r} {g} {b}")
    
    header = "\n".join([
        "ply",
        "format ascii 1.0",
        f"element vertex {len(points)}",
        "property float x",
        "property float y",
        "property float z",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        "end_header",
    ])
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(header + "\n" + "\n".join(points))
    
    return output_path


def save_depth_npy(
    depth: NDArray[np.float32],
    output_path: str | Path,
) -> Path:
    """Save depth map as NumPy .npy file.
    
    Preserves full floating-point precision.
    
    Args:
        depth: Depth map.
        output_path: Output .npy file path.
    
    Returns:
        Path to saved file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(output_path), depth)
    return output_path


def load_depth_npy(input_path: str | Path) -> NDArray[np.float32]:
    """Load depth map from NumPy .npy file.
    
    Args:
        input_path: Path to .npy file.
    
    Returns:
        Depth map as float32 array.
    """
    return np.load(str(input_path)).astype(np.float32)
