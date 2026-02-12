# -*- coding: utf-8 -*-

"""
Smart Image Compression
=======================

Utilities for intelligent image compression with target file size.

"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


CompressionFormat = Literal["jpeg", "webp", "png"]


@dataclass
class CompressionConfig:
    """Configuration for smart compression.
    
    Attributes:
        max_size_kb: Maximum file size in kilobytes.
        min_quality: Minimum acceptable quality (1-100).
        max_quality: Starting/maximum quality (1-100).
        format: Output format.
        quality_step: Quality reduction step per iteration.
        max_iterations: Maximum optimization iterations.
    """
    
    max_size_kb: int = 500
    min_quality: int = 20
    max_quality: int = 95
    format: CompressionFormat = "jpeg"
    quality_step: int = 5
    max_iterations: int = 20


@dataclass
class CompressionResult:
    """Result from compression operation.
    
    Attributes:
        data: Compressed image bytes.
        size_kb: Final file size in kilobytes.
        quality: Final quality setting used.
        format: Output format.
        iterations: Number of optimization iterations.
    """
    
    data: bytes
    size_kb: float
    quality: int
    format: str
    iterations: int
    
    def save(self, path: str | Path) -> Path:
        """Save compressed data to file.
        
        Args:
            path: Output file path.
        
        Returns:
            Path to saved file.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self.data)
        return path


def compress_to_size(
    image: NDArray[np.uint8],
    max_size_kb: int = 500,
    *,
    format: CompressionFormat = "jpeg",
    min_quality: int = 20,
    max_quality: int = 95,
) -> CompressionResult:
    """Compress image to fit within a target file size.
    
    Iteratively reduces quality until the image fits within
    the specified size limit.
    
    Args:
        image: Input image (BGR, 3 channels).
        max_size_kb: Maximum file size in kilobytes.
        format: Output format (jpeg, webp, png).
        min_quality: Minimum acceptable quality.
        max_quality: Starting quality.
    
    Returns:
        CompressionResult with compressed data and metadata.
    
    Example:
        >>> from dreamstack.raster.io import compress_to_size
        >>> result = compress_to_size(image, max_size_kb=200)
        >>> result.save("optimized.jpg")
        >>> print(f"Final size: {result.size_kb:.1f}KB at quality {result.quality}")
    """
    from PIL import Image
    import cv2
    
    # Convert BGR to RGB for PIL
    if image.ndim == 3 and image.shape[2] == 3:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif image.ndim == 3 and image.shape[2] == 4:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
    else:
        rgb = image
    
    pil_image = Image.fromarray(rgb)
    
    # Determine format-specific settings
    if format == "jpeg":
        pil_format = "JPEG"
        quality_key = "quality"
    elif format == "webp":
        pil_format = "WEBP"
        quality_key = "quality"
    else:
        # PNG uses compression level, return immediately
        buffer = BytesIO()
        pil_image.save(buffer, format="PNG", compress_level=9)
        data = buffer.getvalue()
        return CompressionResult(
            data=data,
            size_kb=len(data) / 1024,
            quality=100,
            format="png",
            iterations=1,
        )
    
    # Binary search for optimal quality
    quality = max_quality
    iterations = 0
    best_data = None
    best_quality = max_quality
    
    max_iterations = (max_quality - min_quality) // 5 + 5
    
    while quality >= min_quality and iterations < max_iterations:
        iterations += 1
        
        buffer = BytesIO()
        
        # Handle RGBA for JPEG
        if pil_format == "JPEG" and pil_image.mode == "RGBA":
            # Create white background
            bg = Image.new("RGB", pil_image.size, (255, 255, 255))
            bg.paste(pil_image, mask=pil_image.split()[3])
            bg.save(buffer, format=pil_format, **{quality_key: quality})
        else:
            pil_image.save(buffer, format=pil_format, **{quality_key: quality})
        
        data = buffer.getvalue()
        size_kb = len(data) / 1024
        
        if size_kb <= max_size_kb:
            # Found acceptable size
            best_data = data
            best_quality = quality
            break
        
        # Reduce quality
        quality -= 5
        best_data = data
        best_quality = quality + 5
    
    # Use last successful compression
    if best_data is None:
        # Even at min quality, use whatever we got
        buffer = BytesIO()
        if pil_format == "JPEG" and pil_image.mode == "RGBA":
            bg = Image.new("RGB", pil_image.size, (255, 255, 255))
            bg.paste(pil_image, mask=pil_image.split()[3])
            bg.save(buffer, format=pil_format, **{quality_key: min_quality})
        else:
            pil_image.save(buffer, format=pil_format, **{quality_key: min_quality})
        best_data = buffer.getvalue()
        best_quality = min_quality
    
    return CompressionResult(
        data=best_data,
        size_kb=len(best_data) / 1024,
        quality=best_quality,
        format=format,
        iterations=iterations,
    )


def compress_image(
    image: NDArray[np.uint8],
    quality: int = 85,
    *,
    format: CompressionFormat = "jpeg",
) -> bytes:
    """Compress image at a specific quality level.
    
    Simple compression without file size targeting.
    
    Args:
        image: Input image (BGR, 3 channels).
        quality: Compression quality (1-100).
        format: Output format.
    
    Returns:
        Compressed image bytes.
    
    Example:
        >>> data = compress_image(image, quality=80)
        >>> with open("output.jpg", "wb") as f:
        ...     f.write(data)
    """
    from PIL import Image
    import cv2
    
    # Convert BGR to RGB
    if image.ndim == 3 and image.shape[2] == 3:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        rgb = image
    
    pil_image = Image.fromarray(rgb)
    buffer = BytesIO()
    
    if format == "jpeg":
        if pil_image.mode == "RGBA":
            bg = Image.new("RGB", pil_image.size, (255, 255, 255))
            bg.paste(pil_image, mask=pil_image.split()[3])
            pil_image = bg
        pil_image.save(buffer, format="JPEG", quality=quality)
    elif format == "webp":
        pil_image.save(buffer, format="WEBP", quality=quality)
    else:
        pil_image.save(buffer, format="PNG", compress_level=9)
    
    return buffer.getvalue()


def estimate_file_size(
    image: NDArray[np.uint8],
    quality: int = 85,
    *,
    format: CompressionFormat = "jpeg",
) -> float:
    """Estimate file size without saving.
    
    Args:
        image: Input image.
        quality: Compression quality.
        format: Output format.
    
    Returns:
        Estimated file size in kilobytes.
    """
    data = compress_image(image, quality, format=format)
    return len(data) / 1024


def optimize_for_web(
    image: NDArray[np.uint8],
    max_dimension: int = 1920,
    max_size_kb: int = 500,
    *,
    format: CompressionFormat = "webp",
) -> CompressionResult:
    """Optimize image for web delivery.
    
    Resizes if needed and compresses to target size.
    Defaults to WebP for best compression/quality ratio.
    
    Args:
        image: Input image.
        max_dimension: Maximum width or height.
        max_size_kb: Maximum file size.
        format: Output format (webp recommended).
    
    Returns:
        CompressionResult with optimized image.
    
    Example:
        >>> result = optimize_for_web(large_image, max_dimension=1200)
        >>> result.save("web_ready.webp")
    """
    import cv2
    
    h, w = image.shape[:2]
    
    # Resize if needed
    if max(w, h) > max_dimension:
        scale = max_dimension / max(w, h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    
    return compress_to_size(image, max_size_kb, format=format)
