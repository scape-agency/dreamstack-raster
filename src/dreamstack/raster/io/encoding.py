# -*- coding: utf-8 -*-

"""
Base64 Encoding Utilities
=========================

Encode images to base64 for web embedding and data URIs.

"""

from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


ImageFormat = Literal["png", "jpeg", "webp", "gif"]


def image_to_base64(
    image: NDArray[np.uint8],
    format: ImageFormat = "png",
    *,
    quality: int = 85,
) -> str:
    """Encode image array to base64 string.
    
    Args:
        image: Input image (BGR or RGB, 3-4 channels).
        format: Output format.
        quality: Compression quality for JPEG/WebP.
    
    Returns:
        Base64 encoded string.
    
    Example:
        >>> b64 = image_to_base64(image, format="jpeg")
        >>> # Use for embedding in HTML/JSON
    """
    from PIL import Image
    import cv2
    
    # Convert BGR to RGB for PIL
    if image.ndim == 3:
        if image.shape[2] == 3:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif image.shape[2] == 4:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
        else:
            rgb = image
    else:
        rgb = image
    
    pil_image = Image.fromarray(rgb)
    buffer = BytesIO()
    
    # Save with appropriate format
    format_upper = format.upper()
    if format_upper == "JPEG":
        if pil_image.mode == "RGBA":
            bg = Image.new("RGB", pil_image.size, (255, 255, 255))
            bg.paste(pil_image, mask=pil_image.split()[3])
            pil_image = bg
        pil_image.save(buffer, format="JPEG", quality=quality)
    elif format_upper == "WEBP":
        pil_image.save(buffer, format="WEBP", quality=quality)
    elif format_upper == "GIF":
        pil_image.save(buffer, format="GIF")
    else:
        pil_image.save(buffer, format="PNG")
    
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def image_to_data_uri(
    image: NDArray[np.uint8],
    format: ImageFormat = "png",
    *,
    quality: int = 85,
) -> str:
    """Encode image to data URI for HTML embedding.
    
    Returns a complete data URI that can be used directly in HTML img src.
    
    Args:
        image: Input image.
        format: Output format.
        quality: Compression quality for JPEG/WebP.
    
    Returns:
        Data URI string (data:image/format;base64,...).
    
    Example:
        >>> uri = image_to_data_uri(image, format="jpeg", quality=80)
        >>> html = f'<img src="{uri}" />'
    """
    b64 = image_to_base64(image, format, quality=quality)
    mime_type = f"image/{format}"
    return f"data:{mime_type};base64,{b64}"


def file_to_base64(path: str | Path) -> str:
    """Read image file and encode to base64.
    
    Args:
        path: Path to image file.
    
    Returns:
        Base64 encoded string.
    """
    path = Path(path)
    return base64.b64encode(path.read_bytes()).decode("utf-8")


def file_to_data_uri(path: str | Path) -> str:
    """Read image file and encode to data URI.
    
    Automatically detects format from file extension.
    
    Args:
        path: Path to image file.
    
    Returns:
        Data URI string.
    
    Example:
        >>> uri = file_to_data_uri("photo.jpg")
        >>> # Returns: data:image/jpeg;base64,...
    """
    path = Path(path)
    
    # Detect MIME type from extension
    ext_to_mime = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp",
        ".svg": "image/svg+xml",
        ".bmp": "image/bmp",
    }
    
    ext = path.suffix.lower()
    mime_type = ext_to_mime.get(ext, "application/octet-stream")
    
    b64 = file_to_base64(path)
    return f"data:{mime_type};base64,{b64}"


def base64_to_image(b64_string: str) -> NDArray[np.uint8]:
    """Decode base64 string to image array.
    
    Handles both raw base64 and data URIs.
    
    Args:
        b64_string: Base64 encoded string or data URI.
    
    Returns:
        Image as numpy array (RGB).
    
    Example:
        >>> image = base64_to_image(b64_data)
        >>> # Or with data URI
        >>> image = base64_to_image("data:image/png;base64,...")
    """
    from PIL import Image
    
    # Handle data URI
    if b64_string.startswith("data:"):
        # Extract base64 part after the comma
        _, b64_string = b64_string.split(",", 1)
    
    # Decode
    data = base64.b64decode(b64_string)
    buffer = BytesIO(data)
    pil_image = Image.open(buffer)
    
    return np.array(pil_image)


def data_uri_to_image(data_uri: str) -> NDArray[np.uint8]:
    """Decode data URI to image array.
    
    Alias for base64_to_image that explicitly expects a data URI.
    
    Args:
        data_uri: Data URI string.
    
    Returns:
        Image as numpy array (RGB).
    """
    return base64_to_image(data_uri)
