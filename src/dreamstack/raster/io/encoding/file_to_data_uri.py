"""
File to Data URI
================

Read image file and encode to data URI.

"""

from __future__ import annotations

from pathlib import Path

from dreamstack.raster.io.encoding.file_to_base64 import file_to_base64


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
