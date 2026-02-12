"""
Image Extension Validation
==========================

Validate that a file has a supported image extension.

"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from dreamstack.raster.io.validation.constants import (
    SUPPORTED_EXTENSIONS,
    SUPPORTED_RAW_EXTENSIONS,
)


def validate_image_extension(
    path: str | Path,
    *,
    extensions: Sequence[str] | None = None,
    include_raw: bool = False,
) -> Path:
    """Validate that a file has a supported image extension.

    Args:
        path: Path to the file.
        extensions: Custom list of allowed extensions.
        include_raw: Include RAW camera formats.

    Returns:
        Validated Path object.

    Raises:
        InvalidImageTypeError: If extension is not supported.

    Example:
        >>> path = validate_image_extension("photo.jpg")
    """
    from dreamstack.raster.core.exceptions import InvalidImageTypeError

    path = Path(path)
    ext = path.suffix.lower().lstrip(".")

    if extensions is None:
        allowed = set(SUPPORTED_EXTENSIONS)
        if include_raw:
            allowed.update(SUPPORTED_RAW_EXTENSIONS)
    else:
        allowed = {e.lower().lstrip(".") for e in extensions}

    if ext not in allowed:
        allowed_str = ", ".join(sorted(allowed))
        raise InvalidImageTypeError(
            f"Unsupported image extension '.{ext}'. Supported: {allowed_str}"
        )

    return path
