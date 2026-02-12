# -*- coding: utf-8 -*-

"""
File and Image Validation Utilities
===================================

Validation functions for paths, files, and image data.

"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray


# Default supported image extensions
SUPPORTED_EXTENSIONS: tuple[str, ...] = (
    "jpg", "jpeg", "png", "gif", "bmp", "tiff", "tif",
    "webp", "ico", "ppm", "pgm", "pbm", "hdr", "exr",
)

SUPPORTED_RAW_EXTENSIONS: tuple[str, ...] = (
    "raw", "cr2", "cr3", "nef", "arw", "dng", "orf", "rw2",
)


def validate_path(path: str | Path) -> Path:
    """Validate that a path exists.
    
    Args:
        path: Path string or Path object.
    
    Returns:
        Validated Path object.
    
    Raises:
        FileNotFoundError: If path does not exist.
    
    Example:
        >>> path = validate_path("/path/to/file.png")
    """
    from dreamstack.raster.core.exceptions import InvalidPathError
    
    path = Path(path)
    
    if not path.exists():
        raise InvalidPathError(f"Path does not exist: {path}")
    
    return path


def validate_file(path: str | Path) -> Path:
    """Validate that a path is an existing file.
    
    Args:
        path: Path to validate.
    
    Returns:
        Validated Path object.
    
    Raises:
        FileNotFoundError: If path does not exist.
        NotAFileError: If path exists but is not a file.
    
    Example:
        >>> file_path = validate_file("image.png")
    """
    from dreamstack.raster.core.exceptions import NotAFileError
    
    path = validate_path(path)
    
    if not path.is_file():
        raise NotAFileError(f"Path exists but is not a file: {path}")
    
    return path


def validate_directory(path: str | Path) -> Path:
    """Validate that a path is an existing directory.
    
    Args:
        path: Path to validate.
    
    Returns:
        Validated Path object.
    
    Raises:
        FileNotFoundError: If path does not exist.
        DirectoryNotFoundError: If path exists but is not a directory.
    
    Example:
        >>> dir_path = validate_directory("/path/to/images")
    """
    from dreamstack.raster.core.exceptions import DirectoryNotFoundError
    
    path = validate_path(path)
    
    if not path.is_dir():
        raise DirectoryNotFoundError(f"Path exists but is not a directory: {path}")
    
    return path


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
            f"Unsupported image extension '.{ext}'. "
            f"Supported: {allowed_str}"
        )
    
    return path


def validate_image_file(
    path: str | Path,
    *,
    extensions: Sequence[str] | None = None,
    include_raw: bool = False,
) -> Path:
    """Validate that a path is an existing image file.
    
    Combines file existence and extension validation.
    
    Args:
        path: Path to the image file.
        extensions: Custom allowed extensions.
        include_raw: Include RAW formats.
    
    Returns:
        Validated Path object.
    
    Example:
        >>> img = validate_image_file("photo.jpg")
    """
    path = validate_file(path)
    path = validate_image_extension(path, extensions=extensions, include_raw=include_raw)
    return path


def validate_image_array(
    image: NDArray,
    *,
    min_channels: int = 1,
    max_channels: int = 4,
    require_2d: bool = False,
) -> NDArray:
    """Validate an image numpy array.
    
    Args:
        image: Image array to validate.
        min_channels: Minimum required channels.
        max_channels: Maximum allowed channels.
        require_2d: Require grayscale (2D) image.
    
    Returns:
        Validated image array.
    
    Raises:
        InvalidImageError: If image is invalid.
    
    Example:
        >>> validated = validate_image_array(image)
    """
    import numpy as np
    from dreamstack.raster.core.exceptions import InvalidImageError
    
    if not isinstance(image, np.ndarray):
        raise InvalidImageError("Image must be a numpy array")
    
    if image.size == 0:
        raise InvalidImageError("Image array is empty")
    
    if require_2d:
        if len(image.shape) != 2:
            raise InvalidImageError("Image must be 2D (grayscale)")
    else:
        if len(image.shape) not in (2, 3):
            raise InvalidImageError(
                f"Image must be 2D or 3D, got shape: {image.shape}"
            )
        
        if len(image.shape) == 3:
            channels = image.shape[2]
            if channels < min_channels or channels > max_channels:
                raise InvalidImageError(
                    f"Image must have {min_channels}-{max_channels} channels, "
                    f"got {channels}"
                )
    
    return image


def is_valid_image_file(path: str | Path) -> bool:
    """Check if a path is a valid image file.
    
    Non-raising version of validate_image_file.
    
    Args:
        path: Path to check.
    
    Returns:
        True if valid image file, False otherwise.
    
    Example:
        >>> if is_valid_image_file("photo.jpg"):
        ...     process_image("photo.jpg")
    """
    try:
        validate_image_file(path)
        return True
    except Exception:
        return False


def get_image_files(
    directory: str | Path,
    *,
    extensions: Sequence[str] | None = None,
    include_raw: bool = False,
    recursive: bool = False,
) -> list[Path]:
    """Get all image files in a directory.
    
    Args:
        directory: Directory to search.
        extensions: Specific extensions to match.
        include_raw: Include RAW camera formats.
        recursive: Search subdirectories.
    
    Returns:
        List of image file paths.
    
    Example:
        >>> images = get_image_files("/photos", recursive=True)
    """
    directory = validate_directory(directory)
    
    if extensions is None:
        exts = list(SUPPORTED_EXTENSIONS)
        if include_raw:
            exts.extend(SUPPORTED_RAW_EXTENSIONS)
    else:
        exts = [e.lower().lstrip(".") for e in extensions]
    
    files = []
    search_fn = directory.rglob if recursive else directory.glob
    
    for ext in exts:
        # Case-insensitive matching
        pattern = f"*.{ext}"
        files.extend(search_fn(pattern))
        files.extend(search_fn(pattern.upper()))
    
    # Remove duplicates and sort
    return sorted(set(files))


def ensure_directory(path: str | Path) -> Path:
    """Ensure a directory exists, creating if necessary.
    
    Args:
        path: Directory path.
    
    Returns:
        Path object (guaranteed to exist).
    
    Example:
        >>> output_dir = ensure_directory("output/processed")
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
