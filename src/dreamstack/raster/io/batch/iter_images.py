# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - iter_images
===========

Iterate over images in a directory.

"""


# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING

from dreamstack.raster.io.batch.find_images import find_images

if TYPE_CHECKING:
    # pylint: disable=import-outside-toplevel
    from numpy.typing import NDArray


def iter_images(
    source: str | Path,
    *,
    extensions: tuple[str, ...] | None = None,
    recursive: bool = False,
    load: bool = True,
) -> Iterator[tuple[Path, NDArray | None]]:
    """Iterate over images in a directory.

    Generator that yields image paths and optionally loaded data.

    Args:
        source: Directory path.
        extensions: Allowed extensions.
        recursive: Search subdirectories.
        load: If True, load and yield image data.

    Yields:
        Tuples of (path, image_data or None).

    Example:
        >>> for path, image in iter_images("photos/", load=True):
        ...     processed = my_filter(image)
        ...     save_image(processed, f"out/{path.name}")
    """
    import cv2

    images = find_images(source, extensions=extensions, recursive=recursive)

    for path in images:
        if load:
            image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            yield path, image
        else:
            yield path, None
