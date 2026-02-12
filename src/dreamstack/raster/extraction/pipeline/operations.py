"""
Pipeline Operations
===================

Functional API for batch extraction operations.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from pathlib import Path

from dreamstack.raster.extraction.extracted_object import ExtractedObject
from dreamstack.raster.extraction.object_extractor import ObjectExtractor

logger = logging.getLogger(__name__)

# Supported image formats
DEFAULT_IMAGE_FORMATS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".tiff",
    ".tif",
    ".bmp",
    ".webp",
)


def find_images(
    directory: str | Path,
    extensions: tuple = DEFAULT_IMAGE_FORMATS,
    recursive: bool = False,
) -> list[Path]:
    """Find all image files in a directory.

    Parameters
    ----------
    directory : str | Path
        Directory to search.
    extensions : tuple, optional
        File extensions to include. Default common image formats.
    recursive : bool, optional
        Whether to search subdirectories. Default False.

    Returns
    -------
    list[Path]
        List of image file paths, sorted alphabetically.

    Examples
    --------
    >>> images = find_images("scans/", recursive=True)
    >>> print(f"Found {len(images)} images")
    """
    directory = Path(directory)
    pattern = "**/*" if recursive else "*"

    image_files = []
    for ext in extensions:
        image_files.extend(directory.glob(f"{pattern}{ext}"))
        image_files.extend(directory.glob(f"{pattern}{ext.upper()}"))

    return sorted(set(image_files))


def process_image(
    image_path: str | Path,
    output_dir: str | Path | None = None,
    extractor: ObjectExtractor | None = None,
    prefix: str | None = None,
    extension: str = ".png",
) -> list[ExtractedObject]:
    """Process a single image and extract objects.

    Parameters
    ----------
    image_path : str | Path
        Path to the input image.
    output_dir : str | Path | None, optional
        Directory to save extracted objects. None = don't save.
    extractor : ObjectExtractor | None, optional
        Extractor instance. Creates default if None.
    prefix : str | None, optional
        Filename prefix. Uses source stem if None.
    extension : str, optional
        Output file extension. Default ".png".

    Returns
    -------
    list[ExtractedObject]
        List of extracted objects.

    Examples
    --------
    >>> objects = process_image("scan.jpg", "output/")
    >>> print(f"Extracted {len(objects)} objects")
    """
    image_path = Path(image_path)
    extractor = extractor or ObjectExtractor()
    prefix = prefix or image_path.stem

    logger.info(f"Processing: {image_path.name}")

    objects = extractor.extract_from_file(image_path)
    logger.info(f"Found {len(objects)} objects")

    if output_dir:
        output_dir = Path(output_dir)
        saved = extractor.save_objects(
            objects,
            output_dir,
            prefix=prefix,
            extension=extension,
        )
        logger.info(f"Saved {len(saved)} objects to {output_dir}")

    return objects


def process_directory(
    input_dir: str | Path,
    output_dir: str | Path,
    extractor: ObjectExtractor | None = None,
    recursive: bool = False,
    extensions: tuple = DEFAULT_IMAGE_FORMATS,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> list[ExtractedObject]:
    """Process all images in a directory.

    Parameters
    ----------
    input_dir : str | Path
        Input directory containing images.
    output_dir : str | Path
        Output directory for extracted objects.
    extractor : ObjectExtractor | None, optional
        Extractor instance. Creates default if None.
    recursive : bool, optional
        Process subdirectories. Default False.
    extensions : tuple, optional
        Image file extensions.
    progress_callback : callable | None, optional
        Progress callback: callback(current, total, filename).

    Returns
    -------
    list[ExtractedObject]
        All extracted objects from all images.

    Examples
    --------
    >>> def show_progress(current, total, name):
    ...     print(f"[{current}/{total}] {name}")
    >>>
    >>> objects = process_directory(
    ...     "scans/",
    ...     "output/",
    ...     progress_callback=show_progress,
    ... )
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    extractor = extractor or ObjectExtractor()
    image_files = find_images(input_dir, extensions, recursive)

    total = len(image_files)
    logger.info(f"Found {total} images to process")

    all_objects = []
    for idx, image_path in enumerate(image_files, 1):
        if progress_callback:
            progress_callback(idx, total, image_path.name)

        try:
            objects = process_image(
                image_path,
                output_dir,
                extractor=extractor,
                prefix=image_path.stem,
            )
            all_objects.extend(objects)
        except Exception as e:
            logger.error(f"Failed to process {image_path}: {e}")

    logger.info(f"Extracted {len(all_objects)} total objects")
    return all_objects


def process_directory_iter(
    input_dir: str | Path,
    extractor: ObjectExtractor | None = None,
    recursive: bool = False,
    extensions: tuple = DEFAULT_IMAGE_FORMATS,
) -> Iterator[tuple[Path, list[ExtractedObject]]]:
    """Process directory and yield results incrementally.

    Useful for processing large directories where you want
    to handle results as they're generated.

    Parameters
    ----------
    input_dir : str | Path
        Input directory.
    extractor : ObjectExtractor | None, optional
        Extractor instance.
    recursive : bool, optional
        Process subdirectories.
    extensions : tuple, optional
        Image file extensions.

    Yields
    ------
    tuple[Path, list[ExtractedObject]]
        Tuples of (image_path, extracted_objects).

    Examples
    --------
    >>> for image_path, objects in process_directory_iter("scans/"):
    ...     print(f"{image_path.name}: {len(objects)} objects")
    """
    input_dir = Path(input_dir)
    extractor = extractor or ObjectExtractor()
    image_files = find_images(input_dir, extensions, recursive)

    for image_path in image_files:
        try:
            objects = extractor.extract_from_file(image_path)
            yield image_path, objects
        except Exception as e:
            logger.error(f"Failed to process {image_path}: {e}")
            yield image_path, []
