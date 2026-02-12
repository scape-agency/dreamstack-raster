# -*- coding: utf-8 -*-

"""
Batch Image Processing Utilities
================================

Utilities for processing multiple images in batch with
parallel execution and progress tracking.

"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Iterator

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray


logger = logging.getLogger(__name__)


@dataclass
class BatchConfig:
    """Configuration for batch processing.
    
    Attributes:
        max_workers: Maximum parallel workers (None = CPU count).
        recursive: Search directories recursively.
        extensions: Allowed file extensions.
        skip_errors: Continue on errors instead of stopping.
        verbose: Print progress messages.
        overwrite: Overwrite existing output files.
    """
    
    max_workers: int | None = None
    recursive: bool = False
    extensions: tuple[str, ...] | None = None
    skip_errors: bool = True
    verbose: bool = True
    overwrite: bool = False


@dataclass
class BatchResult:
    """Result of a batch processing operation.
    
    Attributes:
        total: Total number of files processed.
        successful: Number of successfully processed files.
        failed: Number of failed files.
        skipped: Number of skipped files.
        output_paths: List of output file paths.
        errors: Dictionary of path -> error message.
    """
    
    total: int = 0
    successful: int = 0
    failed: int = 0
    skipped: int = 0
    output_paths: list[Path] = field(default_factory=list)
    errors: dict[Path, str] = field(default_factory=dict)
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        if self.total == 0:
            return 0.0
        return (self.successful / self.total) * 100


def find_images(
    source: str | Path,
    *,
    extensions: tuple[str, ...] | None = None,
    recursive: bool = False,
) -> list[Path]:
    """Find all image files in a directory.
    
    Args:
        source: Directory path or single file.
        extensions: Allowed extensions (default: common formats).
        recursive: Search subdirectories.
    
    Returns:
        List of image file paths.
    
    Example:
        >>> images = find_images("/photos", recursive=True)
        >>> print(f"Found {len(images)} images")
    """
    from dreamstack.raster.io.validation import (
        SUPPORTED_EXTENSIONS,
        get_image_files,
    )
    
    source = Path(source)
    
    if source.is_file():
        return [source]
    
    exts = extensions if extensions else SUPPORTED_EXTENSIONS
    
    return get_image_files(source, extensions=exts, recursive=recursive)


def batch_process(
    input_dir: str | Path,
    output_dir: str | Path,
    processor: Callable[[NDArray], NDArray],
    *,
    config: BatchConfig | None = None,
    output_format: str = "png",
    output_suffix: str = "",
) -> BatchResult:
    """Process multiple images with a given function.
    
    Applies a processing function to all images in a directory,
    saving results to an output directory.
    
    Args:
        input_dir: Input directory with images.
        output_dir: Output directory for processed images.
        processor: Function that takes image array and returns processed array.
        config: Batch processing configuration.
        output_format: Output file format.
        output_suffix: Suffix to add to output filenames.
    
    Returns:
        BatchResult with processing statistics.
    
    Example:
        >>> from dreamstack.raster.filters.blur import gaussian_blur
        >>> 
        >>> def blur_image(img):
        ...     return gaussian_blur(img, sigma=2.0)
        >>> 
        >>> result = batch_process(
        ...     "input/",
        ...     "output/",
        ...     blur_image,
        ...     output_format="jpg"
        ... )
        >>> print(f"Processed {result.successful}/{result.total} images")
    """
    import cv2
    
    if config is None:
        config = BatchConfig()
    
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find images
    images = find_images(input_dir, extensions=config.extensions, recursive=config.recursive)
    
    result = BatchResult(total=len(images))
    
    if config.verbose:
        logger.info("Found %d images to process", len(images))
    
    def process_single(image_path: Path) -> tuple[Path | None, str | None]:
        """Process a single image."""
        try:
            # Build output path
            rel_path = image_path.relative_to(input_dir)
            out_stem = image_path.stem + output_suffix
            out_name = f"{out_stem}.{output_format}"
            out_path = output_dir / rel_path.parent / out_name
            
            # Check if should skip
            if out_path.exists() and not config.overwrite:
                return None, "skipped"
            
            # Ensure output directory
            out_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Load image
            image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
            if image is None:
                return None, f"Failed to load: {image_path}"
            
            # Process
            processed = processor(image)
            
            # Save
            cv2.imwrite(str(out_path), processed)
            
            return out_path, None
            
        except Exception as e:
            return None, str(e)
    
    # Process in parallel
    with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
        futures = {
            executor.submit(process_single, path): path
            for path in images
        }
        
        for future in as_completed(futures):
            path = futures[future]
            out_path, error = future.result()
            
            if error == "skipped":
                result.skipped += 1
            elif error:
                result.failed += 1
                result.errors[path] = error
                if config.verbose:
                    logger.warning("Failed: %s - %s", path.name, error)
            else:
                result.successful += 1
                if out_path:
                    result.output_paths.append(out_path)
                if config.verbose:
                    logger.info("Processed: %s", path.name)
    
    if config.verbose:
        logger.info(
            "Completed: %d/%d successful (%.1f%%)",
            result.successful,
            result.total,
            result.success_rate,
        )
    
    return result


def batch_resize(
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    width: int = 800,
    height: int = 600,
    maintain_aspect: bool = True,
    config: BatchConfig | None = None,
    output_format: str = "jpg",
    quality: int = 90,
) -> BatchResult:
    """Batch resize images to specified dimensions.
    
    Args:
        input_dir: Input directory.
        output_dir: Output directory.
        width: Target width.
        height: Target height.
        maintain_aspect: Preserve aspect ratio (fit within dimensions).
        config: Batch configuration.
        output_format: Output format.
        quality: JPEG quality (1-100).
    
    Returns:
        BatchResult with processing statistics.
    
    Example:
        >>> result = batch_resize(
        ...     "photos/",
        ...     "thumbnails/",
        ...     width=200,
        ...     height=200
        ... )
    """
    import cv2
    
    def resize_processor(image):
        h, w = image.shape[:2]
        
        if maintain_aspect:
            # Fit within dimensions
            scale = min(width / w, height / h)
            new_w = int(w * scale)
            new_h = int(h * scale)
        else:
            new_w, new_h = width, height
        
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    
    return batch_process(
        input_dir,
        output_dir,
        resize_processor,
        config=config,
        output_format=output_format,
    )


def batch_convert(
    input_dir: str | Path,
    output_dir: str | Path,
    output_format: str = "png",
    *,
    config: BatchConfig | None = None,
) -> BatchResult:
    """Batch convert images to a different format.
    
    Args:
        input_dir: Input directory.
        output_dir: Output directory.
        output_format: Target format (png, jpg, webp, etc.).
        config: Batch configuration.
    
    Returns:
        BatchResult with processing statistics.
    
    Example:
        >>> result = batch_convert("raw/", "converted/", output_format="png")
    """
    # Identity processor - just reads and writes in new format
    return batch_process(
        input_dir,
        output_dir,
        lambda img: img,
        config=config,
        output_format=output_format,
    )


def batch_apply(
    images: list[Path] | list[str],
    processor: Callable[[NDArray], NDArray],
    *,
    output_dir: str | Path | None = None,
    output_suffix: str = "_processed",
    output_format: str | None = None,
    config: BatchConfig | None = None,
) -> BatchResult:
    """Apply a processor function to a list of images.
    
    More flexible than batch_process - takes explicit list of paths.
    
    Args:
        images: List of image paths.
        processor: Processing function.
        output_dir: Output directory (default: same as input).
        output_suffix: Suffix for output files.
        output_format: Output format (default: same as input).
        config: Batch configuration.
    
    Returns:
        BatchResult with processing statistics.
    
    Example:
        >>> images = [Path("a.jpg"), Path("b.jpg")]
        >>> result = batch_apply(images, my_filter)
    """
    import cv2
    
    if config is None:
        config = BatchConfig()
    
    result = BatchResult(total=len(images))
    
    def process_single(img_path: Path) -> tuple[Path | None, str | None]:
        try:
            img_path = Path(img_path)
            
            # Determine output path
            if output_dir:
                out_base = Path(output_dir)
                out_base.mkdir(parents=True, exist_ok=True)
            else:
                out_base = img_path.parent
            
            fmt = output_format if output_format else img_path.suffix.lstrip(".")
            out_name = f"{img_path.stem}{output_suffix}.{fmt}"
            out_path = out_base / out_name
            
            if out_path.exists() and not config.overwrite:
                return None, "skipped"
            
            # Load, process, save
            image = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
            if image is None:
                return None, f"Failed to load: {img_path}"
            
            processed = processor(image)
            cv2.imwrite(str(out_path), processed)
            
            return out_path, None
            
        except Exception as e:
            return None, str(e)
    
    with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
        futures = {
            executor.submit(process_single, Path(p)): p
            for p in images
        }
        
        for future in as_completed(futures):
            path = futures[future]
            out_path, error = future.result()
            
            if error == "skipped":
                result.skipped += 1
            elif error:
                result.failed += 1
                result.errors[Path(path)] = error
            else:
                result.successful += 1
                if out_path:
                    result.output_paths.append(out_path)
    
    return result


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
