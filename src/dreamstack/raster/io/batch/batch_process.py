# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - batch_process
=============

Process multiple images with a given function.

"""


# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

import logging
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING

from dreamstack.raster.io.batch.batch_config import BatchConfig
from dreamstack.raster.io.batch.batch_result import BatchResult
from dreamstack.raster.io.batch.find_images import find_images

# =============================================================================
# Type Checking Imports
# =============================================================================

if TYPE_CHECKING:
    # pylint: disable=import-outside-toplevel
    from numpy.typing import NDArray


logger = logging.getLogger(__name__)


def batch_process(
    input_dir: str | Path,
    output_dir: str | Path,
    processor: Callable[[NDArray], NDArray],
    *,
    config: BatchConfig | None = None,
    output_format: str = "png",
    output_suffix: str = "",
) -> BatchResult:
    """
    Process multiple images with a given function.

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
    import cv2  # pylint: disable=import-outside-toplevel

    if config is None:
        config = BatchConfig()

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find images
    images = find_images(
        input_dir, extensions=config.extensions, recursive=config.recursive
    )

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

        except OSError as e:
            return None, str(e)
        except Exception as e:  # pylint: disable=broad-exception-caught
            # Catch cv2.error and other processing errors gracefully
            return None, str(e)

    # Process in parallel
    with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
        futures = {
            executor.submit(process_single, path): path for path in images
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
