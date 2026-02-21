# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - batch_apply
===========

Apply a processor function to a list of images.

"""


# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING

from dreamstack.raster.io.batch.batch_config import BatchConfig
from dreamstack.raster.io.batch.batch_result import BatchResult

if TYPE_CHECKING:
    from numpy.typing import NDArray


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

            fmt = (
                output_format if output_format else img_path.suffix.lstrip(".")
            )
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

        except (OSError, Exception) as e:  # cv2.error, etc.
            return None, str(e)

    with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
        futures = {executor.submit(process_single, Path(p)): p for p in images}

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
