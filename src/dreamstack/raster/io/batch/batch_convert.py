"""
batch_convert
=============

Batch convert images to a different format.

"""

from __future__ import annotations

from pathlib import Path

from dreamstack.raster.io.batch.batch_config import BatchConfig
from dreamstack.raster.io.batch.batch_process import batch_process
from dreamstack.raster.io.batch.batch_result import BatchResult


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
