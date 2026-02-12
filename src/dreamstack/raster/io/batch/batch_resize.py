"""
batch_resize
============

Batch resize images to specified dimensions.

"""

from __future__ import annotations

from pathlib import Path

from dreamstack.raster.io.batch.batch_config import BatchConfig
from dreamstack.raster.io.batch.batch_process import batch_process
from dreamstack.raster.io.batch.batch_result import BatchResult


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
