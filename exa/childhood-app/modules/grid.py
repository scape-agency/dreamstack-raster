"""
Grid Segmentation
=================

Divide images into randomized grid segments for the art installation.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

if TYPE_CHECKING:
    from numpy.typing import NDArray

from modules.config import SegmentConfig


@dataclass
class GridSegment:
    """A single grid segment from an image.

    Attributes
    ----------
    image : Image.Image
        The segment image (PIL).
    row : int
        Row index in grid.
    col : int
        Column index in grid.
    x : int
        X position in source image.
    y : int
        Y position in source image.
    width : int
        Segment width.
    height : int
        Segment height.
    offset_x : int
        Random X offset applied.
    offset_y : int
        Random Y offset applied.
    has_empty_pixels : bool
        True if segment contains transparent/empty pixels.
    """

    image: Image.Image
    row: int
    col: int
    x: int
    y: int
    width: int
    height: int
    offset_x: int = 0
    offset_y: int = 0
    has_empty_pixels: bool = False
    inbetween_type: str | None = (
        None  # None, 'h' (horizontal), or 'v' (vertical)
    )

    @property
    def filename(self) -> str:
        """Generate filename for this segment."""
        suffix = f"_{self.inbetween_type}" if self.inbetween_type else ""
        return f"seg_{self.row}_{self.col}{suffix}.png"

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        result = {
            "row": self.row,
            "col": self.col,
            "position": [self.x, self.y],
            "size": [self.width, self.height],
            "offset": [self.offset_x, self.offset_y],
            "has_empty_pixels": self.has_empty_pixels,
        }
        if self.inbetween_type:
            result["inbetween_type"] = self.inbetween_type
        return result


def segment_image(
    image: Image.Image | NDArray | Path | str,
    config: SegmentConfig | None = None,
    seed: int | None = None,
) -> list[GridSegment]:
    """Segment an image into a randomized grid.

    Parameters
    ----------
    image : Image.Image | NDArray | Path | str
        Input image (PIL, numpy array, or path).
    config : SegmentConfig | None
        Segmentation configuration. Uses defaults if None.
    seed : int | None
        Random seed for reproducibility. None for random.

    Returns
    -------
    list[GridSegment]
        List of grid segments.

    Example
    -------
    >>> from PIL import Image
    >>> img = Image.open("cutout.png")
    >>> segments = segment_image(img)
    >>> for seg in segments:
    ...     seg.image.save(f"segments/{seg.filename}")
    """
    config = config or SegmentConfig()

    # Handle different input types
    if isinstance(image, (str, Path)):
        image = Image.open(image)
    elif isinstance(image, np.ndarray):
        # Convert BGR to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = Image.fromarray(image[:, :, ::-1])
        elif len(image.shape) == 3 and image.shape[2] == 4:
            # BGRA to RGBA
            image = Image.fromarray(
                np.concatenate([image[:, :, 2::-1], image[:, :, 3:4]], axis=2)
            )
        else:
            image = Image.fromarray(image)

    # Ensure RGBA for transparency support
    if image.mode != "RGBA":
        image = image.convert("RGBA")

    # Set random seed
    if seed is not None:
        random.seed(seed)

    img_width, img_height = image.size
    seg_width, seg_height = config.segment_size
    alpha_threshold = config.empty_alpha_threshold

    # Calculate grid dimensions (step = segment size, no overlap)
    step_x = seg_width
    step_y = seg_height

    num_cols = max(1, (img_width + step_x - 1) // step_x)
    num_rows = max(1, (img_height + step_y - 1) // step_y)

    # Collect all segment positions: (row, col, base_x, base_y, inbetween_type)
    positions: list[tuple[int, int, int, int, str | None]] = []

    # 1. Regular grid positions
    for row in range(num_rows):
        for col in range(num_cols):
            base_x = col * step_x
            base_y = row * step_y
            positions.append((row, col, base_x, base_y, None))

    # 2. In-between segments (if enabled)
    if config.generate_inbetweens:
        half_x = seg_width // 2
        half_y = seg_height // 2

        # Horizontal in-betweens: between adjacent columns (same row)
        for row in range(num_rows):
            for col in range(num_cols - 1):  # One less than total columns
                base_x = col * step_x + half_x
                base_y = row * step_y
                # Only add if there's enough space for a full segment
                if base_x + seg_width <= img_width:
                    positions.append((row, col, base_x, base_y, "h"))

        # Vertical in-betweens: between adjacent rows (same column)
        for row in range(num_rows - 1):  # One less than total rows
            for col in range(num_cols):
                base_x = col * step_x
                base_y = row * step_y + half_y
                # Only add if there's enough space for a full segment
                if base_y + seg_height <= img_height:
                    positions.append((row, col, base_x, base_y, "v"))

    segments: list[GridSegment] = []

    # 3. Create segments from all positions (with random offsets)
    for row, col, base_x, base_y, inbetween_type in positions:
        # Apply random offset if enabled
        if config.randomize_offset:
            offset_x = random.randint(-config.max_offset, config.max_offset)
            offset_y = random.randint(-config.max_offset, config.max_offset)
        else:
            offset_x = 0
            offset_y = 0

        # Calculate final position with bounds checking
        x = max(0, min(base_x + offset_x, img_width - seg_width))
        y = max(0, min(base_y + offset_y, img_height - seg_height))

        # Adjust segment size if at edge
        actual_width = min(seg_width, img_width - x)
        actual_height = min(seg_height, img_height - y)

        # Skip tiny segments
        if actual_width < seg_width // 2 or actual_height < seg_height // 2:
            continue

        # Crop segment
        segment_img = image.crop((x, y, x + actual_width, y + actual_height))

        # Check for empty/transparent pixels (only padded pixels, not semi-transparent)
        has_empty = False
        if segment_img.mode == "RGBA":
            alpha = np.array(segment_img.split()[3])
            has_empty = bool(np.any(alpha <= alpha_threshold))

        # Pad to full size if needed
        if actual_width < seg_width or actual_height < seg_height:
            padded = Image.new("RGBA", (seg_width, seg_height), (0, 0, 0, 0))
            padded.paste(segment_img, (0, 0))
            segment_img = padded
            has_empty = True  # Padded segments always have empty pixels

        segments.append(
            GridSegment(
                image=segment_img,
                row=row,
                col=col,
                x=x,
                y=y,
                width=actual_width,
                height=actual_height,
                offset_x=offset_x,
                offset_y=offset_y,
                has_empty_pixels=has_empty,
                inbetween_type=inbetween_type,
            )
        )

    return segments


def save_segments(
    segments: list[GridSegment],
    output_dir: Path | str,
    prefix: str = "",
) -> list[Path]:
    """Save segments to disk.

    Parameters
    ----------
    segments : list[GridSegment]
        Segments to save.
    output_dir : Path | str
        Output directory.
    prefix : str
        Filename prefix.

    Returns
    -------
    list[Path]
        Paths to saved files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_paths = []
    for seg in segments:
        filename = f"{prefix}{seg.filename}" if prefix else seg.filename
        path = output_dir / filename
        seg.image.save(path)
        saved_paths.append(path)

    return saved_paths
