"""
Segment Image Service
=====================

Divide images into randomized grid segments.
Supports both uniform grids and organic fluid grids.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

if TYPE_CHECKING:
    from numpy.typing import NDArray

from models.model_segment_config import SegmentConfig
from models.model_grid_segment import GridSegment


def _generate_fluid_divisions(
    total_size: int,
    target_segment_size: int,
    size_variation: float,
    min_segments: int = 2,
    max_segments: int = 6,
) -> list[tuple[int, int]]:
    """Generate irregular divisions for one dimension.
    
    Returns list of (start, end) tuples representing segment boundaries.
    Each segment varies in size around the target, creating organic feel.
    """
    # Determine number of segments (varies based on image size vs target)
    ideal_count = max(min_segments, total_size // target_segment_size)
    # Add some randomness to count
    num_segments = random.randint(
        max(min_segments, ideal_count - 1),
        min(max_segments, ideal_count + 1)
    )
    
    # Generate random weights for each segment
    weights = [random.uniform(1 - size_variation, 1 + size_variation) for _ in range(num_segments)]
    total_weight = sum(weights)
    
    # Convert weights to actual sizes
    divisions = []
    current_pos = 0
    
    for i, weight in enumerate(weights):
        # Calculate this segment's size proportionally
        if i == len(weights) - 1:
            # Last segment takes remaining space
            segment_size = total_size - current_pos
        else:
            segment_size = int((weight / total_weight) * total_size)
            # Ensure minimum size
            segment_size = max(target_segment_size // 2, segment_size)
        
        end_pos = min(current_pos + segment_size, total_size)
        divisions.append((current_pos, end_pos))
        current_pos = end_pos
        
        if current_pos >= total_size:
            break
    
    return divisions


def _generate_fluid_grid_positions(
    img_width: int,
    img_height: int,
    target_width: int,
    target_height: int,
    size_variation: float,
    layer: int,
) -> list[tuple[int, int, int, int, int, int, str | None, int]]:
    """Generate fluid grid positions for one layer.
    
    Returns list of (row, col, x, y, width, height, inbetween_type, layer).
    """
    # Generate irregular column divisions
    col_divisions = _generate_fluid_divisions(
        img_width, target_width, size_variation, min_segments=2, max_segments=5
    )
    
    positions = []
    
    for col_idx, (col_start, col_end) in enumerate(col_divisions):
        col_width = col_end - col_start
        
        # Each column can have different number of rows (2-6)
        row_divisions = _generate_fluid_divisions(
            img_height, target_height, size_variation, min_segments=2, max_segments=6
        )
        
        for row_idx, (row_start, row_end) in enumerate(row_divisions):
            row_height = row_end - row_start
            
            # Skip if segment is too small
            if col_width < target_width // 3 or row_height < target_height // 3:
                continue
            
            positions.append((
                row_idx,      # row
                col_idx,      # col
                col_start,    # x
                row_start,    # y
                col_width,    # width
                row_height,   # height
                None,         # inbetween_type (not used in fluid mode)
                layer,        # layer index
            ))
    
    return positions


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

    # Choose grid generation strategy
    all_positions: list  # type: ignore[type-arg]  # Mixed tuple types
    if config.fluid_grid:
        # Generate fluid grid with multiple layers
        all_positions = []
        
        for layer in range(config.layer_count):
            # Use different seed offset for each layer
            if seed is not None:
                random.seed(seed + layer * 1000)
            
            layer_positions = _generate_fluid_grid_positions(
                img_width,
                img_height,
                seg_width,
                seg_height,
                config.size_variation,
                layer,
            )
            all_positions.extend(layer_positions)
        
        # Reset seed for rotation assignment
        if seed is not None:
            random.seed(seed)
    else:
        # Legacy uniform grid mode
        all_positions = _generate_uniform_grid_positions(
            img_width, img_height, seg_width, seg_height, config
        )

    segments: list[GridSegment] = []

    # Create segments from all positions
    for pos_data in all_positions:
        if config.fluid_grid:
            # Fluid grid: 8-tuple (row, col, x, y, width, height, inbetween_type, layer)
            row = int(pos_data[0])
            col = int(pos_data[1])
            x = int(pos_data[2])
            y = int(pos_data[3])
            actual_width = int(pos_data[4])
            actual_height = int(pos_data[5])
            inbetween_type = pos_data[6] if isinstance(pos_data[6], str) else None
            layer = int(pos_data[7])
        else:
            # Legacy uniform grid: 5-tuple (row, col, base_x, base_y, inbetween_type)
            row = int(pos_data[0])
            col = int(pos_data[1])
            base_x = int(pos_data[2])
            base_y = int(pos_data[3])
            inbetween_type = pos_data[4] if isinstance(pos_data[4], str) else None
            layer = 0
            # Apply random offset if enabled (legacy mode)
            if config.randomize_offset:
                offset_x = random.randint(-config.max_offset, config.max_offset)
                offset_y = random.randint(-config.max_offset, config.max_offset)
            else:
                offset_x = 0
                offset_y = 0
            x = max(0, min(base_x + offset_x, img_width - seg_width))
            y = max(0, min(base_y + offset_y, img_height - seg_height))
            actual_width = min(seg_width, img_width - x)
            actual_height = min(seg_height, img_height - y)

        # Skip tiny segments
        if actual_width < seg_width // 3 or actual_height < seg_height // 3:
            continue

        # Crop segment
        segment_img = image.crop((x, y, x + actual_width, y + actual_height))

        # Check for empty/transparent pixels
        has_empty = False
        if segment_img.mode == "RGBA":
            alpha = np.array(segment_img.split()[3])
            has_empty = bool(np.any(alpha <= alpha_threshold))

        # Assign random rotation within range
        rotation = random.uniform(-config.rotation_range, config.rotation_range)

        segments.append(
            GridSegment(
                image=segment_img,
                row=row,
                col=col,
                x=x,
                y=y,
                width=actual_width,
                height=actual_height,
                offset_x=0,  # Not used in fluid mode
                offset_y=0,
                has_empty_pixels=has_empty,
                inbetween_type=inbetween_type,
                layer=layer,
                rotation=rotation,
            )
        )

    return segments


def _generate_uniform_grid_positions(
    img_width: int,
    img_height: int,
    seg_width: int,
    seg_height: int,
    config: SegmentConfig,
) -> list[tuple[int, int, int, int, str | None]]:
    """Generate uniform grid positions (legacy mode).
    
    Returns list of (row, col, base_x, base_y, inbetween_type).
    """
    step_x = seg_width
    step_y = seg_height

    num_cols = max(1, (img_width + step_x - 1) // step_x)
    num_rows = max(1, (img_height + step_y - 1) // step_y)

    positions: list[tuple[int, int, int, int, str | None]] = []

    # Regular grid positions
    for row in range(num_rows):
        for col in range(num_cols):
            base_x = col * step_x
            base_y = row * step_y
            positions.append((row, col, base_x, base_y, None))

    # In-between segments (if enabled)
    if config.generate_inbetweens:
        half_x = seg_width // 2
        half_y = seg_height // 2

        # Horizontal in-betweens
        for row in range(num_rows):
            for col in range(num_cols - 1):
                base_x = col * step_x + half_x
                base_y = row * step_y
                if base_x + seg_width <= img_width:
                    positions.append((row, col, base_x, base_y, "h"))

        # Vertical in-betweens
        for row in range(num_rows - 1):
            for col in range(num_cols):
                base_x = col * step_x
                base_y = row * step_y + half_y
                if base_y + seg_height <= img_height:
                    positions.append((row, col, base_x, base_y, "v"))

    # Diagonal in-between segments (if enabled)
    if config.generate_diagonal_inbetweens:
        half_x = seg_width // 2
        half_y = seg_height // 2

        for row in range(num_rows - 1):
            for col in range(num_cols - 1):
                base_x = col * step_x + half_x
                base_y = row * step_y + half_y
                if base_x + seg_width <= img_width and base_y + seg_height <= img_height:
                    positions.append((row, col, base_x, base_y, "d"))

    return positions
