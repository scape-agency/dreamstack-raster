"""
Grid Segment
============

Dataclass for a single grid segment from an image.
"""

from __future__ import annotations

from dataclasses import dataclass

from PIL import Image


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
    inbetween_type : str | None
        None, 'h' (horizontal), or 'v' (vertical).
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
    inbetween_type: str | None = None

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
