"""
Placed Item
===========

Dataclass for a placed item on the canvas.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class PlacedItem:
    """A placed item on the canvas.

    Attributes
    ----------
    path : Path
        Path to the image file.
    x : int
        X position (from left).
    y : int
        Y position (from bottom in canvas coords, converted to top for PIL).
    width : int
        Image width.
    height : int
        Image height.
    layer : int
        Z-order layer (higher = on top).
    rotation : float
        Rotation angle in degrees (counter-clockwise).
    """

    path: Path
    x: int
    y: int
    width: int
    height: int
    layer: int = 0
    rotation: float = 0.0

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "path": str(self.path),
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "layer": self.layer,
            "rotation": self.rotation,
        }
