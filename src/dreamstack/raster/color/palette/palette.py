# -*- coding: utf-8 -*-

"""Palette class definition."""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from dreamstack.raster.color.palette.color import Color


@dataclass
class Palette:
    """
    Color palette.

    Attributes:
        colors: List of colors in the palette
        name: Palette name
    """

    colors: List[Color] = field(default_factory=list)
    name: str = ""

    def __len__(self) -> int:
        return len(self.colors)

    def __getitem__(self, index: int) -> Color:
        return self.colors[index]

    def __iter__(self):
        return iter(self.colors)

    def add(self, color: Color) -> None:
        """Add a color to the palette."""
        self.colors.append(color)

    def remove(self, index: int) -> Color:
        """Remove and return a color."""
        return self.colors.pop(index)

    def clear(self) -> None:
        """Clear all colors."""
        self.colors.clear()

    def sort_by_hue(self) -> None:
        """Sort colors by hue."""
        self.colors.sort(key=lambda c: c.to_hsv()[0])

    def sort_by_luminance(self) -> None:
        """Sort colors by luminance."""
        self.colors.sort(key=lambda c: c.luminance())

    @classmethod
    def from_hex_list(cls, hex_colors: List[str], name: str = "") -> Palette:
        """Create palette from hex color strings."""
        colors = [Color.from_hex(h) for h in hex_colors]
        return cls(colors=colors, name=name)

    def to_hex_list(self) -> List[str]:
        """Get palette as list of hex strings."""
        return [c.to_hex() for c in self.colors]

    @classmethod
    def load_aco(cls, path: str) -> Palette:
        """Load Adobe Color (ACO) palette."""
        path = Path(path)
        data = path.read_bytes()

        colors = []
        offset = 0

        # Version
        version = struct.unpack(">H", data[offset : offset + 2])[0]
        offset += 2

        # Color count
        count = struct.unpack(">H", data[offset : offset + 2])[0]
        offset += 2

        for _ in range(count):
            color_space = struct.unpack(">H", data[offset : offset + 2])[0]
            offset += 2

            if color_space == 0:  # RGB
                r = struct.unpack(">H", data[offset : offset + 2])[0] / 256
                g = struct.unpack(">H", data[offset + 2 : offset + 4])[0] / 256
                b = struct.unpack(">H", data[offset + 4 : offset + 6])[0] / 256
                colors.append(Color(r, g, b))

            offset += 8

            # Skip name in v2
            if version == 2:
                name_length = struct.unpack(">I", data[offset : offset + 4])[0]
                offset += 4 + name_length * 2

        return cls(colors=colors, name=path.stem)

    @classmethod
    def load_gpl(cls, path: str) -> Palette:
        """Load GIMP palette (GPL)."""
        path = Path(path)
        lines = path.read_text().splitlines()

        colors = []
        name = path.stem

        for line in lines:
            line = line.strip()

            if line.startswith("Name:"):
                name = line.split(":", 1)[1].strip()
                continue

            if line.startswith("#") or not line or line.startswith("GIMP"):
                continue

            if line.startswith("Columns:"):
                continue

            # Parse color line
            parts = line.split()
            if len(parts) >= 3:
                try:
                    r = int(parts[0])
                    g = int(parts[1])
                    b = int(parts[2])
                    colors.append(Color(r, g, b))
                except ValueError:
                    continue

        return cls(colors=colors, name=name)

    def save_gpl(self, path: str) -> None:
        """Save as GIMP palette."""
        path = Path(path)

        lines = [
            "GIMP Palette",
            f"Name: {self.name or path.stem}",
            "Columns: 16",
            "#",
        ]

        for color in self.colors:
            r, g, b = color.to_rgb()
            lines.append(f"{r:3d} {g:3d} {b:3d}")

        path.write_text("\n".join(lines))
