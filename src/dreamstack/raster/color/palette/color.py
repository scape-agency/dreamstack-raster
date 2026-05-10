# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""Single-color helpers for palette and swatch operations."""

# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dreamstack.raster.color.convert import (
    hsl_to_rgb,
    hsv_to_rgb,
    rgb_to_hsl,
    rgb_to_hsv,
)


@dataclass
class Color:
    """
    Represents a color value.

    Attributes:
        r: Red component (0-255 or 0-1)
        g: Green component
        b: Blue component
        a: Alpha component
        normalized: Whether values are in 0-1 range
    """

    r: float
    g: float
    b: float
    a: float = 1.0
    normalized: bool = False

    def __post_init__(self):
        self.r = float(self.r)
        self.g = float(self.g)
        self.b = float(self.b)
        self.a = float(self.a)

    @classmethod
    def from_hex(cls, hex_string: str) -> Color:
        """Create color from hex string."""
        hex_string = hex_string.lstrip("#")

        if len(hex_string) == 3:
            hex_string = "".join(c * 2 for c in hex_string)

        if len(hex_string) == 6:
            r = int(hex_string[0:2], 16)
            g = int(hex_string[2:4], 16)
            b = int(hex_string[4:6], 16)
            return cls(r, g, b)
        elif len(hex_string) == 8:
            r = int(hex_string[0:2], 16)
            g = int(hex_string[2:4], 16)
            b = int(hex_string[4:6], 16)
            a = int(hex_string[6:8], 16)
            return cls(r, g, b, a)
        else:
            raise ValueError(f"Invalid hex color: {hex_string}")

    @classmethod
    def from_hsv(cls, h: float, s: float, v: float, a: float = 1.0) -> Color:
        """Create color from HSV values."""
        rgb = hsv_to_rgb(np.array([h, s, v], dtype=np.float64))
        return cls(
            int(round(rgb[0] * 255)),
            int(round(rgb[1] * 255)),
            int(round(rgb[2] * 255)),
            a * 255,
        )

    @classmethod
    def from_hsl(cls, h: float, s: float, l: float, a: float = 1.0) -> Color:
        """Create color from HSL values."""
        rgb = hsl_to_rgb(np.array([h, s, l], dtype=np.float64))
        return cls(
            int(round(rgb[0] * 255)),
            int(round(rgb[1] * 255)),
            int(round(rgb[2] * 255)),
            a * 255,
        )

    @classmethod
    def from_array(cls, array: np.ndarray, normalized: bool = False) -> Color:
        """Create color from numpy array."""
        if len(array) == 3:
            return cls(array[0], array[1], array[2], normalized=normalized)
        else:
            return cls(
                array[0], array[1], array[2], array[3], normalized=normalized
            )

    def to_rgb(self) -> tuple[int, int, int]:
        """Get as RGB tuple (0-255)."""
        if self.normalized:
            return (int(self.r * 255), int(self.g * 255), int(self.b * 255))
        return (int(self.r), int(self.g), int(self.b))

    def to_rgba(self) -> tuple[int, int, int, int]:
        """Get as RGBA tuple (0-255)."""
        if self.normalized:
            return (
                int(self.r * 255),
                int(self.g * 255),
                int(self.b * 255),
                int(self.a * 255),
            )
        return (int(self.r), int(self.g), int(self.b), int(self.a))

    def to_normalized(self) -> tuple[float, float, float, float]:
        """Get as normalized RGBA tuple (0-1)."""
        if self.normalized:
            return (self.r, self.g, self.b, self.a)
        return (self.r / 255, self.g / 255, self.b / 255, self.a / 255)

    def to_hex(self, include_alpha: bool = False) -> str:
        """Get as hex string."""
        r, g, b = self.to_rgb()
        if include_alpha:
            a = int(self.a * 255) if self.normalized else int(self.a)
            return f"#{r:02x}{g:02x}{b:02x}{a:02x}"
        return f"#{r:02x}{g:02x}{b:02x}"

    def to_array(self, include_alpha: bool = True) -> np.ndarray:
        """Get as numpy array."""
        if include_alpha:
            return np.array([self.r, self.g, self.b, self.a])
        return np.array([self.r, self.g, self.b])

    def to_hsv(self) -> tuple[float, float, float]:
        """Get as HSV values (h: 0-360, s: 0-1, v: 0-1)."""
        hsv = rgb_to_hsv(np.array(self.to_normalized()[:3], dtype=np.float64))
        return (float(hsv[0]), float(hsv[1]), float(hsv[2]))

    def to_hsl(self) -> tuple[float, float, float]:
        """Get as HSL values (h: 0-360, s: 0-1, l: 0-1)."""
        hsl = rgb_to_hsl(np.array(self.to_normalized()[:3], dtype=np.float64))
        return (float(hsl[0]), float(hsl[1]), float(hsl[2]))

    def luminance(self) -> float:
        """Calculate relative luminance in encoded RGB."""
        red, green, blue, _alpha = self.to_normalized()
        return 0.2126 * red + 0.7152 * green + 0.0722 * blue

    def blend(self, other: Color, factor: float = 0.5) -> Color:
        """
        Blend with another color.

        Args:
            other: Color to blend with
            factor: Blend factor (0-1), where 0 is self and 1 is other

        Returns:
            Blended color
        """
        start = np.array(self.to_normalized(), dtype=np.float64)
        end = np.array(other.to_normalized(), dtype=np.float64)
        mixed = start + (end - start) * factor
        return self.from_array(mixed, normalized=True)

    def lighten(self, amount: float = 0.1) -> Color:
        """
        Lighten color in HSL space.

        Args:
            amount: Amount to lighten (0-1 range, will be scaled to percent)

        Returns:
            Lightened color
        """
        hue, saturation, lightness = self.to_hsl()
        return Color.from_hsl(
            hue,
            saturation,
            np.clip(lightness + amount, 0.0, 1.0),
            self.to_normalized()[3],
        )

    def darken(self, amount: float = 0.1) -> Color:
        """
        Darken color in HSL space.

        Args:
            amount: Amount to darken (0-1 range, will be scaled to percent)

        Returns:
            Darkened color
        """
        hue, saturation, lightness = self.to_hsl()
        return Color.from_hsl(
            hue,
            saturation,
            np.clip(lightness - amount, 0.0, 1.0),
            self.to_normalized()[3],
        )

    def saturate(self, amount: float = 0.1) -> Color:
        """
        Increase saturation in HSL space.

        Args:
            amount: Amount to saturate (0-1 range, will be scaled to percent)

        Returns:
            More saturated color
        """
        hue, saturation, lightness = self.to_hsl()
        return Color.from_hsl(
            hue,
            np.clip(saturation + amount, 0.0, 1.0),
            lightness,
            self.to_normalized()[3],
        )

    def desaturate(self, amount: float = 0.1) -> Color:
        """
        Decrease saturation in HSL space.

        Args:
            amount: Amount to desaturate (0-1 range, will be scaled to percent)

        Returns:
            Less saturated color
        """
        hue, saturation, lightness = self.to_hsl()
        return Color.from_hsl(
            hue,
            np.clip(saturation - amount, 0.0, 1.0),
            lightness,
            self.to_normalized()[3],
        )

    def complement(self) -> Color:
        """
        Get the complementary color.

        Returns:
            Complementary color
        """
        return self.adjust_hue(180.0)

    def grayscale(self) -> Color:
        """
        Convert to grayscale.

        Returns:
            Grayscale color
        """
        luminance = self.luminance()
        alpha = self.to_normalized()[3]
        gray = int(round(luminance * 255))
        return Color(gray, gray, gray, alpha * 255)

    def invert(self) -> Color:
        """
        Invert color channels.

        Returns:
            Inverted color
        """
        red, green, blue, alpha = self.to_normalized()
        return Color.from_array(
            np.array([1.0 - red, 1.0 - green, 1.0 - blue, alpha]),
            normalized=True,
        )

    def adjust_hue(self, degrees: float) -> Color:
        """
        Adjust hue in HSL space.

        Args:
            degrees: Degrees to rotate hue (-360 to 360)

        Returns:
            Color with adjusted hue
        """
        hue, saturation, lightness = self.to_hsl()
        alpha = self.to_normalized()[3]
        return Color.from_hsl(
            (hue + degrees) % 360.0, saturation, lightness, alpha
        )
