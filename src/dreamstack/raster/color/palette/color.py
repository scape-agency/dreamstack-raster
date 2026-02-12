"""
Color Class Definition
======================

This module provides a Color class that bridges between numpy arrays
(for raster image processing) and dreamstack.color models (for single-color
manipulation operations).

"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Import dreamstack.color for manipulation operations
from dreamstack.color import HSLColorModel, HSVColorModel
from dreamstack.color import RGBColorModel as DreamstackRGB
from dreamstack.color import adjust_hue as ds_adjust_hue
from dreamstack.color import complement as ds_complement
from dreamstack.color import darken as ds_darken
from dreamstack.color import desaturate as ds_desaturate
from dreamstack.color import grayscale as ds_grayscale
from dreamstack.color import hsl_to_rgb as ds_hsl_to_rgb
from dreamstack.color import hsv_to_rgb as ds_hsv_to_rgb
from dreamstack.color import invert as ds_invert
from dreamstack.color import lighten as ds_lighten
from dreamstack.color import mix as ds_mix
from dreamstack.color import rgb_to_hsl as ds_rgb_to_hsl
from dreamstack.color import rgb_to_hsv as ds_rgb_to_hsv
from dreamstack.color import saturate as ds_saturate


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
        if not self.normalized:
            # Assume 0-255 range
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
        """
        Create color from HSV values using dreamstack.color.

        Args:
            h: Hue (0-360)
            s: Saturation (0-1)
            v: Value (0-1)
            a: Alpha (0-1)

        Returns:
            Color instance
        """
        hsv = HSVColorModel(
            h, s * 100, v * 100, a
        )  # dreamstack.color uses 0-100 for S/V
        rgb = ds_hsv_to_rgb(hsv)
        return cls(rgb.r, rgb.g, rgb.b, rgb.a)

    @classmethod
    def from_hsl(cls, h: float, s: float, l: float, a: float = 1.0) -> Color:
        """
        Create color from HSL values using dreamstack.color.

        Args:
            h: Hue (0-360)
            s: Saturation (0-1)
            l: Lightness (0-1)
            a: Alpha (0-1)

        Returns:
            Color instance
        """
        hsl = HSLColorModel(
            h, s * 100, l * 100, a
        )  # dreamstack.color uses 0-100 for S/L
        rgb = ds_hsl_to_rgb(hsl)
        return cls(rgb.r, rgb.g, rgb.b, rgb.a)

    @classmethod
    def from_array(cls, array: np.ndarray, normalized: bool = False) -> Color:
        """Create color from numpy array."""
        if len(array) == 3:
            return cls(array[0], array[1], array[2], normalized=normalized)
        else:
            return cls(array[0], array[1], array[2], array[3], normalized=normalized)

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
        ds_rgb = self.to_dreamstack_rgb()
        hsv = ds_rgb_to_hsv(ds_rgb)
        return (hsv.h, hsv.s / 100, hsv.v / 100)

    def to_hsl(self) -> tuple[float, float, float]:
        """Get as HSL values (h: 0-360, s: 0-1, l: 0-1)."""
        ds_rgb = self.to_dreamstack_rgb()
        hsl = ds_rgb_to_hsl(ds_rgb)
        return (hsl.h, hsl.s / 100, hsl.l / 100)

    def to_dreamstack_rgb(self) -> DreamstackRGB:
        """Convert to dreamstack.color RGB model."""
        r, g, b = self.to_rgb()
        return DreamstackRGB(r, g, b, self.a if self.normalized else self.a / 255)

    @classmethod
    def from_dreamstack_rgb(cls, rgb: DreamstackRGB | HSLColorModel) -> Color:
        """Create Color from dreamstack.color RGB or HSL model."""
        if isinstance(rgb, HSLColorModel):
            # Convert HSL to RGB first
            rgb = ds_hsl_to_rgb(rgb)
        return cls(rgb.r, rgb.g, rgb.b, rgb.a)

    def luminance(self) -> float:
        """Calculate perceptual luminance using dreamstack.color."""
        from dreamstack.color import luminance as ds_luminance

        return ds_luminance(self.to_dreamstack_rgb())

    def blend(self, other: Color, factor: float = 0.5) -> Color:
        """
        Blend with another color using dreamstack.color.

        Args:
            other: Color to blend with
            factor: Blend factor (0-1), where 0 is self and 1 is other

        Returns:
            Blended color
        """
        result = ds_mix(self.to_dreamstack_rgb(), other.to_dreamstack_rgb(), factor)
        return Color.from_dreamstack_rgb(result)

    def lighten(self, amount: float = 0.1) -> Color:
        """
        Lighten color using dreamstack.color.

        Args:
            amount: Amount to lighten (0-1 range, will be scaled to percent)

        Returns:
            Lightened color
        """
        # dreamstack.color.lighten uses percentage (0-100)
        result = ds_lighten(self.to_dreamstack_rgb(), amount * 100)
        return Color.from_dreamstack_rgb(result)

    def darken(self, amount: float = 0.1) -> Color:
        """
        Darken color using dreamstack.color.

        Args:
            amount: Amount to darken (0-1 range, will be scaled to percent)

        Returns:
            Darkened color
        """
        result = ds_darken(self.to_dreamstack_rgb(), amount * 100)
        return Color.from_dreamstack_rgb(result)

    def saturate(self, amount: float = 0.1) -> Color:
        """
        Increase saturation using dreamstack.color.

        Args:
            amount: Amount to saturate (0-1 range, will be scaled to percent)

        Returns:
            More saturated color
        """
        result = ds_saturate(self.to_dreamstack_rgb(), amount * 100)
        return Color.from_dreamstack_rgb(result)

    def desaturate(self, amount: float = 0.1) -> Color:
        """
        Decrease saturation using dreamstack.color.

        Args:
            amount: Amount to desaturate (0-1 range, will be scaled to percent)

        Returns:
            Less saturated color
        """
        result = ds_desaturate(self.to_dreamstack_rgb(), amount * 100)
        return Color.from_dreamstack_rgb(result)

    def complement(self) -> Color:
        """
        Get complementary color using dreamstack.color.

        Returns:
            Complementary color
        """
        result = ds_complement(self.to_dreamstack_rgb())
        return Color.from_dreamstack_rgb(result)

    def grayscale(self) -> Color:
        """
        Convert to grayscale using dreamstack.color.

        Returns:
            Grayscale color
        """
        result = ds_grayscale(self.to_dreamstack_rgb())
        return Color.from_dreamstack_rgb(result)

    def invert(self) -> Color:
        """
        Invert color using dreamstack.color.

        Returns:
            Inverted color
        """
        result = ds_invert(self.to_dreamstack_rgb())
        return Color.from_dreamstack_rgb(result)

    def adjust_hue(self, degrees: float) -> Color:
        """
        Adjust hue using dreamstack.color.

        Args:
            degrees: Degrees to rotate hue (-360 to 360)

        Returns:
            Color with adjusted hue
        """
        result = ds_adjust_hue(self.to_dreamstack_rgb(), degrees)
        return Color.from_dreamstack_rgb(result)
