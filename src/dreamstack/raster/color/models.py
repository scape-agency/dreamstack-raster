# -*- coding: utf-8 -*-

"""Lightweight single-color models used by dreamstack-raster."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dreamstack.raster.color.convert import rgb_to_hsl, rgb_to_hsv


def _clip_unit(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def _normalized_rgb(
    red: float, green: float, blue: float
) -> tuple[float, float, float]:
    return red / 255.0, green / 255.0, blue / 255.0


@dataclass
class RGBColorModel:
    """Simple RGB color model using 8-bit channels and unit alpha."""

    r: int
    g: int
    b: int
    a: float = 1.0

    def __post_init__(self) -> None:
        self.r = int(np.clip(round(self.r), 0, 255))
        self.g = int(np.clip(round(self.g), 0, 255))
        self.b = int(np.clip(round(self.b), 0, 255))
        self.a = _clip_unit(self.a)

    @classmethod
    def from_hex(cls, hex_string: str) -> RGBColorModel:
        """Create an RGB model from a hex string."""
        value = hex_string.lstrip("#")
        if len(value) == 3:
            value = "".join(channel * 2 for channel in value)

        if len(value) == 6:
            return cls(
                int(value[0:2], 16),
                int(value[2:4], 16),
                int(value[4:6], 16),
            )
        if len(value) == 8:
            return cls(
                int(value[0:2], 16),
                int(value[2:4], 16),
                int(value[4:6], 16),
                int(value[6:8], 16) / 255.0,
            )
        raise ValueError(f"Invalid hex color: {hex_string}")

    def to_normalized(self) -> tuple[float, float, float]:
        """Return normalized RGB components in the range [0, 1]."""
        return _normalized_rgb(self.r, self.g, self.b)

    def to_array(
        self, normalized: bool = True, include_alpha: bool = False
    ) -> np.ndarray:
        """Return the model as a numpy array."""
        if normalized:
            red, green, blue = self.to_normalized()
            if include_alpha:
                return np.array([red, green, blue, self.a], dtype=np.float64)
            return np.array([red, green, blue], dtype=np.float64)

        if include_alpha:
            return np.array(
                [self.r, self.g, self.b, int(round(self.a * 255))],
                dtype=np.uint8,
            )
        return np.array([self.r, self.g, self.b], dtype=np.uint8)

    def to_hsv(self) -> tuple[float, float, float]:
        """Return HSV with hue in degrees and saturation/value in [0, 1]."""
        hsv = rgb_to_hsv(np.array(self.to_normalized(), dtype=np.float64))
        return float(hsv[0]), float(hsv[1]), float(hsv[2])

    def to_hsl(self) -> tuple[float, float, float]:
        """Return HSL with hue in degrees and saturation/lightness in [0, 1]."""
        hsl = rgb_to_hsl(np.array(self.to_normalized(), dtype=np.float64))
        return float(hsl[0]), float(hsl[1]), float(hsl[2])


@dataclass
class HSLColorModel:
    """HSL color model with hue in degrees and saturation/lightness in percent."""

    h: float
    s: float
    l: float
    a: float = 1.0

    def to_array(
        self, normalized: bool = True, include_alpha: bool = False
    ) -> np.ndarray:
        """Return the model as a numpy array."""
        if normalized:
            values = [self.h / 360.0, self.s / 100.0, self.l / 100.0]
            if include_alpha:
                values.append(self.a)
            return np.array(values, dtype=np.float64)

        values = [self.h, self.s, self.l]
        if include_alpha:
            values.append(self.a)
        return np.array(values, dtype=np.float64)


@dataclass
class HSVColorModel:
    """HSV color model with hue in degrees and saturation/value in percent."""

    h: float
    s: float
    v: float
    a: float = 1.0

    def to_array(
        self, normalized: bool = True, include_alpha: bool = False
    ) -> np.ndarray:
        """Return the model as a numpy array."""
        if normalized:
            values = [self.h / 360.0, self.s / 100.0, self.v / 100.0]
            if include_alpha:
                values.append(self.a)
            return np.array(values, dtype=np.float64)

        values = [self.h, self.s, self.v]
        if include_alpha:
            values.append(self.a)
        return np.array(values, dtype=np.float64)


@dataclass
class CMYKColorModel:
    """CMYK color model with channels stored in percent."""

    c: float
    m: float
    y: float
    k: float
    a: float = 1.0

    def to_array(
        self, normalized: bool = True, include_alpha: bool = False
    ) -> np.ndarray:
        """Return the model as a numpy array."""
        if normalized:
            values = [
                self.c / 100.0,
                self.m / 100.0,
                self.y / 100.0,
                self.k / 100.0,
            ]
        else:
            values = [self.c, self.m, self.y, self.k]

        if include_alpha:
            values.append(self.a)
        return np.array(values, dtype=np.float64)


__all__: list[str] = [
    "RGBColorModel",
    "HSLColorModel",
    "HSVColorModel",
    "CMYKColorModel",
]
