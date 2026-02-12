"""ColorSpace dataclass definition."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dreamstack.raster.color.spaces.gamma_type import GammaType
from dreamstack.raster.color.spaces.transfer_functions import (
    _hlg_eotf,
    _hlg_oetf,
    _pq_eotf,
    _pq_oetf,
)
from dreamstack.raster.color.spaces.xyz_matrix import (
    _compute_rgb_to_xyz_matrix,
)


@dataclass
class ColorSpace:
    """
    Defines a color space with primaries and transfer function.

    Attributes:
        name: Color space name
        primaries: CIE xy chromaticity coordinates for RGB primaries
        white_point: CIE xy chromaticity of white point
        gamma_type: Type of gamma/transfer function
        gamma: Gamma value (for power gamma)
        description: Human-readable description
    """

    name: str
    primaries: np.ndarray  # Shape (3, 2) - RGB xy coordinates
    white_point: np.ndarray  # Shape (2,) - xy coordinates
    gamma_type: GammaType = GammaType.SRGB
    gamma: float = 2.2
    description: str = ""

    def __post_init__(self):
        self.primaries = np.asarray(self.primaries)
        self.white_point = np.asarray(self.white_point)

    @property
    def rgb_to_xyz_matrix(self) -> np.ndarray:
        """Get RGB to XYZ conversion matrix."""
        return _compute_rgb_to_xyz_matrix(self.primaries, self.white_point)

    @property
    def xyz_to_rgb_matrix(self) -> np.ndarray:
        """Get XYZ to RGB conversion matrix."""
        return np.linalg.inv(self.rgb_to_xyz_matrix)

    def linearize(self, encoded: np.ndarray) -> np.ndarray:
        """
        Apply inverse transfer function (linearize).

        Args:
            encoded: Gamma-encoded values

        Returns:
            Linear values
        """
        if self.gamma_type == GammaType.LINEAR:
            return encoded

        elif self.gamma_type == GammaType.SRGB:
            return np.where(
                encoded <= 0.04045,
                encoded / 12.92,
                np.power((encoded + 0.055) / 1.055, 2.4),
            )

        elif self.gamma_type == GammaType.POWER:
            return np.power(np.maximum(encoded, 0), self.gamma)

        elif self.gamma_type == GammaType.PQ:
            return _pq_eotf(encoded)

        elif self.gamma_type == GammaType.HLG:
            return _hlg_eotf(encoded)

        elif self.gamma_type == GammaType.LOG:
            # Generic log curve
            return (np.power(10, encoded) - 1) / 9

        return encoded

    def encode(self, linear: np.ndarray) -> np.ndarray:
        """
        Apply transfer function (gamma encode).

        Args:
            linear: Linear values

        Returns:
            Gamma-encoded values
        """
        if self.gamma_type == GammaType.LINEAR:
            return linear

        elif self.gamma_type == GammaType.SRGB:
            return np.where(
                linear <= 0.0031308,
                12.92 * linear,
                1.055 * np.power(np.maximum(linear, 0), 1 / 2.4) - 0.055,
            )

        elif self.gamma_type == GammaType.POWER:
            return np.power(np.maximum(linear, 0), 1 / self.gamma)

        elif self.gamma_type == GammaType.PQ:
            return _pq_oetf(linear)

        elif self.gamma_type == GammaType.HLG:
            return _hlg_oetf(linear)

        elif self.gamma_type == GammaType.LOG:
            return np.log10(linear * 9 + 1)

        return linear
