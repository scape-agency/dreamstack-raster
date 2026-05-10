# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Bit Depth
=============================

Supported bit depth enumeration and dtype mapping.

"""

from enum import Enum, auto

import numpy as np


class BitDepth(Enum):
    """Supported bit depths."""

    UINT8 = auto()  # 8-bit unsigned integer (0-255)
    UINT16 = auto()  # 16-bit unsigned integer (0-65535)
    FLOAT16 = auto()  # 16-bit float (half precision)
    FLOAT32 = auto()  # 32-bit float (single precision)
    FLOAT64 = auto()  # 64-bit float (double precision)

    @property
    def max_value(self) -> float:
        """Maximum representable value of the canonical [0, max] range.

        Integer depths return their dtype maximum (255 / 65535). Float
        depths use the conventional ``1.0`` upper bound for normalized
        pixel data; values may exceed this for HDR data.
        """
        return _MAX_VALUE_MAP[self]


_MAX_VALUE_MAP: dict[BitDepth, float] = {
    BitDepth.UINT8: 255.0,
    BitDepth.UINT16: 65535.0,
    BitDepth.FLOAT16: 1.0,
    BitDepth.FLOAT32: 1.0,
    BitDepth.FLOAT64: 1.0,
}


DTYPE_MAP = {
    BitDepth.UINT8: np.uint8,
    BitDepth.UINT16: np.uint16,
    BitDepth.FLOAT16: np.float16,
    BitDepth.FLOAT32: np.float32,
    BitDepth.FLOAT64: np.float64,
}
