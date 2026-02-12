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


DTYPE_MAP = {
    BitDepth.UINT8: np.uint8,
    BitDepth.UINT16: np.uint16,
    BitDepth.FLOAT16: np.float16,
    BitDepth.FLOAT32: np.float32,
    BitDepth.FLOAT64: np.float64,
}
