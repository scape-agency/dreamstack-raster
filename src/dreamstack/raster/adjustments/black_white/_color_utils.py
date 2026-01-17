# -*- coding: utf-8 -*-

"""Internal hue to RGB conversion utility."""

from __future__ import annotations

import numpy as np


def _hue_to_rgb(hue: float) -> np.ndarray:
    """Convert hue (0-360) to RGB (0-1)."""
    h = (hue % 360) / 60
    x = 1 - abs(h % 2 - 1)

    if h < 1:
        return np.array([1, x, 0])
    elif h < 2:
        return np.array([x, 1, 0])
    elif h < 3:
        return np.array([0, 1, x])
    elif h < 4:
        return np.array([0, x, 1])
    elif h < 5:
        return np.array([x, 0, 1])
    else:
        return np.array([1, 0, x])
