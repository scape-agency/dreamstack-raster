"""Internal color conversion utilities for basic adjustments."""

from __future__ import annotations

import numpy as np


def _rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    """Convert RGB (0-1) to HSV (0-1)."""
    hsv = np.zeros_like(rgb)

    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]

    max_c = np.maximum(np.maximum(r, g), b)
    min_c = np.minimum(np.minimum(r, g), b)
    diff = max_c - min_c

    # Value
    hsv[:, :, 2] = max_c

    # Saturation
    hsv[:, :, 1] = np.where(max_c > 0, diff / max_c, 0)

    # Hue
    with np.errstate(divide="ignore", invalid="ignore"):
        hue = np.zeros_like(max_c)

        mask = diff > 0

        # Red is max
        mask_r = mask & (max_c == r)
        hue[mask_r] = ((g[mask_r] - b[mask_r]) / diff[mask_r]) % 6

        # Green is max
        mask_g = mask & (max_c == g)
        hue[mask_g] = (b[mask_g] - r[mask_g]) / diff[mask_g] + 2

        # Blue is max
        mask_b = mask & (max_c == b)
        hue[mask_b] = (r[mask_b] - g[mask_b]) / diff[mask_b] + 4

        hsv[:, :, 0] = hue / 6

    return hsv


def _hsv_to_rgb(hsv: np.ndarray) -> np.ndarray:
    """Convert HSV (0-1) to RGB (0-1)."""
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    h = h * 6
    i = np.floor(h).astype(int)
    f = h - i

    p = v * (1 - s)
    q = v * (1 - s * f)
    t = v * (1 - s * (1 - f))

    rgb = np.zeros_like(hsv)

    mask = i % 6 == 0
    rgb[mask] = np.stack([v[mask], t[mask], p[mask]], axis=-1)

    mask = i % 6 == 1
    rgb[mask] = np.stack([q[mask], v[mask], p[mask]], axis=-1)

    mask = i % 6 == 2
    rgb[mask] = np.stack([p[mask], v[mask], t[mask]], axis=-1)

    mask = i % 6 == 3
    rgb[mask] = np.stack([p[mask], q[mask], v[mask]], axis=-1)

    mask = i % 6 == 4
    rgb[mask] = np.stack([t[mask], p[mask], v[mask]], axis=-1)

    mask = i % 6 == 5
    rgb[mask] = np.stack([v[mask], p[mask], q[mask]], axis=-1)

    return rgb
