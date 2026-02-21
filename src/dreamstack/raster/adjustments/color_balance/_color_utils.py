# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Internal color conversion utilities for color balance adjustments.
"""


# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

import numpy as np


def _rgb_to_hsl(rgb: np.ndarray) -> np.ndarray:
    """Convert RGB (0-1) to HSL (0-1)."""
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]

    max_c = np.maximum(np.maximum(r, g), b)
    min_c = np.minimum(np.minimum(r, g), b)
    diff = max_c - min_c

    hsl = np.zeros_like(rgb)

    # Lightness
    hsl[:, :, 2] = (max_c + min_c) / 2

    # Saturation
    mask = diff > 0
    hsl[:, :, 1] = np.where(
        mask,
        np.where(
            hsl[:, :, 2] <= 0.5,
            diff / (max_c + min_c),
            diff / (2 - max_c - min_c),
        ),
        0,
    )

    # Hue
    with np.errstate(divide="ignore", invalid="ignore"):
        hue = np.zeros_like(max_c)

        mask_r = mask & (max_c == r)
        hue[mask_r] = ((g[mask_r] - b[mask_r]) / diff[mask_r]) % 6

        mask_g = mask & (max_c == g)
        hue[mask_g] = (b[mask_g] - r[mask_g]) / diff[mask_g] + 2

        mask_b = mask & (max_c == b)
        hue[mask_b] = (r[mask_b] - g[mask_b]) / diff[mask_b] + 4

        hsl[:, :, 0] = hue / 6

    return hsl


def _hsl_to_rgb(hsl: np.ndarray) -> np.ndarray:
    """Convert HSL (0-1) to RGB (0-1)."""
    h, s, l = hsl[:, :, 0], hsl[:, :, 1], hsl[:, :, 2]

    c = (1 - np.abs(2 * l - 1)) * s
    x = c * (1 - np.abs((h * 6) % 2 - 1))
    m = l - c / 2

    h_sector = (h * 6).astype(int) % 6

    rgb = np.zeros_like(hsl)

    for sector in range(6):
        mask = h_sector == sector
        if sector == 0:
            rgb[mask] = np.stack(
                [c[mask], x[mask], np.zeros_like(c[mask])], axis=-1
            )
        elif sector == 1:
            rgb[mask] = np.stack(
                [x[mask], c[mask], np.zeros_like(c[mask])], axis=-1
            )
        elif sector == 2:
            rgb[mask] = np.stack(
                [np.zeros_like(c[mask]), c[mask], x[mask]], axis=-1
            )
        elif sector == 3:
            rgb[mask] = np.stack(
                [np.zeros_like(c[mask]), x[mask], c[mask]], axis=-1
            )
        elif sector == 4:
            rgb[mask] = np.stack(
                [x[mask], np.zeros_like(c[mask]), c[mask]], axis=-1
            )
        else:
            rgb[mask] = np.stack(
                [c[mask], np.zeros_like(c[mask]), x[mask]], axis=-1
            )

    rgb = rgb + m[:, :, np.newaxis]

    return rgb


def _get_color_mask(rgb: np.ndarray, color: str):
    """Get mask for specific color range."""

    _r, _g, _b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]

    # Convert to HSL for color detection
    hsl = _rgb_to_hsl(rgb)
    h, s, l = hsl[:, :, 0], hsl[:, :, 1], hsl[:, :, 2]

    color_ranges = {
        "reds": (0, 30, 330, 360),  # 0-30 and 330-360
        "yellows": (30, 90, -1, -1),
        "greens": (90, 150, -1, -1),
        "cyans": (150, 210, -1, -1),
        "blues": (210, 270, -1, -1),
        "magentas": (270, 330, -1, -1),
    }

    mask: np.ndarray | None = None

    if color in color_ranges:
        h_deg = h * 360
        h1_start, h1_end, h2_start, h2_end = color_ranges[color]

        if h2_start >= 0:
            # Two ranges (for reds)
            mask = ((h_deg >= h1_start) & (h_deg < h1_end)) | (
                (h_deg >= h2_start) & (h_deg <= h2_end)
            )
        else:
            mask = (h_deg >= h1_start) & (h_deg < h1_end)

        # Weight by saturation
        mask = mask.astype(np.float32) * s

    elif color == "whites":
        mask = l * (1 - s)
        mask = np.where(l > 0.5, mask, 0)

    elif color == "neutrals":
        mask = (1 - s) * (1 - np.abs(l - 0.5) * 2)

    elif color == "blacks":
        mask = (1 - l) * (1 - s)
        mask = np.where(l < 0.5, mask, 0)
    else:
        return None

    return mask.astype(np.float32)
