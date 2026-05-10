# -*- coding: utf-8 -*-
# pylint: disable=invalid-name

"""Polar conversions: Lab ↔ LCh and OKLab ↔ OKLCh.

LCh is the cylindrical (perceptual) form of a Cartesian Lab-style space:
``L`` (lightness) is unchanged, ``C = sqrt(a² + b²)`` is chroma, and
``h = atan2(b, a)`` is hue in degrees ``[0, 360)``.

These helpers are space-agnostic: feed them ``Lab`` to get ``LCh(ab)`` or
``OKLab`` to get ``OKLCh``. The third axis layout is preserved (alpha
passes through).
"""

from __future__ import annotations

import numpy as np


def _split_alpha(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    if arr.shape[-1] == 4:
        return arr[..., :3], arr[..., 3:4]
    return arr, None


def _rejoin_alpha(arr: np.ndarray, alpha: np.ndarray | None) -> np.ndarray:
    if alpha is None:
        return arr
    return np.concatenate([arr, alpha], axis=-1)


def lab_to_lch(lab: np.ndarray) -> np.ndarray:
    """Cartesian Lab → cylindrical LCh.

    ``h`` is returned in degrees, wrapped to ``[0, 360)``.
    """
    arr = np.asarray(lab, dtype=np.float64)
    cart, alpha = _split_alpha(arr)

    L = cart[..., 0]
    a = cart[..., 1]
    b = cart[..., 2]
    C = np.hypot(a, b)
    h = np.degrees(np.arctan2(b, a))
    h = np.where(h < 0, h + 360.0, h)

    out = np.stack([L, C, h], axis=-1)
    return _rejoin_alpha(out, alpha)


def lch_to_lab(lch: np.ndarray) -> np.ndarray:
    """Cylindrical LCh → Cartesian Lab.

    ``h`` is interpreted in degrees.
    """
    arr = np.asarray(lch, dtype=np.float64)
    cyl, alpha = _split_alpha(arr)

    L = cyl[..., 0]
    C = cyl[..., 1]
    h = np.radians(cyl[..., 2])
    a = C * np.cos(h)
    b = C * np.sin(h)

    out = np.stack([L, a, b], axis=-1)
    return _rejoin_alpha(out, alpha)


# OKLab is just another Lab-style Cartesian space, so the polar
# transform is identical. Aliases keep call sites self-documenting.
oklab_to_oklch = lab_to_lch
oklch_to_oklab = lch_to_lab
