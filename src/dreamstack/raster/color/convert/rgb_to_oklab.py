# -*- coding: utf-8 -*-

"""Linear sRGB ↔ OKLab (Björn Ottosson, 2020).

OKLab is a perceptually uniform color space derived from cone-LMS, well
suited to gradients, mixing, and uniform-perceived hue/chroma editing.

References
----------
- Ottosson, B. (2020). "A perceptual color space for image processing."
  https://bottosson.github.io/posts/oklab/
"""

from __future__ import annotations

import numpy as np

from dreamstack.raster.color.spaces.color_space_instances import sRGB

# Linear-sRGB → LMS (M1 in Ottosson's notation).
_M1 = np.array(
    [
        [0.4122214708, 0.5363325363, 0.0514459929],
        [0.2119034982, 0.6806995451, 0.1073969566],
        [0.0883024619, 0.2817188376, 0.6299787005],
    ]
)

# LMS' → OKLab (M2).
_M2 = np.array(
    [
        [0.2104542553, 0.7936177850, -0.0040720468],
        [1.9779984951, -2.4285922050, 0.4505937099],
        [0.0259040371, 0.7827717662, -0.8086757660],
    ]
)

_M1_INV = np.linalg.inv(_M1)
_M2_INV = np.linalg.inv(_M2)


def _split_alpha(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    if arr.shape[-1] == 4:
        return arr[..., :3], arr[..., 3:4]
    return arr, None


def _rejoin_alpha(arr: np.ndarray, alpha: np.ndarray | None) -> np.ndarray:
    if alpha is None:
        return arr
    return np.concatenate([arr, alpha], axis=-1)


def rgb_to_oklab(rgb: np.ndarray, *, linear: bool = False) -> np.ndarray:
    """Convert sRGB to OKLab.

    Args:
        rgb: sRGB array, values in ``[0, 1]``. Last axis is the channel
            axis (3 or 4 if alpha is present; alpha is passed through).
        linear: ``True`` if ``rgb`` is already linear-light sRGB.

    Returns:
        OKLab array with the same shape as ``rgb``. ``L`` is roughly in
        ``[0, 1]`` for in-gamut sRGB; ``a``/``b`` typically in
        ``[-0.4, 0.4]``.
    """
    arr = np.asarray(rgb, dtype=np.float64)
    color, alpha = _split_alpha(arr)

    lin = color if linear else sRGB.linearize(color)
    lms = np.einsum("...j,ij->...i", lin, _M1)
    lms_prime = np.cbrt(lms)
    lab = np.einsum("...j,ij->...i", lms_prime, _M2)

    return _rejoin_alpha(lab, alpha)


def oklab_to_rgb(oklab: np.ndarray, *, linear: bool = False) -> np.ndarray:
    """Convert OKLab to sRGB.

    Args:
        oklab: OKLab array.
        linear: If ``True``, return linear-light sRGB; otherwise apply the
            sRGB OETF so values are display-encoded.

    Returns:
        sRGB array.
    """
    arr = np.asarray(oklab, dtype=np.float64)
    lab, alpha = _split_alpha(arr)

    lms_prime = np.einsum("...j,ij->...i", lab, _M2_INV)
    lms = lms_prime**3
    lin = np.einsum("...j,ij->...i", lms, _M1_INV)
    rgb = lin if linear else sRGB.encode(lin)

    return _rejoin_alpha(rgb, alpha)
