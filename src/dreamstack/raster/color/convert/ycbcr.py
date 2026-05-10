# -*- coding: utf-8 -*-

"""RGB ↔ YCbCr (Rec.601 / Rec.709 / Rec.2020).

YCbCr is a luma+chroma representation widely used in JPEG and video.
By convention this module operates on **gamma-encoded** RGB
(matching JPEG/video pipelines), in normalized ``[0, 1]`` floats and
"full range" (PC-range) values. Studio-range conversion can be added
later if needed.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

Standard = Literal["rec601", "rec709", "rec2020"]


# Luma coefficients (Kr, Kg, Kb).
_KR_KG_KB: dict[str, tuple[float, float, float]] = {
    "rec601": (0.299, 0.587, 0.114),
    "rec709": (0.2126, 0.7152, 0.0722),
    "rec2020": (0.2627, 0.6780, 0.0593),
}


def _matrices(standard: Standard) -> tuple[np.ndarray, np.ndarray]:
    if standard not in _KR_KG_KB:
        raise ValueError(
            f"Unknown YCbCr standard: {standard!r} "
            f"(expected one of {list(_KR_KG_KB)})"
        )
    kr, kg, kb = _KR_KG_KB[standard]
    # Full-range RGB→YCbCr (Cb,Cr are zero-centered ±0.5).
    forward = np.array(
        [
            [kr, kg, kb],
            [-kr / (2 * (1 - kb)), -kg / (2 * (1 - kb)), 0.5],
            [0.5, -kg / (2 * (1 - kr)), -kb / (2 * (1 - kr))],
        ]
    )
    inverse = np.linalg.inv(forward)
    return forward, inverse


def _split_alpha(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    if arr.shape[-1] == 4:
        return arr[..., :3], arr[..., 3:4]
    return arr, None


def _rejoin_alpha(arr: np.ndarray, alpha: np.ndarray | None) -> np.ndarray:
    if alpha is None:
        return arr
    return np.concatenate([arr, alpha], axis=-1)


def rgb_to_ycbcr(rgb: np.ndarray, standard: Standard = "rec709") -> np.ndarray:
    """RGB (encoded, ``[0,1]``) → YCbCr (full range).

    Returns ``Y`` in ``[0, 1]`` and ``Cb``/``Cr`` in ``[-0.5, 0.5]``.
    """
    arr = np.asarray(rgb, dtype=np.float64)
    color, alpha = _split_alpha(arr)
    forward, _ = _matrices(standard)
    ycbcr = np.einsum("...j,ij->...i", color, forward)
    return _rejoin_alpha(ycbcr, alpha)


def ycbcr_to_rgb(
    ycbcr: np.ndarray, standard: Standard = "rec709"
) -> np.ndarray:
    """YCbCr (full range) → RGB (encoded, ``[0,1]``)."""
    arr = np.asarray(ycbcr, dtype=np.float64)
    color, alpha = _split_alpha(arr)
    _, inverse = _matrices(standard)
    rgb = np.einsum("...j,ij->...i", color, inverse)
    return _rejoin_alpha(rgb, alpha)
