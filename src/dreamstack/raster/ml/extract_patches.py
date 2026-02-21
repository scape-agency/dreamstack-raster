# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""Extract patches operation."""


# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def extract_patches(
    image: NDArray[np.uint8],
    patch_size: tuple[int, int],
    stride: tuple[int, int] | None = None,
) -> NDArray[np.uint8]:
    """Extract patches from image.

    Parameters
    ----------
    image : NDArray[np.uint8]
        Input image.
    patch_size : tuple[int, int]
        Patch (width, height).
    stride : tuple[int, int], optional
        Step between patches. Default = patch_size (no overlap).

    Returns
    -------
    NDArray[np.uint8]
        Array of patches (N, patch_h, patch_w, C).

    Examples
    --------
    >>> patches = extract_patches(img, (64, 64), stride=(32, 32))
    >>> print(f"Extracted {len(patches)} patches")
    """
    if stride is None:
        stride = patch_size

    h, w = image.shape[:2]
    patch_w, patch_h = patch_size
    stride_w, stride_h = stride

    patches = []
    for y in range(0, h - patch_h + 1, stride_h):
        for x in range(0, w - patch_w + 1, stride_w):
            patch = image[y : y + patch_h, x : x + patch_w]
            patches.append(patch)

    return np.array(patches)
