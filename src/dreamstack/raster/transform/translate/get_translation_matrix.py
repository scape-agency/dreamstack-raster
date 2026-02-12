"""Get translation matrix."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def get_translation_matrix(tx: int, ty: int) -> NDArray[np.float32]:
    """Get affine translation matrix.

    Creates a 2x3 affine transformation matrix for translation.
    Can be combined with other transformations.

    Parameters
    ----------
    tx : int
        Translation in x direction.
    ty : int
        Translation in y direction.

    Returns
    -------
    NDArray[np.float32]
        2x3 translation matrix.

    Examples
    --------
    >>> matrix = get_translation_matrix(100, 50)
    >>> translated = cv2.warpAffine(img, matrix, (w, h))
    """
    return np.float32([[1, 0, tx], [0, 1, ty]])
