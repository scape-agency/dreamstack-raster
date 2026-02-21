"""Translation operation."""

# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray


def translate(
    image: NDArray[np.uint8],
    tx: int = 0,
    ty: int = 0,
    *,
    border_mode: str = "constant",
    border_value: int | tuple[int, int, int] = 0,
) -> NDArray[np.uint8]:
    """Translate (shift) an image by pixel offset.

    Moves the image by the specified number of pixels in
    the x and y directions using an affine transformation.

    Parameters
    ----------
    image : NDArray[np.uint8]
        Input image (H, W) or (H, W, C).
    tx : int, optional
        Translation in x direction (positive = right). Default is 0.
    ty : int, optional
        Translation in y direction (positive = down). Default is 0.
    border_mode : str, optional
        How to fill empty areas:
        - "constant": Fill with border_value (default)
        - "reflect": Reflect the image at the border
        - "replicate": Replicate edge pixels
        - "wrap": Wrap around to opposite edge
    border_value : int or tuple, optional
        Fill value for constant border mode. Default is 0 (black).

    Returns
    -------
    NDArray[np.uint8]
        Translated image, same shape as input.

    Examples
    --------
    >>> # Shift image 100 pixels right and 50 pixels down
    >>> translated = translate(img, tx=100, ty=50)

    >>> # Shift with reflection at borders
    >>> translated = translate(img, tx=50, ty=25, border_mode="reflect")
    """
    h, w = image.shape[:2]

    # Create translation matrix
    translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])

    # Get border mode
    border_modes = {
        "constant": cv2.BORDER_CONSTANT,
        "reflect": cv2.BORDER_REFLECT,
        "replicate": cv2.BORDER_REPLICATE,
        "wrap": cv2.BORDER_WRAP,
    }
    border = border_modes.get(border_mode, cv2.BORDER_CONSTANT)

    return cv2.warpAffine(  # type: ignore[call-overload]
        image,
        translation_matrix,
        (w, h),
        borderMode=border,
        borderValue=border_value,
    )
