"""
Remove Background
=================

AI-based background removal using rembg.

"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from dreamstack.raster.effects.background._check_rembg_available import (
    _check_rembg_available,
)
from dreamstack.raster.effects.background.removal_config import (
    ModelName,
    RemovalConfig,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray


def remove_background(
    image: NDArray[np.uint8],
    config: RemovalConfig | None = None,
    *,
    model_name: ModelName | None = None,
    alpha_matting: bool | None = None,
) -> NDArray[np.uint8]:
    """Remove background from an image using AI segmentation.

    Uses rembg library for AI-based background removal. Returns RGBA image
    with transparent background.

    Args:
        image: Input image as numpy array (BGR or RGB, 3 channels).
        config: Optional RemovalConfig for detailed settings.
        model_name: Override model name (u2net, isnet-general-use, etc.).
        alpha_matting: Override alpha matting setting.

    Returns:
        RGBA image with background removed (4 channels).

    Raises:
        ImportError: If rembg is not installed.
        ValueError: If image format is invalid.

    Example:
        >>> from dreamstack.raster.effects.background import remove_background
        >>> result = remove_background(image, model_name="u2net")
        >>> # result is RGBA with transparent background
    """
    if not _check_rembg_available():
        raise ImportError(
            "rembg is required for background removal. Install with: pip install rembg"
        )

    import rembg

    # Use config or defaults
    cfg = config or RemovalConfig()

    # Override with keyword arguments
    if model_name is not None:
        cfg = RemovalConfig(
            model_name=model_name,
            alpha_matting=(
                cfg.alpha_matting if alpha_matting is None else alpha_matting
            ),
            alpha_matting_foreground_threshold=cfg.alpha_matting_foreground_threshold,
            alpha_matting_background_threshold=cfg.alpha_matting_background_threshold,
            alpha_matting_erode_size=cfg.alpha_matting_erode_size,
            post_process_mask=cfg.post_process_mask,
        )
    elif alpha_matting is not None:
        cfg = RemovalConfig(
            model_name=cfg.model_name,
            alpha_matting=alpha_matting,
            alpha_matting_foreground_threshold=cfg.alpha_matting_foreground_threshold,
            alpha_matting_background_threshold=cfg.alpha_matting_background_threshold,
            alpha_matting_erode_size=cfg.alpha_matting_erode_size,
            post_process_mask=cfg.post_process_mask,
        )

    # Validate input
    if image.ndim != 3:
        raise ValueError(f"Expected 3-channel image, got {image.ndim} dimensions")

    # Convert to RGBA for rembg
    from PIL import Image

    if image.shape[2] == 3:
        # Assume BGR from OpenCV, convert to RGB
        import cv2

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif image.shape[2] == 4:
        import cv2

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)[:, :, :3]
    else:
        raise ValueError(f"Expected 3 or 4 channels, got {image.shape[2]}")

    pil_image = Image.fromarray(rgb_image)

    # Remove background
    result = rembg.remove(
        pil_image,
        session=rembg.new_session(cfg.model_name),
        alpha_matting=cfg.alpha_matting,
        alpha_matting_foreground_threshold=cfg.alpha_matting_foreground_threshold,
        alpha_matting_background_threshold=cfg.alpha_matting_background_threshold,
        alpha_matting_erode_size=cfg.alpha_matting_erode_size,
        post_process_mask=cfg.post_process_mask,
    )

    # Convert back to numpy RGBA
    return np.array(result)
