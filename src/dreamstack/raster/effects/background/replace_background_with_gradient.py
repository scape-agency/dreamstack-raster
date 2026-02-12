"""
Replace Background with Gradient
=================================

Replace background with a color gradient.

"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from dreamstack.raster.effects.background._create_gradient import (
    _create_gradient,
)
from dreamstack.raster.effects.background.gradient_config import (
    GradientConfig,
    GradientDirection,
)
from dreamstack.raster.effects.background.replace_background import (
    replace_background,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray


def replace_background_with_gradient(
    rgba_image: NDArray[np.uint8],
    config: GradientConfig | None = None,
    *,
    start_color: tuple[int, int, int] | None = None,
    end_color: tuple[int, int, int] | None = None,
    direction: GradientDirection | None = None,
) -> NDArray[np.uint8]:
    """Replace background with a color gradient.

    Creates a gradient background behind the RGBA foreground.

    Args:
        rgba_image: RGBA image with alpha channel.
        config: Optional GradientConfig.
        start_color: Starting gradient color.
        end_color: Ending gradient color.
        direction: Gradient direction.

    Returns:
        RGB image with gradient background.

    Example:
        >>> result = replace_background_with_gradient(
        ...     rgba,
        ...     start_color=(255, 200, 200),
        ...     end_color=(200, 200, 255),
        ...     direction="horizontal",
        ... )
    """
    cfg = config or GradientConfig()

    # Override with keyword arguments
    s_color = start_color or cfg.start_color
    e_color = end_color or cfg.end_color
    grad_dir = direction or cfg.direction

    if rgba_image.ndim != 3 or rgba_image.shape[2] != 4:
        raise ValueError("Expected RGBA image with 4 channels")

    h, w = rgba_image.shape[:2]

    # Create gradient
    gradient = _create_gradient(w, h, s_color, e_color, grad_dir, cfg.center)

    # Composite
    return replace_background(rgba_image, gradient)
