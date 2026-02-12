"""
Dreamstack Raster - Diffuse
===========================

Diffuse glow effect implementation.

"""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from dreamstack.raster.core.image import Image


def diffuse(image: Image, mode: str = "normal", distance: int = 10) -> Image:
    """
    Apply diffuse glow effect.

    Args:
        image: Input image
        mode: Diffusion mode ('normal', 'darken', 'lighten', 'anisotropic')
        distance: Diffusion distance

    Returns:
        Diffused image
    """
    from dreamstack.raster.core.pixel import PixelData

    data = image.data.copy()
    h, w = data.shape[:2]

    if mode == "anisotropic":
        # Anisotropic diffusion (edge-preserving)
        if data.dtype != np.float32:
            data = data.astype(np.float32)

        for _ in range(distance):
            # Compute gradients
            gx = cv2.Sobel(data, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(data, cv2.CV_32F, 0, 1, ksize=3)

            # Perona-Malik diffusion coefficient
            k = 25
            c = 1 / (1 + (np.sqrt(gx**2 + gy**2) / k) ** 2)

            # Diffuse
            laplacian = cv2.Laplacian(data, cv2.CV_32F)
            data = data + 0.25 * c * laplacian

    else:
        # Random displacement
        for y in range(h):
            for x in range(w):
                dx = np.random.randint(-distance, distance + 1)
                dy = np.random.randint(-distance, distance + 1)

                src_x = np.clip(x + dx, 0, w - 1)
                src_y = np.clip(y + dy, 0, h - 1)

                if mode == "normal":
                    data[y, x] = image.data[src_y, src_x]
                elif mode == "darken":
                    if data.ndim == 3:
                        data[y, x] = np.minimum(data[y, x], image.data[src_y, src_x])
                    else:
                        data[y, x] = min(data[y, x], image.data[src_y, src_x])
                elif mode == "lighten":
                    if data.ndim == 3:
                        data[y, x] = np.maximum(data[y, x], image.data[src_y, src_x])
                    else:
                        data[y, x] = max(data[y, x], image.data[src_y, src_x])

    result_image = image.copy()
    result_image._pixel_data = PixelData(
        data=data.astype(image.data.dtype),
        pixel_format=image.pixel_format,
        bit_depth=image.bit_depth,
    )

    return result_image
