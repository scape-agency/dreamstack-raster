"""Augment operation."""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

from .models.augmentation_config import AugmentationConfig


def augment(
    image: NDArray[np.uint8],
    config: AugmentationConfig | None = None,
    *,
    seed: int | None = None,
) -> NDArray[np.uint8]:
    """Apply random augmentation to image.

    Parameters
    ----------
    image : NDArray[np.uint8]
        Input image.
    config : AugmentationConfig, optional
        Augmentation configuration.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    NDArray[np.uint8]
        Augmented image.

    Examples
    --------
    >>> # Default augmentation
    >>> augmented = augment(img)

    >>> # Custom augmentation
    >>> config = AugmentationConfig(
    ...     rotation_range=30,
    ...     horizontal_flip=True,
    ...     brightness_range=(0.7, 1.3)
    ... )
    >>> augmented = augment(img, config)
    """
    if config is None:
        config = AugmentationConfig()

    if seed is not None:
        np.random.seed(seed)

    result = image.copy()
    h, w = result.shape[:2]

    # Horizontal flip
    if config.horizontal_flip and np.random.random() > 0.5:
        result = cv2.flip(result, 1)

    # Vertical flip
    if config.vertical_flip and np.random.random() > 0.5:
        result = cv2.flip(result, 0)

    # Rotation
    if config.rotation_range > 0:
        angle = np.random.uniform(-config.rotation_range, config.rotation_range)
        matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        result = cv2.warpAffine(result, matrix, (w, h), borderMode=cv2.BORDER_REFLECT)

    # Scale
    if config.scale_range != (1.0, 1.0):
        scale = np.random.uniform(*config.scale_range)
        new_h, new_w = int(h * scale), int(w * scale)
        scaled = cv2.resize(result, (new_w, new_h))
        # Crop or pad to original size
        if scale > 1.0:
            # Crop center
            start_x = (new_w - w) // 2
            start_y = (new_h - h) // 2
            result = scaled[start_y : start_y + h, start_x : start_x + w]
        else:
            # Pad
            pad_x = (w - new_w) // 2
            pad_y = (h - new_h) // 2
            if result.ndim == 3:
                result = np.zeros((h, w, result.shape[2]), dtype=np.uint8)
            else:
                result = np.zeros((h, w), dtype=np.uint8)
            result[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = scaled

    # Translation
    if config.translation_range != (0.0, 0.0):
        max_tx = int(w * config.translation_range[0])
        max_ty = int(h * config.translation_range[1])
        tx = np.random.randint(-max_tx, max_tx + 1)
        ty = np.random.randint(-max_ty, max_ty + 1)
        matrix = np.float32([[1, 0, tx], [0, 1, ty]])
        result = cv2.warpAffine(result, matrix, (w, h), borderMode=cv2.BORDER_REFLECT)

    # Brightness
    if config.brightness_range != (1.0, 1.0):
        factor = np.random.uniform(*config.brightness_range)
        result = np.clip(result.astype(np.float32) * factor, 0, 255).astype(np.uint8)

    # Contrast
    if config.contrast_range != (1.0, 1.0):
        factor = np.random.uniform(*config.contrast_range)
        mean = np.mean(result)
        result = np.clip((result - mean) * factor + mean, 0, 255).astype(np.uint8)

    # Gaussian noise
    if config.noise_std > 0:
        noise = np.random.normal(0, config.noise_std * 255, result.shape)
        result = np.clip(result.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # Blur
    if config.blur_range != (0.0, 0.0):
        sigma = np.random.uniform(*config.blur_range)
        if sigma > 0:
            ksize = int(sigma * 4) | 1  # Ensure odd
            if ksize >= 3:
                result = cv2.GaussianBlur(result, (ksize, ksize), sigma)

    return result
