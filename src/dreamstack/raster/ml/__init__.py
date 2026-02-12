# -*- coding: utf-8 -*-

"""
Machine Learning Image Preprocessing
====================================

This module provides image preprocessing utilities specifically designed
for machine learning and deep learning tasks. Includes normalization,
augmentation, batch processing, and common transformations.

Example Usage
-------------
>>> from dreamstack.raster.ml import preprocess
>>>
>>> # Preprocess image for neural network
>>> img = cv2.imread("image.jpg")
>>> processed = preprocess.normalize_for_model(img, model_type="resnet")
>>>
>>> # Apply augmentation pipeline
>>> augmented = preprocess.augment(img, rotation=True, flip=True)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, List, Literal, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
from numpy.typing import NDArray


class NormalizationType(str, Enum):
    """Image normalization methods for ML models.

    Attributes
    ----------
    MINMAX : Scale to [0, 1] range
    STANDARDIZE : Zero mean, unit variance
    IMAGENET : ImageNet mean/std normalization
    CAFFE : Caffe-style BGR mean subtraction
    TORCH : PyTorch-style normalization
    TF : TensorFlow-style [-1, 1] scaling
    """

    MINMAX = "minmax"
    STANDARDIZE = "standardize"
    IMAGENET = "imagenet"
    CAFFE = "caffe"
    TORCH = "torch"
    TF = "tensorflow"


@dataclass
class AugmentationConfig:
    """Configuration for image augmentation.

    Attributes
    ----------
    horizontal_flip : bool
        Random horizontal flip.
    vertical_flip : bool
        Random vertical flip.
    rotation_range : float
        Maximum rotation angle in degrees.
    scale_range : tuple[float, float]
        Min/max scale factors.
    translation_range : tuple[float, float]
        Max translation as fraction of image size.
    brightness_range : tuple[float, float]
        Min/max brightness adjustment.
    contrast_range : tuple[float, float]
        Min/max contrast adjustment.
    noise_std : float
        Standard deviation of Gaussian noise.
    blur_range : tuple[float, float]
        Min/max Gaussian blur sigma.
    """

    horizontal_flip: bool = True
    vertical_flip: bool = False
    rotation_range: float = 15.0
    scale_range: Tuple[float, float] = (0.9, 1.1)
    translation_range: Tuple[float, float] = (0.1, 0.1)
    brightness_range: Tuple[float, float] = (0.8, 1.2)
    contrast_range: Tuple[float, float] = (0.8, 1.2)
    noise_std: float = 0.0
    blur_range: Tuple[float, float] = (0.0, 0.0)


@dataclass
class PreprocessingPipeline:
    """Reusable preprocessing pipeline.

    Attributes
    ----------
    target_size : tuple[int, int]
        Target image size (width, height).
    normalization : NormalizationType
        Normalization method.
    color_mode : str
        Color mode: "rgb", "bgr", or "gray".
    preserve_aspect : bool
        Whether to preserve aspect ratio.
    pad_color : tuple
        Padding color when preserving aspect.
    """

    target_size: Optional[Tuple[int, int]] = None
    normalization: NormalizationType = NormalizationType.MINMAX
    color_mode: str = "rgb"
    preserve_aspect: bool = True
    pad_color: Tuple[int, int, int] = (0, 0, 0)

    def __call__(self, image: NDArray) -> NDArray:
        """Apply the pipeline to an image."""
        return preprocess(
            image,
            target_size=self.target_size,
            normalization=self.normalization,
            color_mode=self.color_mode,
            preserve_aspect=self.preserve_aspect,
            pad_color=self.pad_color,
        )


# ImageNet statistics
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])

# Caffe-style BGR mean
CAFFE_MEAN = np.array([103.939, 116.779, 123.68])


def normalize(
    image: NDArray,
    method: Union[NormalizationType, str] = NormalizationType.MINMAX,
) -> NDArray[np.float32]:
    """Normalize image for ML model input.

    Parameters
    ----------
    image : NDArray
        Input image (uint8 0-255 or float 0-1).
    method : NormalizationType or str
        Normalization method:
        - "minmax": Scale to [0, 1]
        - "standardize": Zero mean, unit variance
        - "imagenet": ImageNet normalization (RGB)
        - "caffe": Caffe BGR mean subtraction
        - "torch": PyTorch normalization
        - "tensorflow": Scale to [-1, 1]

    Returns
    -------
    NDArray[np.float32]
        Normalized image.

    Examples
    --------
    >>> # For ResNet, VGG, etc.
    >>> normalized = normalize(img, "imagenet")

    >>> # For TensorFlow models
    >>> normalized = normalize(img, "tensorflow")
    """
    if isinstance(method, str):
        method = NormalizationType(method.lower())

    # Convert to float
    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 255.0
    else:
        image = image.astype(np.float32)

    if method == NormalizationType.MINMAX:
        return image

    elif method == NormalizationType.STANDARDIZE:
        mean = np.mean(image)
        std = np.std(image) + 1e-7
        return (image - mean) / std

    elif method == NormalizationType.IMAGENET:
        # Assumes RGB input
        return (image - IMAGENET_MEAN) / IMAGENET_STD

    elif method == NormalizationType.TORCH:
        # Same as ImageNet but explicit
        return (image - IMAGENET_MEAN) / IMAGENET_STD

    elif method == NormalizationType.CAFFE:
        # BGR input, mean subtraction
        return (image * 255.0) - CAFFE_MEAN

    elif method == NormalizationType.TF:
        # TensorFlow: scale to [-1, 1]
        return (image - 0.5) * 2.0

    return image


def denormalize(
    image: NDArray[np.float32],
    method: Union[NormalizationType, str] = NormalizationType.MINMAX,
) -> NDArray[np.uint8]:
    """Reverse normalization for visualization.

    Parameters
    ----------
    image : NDArray[np.float32]
        Normalized image.
    method : NormalizationType or str
        Method used for normalization.

    Returns
    -------
    NDArray[np.uint8]
        Image in uint8 format (0-255).
    """
    if isinstance(method, str):
        method = NormalizationType(method.lower())

    if method == NormalizationType.MINMAX:
        result = image

    elif (
        method == NormalizationType.IMAGENET
        or method == NormalizationType.TORCH
    ):
        result = image * IMAGENET_STD + IMAGENET_MEAN

    elif method == NormalizationType.CAFFE:
        result = (image + CAFFE_MEAN) / 255.0

    elif method == NormalizationType.TF:
        result = (image / 2.0) + 0.5

    else:
        result = image

    result = np.clip(result * 255.0, 0, 255)
    return result.astype(np.uint8)


def resize_for_model(
    image: NDArray[np.uint8],
    target_size: Tuple[int, int],
    *,
    preserve_aspect: bool = True,
    pad_color: Tuple[int, int, int] = (0, 0, 0),
    interpolation: int = cv2.INTER_LINEAR,
) -> NDArray[np.uint8]:
    """Resize image for model input with optional padding.

    Parameters
    ----------
    image : NDArray[np.uint8]
        Input image.
    target_size : tuple[int, int]
        Target (width, height).
    preserve_aspect : bool, optional
        Maintain aspect ratio. Default is True.
    pad_color : tuple, optional
        Padding color. Default is black.
    interpolation : int, optional
        OpenCV interpolation method.

    Returns
    -------
    NDArray[np.uint8]
        Resized image.
    """
    if not preserve_aspect:
        return cv2.resize(image, target_size, interpolation=interpolation)

    h, w = image.shape[:2]
    target_w, target_h = target_size

    # Calculate scale to fit
    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    # Resize
    resized = cv2.resize(image, (new_w, new_h), interpolation=interpolation)

    # Create padded output
    if image.ndim == 3:
        result = np.full(
            (target_h, target_w, image.shape[2]), pad_color, dtype=np.uint8
        )
    else:
        result = np.full((target_h, target_w), pad_color[0], dtype=np.uint8)

    # Center the resized image
    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2
    result[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = resized

    return result


def to_grayscale(
    image: NDArray[np.uint8],
    method: str = "luminance",
) -> NDArray[np.uint8]:
    """Convert image to grayscale.

    Parameters
    ----------
    image : NDArray[np.uint8]
        Input BGR/RGB image.
    method : str, optional
        Conversion method:
        - "luminance": Perceptual weights (default)
        - "average": Simple average
        - "cv2": OpenCV default

    Returns
    -------
    NDArray[np.uint8]
        Grayscale image.
    """
    if image.ndim == 2:
        return image

    if method == "cv2":
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    elif method == "average":
        return np.mean(image, axis=2).astype(np.uint8)

    else:  # luminance
        # Assumes BGR
        b, g, r = image[:, :, 0], image[:, :, 1], image[:, :, 2]
        return (0.299 * r + 0.587 * g + 0.114 * b).astype(np.uint8)


def bgr_to_rgb(image: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Convert BGR to RGB color space.

    Parameters
    ----------
    image : NDArray[np.uint8]
        BGR image (OpenCV default).

    Returns
    -------
    NDArray[np.uint8]
        RGB image.
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def rgb_to_bgr(image: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Convert RGB to BGR color space.

    Parameters
    ----------
    image : NDArray[np.uint8]
        RGB image.

    Returns
    -------
    NDArray[np.uint8]
        BGR image (OpenCV default).
    """
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def preprocess(
    image: NDArray[np.uint8],
    *,
    target_size: Optional[Tuple[int, int]] = None,
    normalization: Union[NormalizationType, str] = NormalizationType.MINMAX,
    color_mode: str = "rgb",
    preserve_aspect: bool = True,
    pad_color: Tuple[int, int, int] = (0, 0, 0),
) -> NDArray[np.float32]:
    """Complete preprocessing pipeline for ML models.

    Applies resizing, color conversion, and normalization in one step.

    Parameters
    ----------
    image : NDArray[np.uint8]
        Input image (BGR from OpenCV).
    target_size : tuple[int, int], optional
        Target (width, height). None = no resize.
    normalization : NormalizationType or str, optional
        Normalization method. Default is minmax.
    color_mode : str, optional
        Output color mode: "rgb", "bgr", or "gray".
    preserve_aspect : bool, optional
        Maintain aspect ratio when resizing.
    pad_color : tuple, optional
        Padding color.

    Returns
    -------
    NDArray[np.float32]
        Preprocessed image ready for model input.

    Examples
    --------
    >>> # Preprocess for ResNet
    >>> preprocessed = preprocess(
    ...     img,
    ...     target_size=(224, 224),
    ...     normalization="imagenet",
    ...     color_mode="rgb"
    ... )
    """
    result = image.copy()

    # Color conversion
    if color_mode == "rgb" and result.ndim == 3:
        result = bgr_to_rgb(result)
    elif color_mode == "gray":
        result = to_grayscale(result)

    # Resize
    if target_size is not None:
        result = resize_for_model(
            result,
            target_size,
            preserve_aspect=preserve_aspect,
            pad_color=pad_color,
        )

    # Normalize
    result = normalize(result, normalization)

    return result


def augment(
    image: NDArray[np.uint8],
    config: Optional[AugmentationConfig] = None,
    *,
    seed: Optional[int] = None,
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
        angle = np.random.uniform(
            -config.rotation_range, config.rotation_range
        )
        matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        result = cv2.warpAffine(
            result, matrix, (w, h), borderMode=cv2.BORDER_REFLECT
        )

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
        result = cv2.warpAffine(
            result, matrix, (w, h), borderMode=cv2.BORDER_REFLECT
        )

    # Brightness
    if config.brightness_range != (1.0, 1.0):
        factor = np.random.uniform(*config.brightness_range)
        result = np.clip(result.astype(np.float32) * factor, 0, 255).astype(
            np.uint8
        )

    # Contrast
    if config.contrast_range != (1.0, 1.0):
        factor = np.random.uniform(*config.contrast_range)
        mean = np.mean(result)
        result = np.clip((result - mean) * factor + mean, 0, 255).astype(
            np.uint8
        )

    # Gaussian noise
    if config.noise_std > 0:
        noise = np.random.normal(0, config.noise_std * 255, result.shape)
        result = np.clip(result.astype(np.float32) + noise, 0, 255).astype(
            np.uint8
        )

    # Blur
    if config.blur_range != (0.0, 0.0):
        sigma = np.random.uniform(*config.blur_range)
        if sigma > 0:
            ksize = int(sigma * 4) | 1  # Ensure odd
            if ksize >= 3:
                result = cv2.GaussianBlur(result, (ksize, ksize), sigma)

    return result


def batch_preprocess(
    images: Sequence[NDArray[np.uint8]],
    pipeline: Optional[PreprocessingPipeline] = None,
    **kwargs,
) -> NDArray[np.float32]:
    """Preprocess a batch of images.

    Parameters
    ----------
    images : Sequence[NDArray]
        List of input images.
    pipeline : PreprocessingPipeline, optional
        Reusable preprocessing pipeline.
    **kwargs
        Arguments passed to preprocess() if no pipeline.

    Returns
    -------
    NDArray[np.float32]
        Batch array of shape (N, H, W, C) or (N, H, W).
    """
    if pipeline is not None:
        processed = [pipeline(img) for img in images]
    else:
        processed = [preprocess(img, **kwargs) for img in images]

    return np.stack(processed, axis=0)


def extract_patches(
    image: NDArray[np.uint8],
    patch_size: Tuple[int, int],
    stride: Optional[Tuple[int, int]] = None,
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


def add_channel_dim(image: NDArray) -> NDArray:
    """Add channel dimension for single-channel images.

    Ensures images have shape (H, W, C) for model input.

    Parameters
    ----------
    image : NDArray
        Input image.

    Returns
    -------
    NDArray
        Image with channel dimension.
    """
    if image.ndim == 2:
        return image[:, :, np.newaxis]
    return image


def add_batch_dim(image: NDArray) -> NDArray:
    """Add batch dimension for single image.

    Converts (H, W, C) to (1, H, W, C).

    Parameters
    ----------
    image : NDArray
        Single image.

    Returns
    -------
    NDArray
        Image with batch dimension.
    """
    return image[np.newaxis, ...]


def channels_first(image: NDArray) -> NDArray:
    """Convert from channels-last to channels-first format.

    Converts (H, W, C) to (C, H, W) for PyTorch.

    Parameters
    ----------
    image : NDArray
        Channels-last image.

    Returns
    -------
    NDArray
        Channels-first image.
    """
    if image.ndim == 3:
        return np.transpose(image, (2, 0, 1))
    elif image.ndim == 4:
        return np.transpose(image, (0, 3, 1, 2))
    return image


def channels_last(image: NDArray) -> NDArray:
    """Convert from channels-first to channels-last format.

    Converts (C, H, W) to (H, W, C) for TensorFlow.

    Parameters
    ----------
    image : NDArray
        Channels-first image.

    Returns
    -------
    NDArray
        Channels-last image.
    """
    if image.ndim == 3:
        return np.transpose(image, (1, 2, 0))
    elif image.ndim == 4:
        return np.transpose(image, (0, 2, 3, 1))
    return image
