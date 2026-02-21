# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Machine Learning Image Preprocessing
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

from .add_batch_dim import add_batch_dim
from .add_channel_dim import add_channel_dim
from .augment import augment
from .batch_preprocess import batch_preprocess
from .bgr_to_rgb import bgr_to_rgb
from .channels_first import channels_first
from .channels_last import channels_last
from .constants import CAFFE_MEAN, IMAGENET_MEAN, IMAGENET_STD
from .denormalize import denormalize
from .extract_patches import extract_patches
from .models import (
    AugmentationConfig,
    NormalizationType,
    PreprocessingPipeline,
)
from .normalize import normalize
from .preprocess import preprocess
from .resize_for_model import resize_for_model
from .rgb_to_bgr import rgb_to_bgr
from .to_grayscale import to_grayscale

__all__ = [
    "add_batch_dim",
    "add_channel_dim",
    "augment",
    "AugmentationConfig",
    "batch_preprocess",
    "bgr_to_rgb",
    "CAFFE_MEAN",
    "channels_first",
    "channels_last",
    "denormalize",
    "extract_patches",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "normalize",
    "NormalizationType",
    "preprocess",
    "PreprocessingPipeline",
    "resize_for_model",
    "rgb_to_bgr",
    "to_grayscale",
]
