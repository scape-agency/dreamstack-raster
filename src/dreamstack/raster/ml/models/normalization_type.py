"""Normalization type enum."""

from enum import StrEnum


class NormalizationType(StrEnum):
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
