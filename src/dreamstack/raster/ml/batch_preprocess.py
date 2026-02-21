# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Batch preprocess operation."""


# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from .models.preprocessing_pipeline import PreprocessingPipeline
from .preprocess import preprocess


def batch_preprocess(
    images: Sequence[NDArray[np.uint8]],
    pipeline: PreprocessingPipeline | None = None,
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
