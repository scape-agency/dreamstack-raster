# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Preprocessing pipeline dataclass."""


# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

from dataclasses import dataclass

from numpy.typing import NDArray

from .normalization_type import NormalizationType


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

    target_size: tuple[int, int] | None = None
    normalization: NormalizationType = NormalizationType.MINMAX
    color_mode: str = "rgb"
    preserve_aspect: bool = True
    pad_color: tuple[int, int, int] = (0, 0, 0)

    def __call__(self, image: NDArray) -> NDArray:
        """Apply the pipeline to an image."""
        from ..preprocess import preprocess

        return preprocess(
            image,
            target_size=self.target_size,
            normalization=self.normalization,
            color_mode=self.color_mode,
            preserve_aspect=self.preserve_aspect,
            pad_color=self.pad_color,
        )
