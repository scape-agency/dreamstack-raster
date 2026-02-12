"""
Dreamstack Raster - Curves Adjustments
======================================

Professional curves adjustment with spline interpolation.

"""

from dreamstack.raster.adjustments.curves.apply_curve import apply_curve
from dreamstack.raster.adjustments.curves.create_curve import create_curve
from dreamstack.raster.adjustments.curves.curve import Curve
from dreamstack.raster.adjustments.curves.curve_point import CurvePoint
from dreamstack.raster.adjustments.curves.curves import curves
from dreamstack.raster.adjustments.curves.linear_contrast import (
    linear_contrast,
)
from dreamstack.raster.adjustments.curves.preset_curves import preset_curves
from dreamstack.raster.adjustments.curves.s_curve import s_curve

__all__ = [
    "CurvePoint",
    "Curve",
    "create_curve",
    "curves",
    "apply_curve",
    "s_curve",
    "linear_contrast",
    "preset_curves",
]
