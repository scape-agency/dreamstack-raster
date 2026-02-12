"""
Dreamstack Raster - Compositing Merge Module
=============================================

Image merging and arithmetic operations.

"""

from __future__ import annotations

from typing import Literal

from dreamstack.raster.compositing.merge.add import add
from dreamstack.raster.compositing.merge.average import average
from dreamstack.raster.compositing.merge.difference import difference
from dreamstack.raster.compositing.merge.divide import divide
from dreamstack.raster.compositing.merge.maximum import maximum
from dreamstack.raster.compositing.merge.merge import MergeMode, merge
from dreamstack.raster.compositing.merge.minimum import minimum
from dreamstack.raster.compositing.merge.multiply import multiply
from dreamstack.raster.compositing.merge.over import over
from dreamstack.raster.compositing.merge.screen import screen
from dreamstack.raster.compositing.merge.subtract import subtract

__all__: list[str] = [
    "add",
    "subtract",
    "multiply",
    "divide",
    "screen",
    "difference",
    "average",
    "maximum",
    "minimum",
    "merge",
    "over",
    "MergeMode",
]
