# -*- coding: utf-8 -*-

"""
Dreamstack Raster - Core Module
===============================

Core classes for image processing including Image, Canvas, Layer, and History.

"""

from __future__ import annotations

from dreamstack.raster.core.bounds import Bounds, Point, Size
from dreamstack.raster.core.canvas import Canvas
from dreamstack.raster.core.channel import Channel, ChannelType
from dreamstack.raster.core.document import Document
from dreamstack.raster.core.history import History, HistoryState
from dreamstack.raster.core.image import Image
from dreamstack.raster.core.layer import Layer, LayerGroup
from dreamstack.raster.core.pixel import PixelData

__all__: list[str] = [
    "Image",
    "Canvas",
    "Layer",
    "LayerGroup",
    "History",
    "HistoryState",
    "Document",
    "PixelData",
    "Channel",
    "ChannelType",
    "Bounds",
    "Point",
    "Size",
]
