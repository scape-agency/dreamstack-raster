# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Professional Image Processing Library
=========================================================

A comprehensive, professional-grade raster image processing library
designed to compete with Adobe Photoshop. Built with Python, leveraging
NumPy, OpenCV, scikit-image, and Pillow for high-performance image
manipulation.

Features
--------
- **Core**: Image, Layer, Canvas, Document management with full undo/redo
- **IO**: Support for 25+ formats including PSD, RAW, EXR, HDR
- **Color**: Color space conversions (sRGB, Adobe RGB, ProPhoto, ACES, HDR)
- **Filters**: Blur, sharpen, noise, edge detection, artistic effects
- **Adjustments**: Levels, curves, color balance, tone mapping
- **Transform**: Resize, rotate, crop, perspective, distortion
- **Selection**: Magic wand, lasso, color range, feathering
- **Drawing**: Brushes with dynamics, shapes, text, gradients
- **Effects**: Drop shadow, glow, bevel, emboss, overlays
- **Compositing**: 27 blend modes, alpha compositing, masks
- **Analysis**: Histogram, statistics, measurements

Example Usage
-------------
>>> from dreamstack.raster import Image, Document
>>> from dreamstack.raster.io import load_image, save_image
>>> from dreamstack.raster.filters import gaussian_blur
>>> from dreamstack.raster.adjustments import adjust_brightness
>>>
>>> # Load and process an image
>>> img = load_image("photo.jpg")
>>> img = gaussian_blur(img, sigma=2.0)
>>> img = adjust_brightness(img, factor=1.2)
>>> save_image(img, "output.png")

"""


# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

# Import | Standard Library
from typing import TYPE_CHECKING

# Import | Sub-modules for namespace access
from dreamstack.raster import (
    adjustments,
    analysis,
    color,
    compositing,
    core,
    drawing,
    effects,
    extraction,
    filters,
    io,
    selection,
    transform,
)

# Import | Version
from dreamstack.raster.__version__ import __version__

# Import | Common adjustment functions
from dreamstack.raster.adjustments import (
    adjust_brightness,
    adjust_contrast,
    adjust_saturation,
    apply_curves,
    auto_levels,
)

# Import | Common compositing functions
from dreamstack.raster.compositing import (
    alpha_composite,
    apply_mask,
    blend_multiply,
    blend_normal,
    blend_overlay,
    blend_screen,
)

# Import | Core Module - Primary classes
from dreamstack.raster.core import (
    BlendMode,
    Bounds,
    Canvas,
    Channel,
    ChannelType,
    Document,
    HistoryManager,
    HistoryState,
    Image,
    Layer,
    LayerType,
    Pixel,
)

# Import | Common filter functions
from dreamstack.raster.filters import (
    box_blur,
    gaussian_blur,
    motion_blur,
    smart_sharpen,
    unsharp_mask,
)

# Import | Convenience functions from IO
from dreamstack.raster.io import get_supported_formats, load_image, save_image

# Import | Common transform functions
from dreamstack.raster.transform import (
    crop,
    flip_horizontal,
    flip_vertical,
    resize,
    rotate,
)

# Import | Object extraction
from dreamstack.raster.extraction import (
    ExtractedObject,
    ExtractionConfig,
    ObjectExtractor,
)
from dreamstack.raster.extraction.pipeline import (
    BatchPipeline,
    BatchResult,
    PipelineConfig,
)

# =============================================================================
# Variables
# =============================================================================

__author__ = "Dreamstack Team"
__email__ = "info@dreamstack.dev"
__license__ = "MIT"

__all__: list[str] = [
    # Version
    "__version__",
    # Core classes
    "Bounds",
    "Pixel",
    "Channel",
    "ChannelType",
    "Image",
    "Layer",
    "LayerType",
    "BlendMode",
    "HistoryManager",
    "HistoryState",
    "Canvas",
    "Document",
    # Sub-modules
    "core",
    "io",
    "color",
    "filters",
    "adjustments",
    "transform",
    "selection",
    "drawing",
    "effects",
    "compositing",
    "analysis",
    "extraction",
    # IO convenience
    "load_image",
    "save_image",
    "get_supported_formats",
    # Filter convenience
    "gaussian_blur",
    "box_blur",
    "motion_blur",
    "unsharp_mask",
    "smart_sharpen",
    # Adjustment convenience
    "adjust_brightness",
    "adjust_contrast",
    "adjust_saturation",
    "auto_levels",
    "apply_curves",
    # Transform convenience
    "resize",
    "rotate",
    "crop",
    "flip_horizontal",
    "flip_vertical",
    # Compositing convenience
    "alpha_composite",
    "apply_mask",
    "blend_normal",
    "blend_multiply",
    "blend_screen",
    "blend_overlay",
    # Extraction convenience
    "ObjectExtractor",
    "ExtractedObject",
    "ExtractionConfig",
    "BatchPipeline",
    "BatchResult",
    "PipelineConfig",
]


# =============================================================================
# Module Information
# =============================================================================


def get_info() -> dict:
    """
    Get information about the Dreamstack Raster library.

    Returns
    -------
    dict
        Dictionary containing library information.
    """
    return {
        "name": "Dreamstack Raster",
        "version": __version__,
        "description": "Professional-grade raster image processing library",
        "author": __author__,
        "license": __license__,
        "modules": [
            "core",
            "io",
            "color",
            "filters",
            "adjustments",
            "transform",
            "selection",
            "drawing",
            "effects",
            "compositing",
            "analysis",
            "extraction",
        ],
        "features": {
            "formats_supported": "25+",
            "blend_modes": 27,
            "color_spaces": [
                "sRGB",
                "Adobe RGB",
                "ProPhoto RGB",
                "Display P3",
                "Rec. 709",
                "Rec. 2020",
                "ACES",
                "DCI-P3",
            ],
            "hdr_support": True,
            "raw_support": True,
            "psd_support": True,
            "gpu_acceleration": "planned",
        },
    }


def print_info() -> None:
    """Print library information to stdout."""
    info = get_info()
    print(f"\n{'='*60}")
    print(f"  {info['name']} v{info['version']}")
    print(f"{'='*60}")
    print(f"  {info['description']}")
    print(f"  License: {info['license']}")
    print(f"\n  Modules:")
    for module in info["modules"]:
        print(f"    - {module}")
    print(f"\n  Key Features:")
    print(f"    - {info['features']['formats_supported']} image formats")
    print(f"    - {info['features']['blend_modes']} blend modes")
    print(f"    - HDR/RAW/PSD support")
    print(f"    - Professional color management")
    print(f"{'='*60}\n")
