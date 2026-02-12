"""
Dreamstack Raster - Filters Module
==================================

Professional image filters including blur, sharpen,
noise reduction, and artistic effects.

"""

from __future__ import annotations

from dreamstack.raster.filters.artistic import (
    cartoon,
    glitch,
    halftone,
    oil_paint,
    pixelate,
    posterize,
    sketch,
    stipple,
    vignette,
    watercolor,
)
from dreamstack.raster.filters.blur import (
    bilateral_blur,
    box_blur,
    gaussian_blur,
    lens_blur,
    motion_blur,
    radial_blur,
    surface_blur,
    zoom_blur,
)
from dreamstack.raster.filters.distort import (
    bulge,
    fisheye,
    pinch,
    polar_coordinates,
    ripple,
    sphere,
    twirl,
    wave,
)
from dreamstack.raster.filters.edge import (
    canny,
    edge_detect,
    emboss,
    find_edges,
    laplacian,
    sobel,
)
from dreamstack.raster.filters.noise import (
    add_noise,
    denoise_bilateral,
    denoise_nlmeans,
    despeckle,
    median_filter,
    reduce_noise,
)
from dreamstack.raster.filters.sharpen import (
    high_pass,
    sharpen,
    smart_sharpen,
    unsharp_mask,
)
from dreamstack.raster.filters.stylize import diffuse, extrude, solarize, tiles, wind
from dreamstack.raster.filters.stylize import find_edges as stylize_edges

__all__: list[str] = [
    # Blur
    "gaussian_blur",
    "box_blur",
    "motion_blur",
    "radial_blur",
    "zoom_blur",
    "lens_blur",
    "surface_blur",
    "bilateral_blur",
    # Sharpen
    "unsharp_mask",
    "sharpen",
    "high_pass",
    "smart_sharpen",
    # Noise
    "add_noise",
    "reduce_noise",
    "median_filter",
    "despeckle",
    "denoise_bilateral",
    "denoise_nlmeans",
    # Edge
    "edge_detect",
    "sobel",
    "canny",
    "laplacian",
    "find_edges",
    "emboss",
    # Artistic
    "oil_paint",
    "watercolor",
    "posterize",
    "pixelate",
    "halftone",
    "stipple",
    "sketch",
    "cartoon",
    "glitch",
    "vignette",
    # Distort
    "wave",
    "ripple",
    "twirl",
    "sphere",
    "pinch",
    "bulge",
    "fisheye",
    "polar_coordinates",
    # Stylize
    "diffuse",
    "stylize_edges",
    "solarize",
    "tiles",
    "extrude",
    "wind",
]
