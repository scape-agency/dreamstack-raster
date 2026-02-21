# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Format Constants
====================================

Constants mapping formats to extensions, MIME types, and capabilities.

"""


# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

from dreamstack.raster.io.formats.image_format import ImageFormat

FORMAT_EXTENSIONS = {
    ImageFormat.PNG: [".png"],
    ImageFormat.JPEG: [".jpg", ".jpeg", ".jpe", ".jfif"],
    ImageFormat.TIFF: [".tif", ".tiff"],
    ImageFormat.BMP: [".bmp", ".dib"],
    ImageFormat.GIF: [".gif"],
    ImageFormat.WEBP: [".webp"],
    ImageFormat.ICO: [".ico"],
    ImageFormat.EXR: [".exr"],
    ImageFormat.HDR: [".hdr"],
    ImageFormat.PSD: [".psd"],
    ImageFormat.PSB: [".psb"],
    ImageFormat.XCF: [".xcf"],
    ImageFormat.RAW: [".raw"],
    ImageFormat.CR2: [".cr2"],
    ImageFormat.CR3: [".cr3"],
    ImageFormat.NEF: [".nef"],
    ImageFormat.ARW: [".arw"],
    ImageFormat.DNG: [".dng"],
    ImageFormat.ORF: [".orf"],
    ImageFormat.RW2: [".rw2"],
    ImageFormat.SVG: [".svg", ".svgz"],
    ImageFormat.PDF: [".pdf"],
    ImageFormat.AI: [".ai"],
    ImageFormat.EPS: [".eps"],
    ImageFormat.HEIC: [".heic"],
    ImageFormat.HEIF: [".heif"],
    ImageFormat.AVIF: [".avif"],
    ImageFormat.JXL: [".jxl"],
}

FORMAT_MIME_TYPES = {
    ImageFormat.PNG: "image/png",
    ImageFormat.JPEG: "image/jpeg",
    ImageFormat.TIFF: "image/tiff",
    ImageFormat.BMP: "image/bmp",
    ImageFormat.GIF: "image/gif",
    ImageFormat.WEBP: "image/webp",
    ImageFormat.ICO: "image/x-icon",
    ImageFormat.EXR: "image/x-exr",
    ImageFormat.HDR: "image/vnd.radiance",
    ImageFormat.PSD: "image/vnd.adobe.photoshop",
    ImageFormat.SVG: "image/svg+xml",
    ImageFormat.PDF: "application/pdf",
    ImageFormat.HEIC: "image/heic",
    ImageFormat.HEIF: "image/heif",
    ImageFormat.AVIF: "image/avif",
    ImageFormat.JXL: "image/jxl",
}

# Formats that support reading
READ_FORMATS = {
    ImageFormat.PNG,
    ImageFormat.JPEG,
    ImageFormat.TIFF,
    ImageFormat.BMP,
    ImageFormat.GIF,
    ImageFormat.WEBP,
    ImageFormat.ICO,
    ImageFormat.EXR,
    ImageFormat.HDR,
    ImageFormat.PSD,
    ImageFormat.PSB,
    ImageFormat.RAW,
    ImageFormat.CR2,
    ImageFormat.CR3,
    ImageFormat.NEF,
    ImageFormat.ARW,
    ImageFormat.DNG,
    ImageFormat.ORF,
    ImageFormat.RW2,
    ImageFormat.SVG,
    ImageFormat.HEIC,
    ImageFormat.HEIF,
    ImageFormat.AVIF,
}

# Formats that support writing
WRITE_FORMATS = {
    ImageFormat.PNG,
    ImageFormat.JPEG,
    ImageFormat.TIFF,
    ImageFormat.BMP,
    ImageFormat.GIF,
    ImageFormat.WEBP,
    ImageFormat.ICO,
    ImageFormat.EXR,
    ImageFormat.HDR,
    ImageFormat.PSD,
    ImageFormat.PDF,
    ImageFormat.HEIC,
    ImageFormat.AVIF,
}

# Formats that support layers
LAYER_FORMATS = {
    ImageFormat.PSD,
    ImageFormat.PSB,
    ImageFormat.TIFF,
    ImageFormat.XCF,
}

# Formats that support alpha
ALPHA_FORMATS = {
    ImageFormat.PNG,
    ImageFormat.TIFF,
    ImageFormat.BMP,
    ImageFormat.WEBP,
    ImageFormat.ICO,
    ImageFormat.EXR,
    ImageFormat.PSD,
    ImageFormat.PSB,
    ImageFormat.GIF,
    ImageFormat.AVIF,
}

# Formats that support 16-bit
HIGH_BIT_DEPTH_FORMATS = {
    ImageFormat.PNG,
    ImageFormat.TIFF,
    ImageFormat.EXR,
    ImageFormat.HDR,
    ImageFormat.PSD,
    ImageFormat.PSB,
}
