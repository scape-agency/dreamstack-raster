"""
Dreamstack Raster - Metadata Handling
=====================================

Image metadata reading and writing.

"""

from dreamstack.raster.io.metadata.copy_metadata import copy_metadata
from dreamstack.raster.io.metadata.read_exif import read_exif
from dreamstack.raster.io.metadata.read_iptc import read_iptc
from dreamstack.raster.io.metadata.read_metadata import read_metadata
from dreamstack.raster.io.metadata.read_xmp import read_xmp
from dreamstack.raster.io.metadata.set_dpi import set_dpi
from dreamstack.raster.io.metadata.strip_metadata import strip_metadata
from dreamstack.raster.io.metadata.write_metadata import write_metadata

__all__ = [
    "read_metadata",
    "read_exif",
    "read_xmp",
    "read_iptc",
    "write_metadata",
    "set_dpi",
    "copy_metadata",
    "strip_metadata",
]
