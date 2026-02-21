# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Read IPTC Metadata
======================================

Read IPTC metadata from image files.

"""


# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

from pathlib import Path
from typing import Any


def read_iptc(path: str | Path) -> dict[str, Any]:
    """
    Read IPTC metadata from an image.

    Args:
        path: Path to image file

    Returns:
        Dictionary of IPTC data
    """
    from PIL import Image as PILImage
    from PIL import IptcImagePlugin

    path = Path(path)
    result = {}

    try:
        with PILImage.open(path) as img:
            iptc = IptcImagePlugin.getiptcinfo(img)
            if iptc:
                # IPTC tag names
                tag_names = {
                    (2, 5): "ObjectName",
                    (2, 25): "Keywords",
                    (2, 55): "DateCreated",
                    (2, 80): "By-line",
                    (2, 90): "City",
                    (2, 95): "State",
                    (2, 101): "Country",
                    (2, 105): "Headline",
                    (2, 116): "Copyright",
                    (2, 120): "Caption",
                }

                for tag, value in iptc.items():
                    tag_name = tag_names.get(tag, f"({tag[0]}, {tag[1]})")

                    if isinstance(value, bytes):
                        try:
                            value = value.decode("utf-8")
                        except UnicodeDecodeError:
                            continue

                    result[tag_name] = value
    except Exception:
        pass

    return result
