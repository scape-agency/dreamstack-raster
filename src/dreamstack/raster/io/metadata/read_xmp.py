"""
Dreamstack Raster - Read XMP Metadata
=====================================

Read XMP metadata from image files.

"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def read_xmp(path: str | Path) -> dict[str, Any]:
    """
    Read XMP metadata from an image.

    Args:
        path: Path to image file

    Returns:
        Dictionary of XMP data
    """
    import xml.etree.ElementTree as ET

    from PIL import Image as PILImage

    path = Path(path)
    result = {}

    with PILImage.open(path) as img:
        # Check for XMP in info
        if "XML:com.adobe.xmp" in img.info:
            xmp_str = img.info["XML:com.adobe.xmp"]
        elif hasattr(img, "applist"):
            # Look in APP1 segments
            xmp_str = None
            for segment in img.applist:  # type: ignore[attr-defined]
                if (
                    segment[0] == "APP1"
                    and b"http://ns.adobe.com/xap/" in segment[1]
                ):
                    xmp_str = segment[1].decode("utf-8", errors="ignore")
                    break
            if xmp_str is None:
                return {}
        else:
            return {}

        # Parse XMP XML
        try:
            # Extract XMP packet
            start = xmp_str.find("<x:xmpmeta")
            end = xmp_str.find("</x:xmpmeta>") + len("</x:xmpmeta>")
            if start >= 0 and end > start:
                xmp_xml = xmp_str[start:end]
                root = ET.fromstring(xmp_xml)

                # Extract common namespaces
                namespaces = {
                    "dc": "http://purl.org/dc/elements/1.1/",
                    "xmp": "http://ns.adobe.com/xap/1.0/",
                    "xmpRights": "http://ns.adobe.com/xap/1.0/rights/",
                    "photoshop": "http://ns.adobe.com/photoshop/1.0/",
                    "lr": "http://ns.adobe.com/lightroom/1.0/",
                }

                # Extract values
                for ns_name, ns_uri in namespaces.items():
                    for elem in root.iter():
                        if elem.tag.startswith("{" + ns_uri + "}"):
                            tag_name = elem.tag.split("}")[1]
                            if elem.text and elem.text.strip():
                                result[f"{ns_name}:{tag_name}"] = (
                                    elem.text.strip()
                                )
        except ET.ParseError:
            pass

    return result
