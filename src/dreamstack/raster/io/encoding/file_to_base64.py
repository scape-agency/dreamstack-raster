"""
File to Base64
==============

Read image file and encode to base64.

"""

from __future__ import annotations

import base64
from pathlib import Path


def file_to_base64(path: str | Path) -> str:
    """Read image file and encode to base64.

    Args:
        path: Path to image file.

    Returns:
        Base64 encoded string.
    """
    path = Path(path)
    return base64.b64encode(path.read_bytes()).decode("utf-8")
