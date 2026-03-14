"""
Fetch By Type Utility
=====================

Search cutouts by object type.
"""

from __future__ import annotations

from pathlib import Path

from services.service_image_index import ImageIndex
from models.model_cutout_result import CutoutResult


def fetch_by_type(
    object_type: str,
    output_dir: str | Path = "./output",
    limit: int | None = None,
) -> list[CutoutResult]:
    """Search cutouts by object type.

    Parameters
    ----------
    object_type : str
        Object type to search for.
    output_dir : str | Path
        Output directory to search.
    limit : int | None
        Maximum results.

    Returns
    -------
    list[CutoutResult]
        Matching cutouts.
    """
    index = ImageIndex(output_dir)
    return index.search_type(object_type, limit)
