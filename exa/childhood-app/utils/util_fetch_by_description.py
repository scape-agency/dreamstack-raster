"""
Fetch By Description Utility
============================

Search images by description.
"""

from __future__ import annotations

from pathlib import Path

from services.service_image_index import ImageIndex
from models.model_search_result import SearchResult


def fetch_by_description(
    query: str,
    output_dir: str | Path = "./output",
    limit: int | None = None,
) -> list[SearchResult]:
    """Search images by description.

    Parameters
    ----------
    query : str
        Search query.
    output_dir : str | Path
        Output directory to search.
    limit : int | None
        Maximum results.

    Returns
    -------
    list[SearchResult]
        Matching results.
    """
    index = ImageIndex(output_dir)
    return index.search_description(query, limit)
