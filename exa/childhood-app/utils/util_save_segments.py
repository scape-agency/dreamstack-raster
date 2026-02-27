"""
Save Segments Utility
=====================

Save grid segments to disk.
"""

from __future__ import annotations

from pathlib import Path

from models.model_grid_segment import GridSegment


def save_segments(
    segments: list[GridSegment],
    output_dir: Path | str,
    prefix: str = "",
) -> list[Path]:
    """Save segments to disk.

    Parameters
    ----------
    segments : list[GridSegment]
        Segments to save.
    output_dir : Path | str
        Output directory.
    prefix : str
        Filename prefix.

    Returns
    -------
    list[Path]
        Paths to saved files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_paths = []
    for seg in segments:
        filename = f"{prefix}{seg.filename}" if prefix else seg.filename
        path = output_dir / filename
        seg.image.save(path)
        saved_paths.append(path)

    return saved_paths
