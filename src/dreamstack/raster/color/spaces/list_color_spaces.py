# -*- coding: utf-8 -*-

"""List available color spaces."""

from __future__ import annotations

from typing import List

from dreamstack.raster.color.spaces.get_color_space import _color_spaces


def list_color_spaces() -> List[str]:
    """Get list of available color space names."""
    return list(set(cs.name for cs in _color_spaces.values()))
