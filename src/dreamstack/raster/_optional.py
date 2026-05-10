# -*- coding: utf-8 -*-

"""Helpers for guarding optional dependencies behind clear import errors."""

from __future__ import annotations

import importlib
from types import ModuleType


def require(module_name: str, *, extra: str, feature: str) -> ModuleType:
    """Import ``module_name`` or raise an actionable ``ImportError``.

    The raised message tells the user which ``pip install`` extra to use,
    so optional backends always fail with a single, consistent message.

    Example
    -------
    >>> rawpy = require("rawpy", extra="raw", feature="RAW image loading")
    """
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        raise ImportError(
            f"{feature} requires the optional '{module_name}' package. "
            f"Install it with: pip install 'dreamstack-raster[{extra}]'"
        ) from exc
