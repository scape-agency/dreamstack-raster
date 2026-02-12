# =============================================================================
# Docstring
# =============================================================================

"""
Sturnus - Dreamstack Initialization
===================================

This module initializes the Dreamstack package.

"""


# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

from collections.abc import MutableSequence

# Import | Standard Library
from pkgutil import extend_path

# =============================================================================
# Variables
# =============================================================================

__path__: MutableSequence[str] = extend_path(
    path=__path__,  # type: ignore
    name=__name__,
)
