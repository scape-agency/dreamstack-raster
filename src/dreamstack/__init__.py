# -*- coding: utf-8 -*-


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

# Import | Standard Library
from pkgutil import extend_path
from typing import MutableSequence

# =============================================================================
# Variables
# =============================================================================

__path__: MutableSequence[str] = extend_path(
    path=__path__,  # type: ignore
    name=__name__,
)
