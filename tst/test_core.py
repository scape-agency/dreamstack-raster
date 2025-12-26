# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Test Core Module
================

This module contains tests for the core functionalities of the Dreamstack library.
Examples:
    Running the test::

        $ pytest tst/test_core.py

"""


# =============================================================================
# Imports
# =============================================================================

# Import | Local
from dreamstack.core import hello

# =============================================================================
# Tests
# =============================================================================


def test_greet() -> None:
    """
    Test the hello function from dreamstack.core module.
    """
    assert hello("World") == "Hello, World! Welcome to the dreamstack library."


# =============================================================================
# Exports
# =============================================================================

__all__: list[str] = [
    "test_greet",
]
