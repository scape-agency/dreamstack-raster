"""
Rembg Availability Check
========================

Internal helper to check if rembg is available.

"""

from __future__ import annotations


def _check_rembg_available() -> bool:
    """Check if rembg is available."""
    try:
        import rembg  # noqa: F401  # type: ignore[import-not-found]  # pylint: disable=W0611

        return True
    except ImportError:
        return False
