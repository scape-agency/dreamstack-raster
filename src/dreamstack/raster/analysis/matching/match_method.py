"""
MatchMethod Enum
================

Template matching methods.

"""

from __future__ import annotations

from enum import StrEnum


class MatchMethod(StrEnum):
    """Template matching methods.

    Attributes
    ----------
    SQDIFF : Squared difference (best match = minimum value)
    SQDIFF_NORMED : Normalized squared difference
    CCORR : Cross correlation (best match = maximum value)
    CCORR_NORMED : Normalized cross correlation
    CCOEFF : Coefficient correlation (best match = maximum value)
    CCOEFF_NORMED : Normalized coefficient correlation (recommended)
    """

    SQDIFF = "sqdiff"
    SQDIFF_NORMED = "sqdiff_normed"
    CCORR = "ccorr"
    CCORR_NORMED = "ccorr_normed"
    CCOEFF = "ccoeff"
    CCOEFF_NORMED = "ccoeff_normed"
