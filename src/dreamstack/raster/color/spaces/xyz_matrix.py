# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Compute RGB to XYZ conversion matrix."""


# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

import numpy as np


def _compute_rgb_to_xyz_matrix(
    primaries: np.ndarray, white_point: np.ndarray
) -> np.ndarray:
    """
    Compute RGB to XYZ conversion matrix from primaries and white point.
    """

    # Convert xy to XYZ
    def xy_to_XYZ(xy):
        x, y = xy
        return np.array([x / y, 1, (1 - x - y) / y])

    # Primary matrix (columns are XYZ of primaries)
    P = np.column_stack(
        [
            xy_to_XYZ(primaries[0]),
            xy_to_XYZ(primaries[1]),
            xy_to_XYZ(primaries[2]),
        ]
    )

    # White point XYZ
    W = xy_to_XYZ(white_point)

    # Solve for scaling factors
    S = np.linalg.solve(P, W)

    # Final matrix
    M = P * S

    return M
