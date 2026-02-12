"""
Removal Configuration
====================

Configuration dataclass for background removal operations.

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

# Available rembg models
ModelName = Literal[
    "u2net",
    "u2netp",
    "u2net_human_seg",
    "u2net_cloth_seg",
    "silueta",
    "isnet-general-use",
    "isnet-anime",
    "sam",
]


@dataclass
class RemovalConfig:
    """Configuration for background removal.

    Attributes:
        model_name: The rembg model to use for segmentation.
        alpha_matting: Enable alpha matting for refined edges.
        alpha_matting_foreground_threshold: Foreground threshold for matting.
        alpha_matting_background_threshold: Background threshold for matting.
        alpha_matting_erode_size: Erosion size for matting refinement.
        post_process_mask: Apply post-processing to the mask.
    """

    model_name: ModelName = "u2net"
    alpha_matting: bool = False
    alpha_matting_foreground_threshold: int = 240
    alpha_matting_background_threshold: int = 10
    alpha_matting_erode_size: int = 10
    post_process_mask: bool = False
