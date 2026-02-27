"""
Describe Image Service
======================

Describe image using AI vision.
"""

from __future__ import annotations

from pathlib import Path


def describe_image(
    image_path: Path,
    backend: str = "openai",
) -> tuple[str, list[str]]:
    """Describe image using AI vision.

    Parameters
    ----------
    image_path : Path
        Path to the image file.
    backend : str
        AI vision backend to use. Default "openai".

    Returns
    -------
    tuple[str, list[str]]
        A tuple of (description, object_list).
    """
    from dreamstack.raster.detection.describer import (
        DescriptionConfig,
        ImageDescriber,
    )

    config = DescriptionConfig(backend=backend)  # type: ignore[arg-type]
    describer = ImageDescriber(config)

    result = describer.describe(image_path)
    return result.description, result.objects
