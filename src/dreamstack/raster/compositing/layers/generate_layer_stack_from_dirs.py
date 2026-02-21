# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Layer stack generation from directories."""


# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

import itertools
from pathlib import Path

from .stack_layers import stack_layers


def generate_layer_stack_from_dirs(
    base_dir: str | Path,
    output_dir: str | Path,
    num_layers: int,
    variants_per_layer: int,
    *,
    image_format: str = "png",
    layer_prefix: str = "layer",
) -> int:
    """Generate all image combinations from layered directory structure.

    Expects directory structure:
        base_dir/
            layer0/
                0.png, 1.png, 2.png, ...
            layer1/
                0.png, 1.png, 2.png, ...
            ...  # pylint: disable=unnecessary-ellipsis

    Args:
        base_dir: Root directory containing layer subdirectories.
        output_dir: Output directory for combined images.
        num_layers: Number of layers to combine.
        variants_per_layer: Number of variants per layer.
        image_format: Image format extension.
        layer_prefix: Prefix for layer directory names.

    Returns:
        Number of generated combinations.

    Example:
        >>> count = generate_layer_stack_from_dirs(
        ...     "layers/",
        ...     "output/",
        ...     num_layers=3,
        ...     variants_per_layer=4,
        ... )
        >>> print(f"Generated {count} combinations")
    """
    import cv2

    base_dir = Path(base_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load all variants for each layer
    layers = []
    for i in range(num_layers):
        layer_dir = base_dir / f"{layer_prefix}{i}"
        variants = []

        for j in range(variants_per_layer):
            img_path = layer_dir / f"{j}.{image_format}"
            if img_path.exists():
                img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
                if img is not None:
                    variants.append(img)

        if variants:
            layers.append(variants)

    if not layers:
        return 0

    # Generate all combinations
    count = 0
    for idx, combo in enumerate(itertools.product(*layers)):
        result = stack_layers(list(combo))
        output_path = output_dir / f"combined_{idx}.{image_format}"
        cv2.imwrite(str(output_path), result)
        count += 1

    return count
