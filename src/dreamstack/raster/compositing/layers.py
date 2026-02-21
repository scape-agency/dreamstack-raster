# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Layer Stacking Operations
=========================

Multi-layer image compositing and combination utilities.

"""


# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

import itertools
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    # pylint: disable=import-outside-toplevel
    from numpy.typing import NDArray


@dataclass
class LayerStackConfig:
    """Configuration for layer stacking.

    Attributes:
        output_format: Output image format.
        resize_to_base: Resize all layers to match base layer.
        fill_color: Background fill color (RGBA).
    """

    output_format: str = "png"
    resize_to_base: bool = True
    fill_color: tuple[int, int, int, int] = (0, 0, 0, 0)


def stack_layers(
    layers: Sequence[NDArray[np.uint8]],
    *,
    resize_to_base: bool = True,
) -> NDArray[np.uint8]:
    """Stack multiple image layers using alpha compositing.

    Layers are composited from bottom to top (first layer is the base).
    All layers should be RGBA for proper alpha blending.

    Args:
        layers: Sequence of RGBA images (4 channels).
        resize_to_base: Resize all layers to match the first layer's size.

    Returns:
        Composited RGBA image.

    Example:
        >>> from dreamstack.raster.compositing import stack_layers
        >>> result = stack_layers([background, midground, foreground])
    """
    import cv2

    if not layers:
        raise ValueError("At least one layer required")

    # Start with base layer
    base = layers[0].copy()

    # Ensure RGBA
    if base.ndim == 2:
        base = np.asarray(
            cv2.cvtColor(
                base, cv2.COLOR_GRAY2RGBA
            ),  # pylint: disable=no-member
            dtype=np.uint8,
        )
    elif base.shape[2] == 3:
        base = np.asarray(
            cv2.cvtColor(
                base, cv2.COLOR_BGR2BGRA
            ),  # pylint: disable=no-member
            dtype=np.uint8,
        )

    base = base.astype(np.float32)
    h, w = base.shape[:2]

    # Composite each layer
    for layer in layers[1:]:
        overlay = layer.copy()

        # Ensure RGBA
        if overlay.ndim == 2:
            overlay = np.asarray(
                cv2.cvtColor(
                    overlay, cv2.COLOR_GRAY2RGBA
                ),  # pylint: disable=no-member
                dtype=np.uint8,
            )
        elif overlay.shape[2] == 3:
            overlay = np.asarray(
                cv2.cvtColor(
                    overlay, cv2.COLOR_BGR2BGRA
                ),  # pylint: disable=no-member
                dtype=np.uint8,
            )

        # Resize if needed
        if resize_to_base and overlay.shape[:2] != (h, w):
            overlay = np.asarray(
                cv2.resize(
                    overlay, (w, h), interpolation=cv2.INTER_LINEAR
                ),  # pylint: disable=no-member
                dtype=np.uint8,
            )

        overlay = overlay.astype(np.float32)

        # Alpha compositing (Porter-Duff over)
        src_alpha = overlay[:, :, 3:4] / 255.0
        dst_alpha = base[:, :, 3:4] / 255.0

        out_alpha = src_alpha + dst_alpha * (1 - src_alpha)

        # Avoid division by zero
        safe_alpha = np.where(out_alpha > 0, out_alpha, 1)

        rgb = (
            overlay[:, :, :3] * src_alpha
            + base[:, :, :3] * dst_alpha * (1 - src_alpha)
        ) / safe_alpha

        base[:, :, :3] = rgb
        base[:, :, 3:4] = out_alpha * 255

    return base.astype(np.uint8)


def generate_layer_combinations(
    layer_variants: Sequence[Sequence[NDArray[np.uint8]]],
) -> Iterator[NDArray[np.uint8]]:
    """Generate all combinations of layer variants.

    Given multiple layers each with multiple variants, generates all
    possible combinations by stacking one variant from each layer.

    Args:
        layer_variants: List of layers, each containing list of variant images.

    Yields:
        Composited images for each combination.

    Example:
        >>> # 3 variants each for 3 layers = 27 combinations
        >>> variants = [
        ...     [bg1, bg2, bg3],        # Background variants
        ...     [mid1, mid2, mid3],     # Midground variants
        ...     [fg1, fg2, fg3],        # Foreground variants
        ... ]
        >>> for combo in generate_layer_combinations(variants):
        ...     save_image(combo, f"combo_{i}.png")
    """
    for combo in itertools.product(*layer_variants):
        yield stack_layers(list(combo))


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
                img = cv2.imread(
                    str(img_path), cv2.IMREAD_UNCHANGED
                )  # pylint: disable=no-member
                if img is not None:
                    variants.append(np.asarray(img, dtype=np.uint8))

        if variants:
            layers.append(variants)

    if not layers:
        return 0

    # Generate all combinations
    count = 0
    for idx, combo in enumerate(itertools.product(*layers)):
        result = stack_layers(list(combo))
        output_path = output_dir / f"combined_{idx}.{image_format}"
        cv2.imwrite(str(output_path), result)  # pylint: disable=no-member
        count += 1

    return count


def composite_with_mask(
    foreground: NDArray[np.uint8],
    background: NDArray[np.uint8],
    mask: NDArray[np.uint8],
) -> NDArray[np.uint8]:
    """Composite foreground onto background using a mask.

    Args:
        foreground: Foreground image (RGB/RGBA).
        background: Background image (RGB/RGBA).
        mask: Grayscale mask (white = foreground, black = background).

    Returns:
        Composited image.

    Example:
        >>> result = composite_with_mask(person, scene, person_mask)
    """
    import cv2

    # Ensure same size
    h, w = background.shape[:2]
    if foreground.shape[:2] != (h, w):
        foreground = np.asarray(
            cv2.resize(foreground, (w, h)),  # pylint: disable=no-member
            dtype=np.uint8,
        )
    if mask.shape[:2] != (h, w):
        mask = np.asarray(
            cv2.resize(mask, (w, h)),  # pylint: disable=no-member
            dtype=np.uint8,
        )

    # Ensure mask is single channel
    if mask.ndim == 3:
        mask = np.asarray(
            cv2.cvtColor(
                mask, cv2.COLOR_BGR2GRAY
            ),  # pylint: disable=no-member
            dtype=np.uint8,
        )

    # Normalize mask to 0-1
    alpha = mask.astype(np.float32) / 255.0

    # Ensure same channel count
    if foreground.ndim == 2:
        foreground = np.asarray(
            cv2.cvtColor(
                foreground, cv2.COLOR_GRAY2BGR
            ),  # pylint: disable=no-member
            dtype=np.uint8,
        )
    if background.ndim == 2:
        background = np.asarray(
            cv2.cvtColor(
                background, cv2.COLOR_GRAY2BGR
            ),  # pylint: disable=no-member
            dtype=np.uint8,
        )

    # Handle alpha channels
    if foreground.shape[2] == 4:
        foreground = foreground[:, :, :3]
    if background.shape[2] == 4:
        background = background[:, :, :3]

    # Expand alpha for broadcasting
    alpha = alpha[:, :, np.newaxis]

    # Blend
    result = foreground.astype(np.float32) * alpha + background.astype(
        np.float32
    ) * (1 - alpha)

    return result.astype(np.uint8)


def apply_alpha_from_mask(
    image: NDArray[np.uint8],
    mask: NDArray[np.uint8],
) -> NDArray[np.uint8]:
    """Apply mask as alpha channel to create RGBA image.

    Args:
        image: Input RGB image (3 channels).
        mask: Grayscale mask (white = opaque).

    Returns:
        RGBA image with mask as alpha.

    Example:
        >>> rgba = apply_alpha_from_mask(rgb_image, transparency_mask)
    """
    import cv2

    # Ensure image is 3 channel
    if image.ndim == 2:
        image = np.asarray(
            cv2.cvtColor(
                image, cv2.COLOR_GRAY2BGR
            ),  # pylint: disable=no-member
            dtype=np.uint8,
        )
    elif image.shape[2] == 4:
        image = image[:, :, :3]

    # Ensure mask is single channel
    if mask.ndim == 3:
        mask = np.asarray(
            cv2.cvtColor(
                mask, cv2.COLOR_BGR2GRAY
            ),  # pylint: disable=no-member
            dtype=np.uint8,
        )

    # Resize mask if needed
    if mask.shape[:2] != image.shape[:2]:
        mask = np.asarray(
            cv2.resize(
                mask, (image.shape[1], image.shape[0])
            ),  # pylint: disable=no-member
            dtype=np.uint8,
        )

    # Create RGBA
    rgba = np.dstack([image, mask])

    return rgba
