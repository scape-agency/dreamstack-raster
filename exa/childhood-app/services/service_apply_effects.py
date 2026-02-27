"""
Apply Effects Service
=====================

Apply visual effects to image segments.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

if TYPE_CHECKING:
    from numpy.typing import NDArray

from models.model_effect_config import EffectConfig
from models.model_effect_type import EffectType
from models.model_effect_result import EffectResult

logger = logging.getLogger(__name__)


def apply_effects(
    image: Image.Image | NDArray | Path | str,
    config: EffectConfig | None = None,
    effects: list[EffectType | str] | None = None,
) -> EffectResult:
    """Apply visual effects to an image.

    Parameters
    ----------
    image : Image.Image | NDArray | Path | str
        Input image.
    config : EffectConfig | None
        Effect configuration. Uses defaults if None.
    effects : list[EffectType | str] | None
        Specific effects to apply. If None, uses config.filters.

    Returns
    -------
    EffectResult
        Processed image and list of applied effects.
    """
    config = config or EffectConfig()

    # Handle different input types
    if isinstance(image, (str, Path)):
        image = Image.open(image)
    elif isinstance(image, np.ndarray):
        image = numpy_to_pil(image)

    # Ensure RGBA
    if image.mode != "RGBA":
        image = image.convert("RGBA")

    applied: list[str] = []

    # Apply drop shadow first if enabled
    if config.drop_shadow:
        image = apply_drop_shadow(
            image,
            offset=config.shadow_offset,
            blur=config.shadow_blur,
            opacity=config.shadow_opacity,
        )
        applied.append("drop_shadow")

    # Determine which effects to apply
    effect_list = effects if effects is not None else config.filters
    if isinstance(effect_list, str):
        effect_list = [effect_list]

    # Apply each effect
    for effect in effect_list:
        effect_name = (
            effect.value if isinstance(effect, EffectType) else effect
        )

        if effect_name == "warm_filter":
            image = apply_warm_filter(image)
            applied.append("warm_filter")
        elif effect_name == "cool_filter":
            image = apply_cool_filter(image)
            applied.append("cool_filter")
        elif effect_name == "vintage":
            image = apply_vintage(image)
            applied.append("vintage")
        elif effect_name == "high_contrast":
            image = apply_high_contrast(image)
            applied.append("high_contrast")
        elif effect_name == "soft_glow":
            image = apply_soft_glow(image)
            applied.append("soft_glow")
        elif effect_name == "sharpen":
            image = apply_sharpen(image)
            applied.append("sharpen")
        elif effect_name == "blur":
            image = apply_blur(image)
            applied.append("blur")
        elif effect_name == "sepia":
            image = apply_sepia(image)
            applied.append("sepia")

    return EffectResult(image=image, effects_applied=applied)


def numpy_to_pil(arr: NDArray) -> Image.Image:
    """Convert numpy array to PIL Image."""
    if len(arr.shape) == 2:
        return Image.fromarray(arr)
    elif arr.shape[2] == 3:
        # BGR to RGB
        return Image.fromarray(arr[:, :, ::-1])
    elif arr.shape[2] == 4:
        # BGRA to RGBA
        return Image.fromarray(
            np.concatenate([arr[:, :, 2::-1], arr[:, :, 3:4]], axis=2)
        )
    return Image.fromarray(arr)


def apply_drop_shadow(
    image: Image.Image,
    offset: tuple[int, int] = (5, 5),
    blur: int = 10,
    opacity: float = 0.5,
    color: tuple[int, int, int] = (0, 0, 0),
) -> Image.Image:
    """Apply drop shadow effect.

    Uses dreamstack.raster.effects.shadow if available,
    falls back to PIL implementation.
    """
    try:
        # Try dreamstack implementation
        from dreamstack.raster.effects.shadow import drop_shadow

        # Convert to numpy
        arr = np.array(image)
        # RGB to BGR for OpenCV
        if arr.shape[2] == 4:
            bgra = np.concatenate([arr[:, :, 2::-1], arr[:, :, 3:4]], axis=2)
        else:
            bgra = arr[:, :, ::-1]

        result = drop_shadow(
            bgra,
            offset=offset,
            blur=float(blur),
            color=color,
            opacity=opacity,
        )

        # Convert back to RGB
        if result.shape[2] == 4:
            rgba = np.concatenate(
                [result[:, :, 2::-1], result[:, :, 3:4]], axis=2
            )
        else:
            rgba = result[:, :, ::-1]

        return Image.fromarray(rgba)

    except ImportError:
        logger.debug("Using PIL drop shadow fallback")
        return pil_drop_shadow(image, offset, blur, opacity, color)


def pil_drop_shadow(
    image: Image.Image,
    offset: tuple[int, int],
    blur: int,
    opacity: float,
    color: tuple[int, int, int],
) -> Image.Image:
    """PIL-based drop shadow implementation."""
    # Get alpha channel
    if image.mode != "RGBA":
        image = image.convert("RGBA")

    # Create shadow from alpha
    alpha = image.split()[3]

    # Create shadow image
    shadow = Image.new("RGBA", image.size, (*color, 0))
    shadow.putalpha(alpha)

    # Apply opacity
    shadow_alpha = shadow.split()[3]
    shadow_alpha = shadow_alpha.point(lambda x: int(x * opacity))
    shadow.putalpha(shadow_alpha)

    # Apply blur
    shadow = shadow.filter(ImageFilter.GaussianBlur(blur))

    # Create larger canvas for shadow offset
    new_width = image.width + abs(offset[0]) + blur * 2
    new_height = image.height + abs(offset[1]) + blur * 2

    result = Image.new("RGBA", (new_width, new_height), (0, 0, 0, 0))

    # Position shadow
    shadow_x = blur + max(0, offset[0])
    shadow_y = blur + max(0, offset[1])
    result.paste(shadow, (shadow_x, shadow_y), shadow)

    # Position original image
    img_x = blur + max(0, -offset[0])
    img_y = blur + max(0, -offset[1])
    result.paste(image, (img_x, img_y), image)

    return result


def apply_warm_filter(image: Image.Image) -> Image.Image:
    """Apply warm color filter (orange/yellow tint)."""
    if image.mode != "RGBA":
        image = image.convert("RGBA")

    # Split channels
    r, g, b, a = image.split()

    # Increase reds, decrease blues
    # Using lookup tables for type safety with PIL point()
    warm_r_table = [min(255, int(i * 1.1)) for i in range(256)]
    cool_b_table = [int(i * 0.9) for i in range(256)]
    r = r.point(warm_r_table)
    b = b.point(cool_b_table)

    return Image.merge("RGBA", (r, g, b, a))


def apply_cool_filter(image: Image.Image) -> Image.Image:
    """Apply cool color filter (blue tint)."""
    if image.mode != "RGBA":
        image = image.convert("RGBA")

    r, g, b, a = image.split()

    # Decrease reds, increase blues
    # Using lookup tables for type safety with PIL point()
    cool_r_table = [int(i * 0.9) for i in range(256)]
    warm_b_table = [min(255, int(i * 1.1)) for i in range(256)]
    r = r.point(cool_r_table)
    b = b.point(warm_b_table)

    return Image.merge("RGBA", (r, g, b, a))


def apply_vintage(image: Image.Image) -> Image.Image:
    """Apply vintage/retro effect."""
    if image.mode != "RGBA":
        image = image.convert("RGBA")

    # Reduce saturation
    enhancer = ImageEnhance.Color(image)
    image = enhancer.enhance(0.7)

    # Add slight sepia - using lookup tables for type safety
    r, g, b, a = image.split()
    vintage_r_table = [min(255, int(i * 1.05)) for i in range(256)]
    vintage_g_table = [int(i * 0.95) for i in range(256)]
    vintage_b_table = [int(i * 0.85) for i in range(256)]
    r = r.point(vintage_r_table)
    g = g.point(vintage_g_table)
    b = b.point(vintage_b_table)

    # Reduce contrast slightly
    result = Image.merge("RGBA", (r, g, b, a))
    enhancer = ImageEnhance.Contrast(result)
    return enhancer.enhance(0.9)


def apply_high_contrast(image: Image.Image) -> Image.Image:
    """Apply high contrast effect."""
    enhancer = ImageEnhance.Contrast(image)
    return enhancer.enhance(1.5)


def apply_soft_glow(image: Image.Image) -> Image.Image:
    """Apply soft glow effect."""
    if image.mode != "RGBA":
        image = image.convert("RGBA")

    # Create blurred version
    blurred = image.filter(ImageFilter.GaussianBlur(5))

    # Lighten the blur
    enhancer = ImageEnhance.Brightness(blurred)
    blurred = enhancer.enhance(1.2)

    # Blend with original
    return Image.blend(image, blurred, 0.3)


def apply_sharpen(image: Image.Image) -> Image.Image:
    """Apply sharpening filter."""
    return image.filter(ImageFilter.SHARPEN)


def apply_blur(image: Image.Image) -> Image.Image:
    """Apply blur filter."""
    return image.filter(ImageFilter.GaussianBlur(2))


def apply_sepia(image: Image.Image) -> Image.Image:
    """Apply sepia tone effect."""
    if image.mode != "RGBA":
        image = image.convert("RGBA")

    # Convert to grayscale first
    gray = image.convert("L")

    # Apply sepia tone - using lookup tables for type safety
    sepia_r_table = [min(255, int(i * 1.07)) for i in range(256)]
    sepia_g_table = [min(255, int(i * 0.87)) for i in range(256)]
    sepia_b_table = [min(255, int(i * 0.67)) for i in range(256)]
    sepia_r = gray.point(sepia_r_table)
    sepia_g = gray.point(sepia_g_table)
    sepia_b = gray.point(sepia_b_table)

    # Preserve original alpha
    a = image.split()[3]

    return Image.merge("RGBA", (sepia_r, sepia_g, sepia_b, a))
