# -*- coding: utf-8 -*-

"""Extract color palette from image."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from dreamstack.raster.color.palette.color import Color
from dreamstack.raster.color.palette.palette import Palette

if TYPE_CHECKING:
    from dreamstack.raster.core.image import Image


def extract_palette(
    image: "Image",
    num_colors: int = 8,
    method: str = "kmeans",
) -> Palette:
    """
    Extract a color palette from an image.

    Args:
        image: Source image
        num_colors: Number of colors to extract
        method: Extraction method ('kmeans', 'median_cut', 'octree')

    Returns:
        Extracted Palette
    """
    from sklearn.cluster import KMeans

    # Get RGB data
    rgb = image.to_rgb()
    data = rgb.data

    # Normalize to 0-1
    if data.dtype == np.uint8:
        data = data.astype(np.float32) / 255
    elif data.dtype == np.uint16:
        data = data.astype(np.float32) / 65535

    # Flatten to pixel list
    pixels = data.reshape(-1, 3)

    # Remove any fully transparent pixels
    if image.channels == 4:
        alpha = image.data[:, :, 3].flatten()
        if alpha.max() > 1:
            alpha = alpha / 255
        mask = alpha > 0.5
        pixels = pixels[mask]

    if method == "kmeans":
        # K-means clustering
        kmeans = KMeans(n_clusters=num_colors, random_state=42, n_init=10)
        kmeans.fit(pixels)
        centers = kmeans.cluster_centers_

    elif method == "median_cut":
        centers = _median_cut(pixels, num_colors)

    elif method == "octree":
        centers = _octree_quantize(pixels, num_colors)

    else:
        raise ValueError(f"Unknown method: {method}")

    # Create palette
    colors = []
    for center in centers:
        color = Color(
            int(center[0] * 255), int(center[1] * 255), int(center[2] * 255)
        )
        colors.append(color)

    palette = Palette(colors=colors)
    palette.sort_by_luminance()

    return palette


def _median_cut(pixels: np.ndarray, num_colors: int) -> np.ndarray:
    """Median cut color quantization."""

    def split_bucket(bucket):
        # Find channel with greatest range
        ranges = bucket.max(axis=0) - bucket.min(axis=0)
        channel = np.argmax(ranges)

        # Sort by that channel and split
        sorted_bucket = bucket[bucket[:, channel].argsort()]
        mid = len(sorted_bucket) // 2

        return sorted_bucket[:mid], sorted_bucket[mid:]

    buckets = [pixels]

    while len(buckets) < num_colors:
        # Find largest bucket
        largest_idx = max(range(len(buckets)), key=lambda i: len(buckets[i]))
        largest = buckets.pop(largest_idx)

        if len(largest) < 2:
            buckets.append(largest)
            break

        # Split it
        b1, b2 = split_bucket(largest)
        buckets.extend([b1, b2])

    # Get average color of each bucket
    centers = np.array([b.mean(axis=0) for b in buckets])

    return centers


def _octree_quantize(pixels: np.ndarray, num_colors: int) -> np.ndarray:
    """Simplified octree quantization (falls back to k-means)."""
    # Full octree implementation is complex
    # For now, use k-means as fallback
    from sklearn.cluster import KMeans

    kmeans = KMeans(n_clusters=num_colors, random_state=42, n_init=10)
    kmeans.fit(pixels)
    return kmeans.cluster_centers_
