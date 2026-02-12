"""
Dreamstack Raster - Channel Manager
===================================

Multi-channel management for images.

"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from dreamstack.raster.core.channel.channel import Channel
from dreamstack.raster.core.channel.channel_type import ChannelType

if TYPE_CHECKING:
    from dreamstack.raster.core.pixel.pixel_data import PixelData
    from dreamstack.raster.core.pixel.pixel_format import PixelFormat


class ChannelManager:
    """
    Manages multiple channels for an image.

    Provides operations for adding, removing, and manipulating channels.
    """

    def __init__(self, width: int, height: int):
        """
        Initialize channel manager.

        Args:
            width: Image width
            height: Image height
        """
        self._width = width
        self._height = height
        self._channels: list[Channel] = []

    @property
    def width(self) -> int:
        """Get image width."""
        return self._width

    @property
    def height(self) -> int:
        """Get image height."""
        return self._height

    @property
    def count(self) -> int:
        """Get number of channels."""
        return len(self._channels)

    def __len__(self) -> int:
        return len(self._channels)

    def __getitem__(self, index: int | str) -> Channel:
        if isinstance(index, str):
            for channel in self._channels:
                if channel.name == index:
                    return channel
            raise KeyError(f"Channel '{index}' not found")
        return self._channels[index]

    def __iter__(self):
        return iter(self._channels)

    def add(self, channel: Channel) -> None:
        """
        Add a channel.

        Args:
            channel: Channel to add
        """
        if channel.width != self._width or channel.height != self._height:
            raise ValueError("Channel dimensions must match")
        self._channels.append(channel)

    def remove(self, index: int | str) -> Channel:
        """
        Remove a channel.

        Args:
            index: Channel index or name

        Returns:
            Removed channel
        """
        if isinstance(index, str):
            for i, channel in enumerate(self._channels):
                if channel.name == index:
                    return self._channels.pop(i)
            raise KeyError(f"Channel '{index}' not found")
        return self._channels.pop(index)

    def reorder(self, new_order: list[int]) -> None:
        """
        Reorder channels.

        Args:
            new_order: List of new indices
        """
        if len(new_order) != len(self._channels):
            raise ValueError("Order list must contain all channel indices")
        if set(new_order) != set(range(len(self._channels))):
            raise ValueError("Order list must contain each index exactly once")

        self._channels = [self._channels[i] for i in new_order]

    def duplicate(self, index: int | str, new_name: str | None = None) -> Channel:
        """
        Duplicate a channel.

        Args:
            index: Channel to duplicate
            new_name: Name for duplicate

        Returns:
            New duplicated channel
        """
        original = self[index]
        copy = original.copy()

        if new_name:
            copy.name = new_name
        else:
            copy.name = f"{original.name} copy"

        self._channels.append(copy)
        return copy

    def merge(self, indices: list[int], weights: list[float] | None = None) -> Channel:
        """
        Merge multiple channels into one.

        Args:
            indices: Channels to merge
            weights: Optional weights for each channel

        Returns:
            New merged channel
        """
        if weights is None:
            weights = [1.0 / len(indices)] * len(indices)

        if len(weights) != len(indices):
            raise ValueError("Weights must match number of channels")

        result = np.zeros((self._height, self._width), dtype=np.float32)

        for idx, weight in zip(indices, weights):
            channel = self._channels[idx]
            if np.issubdtype(channel.dtype, np.integer):
                max_val = float(np.iinfo(channel.dtype).max)
                result += (channel.data.astype(np.float32) / max_val) * weight
            else:
                result += channel.data.astype(np.float32) * weight

        result = np.clip(result, 0, 1)

        return Channel(name="Merged", channel_type=ChannelType.CUSTOM, data=result)

    def split_from_image(self, pixel_data: PixelData) -> None:
        """
        Split pixel data into individual channels.

        Args:
            pixel_data: PixelData to split
        """
        from dreamstack.raster.core.pixel.pixel_format import PixelFormat

        self._channels.clear()
        self._width = pixel_data.width
        self._height = pixel_data.height

        # Determine channel types based on pixel format
        channel_configs = {
            PixelFormat.GRAY: [(ChannelType.GRAY, "Gray")],
            PixelFormat.GRAY_ALPHA: [
                (ChannelType.GRAY, "Gray"),
                (ChannelType.ALPHA, "Alpha"),
            ],
            PixelFormat.RGB: [
                (ChannelType.RED, "Red"),
                (ChannelType.GREEN, "Green"),
                (ChannelType.BLUE, "Blue"),
            ],
            PixelFormat.RGBA: [
                (ChannelType.RED, "Red"),
                (ChannelType.GREEN, "Green"),
                (ChannelType.BLUE, "Blue"),
                (ChannelType.ALPHA, "Alpha"),
            ],
            PixelFormat.CMYK: [
                (ChannelType.CYAN, "Cyan"),
                (ChannelType.MAGENTA, "Magenta"),
                (ChannelType.YELLOW, "Yellow"),
                (ChannelType.BLACK, "Black"),
            ],
            PixelFormat.LAB: [
                (ChannelType.LIGHTNESS, "Lightness"),
                (ChannelType.A_CHANNEL, "A"),
                (ChannelType.B_CHANNEL, "B"),
            ],
            PixelFormat.HSV: [
                (ChannelType.HUE, "Hue"),
                (ChannelType.SATURATION, "Saturation"),
                (ChannelType.VALUE, "Value"),
            ],
            PixelFormat.HSL: [
                (ChannelType.HUE, "Hue"),
                (ChannelType.SATURATION, "Saturation"),
                (ChannelType.LUMINOSITY, "Luminosity"),
            ],
        }

        configs = channel_configs.get(pixel_data.pixel_format, [])

        for i, (ctype, name) in enumerate(configs):
            channel_data = pixel_data.get_channel(i)
            self._channels.append(
                Channel(name=name, channel_type=ctype, data=channel_data)
            )

    def combine_to_image(self, pixel_format: PixelFormat) -> PixelData:
        """
        Combine channels back into pixel data.

        Args:
            pixel_format: Target pixel format

        Returns:
            Combined PixelData
        """
        from dreamstack.raster.core.pixel.channel_count import CHANNEL_COUNT
        from dreamstack.raster.core.pixel.pixel_data import PixelData

        expected = CHANNEL_COUNT[pixel_format]
        if len(self._channels) != expected:
            raise ValueError(
                f"Need {expected} channels for {pixel_format.name}, have {len(self._channels)}"
            )

        # Stack channels
        channel_data = [ch.data for ch in self._channels]
        stacked = np.stack(channel_data, axis=2)

        return PixelData.from_numpy(stacked, pixel_format=pixel_format)
