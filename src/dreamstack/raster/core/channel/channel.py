# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Channel
===========================

Single image channel representation.

"""


# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from dreamstack.raster.core.channel.channel_type import ChannelType

if TYPE_CHECKING:
    pass


@dataclass
class Channel:
    """
    Represents a single image channel.

    Channels can be used for color components, alpha, masks, or spot colors.

    Attributes:
        name: Display name for the channel
        channel_type: Type of channel
        data: 2D numpy array of channel values
        visible: Whether channel is visible
        color: Display color for channel preview (RGBA tuple)
    """

    name: str
    channel_type: ChannelType
    data: NDArray
    visible: bool = True
    color: tuple[int, int, int, int] = (255, 255, 255, 255)

    def __post_init__(self) -> None:
        """Validate channel data."""
        if self.data.ndim != 2:
            raise ValueError(f"Channel data must be 2D, got {self.data.ndim}D")

    @property
    def height(self) -> int:
        """Get channel height."""
        return self.data.shape[0]

    @property
    def width(self) -> int:
        """Get channel width."""
        return self.data.shape[1]

    @property
    def dtype(self) -> np.dtype:
        """Get data dtype."""
        return self.data.dtype

    def copy(self) -> Channel:
        """Create a deep copy of the channel."""
        return Channel(
            name=self.name,
            channel_type=self.channel_type,
            data=self.data.copy(),
            visible=self.visible,
            color=self.color,
        )

    def get_value(self, x: int, y: int) -> float:
        """Get channel value at coordinates."""
        if not (0 <= x < self.width and 0 <= y < self.height):
            raise IndexError(f"Coordinates ({x}, {y}) out of bounds")
        return float(self.data[y, x])

    def set_value(self, x: int, y: int, value: float) -> None:
        """Set channel value at coordinates."""
        if not (0 <= x < self.width and 0 <= y < self.height):
            raise IndexError(f"Coordinates ({x}, {y}) out of bounds")
        self.data[y, x] = value

    def fill(self, value: float) -> None:
        """Fill entire channel with value."""
        self.data.fill(value)

    def invert(self) -> None:
        """Invert channel values in-place."""
        if np.issubdtype(self.dtype, np.integer):
            max_val = np.iinfo(self.dtype).max
            self.data = max_val - self.data
        else:
            self.data = 1.0 - self.data

    def inverted(self) -> Channel:
        """Return new channel with inverted values."""
        copy = self.copy()
        copy.invert()
        return copy

    def apply_curve(self, curve: NDArray) -> None:
        """
        Apply a curve to channel values.

        Args:
            curve: 256-element lookup table for uint8, or callable for float
        """
        if np.issubdtype(self.dtype, np.integer):
            self.data = curve[self.data]
        else:
            # For float data, interpolate curve
            indices = (self.data * 255).astype(np.uint8)
            self.data = curve[indices].astype(self.dtype)

    def apply_levels(
        self,
        black_point: float = 0.0,
        white_point: float = 1.0,
        gamma: float = 1.0,
        output_black: float = 0.0,
        output_white: float = 1.0,
    ) -> None:
        """
        Apply levels adjustment to channel.

        Args:
            black_point: Input black point (0-1)
            white_point: Input white point (0-1)
            gamma: Gamma correction
            output_black: Output black point
            output_white: Output white point
        """
        # Normalize to float
        if np.issubdtype(self.dtype, np.integer):
            max_val = float(np.iinfo(self.dtype).max)
            data = self.data.astype(np.float32) / max_val
        else:
            data = self.data.astype(np.float32)

        # Apply input levels
        if white_point != black_point:
            data = (data - black_point) / (white_point - black_point)
        data = np.clip(data, 0, 1)

        # Apply gamma
        if gamma != 1.0:
            data = np.power(data, 1.0 / gamma)

        # Apply output levels
        data = output_black + data * (output_white - output_black)

        # Convert back to original dtype
        if np.issubdtype(self.dtype, np.integer):
            max_val = float(np.iinfo(self.dtype).max)
            self.data = (data * max_val).clip(0, max_val).astype(self.dtype)
        else:
            self.data = data.astype(self.dtype)

    def blend(
        self, other: Channel, opacity: float = 1.0, mask: NDArray | None = None
    ) -> None:
        """
        Blend another channel onto this one.

        Args:
            other: Channel to blend
            opacity: Blend opacity (0-1)
            mask: Optional mask for selective blending
        """
        if other.data.shape != self.data.shape:
            raise ValueError("Channel dimensions must match")

        # Convert to float for blending
        max_val: float = 1.0
        if np.issubdtype(self.dtype, np.integer):
            max_val = float(np.iinfo(self.dtype).max)
            self_float = self.data.astype(np.float32) / max_val
            other_float = other.data.astype(np.float32) / max_val
        else:
            self_float = self.data.astype(np.float32)
            other_float = other.data.astype(np.float32)

        # Apply mask and opacity
        blend_factor = np.full_like(self_float, opacity)
        if mask is not None:
            if np.issubdtype(mask.dtype, np.integer):
                mask_float = mask.astype(np.float32) / float(
                    np.iinfo(mask.dtype).max
                )
            else:
                mask_float = mask.astype(np.float32)
            blend_factor *= mask_float

        # Blend
        result = self_float + (other_float - self_float) * blend_factor

        # Convert back
        if np.issubdtype(self.dtype, np.integer):
            self.data = (result * max_val).clip(0, max_val).astype(self.dtype)
        else:
            self.data = result.astype(self.dtype)

    def histogram(self, bins: int = 256) -> NDArray:
        """
        Calculate histogram of channel values.

        Args:
            bins: Number of histogram bins

        Returns:
            Histogram counts as numpy array
        """
        if np.issubdtype(self.dtype, np.integer):
            max_val = np.iinfo(self.dtype).max + 1
            hist, _ = np.histogram(self.data, bins=bins, range=(0, max_val))
        else:
            hist, _ = np.histogram(self.data, bins=bins, range=(0, 1))
        return hist

    def statistics(self) -> dict:
        """
        Calculate channel statistics.

        Returns:
            Dictionary with min, max, mean, std, median
        """
        return {
            "min": float(np.min(self.data)),
            "max": float(np.max(self.data)),
            "mean": float(np.mean(self.data)),
            "std": float(np.std(self.data)),
            "median": float(np.median(self.data)),
        }

    def resize(
        self, width: int, height: int, method: str = "bilinear"
    ) -> Channel:
        """
        Resize channel to new dimensions.

        Args:
            width: New width
            height: New height
            method: Interpolation method

        Returns:
            New resized channel
        """
        import cv2

        interpolation_methods = {
            "nearest": getattr(cv2, "INTER_NEAREST"),
            "bilinear": getattr(cv2, "INTER_LINEAR"),
            "bicubic": getattr(cv2, "INTER_CUBIC"),
            "lanczos": getattr(cv2, "INTER_LANCZOS4"),
            "area": getattr(cv2, "INTER_AREA"),
        }

        interp = interpolation_methods.get(
            method.lower(), cv2.INTER_LINEAR
        )  # pylint: disable=no-member
        resized = cv2.resize(
            self.data, (width, height), interpolation=interp
        )  # pylint: disable=no-member

        return Channel(
            name=self.name,
            channel_type=self.channel_type,
            data=resized,
            visible=self.visible,
            color=self.color,
        )

    @classmethod
    def create(
        cls,
        width: int,
        height: int,
        channel_type: ChannelType = ChannelType.CUSTOM,
        name: str = "Channel",
        dtype: np.dtype | type = np.float32,
        fill_value: float = 0.0,
    ) -> Channel:
        """
        Create a new channel with specified dimensions.

        Args:
            width: Channel width
            height: Channel height
            channel_type: Type of channel
            name: Channel name
            dtype: Data type
            fill_value: Initial fill value

        Returns:
            New Channel instance
        """
        data = np.full((height, width), fill_value, dtype=dtype)

        # Set default display color based on type
        color_map = {
            ChannelType.RED: (255, 0, 0, 255),
            ChannelType.GREEN: (0, 255, 0, 255),
            ChannelType.BLUE: (0, 0, 255, 255),
            ChannelType.ALPHA: (255, 255, 255, 128),
            ChannelType.CYAN: (0, 255, 255, 255),
            ChannelType.MAGENTA: (255, 0, 255, 255),
            ChannelType.YELLOW: (255, 255, 0, 255),
            ChannelType.BLACK: (0, 0, 0, 255),
        }
        color = color_map.get(channel_type, (255, 255, 255, 255))

        return cls(
            name=name, channel_type=channel_type, data=data, color=color
        )

    @classmethod
    def from_array(
        cls,
        data: NDArray,
        channel_type: ChannelType = ChannelType.CUSTOM,
        name: str = "Channel",
    ) -> Channel:
        """
        Create channel from numpy array.

        Args:
            data: 2D numpy array
            channel_type: Type of channel
            name: Channel name

        Returns:
            New Channel instance
        """
        if data.ndim != 2:
            raise ValueError(f"Expected 2D array, got {data.ndim}D")

        return cls(name=name, channel_type=channel_type, data=data.copy())
