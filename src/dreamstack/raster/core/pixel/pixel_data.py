# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Pixel Data
==============================

Main pixel data class for image manipulation.

"""


# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from dreamstack.raster.core.pixel.bit_depth import DTYPE_MAP, BitDepth
from dreamstack.raster.core.pixel.channel_count import CHANNEL_COUNT
from dreamstack.raster.core.pixel.pixel_format import PixelFormat


@dataclass
class PixelData:
    """
    Manages raw pixel data for images.

    This class wraps numpy arrays and provides convenient methods
    for pixel manipulation, color space conversion, and bit depth handling.

    Attributes:
        data: The underlying numpy array (height, width, channels)
        pixel_format: The color format of the pixels
        bit_depth: The bit depth of each channel
        premultiplied_alpha: Whether alpha is premultiplied
    """

    data: NDArray
    pixel_format: PixelFormat = PixelFormat.RGBA
    bit_depth: BitDepth = BitDepth.UINT8
    premultiplied_alpha: bool = False

    def __post_init__(self) -> None:
        """Validate and normalize pixel data."""
        if self.data.ndim == 2:
            # Grayscale - add channel dimension
            self.data = self.data[:, :, np.newaxis]

        if self.data.ndim != 3:
            raise ValueError(
                f"Data must be 2D or 3D array, got {self.data.ndim}D"
            )

        expected_channels = CHANNEL_COUNT[self.pixel_format]
        actual_channels = self.data.shape[2]

        if actual_channels != expected_channels:
            raise ValueError(
                f"Expected {expected_channels} channels for {self.pixel_format.name}, "
                f"got {actual_channels}"
            )

    @property
    def height(self) -> int:
        """Get pixel data height."""
        return self.data.shape[0]

    @property
    def width(self) -> int:
        """Get pixel data width."""
        return self.data.shape[1]

    @property
    def channels(self) -> int:
        """Get number of channels."""
        return self.data.shape[2]

    @property
    def shape(self) -> tuple[int, int, int]:
        """Get data shape (height, width, channels)."""
        return self.data.shape

    @property
    def dtype(self) -> np.dtype:
        """Get numpy dtype."""
        return self.data.dtype

    @property
    def has_alpha(self) -> bool:
        """Check if pixel format includes alpha channel."""
        return self.pixel_format in (PixelFormat.RGBA, PixelFormat.GRAY_ALPHA)

    @property
    def max_value(self) -> float:
        """Get maximum value for current bit depth."""
        if self.bit_depth == BitDepth.UINT8:
            return 255.0
        elif self.bit_depth == BitDepth.UINT16:
            return 65535.0
        else:
            return 1.0

    def copy(self) -> PixelData:
        """Create a deep copy of pixel data."""
        return PixelData(
            data=self.data.copy(),
            pixel_format=self.pixel_format,
            bit_depth=self.bit_depth,
            premultiplied_alpha=self.premultiplied_alpha,
        )

    def get_pixel(self, x: int, y: int) -> NDArray:
        """
        Get pixel value at coordinates.

        Args:
            x: X coordinate
            y: Y coordinate

        Returns:
            Pixel values as numpy array
        """
        if not (0 <= x < self.width and 0 <= y < self.height):
            raise IndexError(
                f"Pixel ({x}, {y}) out of bounds ({self.width}, {self.height})"
            )
        return self.data[y, x].copy()

    def set_pixel(self, x: int, y: int, value: NDArray | list | tuple) -> None:
        """
        Set pixel value at coordinates.

        Args:
            x: X coordinate
            y: Y coordinate
            value: Pixel values (must match channel count)
        """
        if not (0 <= x < self.width and 0 <= y < self.height):
            raise IndexError(
                f"Pixel ({x}, {y}) out of bounds ({self.width}, {self.height})"
            )

        value_array = np.asarray(value, dtype=self.dtype)
        if value_array.shape != (self.channels,):
            raise ValueError(
                f"Expected {self.channels} values, got {len(value)}"
            )

        self.data[y, x] = value_array

    def get_region(self, x: int, y: int, width: int, height: int) -> PixelData:
        """
        Extract a rectangular region.

        Args:
            x: Left edge
            y: Top edge
            width: Region width
            height: Region height

        Returns:
            New PixelData containing the region
        """
        # Clamp to valid bounds
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(self.width, x + width)
        y2 = min(self.height, y + height)

        region_data = self.data[y1:y2, x1:x2].copy()

        return PixelData(
            data=region_data,
            pixel_format=self.pixel_format,
            bit_depth=self.bit_depth,
            premultiplied_alpha=self.premultiplied_alpha,
        )

    def set_region(self, x: int, y: int, region: PixelData) -> None:
        """
        Set a rectangular region with pixel data.

        Args:
            x: Left edge
            y: Top edge
            region: PixelData to copy into region
        """
        # Calculate overlap region
        src_x1 = max(0, -x)
        src_y1 = max(0, -y)
        dst_x1 = max(0, x)
        dst_y1 = max(0, y)

        copy_width = min(region.width - src_x1, self.width - dst_x1)
        copy_height = min(region.height - src_y1, self.height - dst_y1)

        if copy_width <= 0 or copy_height <= 0:
            return

        src_x2 = src_x1 + copy_width
        src_y2 = src_y1 + copy_height
        dst_x2 = dst_x1 + copy_width
        dst_y2 = dst_y1 + copy_height

        self.data[dst_y1:dst_y2, dst_x1:dst_x2] = region.data[
            src_y1:src_y2, src_x1:src_x2
        ]

    def fill(self, value: NDArray | list | tuple) -> None:
        """
        Fill entire pixel data with a single value.

        Args:
            value: Fill value (must match channel count)
        """
        value_array = np.asarray(value, dtype=self.dtype)
        self.data[:, :] = value_array

    def fill_region(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        value: NDArray | list | tuple,
    ) -> None:
        """
        Fill a rectangular region with a value.

        Args:
            x: Left edge
            y: Top edge
            width: Region width
            height: Region height
            value: Fill value
        """
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(self.width, x + width)
        y2 = min(self.height, y + height)

        if x2 <= x1 or y2 <= y1:
            return

        value_array = np.asarray(value, dtype=self.dtype)
        self.data[y1:y2, x1:x2] = value_array

    def get_channel(self, index: int) -> NDArray:
        """
        Get a single channel as 2D array.

        Args:
            index: Channel index

        Returns:
            2D numpy array of channel values
        """
        if not 0 <= index < self.channels:
            raise IndexError(
                f"Channel {index} out of range (0-{self.channels - 1})"
            )
        return self.data[:, :, index].copy()

    def set_channel(self, index: int, values: NDArray) -> None:
        """
        Set a single channel.

        Args:
            index: Channel index
            values: 2D array of values
        """
        if not 0 <= index < self.channels:
            raise IndexError(
                f"Channel {index} out of range (0-{self.channels - 1})"
            )

        if values.shape != (self.height, self.width):
            raise ValueError(
                f"Channel shape mismatch: expected {(self.height, self.width)}, got {values.shape}"
            )

        self.data[:, :, index] = values

    def get_alpha(self) -> NDArray | None:
        """
        Get alpha channel if present.

        Returns:
            2D numpy array of alpha values, or None
        """
        if not self.has_alpha:
            return None
        return self.get_channel(self.channels - 1)

    def set_alpha(self, values: NDArray) -> None:
        """
        Set alpha channel.

        Args:
            values: 2D array of alpha values

        Raises:
            ValueError: If pixel format doesn't have alpha
        """
        if not self.has_alpha:
            raise ValueError(
                f"Pixel format {self.pixel_format.name} has no alpha channel"
            )
        self.set_channel(self.channels - 1, values)

    def to_normalized(self) -> PixelData:
        """
        Convert to normalized float (0.0 - 1.0) representation.

        Returns:
            New PixelData with float32 values in [0, 1]
        """
        if self.bit_depth in (
            BitDepth.FLOAT16,
            BitDepth.FLOAT32,
            BitDepth.FLOAT64,
        ):
            if self.bit_depth == BitDepth.FLOAT32:
                return self.copy()
            return PixelData(
                data=self.data.astype(np.float32),
                pixel_format=self.pixel_format,
                bit_depth=BitDepth.FLOAT32,
                premultiplied_alpha=self.premultiplied_alpha,
            )

        normalized = self.data.astype(np.float32) / self.max_value

        return PixelData(
            data=normalized,
            pixel_format=self.pixel_format,
            bit_depth=BitDepth.FLOAT32,
            premultiplied_alpha=self.premultiplied_alpha,
        )

    def to_bit_depth(self, target_depth: BitDepth) -> PixelData:
        """
        Convert to a different bit depth.

        Args:
            target_depth: Target bit depth

        Returns:
            New PixelData with converted bit depth
        """
        if self.bit_depth == target_depth:
            return self.copy()

        # First normalize to float
        normalized = self.to_normalized().data

        # Get target dtype and max value
        target_dtype = DTYPE_MAP[target_depth]

        if target_depth == BitDepth.UINT8:
            converted = (normalized * 255).clip(0, 255).astype(target_dtype)
        elif target_depth == BitDepth.UINT16:
            converted = (
                (normalized * 65535).clip(0, 65535).astype(target_dtype)
            )
        else:
            converted = normalized.astype(target_dtype)

        return PixelData(
            data=converted,
            pixel_format=self.pixel_format,
            bit_depth=target_depth,
            premultiplied_alpha=self.premultiplied_alpha,
        )

    def premultiply_alpha(self) -> PixelData:
        """
        Premultiply RGB by alpha.

        Returns:
            New PixelData with premultiplied alpha
        """
        if not self.has_alpha or self.premultiplied_alpha:
            return self.copy()

        normalized = self.to_normalized()
        alpha = normalized.data[:, :, -1:]

        result = normalized.data.copy()
        result[:, :, :-1] *= alpha

        return PixelData(
            data=result,
            pixel_format=self.pixel_format,
            bit_depth=BitDepth.FLOAT32,
            premultiplied_alpha=True,
        ).to_bit_depth(self.bit_depth)

    def unpremultiply_alpha(self) -> PixelData:
        """
        Convert from premultiplied to straight alpha.

        Returns:
            New PixelData with straight (non-premultiplied) alpha
        """
        if not self.has_alpha or not self.premultiplied_alpha:
            return self.copy()

        normalized = self.to_normalized()
        alpha = normalized.data[:, :, -1:]

        result = normalized.data.copy()
        # Avoid division by zero
        mask = alpha > 0.0001
        result[:, :, :-1] = np.where(mask, result[:, :, :-1] / alpha, 0)

        return PixelData(
            data=result,
            pixel_format=self.pixel_format,
            bit_depth=BitDepth.FLOAT32,
            premultiplied_alpha=False,
        ).to_bit_depth(self.bit_depth)

    def to_format(self, target_format: PixelFormat) -> PixelData:
        """
        Convert to different pixel format.

        Args:
            target_format: Target pixel format

        Returns:
            New PixelData in target format
        """
        # pylint: disable=import-outside-toplevel
        if self.pixel_format == target_format:
            return self.copy()

        normalized = self.to_normalized()
        data = normalized.data

        # Extract alpha if present
        if self.has_alpha:
            alpha = data[:, :, -1]
            color_data = data[:, :, :-1]
        else:
            alpha = np.ones((self.height, self.width), dtype=np.float32)
            color_data = data

        # Convert color channels based on source format
        if self.pixel_format in (PixelFormat.GRAY, PixelFormat.GRAY_ALPHA):
            if color_data.ndim == 3:
                gray = color_data[:, :, 0]
            else:
                gray = color_data
            rgb = np.stack([gray, gray, gray], axis=2)
        elif self.pixel_format in (PixelFormat.RGB, PixelFormat.RGBA):
            rgb = (
                color_data
                if color_data.shape[2] == 3
                else color_data[:, :, :3]
            )
        elif self.pixel_format == PixelFormat.HSV:
            # HSV to RGB conversion
            from dreamstack.raster.color.convert import hsv_to_rgb

            rgb = hsv_to_rgb(color_data)
        elif self.pixel_format == PixelFormat.HSL:
            from dreamstack.raster.color.convert import hsl_to_rgb

            rgb = hsl_to_rgb(color_data)
        elif self.pixel_format == PixelFormat.LAB:
            from dreamstack.raster.color.convert import lab_to_rgb

            rgb = lab_to_rgb(color_data)
        elif self.pixel_format == PixelFormat.CMYK:
            from dreamstack.raster.color.convert import cmyk_to_rgb

            rgb = cmyk_to_rgb(color_data)
        else:
            raise NotImplementedError(
                f"Conversion from {self.pixel_format} not implemented"
            )

        # Convert to target format
        if target_format == PixelFormat.GRAY:
            # Luminosity formula
            result = (
                0.299 * rgb[:, :, 0]
                + 0.587 * rgb[:, :, 1]
                + 0.114 * rgb[:, :, 2]
            )
            result = result[:, :, np.newaxis]
        elif target_format == PixelFormat.GRAY_ALPHA:
            gray = (
                0.299 * rgb[:, :, 0]
                + 0.587 * rgb[:, :, 1]
                + 0.114 * rgb[:, :, 2]
            )
            result = np.stack([gray, alpha], axis=2)
        elif target_format == PixelFormat.RGB:
            result = rgb
        elif target_format == PixelFormat.RGBA:
            result = np.concatenate([rgb, alpha[:, :, np.newaxis]], axis=2)
        elif target_format == PixelFormat.HSV:
            from dreamstack.raster.color.convert import rgb_to_hsv

            result = rgb_to_hsv(rgb)
        elif target_format == PixelFormat.HSL:
            from dreamstack.raster.color.convert import rgb_to_hsl

            result = rgb_to_hsl(rgb)
        elif target_format == PixelFormat.LAB:
            from dreamstack.raster.color.convert import rgb_to_lab

            result = rgb_to_lab(rgb)
        elif target_format == PixelFormat.CMYK:
            from dreamstack.raster.color.convert import rgb_to_cmyk

            result = rgb_to_cmyk(rgb)
        else:
            raise NotImplementedError(
                f"Conversion to {target_format} not implemented"
            )

        return PixelData(
            data=result.astype(np.float32),
            pixel_format=target_format,
            bit_depth=BitDepth.FLOAT32,
            premultiplied_alpha=False,
        ).to_bit_depth(self.bit_depth)

    @classmethod
    def create(
        cls,
        width: int,
        height: int,
        pixel_format: PixelFormat = PixelFormat.RGBA,
        bit_depth: BitDepth = BitDepth.UINT8,
        fill_value: NDArray | list | tuple | None = None,
    ) -> PixelData:
        """
        Create new pixel data with specified dimensions.

        Args:
            width: Width in pixels
            height: Height in pixels
            pixel_format: Pixel format
            bit_depth: Bit depth
            fill_value: Optional fill value (default is transparent/black)

        Returns:
            New PixelData instance
        """
        channels = CHANNEL_COUNT[pixel_format]
        dtype = DTYPE_MAP[bit_depth]

        data = np.zeros((height, width, channels), dtype=dtype)

        if fill_value is not None:
            fill_array = np.asarray(fill_value, dtype=dtype)
            data[:, :] = fill_array

        return cls(data=data, pixel_format=pixel_format, bit_depth=bit_depth)

    @classmethod
    def from_numpy(
        cls,
        array: NDArray,
        pixel_format: PixelFormat | None = None,
        bit_depth: BitDepth | None = None,
    ) -> PixelData:
        """
        Create PixelData from numpy array.

        Args:
            array: Input numpy array
            pixel_format: Pixel format (auto-detected if None)
            bit_depth: Bit depth (auto-detected if None)

        Returns:
            New PixelData instance
        """
        # Detect bit depth from dtype
        if bit_depth is None:
            if array.dtype == np.uint8:
                bit_depth = BitDepth.UINT8
            elif array.dtype == np.uint16:
                bit_depth = BitDepth.UINT16
            elif array.dtype == np.float16:
                bit_depth = BitDepth.FLOAT16
            elif array.dtype == np.float32:
                bit_depth = BitDepth.FLOAT32
            elif array.dtype == np.float64:
                bit_depth = BitDepth.FLOAT64
            else:
                # Convert to float32
                array = array.astype(np.float32)
                bit_depth = BitDepth.FLOAT32

        # Ensure correct dtype
        target_dtype = DTYPE_MAP[bit_depth]
        if array.dtype != target_dtype:
            array = array.astype(target_dtype)

        # Detect pixel format from shape
        if pixel_format is None:
            if array.ndim == 2:
                pixel_format = PixelFormat.GRAY
            elif array.ndim == 3:
                channels = array.shape[2]
                if channels == 1:
                    pixel_format = PixelFormat.GRAY
                elif channels == 2:
                    pixel_format = PixelFormat.GRAY_ALPHA
                elif channels == 3:
                    pixel_format = PixelFormat.RGB
                elif channels == 4:
                    pixel_format = PixelFormat.RGBA
                else:
                    raise ValueError(
                        f"Cannot detect format for {channels} channels"
                    )
            else:
                raise ValueError(f"Expected 2D or 3D array, got {array.ndim}D")

        return cls(data=array, pixel_format=pixel_format, bit_depth=bit_depth)
