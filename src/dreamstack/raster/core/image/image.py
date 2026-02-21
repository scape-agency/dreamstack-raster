# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Image Class
===============================

Core Image class representing a raster image with full editing capabilities.

"""


# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from dreamstack.raster.core.bounds import Bounds, Size
from dreamstack.raster.core.channel import ChannelManager
from dreamstack.raster.core.image.image_metadata import ImageMetadata
from dreamstack.raster.core.pixel import BitDepth, PixelData, PixelFormat

# =============================================================================
# Type Checking Imports
# =============================================================================

if TYPE_CHECKING:
    # pylint: disable=import-outside-toplevel
    from dreamstack.raster.selection import Selection


class Image:
    """
    Core Image class for raster image manipulation.

    This is the main entry point for working with images. It supports:
    - Multiple pixel formats (RGB, RGBA, CMYK, LAB, etc.)
    - Multiple bit depths (8-bit, 16-bit, 32-bit float)
    - Non-destructive editing via history
    - Channel manipulation
    - Selection support
    - Metadata handling

    Example:
        >>> image = Image.create(1920, 1080)
        >>> image.fill((255, 255, 255, 255))
        >>> image.save("output.png")

        >>> image = Image.open("photo.jpg")
        >>> image.adjust_brightness(1.2)
        >>> image.apply_filter("gaussian_blur", radius=5)
    """

    def __init__(
        self,
        pixel_data: PixelData,
        metadata: ImageMetadata | None = None,
        name: str = "Untitled",
    ):
        """
        Initialize an Image.

        Args:
            pixel_data: The pixel data
            metadata: Optional image metadata
            name: Image name/title
        """
        self._pixel_data = pixel_data
        self._metadata = metadata or ImageMetadata()
        self._name = name
        self._selection: Selection | None = None
        self._channels = ChannelManager(pixel_data.width, pixel_data.height)
        self._dirty = False
        self._path: Path | None = None

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def width(self) -> int:
        """Get image width in pixels."""
        return self._pixel_data.width

    @property
    def height(self) -> int:
        """Get image height in pixels."""
        return self._pixel_data.height

    @property
    def size(self) -> Size:
        """Get image size."""
        return Size(self.width, self.height)

    @property
    def bounds(self) -> Bounds:
        """Get image bounds."""
        return Bounds(0, 0, self.width, self.height)

    @property
    def pixel_format(self) -> PixelFormat:
        """Get pixel format."""
        return self._pixel_data.pixel_format

    @property
    def bit_depth(self) -> BitDepth:
        """Get bit depth."""
        return self._pixel_data.bit_depth

    @property
    def channels(self) -> int:
        """Get number of channels."""
        return self._pixel_data.channels

    @property
    def has_alpha(self) -> bool:
        """Check if image has alpha channel."""
        return self._pixel_data.has_alpha

    @property
    def pixel_data(self) -> PixelData:
        """Get pixel data."""
        return self._pixel_data

    @property
    def data(self) -> NDArray:
        """Get raw numpy array."""
        return self._pixel_data.data

    @property
    def metadata(self) -> ImageMetadata:
        """Get image metadata."""
        return self._metadata

    @property
    def name(self) -> str:
        """Get image name."""
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        """Set image name."""
        self._name = value

    @property
    def path(self) -> Path | None:
        """Get file path if loaded from disk."""
        return self._path

    @property
    def selection(self) -> Selection | None:
        """Get current selection."""
        return self._selection

    @selection.setter
    def selection(self, value: Selection | None) -> None:
        """Set current selection."""
        self._selection = value

    @property
    def is_dirty(self) -> bool:
        """Check if image has unsaved changes."""
        return self._dirty

    @property
    def aspect_ratio(self) -> float:
        """Get aspect ratio (width / height)."""
        return self.width / self.height if self.height > 0 else 0

    @property
    def megapixels(self) -> float:
        """Get megapixel count."""
        return (self.width * self.height) / 1_000_000

    # =========================================================================
    # Pixel Access
    # =========================================================================

    def get_pixel(self, x: int, y: int) -> NDArray:
        """
        Get pixel value at coordinates.

        Args:
            x: X coordinate
            y: Y coordinate

        Returns:
            Pixel values as numpy array
        """
        return self._pixel_data.get_pixel(x, y)

    def set_pixel(self, x: int, y: int, value: NDArray | list | tuple) -> None:
        """
        Set pixel value at coordinates.

        Args:
            x: X coordinate
            y: Y coordinate
            value: Pixel values
        """
        self._pixel_data.set_pixel(x, y, value)
        self._dirty = True

    def get_region(self, bounds: Bounds) -> PixelData:
        """
        Extract a rectangular region.

        Args:
            bounds: Region bounds

        Returns:
            PixelData for the region
        """
        return self._pixel_data.get_region(
            int(bounds.x), int(bounds.y), int(bounds.width), int(bounds.height)
        )

    def set_region(self, x: int, y: int, region: PixelData) -> None:
        """
        Set a rectangular region.

        Args:
            x: Left edge
            y: Top edge
            region: PixelData to copy
        """
        self._pixel_data.set_region(x, y, region)
        self._dirty = True

    # =========================================================================
    # Filling
    # =========================================================================

    def fill(self, color: NDArray | list | tuple) -> None:
        """
        Fill entire image with color.

        Args:
            color: Fill color (must match channel count)
        """
        self._pixel_data.fill(color)
        self._dirty = True

    def fill_selection(self, color: NDArray | list | tuple) -> None:
        """
        Fill current selection with color.

        Args:
            color: Fill color
        """
        if self._selection is None:
            self.fill(color)
            return

        # Apply fill through selection mask
        mask = self._selection.mask
        color_array = np.asarray(color, dtype=self.data.dtype)

        # Broadcast color to image shape
        color_broadcast = np.zeros_like(self.data)
        color_broadcast[:] = color_array

        # Apply masked fill
        for c in range(self.channels):
            self.data[:, :, c] = np.where(
                mask > 0, color_broadcast[:, :, c], self.data[:, :, c]
            )

        self._dirty = True

    # =========================================================================
    # Channel Operations
    # =========================================================================

    def get_channel(self, index: int) -> NDArray:
        """
        Get a single channel.

        Args:
            index: Channel index

        Returns:
            2D numpy array
        """
        return self._pixel_data.get_channel(index)

    def set_channel(self, index: int, values: NDArray) -> None:
        """
        Set a single channel.

        Args:
            index: Channel index
            values: 2D array of values
        """
        self._pixel_data.set_channel(index, values)
        self._dirty = True

    def get_alpha(self) -> NDArray | None:
        """Get alpha channel if present."""
        return self._pixel_data.get_alpha()

    def set_alpha(self, values: NDArray) -> None:
        """Set alpha channel."""
        self._pixel_data.set_alpha(values)
        self._dirty = True

    def split_channels(self) -> list[Image]:
        """
        Split image into separate single-channel images.

        Returns:
            List of grayscale images for each channel
        """
        images = []
        for i in range(self.channels):
            channel_data = self.get_channel(i)
            pixel_data = PixelData(
                data=channel_data[:, :, np.newaxis],
                pixel_format=PixelFormat.GRAY,
                bit_depth=self.bit_depth,
            )
            images.append(Image(pixel_data, name=f"{self.name} - Channel {i}"))
        return images

    @classmethod
    def merge_channels(
        cls, channels: list[Image], pixel_format: PixelFormat
    ) -> Image:
        """
        Merge multiple single-channel images into one.

        Args:
            channels: List of grayscale images
            pixel_format: Target pixel format

        Returns:
            Merged image
        """
        if not channels:
            raise ValueError("No channels to merge")

        # Stack channel data
        channel_arrays = [img.get_channel(0) for img in channels]
        stacked = np.stack(channel_arrays, axis=2)

        pixel_data = PixelData(
            data=stacked,
            pixel_format=pixel_format,
            bit_depth=channels[0].bit_depth,
        )

        return cls(pixel_data, name="Merged")

    # =========================================================================
    # Conversions
    # =========================================================================

    def convert_format(self, target_format: PixelFormat) -> Image:
        """
        Convert to different pixel format.

        Args:
            target_format: Target format

        Returns:
            New image in target format
        """
        converted = self._pixel_data.to_format(target_format)
        return Image(converted, self._metadata.copy(), self._name)

    def convert_bit_depth(self, target_depth: BitDepth) -> Image:
        """
        Convert to different bit depth.

        Args:
            target_depth: Target bit depth

        Returns:
            New image with target bit depth
        """
        converted = self._pixel_data.to_bit_depth(target_depth)
        return Image(converted, self._metadata.copy(), self._name)

    def to_rgb(self) -> Image:
        """Convert to RGB format."""
        return self.convert_format(PixelFormat.RGB)

    def to_rgba(self) -> Image:
        """Convert to RGBA format."""
        return self.convert_format(PixelFormat.RGBA)

    def to_grayscale(self) -> Image:
        """Convert to grayscale."""
        return self.convert_format(PixelFormat.GRAY)

    def to_numpy(self) -> NDArray:
        """Get image data as numpy array."""
        return self.data.copy()

    def to_pil(self):
        """
        Convert to PIL Image.

        Returns:
            PIL.Image object
        """
        # pylint: disable=import-outside-toplevel
        from PIL import Image as PILImage

        # Convert to 8-bit if needed
        if self.bit_depth != BitDepth.UINT8:
            converted = self.convert_bit_depth(BitDepth.UINT8)
            data = converted.data
        else:
            data = self.data

        # Determine PIL mode
        mode_map = {
            PixelFormat.GRAY: "L",
            PixelFormat.GRAY_ALPHA: "LA",
            PixelFormat.RGB: "RGB",
            PixelFormat.RGBA: "RGBA",
            PixelFormat.CMYK: "CMYK",
        }

        mode = mode_map.get(self.pixel_format)
        if mode is None:
            # Convert to RGB first
            converted = self.convert_format(PixelFormat.RGB)
            data = (
                converted.data
                if converted.bit_depth == BitDepth.UINT8
                else converted.convert_bit_depth(BitDepth.UINT8).data
            )
            mode = "RGB"

        # Handle single channel
        if data.ndim == 3 and data.shape[2] == 1:
            data = data[:, :, 0]

        return PILImage.fromarray(data, mode=mode)

    # =========================================================================
    # Copying
    # =========================================================================

    def copy(self) -> Image:
        """Create a deep copy of the image."""
        return Image(
            pixel_data=self._pixel_data.copy(),
            metadata=self._metadata.copy(),
            name=self._name,
        )

    def crop(self, bounds: Bounds) -> Image:
        """
        Crop image to bounds.

        Args:
            bounds: Crop region

        Returns:
            New cropped image
        """
        cropped = self.get_region(bounds)
        return Image(cropped, self._metadata.copy(), f"{self._name} (cropped)")

    # =========================================================================
    # Resizing
    # =========================================================================

    def resize(
        self,
        width: int,
        height: int,
        method: str = "lanczos",
        maintain_aspect: bool = False,
    ) -> Image:
        """
        Resize image.

        Args:
            width: New width
            height: New height
            method: Interpolation method (nearest, bilinear, bicubic, lanczos)
            maintain_aspect: If True, fit within dimensions maintaining aspect

        Returns:
            New resized image
        """
        import cv2  # pylint: disable=import-outside-toplevel

        if maintain_aspect:
            aspect = self.aspect_ratio
            target_aspect = width / height if height > 0 else 0

            if aspect > target_aspect:
                # Width is limiting factor
                height = int(width / aspect)
            else:
                # Height is limiting factor
                width = int(height * aspect)

        interpolation_methods = {
            "nearest": getattr(cv2, "INTER_NEAREST"),
            "bilinear": getattr(cv2, "INTER_LINEAR"),
            "bicubic": getattr(cv2, "INTER_CUBIC"),
            "lanczos": getattr(cv2, "INTER_LANCZOS4"),
            "area": getattr(cv2, "INTER_AREA"),
        }

        interp = interpolation_methods.get(
            method.lower(), cv2.INTER_LANCZOS4
        )  # pylint: disable=no-member

        # Use INTER_AREA for downscaling
        if width < self.width and height < self.height:
            interp = cv2.INTER_AREA  # pylint: disable=no-member

        resized = cv2.resize(
            self.data, (width, height), interpolation=interp
        )  # pylint: disable=no-member

        # Ensure 3D array
        if resized.ndim == 2:
            resized = resized[:, :, np.newaxis]

        pixel_data = PixelData(
            data=resized,
            pixel_format=self.pixel_format,
            bit_depth=self.bit_depth,
        )

        return Image(pixel_data, self._metadata.copy(), self._name)

    def scale(self, factor: float, method: str = "lanczos") -> Image:
        """
        Scale image by factor.

        Args:
            factor: Scale factor
            method: Interpolation method

        Returns:
            New scaled image
        """
        return self.resize(
            int(self.width * factor), int(self.height * factor), method=method
        )

    def thumbnail(self, max_size: int | tuple[int, int]) -> Image:
        """
        Create thumbnail fitting within max_size.

        Args:
            max_size: Maximum dimension(s)

        Returns:
            Thumbnail image
        """
        if isinstance(max_size, int):
            max_size = (max_size, max_size)

        return self.resize(max_size[0], max_size[1], maintain_aspect=True)

    # =========================================================================
    # Rotation and Flipping
    # =========================================================================

    def rotate(
        self,
        angle: float,
        expand: bool = True,
        fill_color: tuple | None = None,
    ) -> Image:
        """
        Rotate image by angle.

        Args:
            angle: Rotation angle in degrees (counter-clockwise)
            expand: If True, expand canvas to fit rotated image
            fill_color: Background fill color

        Returns:
            New rotated image
        """
        import cv2  # pylint: disable=import-outside-toplevel

        # Get rotation matrix
        center = (self.width / 2, self.height / 2)
        matrix = cv2.getRotationMatrix2D(
            center, angle, 1.0
        )  # pylint: disable=no-member

        if expand:
            # Calculate new bounds
            cos = abs(matrix[0, 0])
            sin = abs(matrix[0, 1])
            new_width = int(self.height * sin + self.width * cos)
            new_height = int(self.height * cos + self.width * sin)

            # Adjust matrix for new center
            matrix[0, 2] += (new_width - self.width) / 2
            matrix[1, 2] += (new_height - self.height) / 2
        else:
            new_width = self.width
            new_height = self.height

        # Default fill color
        if fill_color is None:
            fill_color = tuple([0] * self.channels)

        rotated = cv2.warpAffine(  # pylint: disable=no-member
            self.data,
            matrix,
            (new_width, new_height),
            borderMode=cv2.BORDER_CONSTANT,  # pylint: disable=no-member
            borderValue=fill_color,
        )

        if rotated.ndim == 2:
            rotated = rotated[:, :, np.newaxis]

        pixel_data = PixelData(
            data=rotated,
            pixel_format=self.pixel_format,
            bit_depth=self.bit_depth,
        )

        return Image(pixel_data, self._metadata.copy(), self._name)

    def rotate_90(self, times: int = 1) -> Image:
        """
        Rotate image by 90 degree increments.

        Args:
            times: Number of 90-degree rotations (1-3, negative for clockwise)

        Returns:
            New rotated image
        """
        times = times % 4
        if times == 0:
            return self.copy()

        rotated = np.rot90(self.data, k=times)

        pixel_data = PixelData(
            data=rotated.copy(),
            pixel_format=self.pixel_format,
            bit_depth=self.bit_depth,
        )

        return Image(pixel_data, self._metadata.copy(), self._name)

    def flip_horizontal(self) -> Image:
        """Flip image horizontally."""
        flipped = np.fliplr(self.data)

        pixel_data = PixelData(
            data=flipped.copy(),
            pixel_format=self.pixel_format,
            bit_depth=self.bit_depth,
        )

        return Image(pixel_data, self._metadata.copy(), self._name)

    def flip_vertical(self) -> Image:
        """Flip image vertically."""
        flipped = np.flipud(self.data)

        pixel_data = PixelData(
            data=flipped.copy(),
            pixel_format=self.pixel_format,
            bit_depth=self.bit_depth,
        )

        return Image(pixel_data, self._metadata.copy(), self._name)

    # =========================================================================
    # IO
    # =========================================================================

    def save(self, path: str | Path, **options) -> None:
        """
        Save image to file.

        Args:
            path: Output path
            **options: Format-specific options
        """
        # pylint: disable=import-outside-toplevel
        from dreamstack.raster.io import save_image

        save_image(self, path, **options)
        self._path = Path(path)
        self._dirty = False

    @classmethod
    def open(cls, path: str | Path, **options) -> Image:
        """
        Open image from file.

        Args:
            path: Input path
            **options: Format-specific options

        Returns:
            Loaded image
        """
        # pylint: disable=import-outside-toplevel
        from dreamstack.raster.io import load_image

        image = load_image(path, **options)
        image._path = Path(path)  # pylint: disable=protected-access
        return image

    @classmethod
    def create(
        cls,
        width: int,
        height: int,
        pixel_format: PixelFormat = PixelFormat.RGBA,
        bit_depth: BitDepth = BitDepth.UINT8,
        fill_color: tuple | None = None,
        name: str = "Untitled",
    ) -> Image:
        """
        Create a new blank image.

        Args:
            width: Image width
            height: Image height
            pixel_format: Pixel format
            bit_depth: Bit depth
            fill_color: Optional fill color
            name: Image name

        Returns:
            New blank image
        """
        pixel_data = PixelData.create(
            width=width,
            height=height,
            pixel_format=pixel_format,
            bit_depth=bit_depth,
            fill_value=fill_color,
        )

        return cls(pixel_data, name=name)

    @classmethod
    def from_numpy(
        cls,
        array: NDArray,
        pixel_format: PixelFormat | None = None,
        bit_depth: BitDepth | None = None,
        name: str = "Untitled",
    ) -> Image:
        """
        Create image from numpy array.

        Args:
            array: Input array
            pixel_format: Pixel format (auto-detected if None)
            bit_depth: Bit depth (auto-detected if None)
            name: Image name

        Returns:
            New image
        """
        pixel_data = PixelData.from_numpy(array, pixel_format, bit_depth)
        return cls(pixel_data, name=name)

    @classmethod
    def from_pil(cls, pil_image, name: str = "Untitled") -> Image:
        """
        Create image from PIL Image.

        Args:
            pil_image: PIL Image object
            name: Image name

        Returns:
            New image
        """
        array = np.array(pil_image)

        # Determine pixel format from PIL mode
        mode_to_format = {
            "L": PixelFormat.GRAY,
            "LA": PixelFormat.GRAY_ALPHA,
            "RGB": PixelFormat.RGB,
            "RGBA": PixelFormat.RGBA,
            "CMYK": PixelFormat.CMYK,
        }

        pixel_format = mode_to_format.get(pil_image.mode, PixelFormat.RGB)

        return cls.from_numpy(array, pixel_format=pixel_format, name=name)

    # =========================================================================
    # String Representation
    # =========================================================================

    def __repr__(self) -> str:
        return (
            f"Image({self.width}x{self.height}, "
            f"{self.pixel_format.name}, {self.bit_depth.name})"
        )

    def __str__(self) -> str:
        return f"{self._name}: {self.width}x{self.height} {self.pixel_format.name}"
