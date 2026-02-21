# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - ICC Profile class."""


# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

from dreamstack.raster.color.profiles.color_space_type import ColorSpaceType
from dreamstack.raster.color.profiles.profile_class import ProfileClass
from dreamstack.raster.color.profiles.rendering_intent import RenderingIntent


@dataclass
class ICCProfile:
    """
    ICC color profile.

    Attributes:
        data: Raw ICC profile data
        name: Profile description
        version: ICC version
        profile_class: Profile class
        color_space: Color space
        pcs: Profile connection space
        rendering_intent: Default rendering intent
    """

    data: bytes
    name: str = ""
    version: str = ""
    profile_class: ProfileClass | None = None
    color_space: ColorSpaceType | None = None
    pcs: str = ""
    rendering_intent: RenderingIntent = RenderingIntent.PERCEPTUAL
    copyright: str = ""
    manufacturer: str = ""
    model: str = ""

    def __post_init__(self):
        if self.data and not self.name:
            self._parse_header()

    def _parse_header(self):
        """Parse ICC profile header."""
        if len(self.data) < 128:
            return

        # Version
        v = struct.unpack(">I", self.data[8:12])[0]
        major = (v >> 24) & 0xFF
        minor = (v >> 20) & 0x0F
        self.version = f"{major}.{minor}"

        # Profile class
        class_sig = self.data[12:16].decode("ascii", errors="ignore")
        for pc in ProfileClass:
            if pc.value == class_sig:
                self.profile_class = pc
                break

        # Color space
        cs_sig = self.data[16:20].decode("ascii", errors="ignore")
        for cs in ColorSpaceType:
            if cs.value == cs_sig:
                self.color_space = cs
                break

        # PCS
        self.pcs = self.data[20:24].decode("ascii", errors="ignore")

        # Rendering intent
        intent = struct.unpack(">I", self.data[64:68])[0] & 0x03
        self.rendering_intent = RenderingIntent(intent)

        # Get description tag
        self.name = self._get_tag_text("desc")
        self.copyright = self._get_tag_text("cprt")

    def _get_tag_text(self, tag_name: str) -> str:
        """Extract text from a profile tag."""
        if len(self.data) < 132:
            return ""

        tag_count = struct.unpack(">I", self.data[128:132])[0]

        for i in range(tag_count):
            offset = 132 + i * 12
            if offset + 12 > len(self.data):
                break

            sig = self.data[offset : offset + 4].decode(
                "ascii", errors="ignore"
            )
            tag_offset = struct.unpack(
                ">I", self.data[offset + 4 : offset + 8]
            )[0]
            tag_size = struct.unpack(
                ">I", self.data[offset + 8 : offset + 12]
            )[0]

            if sig == tag_name or sig == tag_name.upper():
                if tag_offset + tag_size <= len(self.data):
                    tag_data = self.data[tag_offset : tag_offset + tag_size]
                    return self._parse_text_tag(tag_data)

        return ""

    def _parse_text_tag(self, tag_data: bytes) -> str:
        """Parse a text-type tag."""
        if len(tag_data) < 8:
            return ""

        type_sig = tag_data[0:4].decode("ascii", errors="ignore")

        if type_sig == "mluc":  # Multi-localized Unicode
            # Parse mluc structure
            record_count = struct.unpack(">I", tag_data[8:12])[0]
            if record_count > 0 and len(tag_data) >= 28:
                # Get first record
                str_offset = struct.unpack(">I", tag_data[24:28])[0]
                str_length = struct.unpack(">I", tag_data[20:24])[0]
                if str_offset + str_length <= len(tag_data):
                    try:
                        return (
                            tag_data[str_offset : str_offset + str_length]
                            .decode("utf-16-be")
                            .strip("\x00")
                        )
                    except UnicodeDecodeError:
                        pass

        elif type_sig == "desc":  # Old-style description
            # ASCII string at offset 8, preceded by count
            count = struct.unpack(">I", tag_data[8:12])[0]
            if count > 0 and len(tag_data) >= 12 + count:
                return tag_data[12 : 12 + count - 1].decode(
                    "ascii", errors="ignore"
                )

        elif type_sig == "text":  # Simple text
            return tag_data[8:].decode("ascii", errors="ignore").strip("\x00")

        return ""

    @classmethod
    def from_file(cls, path: str | Path) -> ICCProfile:
        """Load ICC profile from file."""
        path = Path(path)
        data = path.read_bytes()
        return cls(data=data)

    @classmethod
    def srgb(cls) -> ICCProfile:
        """Get sRGB profile."""
        # pylint: disable=import-outside-toplevel
        from dreamstack.raster.color.profiles.get_system_profiles import (
            _get_system_profile_paths,
        )

        # Try to load system sRGB profile
        paths = _get_system_profile_paths()
        for name, path in paths:
            if "srgb" in name.lower():
                try:
                    return cls.from_file(path)
                except (OSError, ValueError):  # File read/parse errors
                    continue

        # Generate minimal sRGB profile
        return cls._generate_srgb()

    @classmethod
    def _generate_srgb(cls) -> ICCProfile:
        """Generate minimal sRGB ICC profile."""
        # This is a simplified implementation
        # A full implementation would generate all required tags
        try:
            from PIL import ImageCms

            profile = ImageCms.createProfile("sRGB")
            data = ImageCms.ImageCmsProfile(profile).tobytes()
            return cls(data=data, name="sRGB IEC61966-2.1")
        except ImportError:
            return cls(data=b"", name="sRGB")

    def save(self, path: str | Path) -> None:
        """Save profile to file."""
        path = Path(path)
        path.write_bytes(self.data)

    def to_pil_profile(self):
        """Convert to PIL/Pillow ImageCmsProfile."""
        # pylint: disable=import-outside-toplevel
        from io import BytesIO

        # pylint: disable=import-outside-toplevel
        from PIL import ImageCms

        return ImageCms.ImageCmsProfile(BytesIO(self.data))
