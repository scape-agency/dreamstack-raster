"""
Dreamstack Raster - Image Metadata
==================================

Image metadata storage class.

"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ImageMetadata:
    """
    Stores image metadata.

    Attributes:
        dpi: Resolution in dots per inch
        color_profile: ICC color profile name
        copyright: Copyright information
        author: Author/creator name
        description: Image description
        keywords: List of keywords/tags
        creation_date: Creation timestamp
        modification_date: Last modification timestamp
        software: Software used to create/edit
        exif: EXIF metadata dictionary
        custom: Custom metadata dictionary
    """

    dpi: tuple[float, float] = (72.0, 72.0)
    color_profile: str = "sRGB"
    copyright: str = ""
    author: str = ""
    description: str = ""
    keywords: list[str] = field(default_factory=list)
    creation_date: str = ""
    modification_date: str = ""
    software: str = "Dreamstack Raster"
    exif: dict = field(default_factory=dict)
    custom: dict = field(default_factory=dict)

    def copy(self) -> ImageMetadata:
        """Create a copy of metadata."""
        return ImageMetadata(
            dpi=self.dpi,
            color_profile=self.color_profile,
            copyright=self.copyright,
            author=self.author,
            description=self.description,
            keywords=self.keywords.copy(),
            creation_date=self.creation_date,
            modification_date=self.modification_date,
            software=self.software,
            exif=self.exif.copy(),
            custom=self.custom.copy(),
        )
