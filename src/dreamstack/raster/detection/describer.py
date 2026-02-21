# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Image Describer
===============

AI-powered image description using vision models.
Supports OpenAI (GPT-4o) and Mistral (Pixtral) backends.
"""


# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

# =============================================================================
# Type Checking Imports
# =============================================================================

if TYPE_CHECKING:
    import numpy as np

    # pylint: disable=import-outside-toplevel
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


Backend = Literal["openai", "mistral"]


@dataclass
class DescriptionConfig:
    """Configuration for image description.

    Attributes
    ----------
    backend : Backend
        Which AI backend to use. Default "openai".
    model : str | None
        Model name. None = use default (gpt-4o / pixtral-12b).
    max_tokens : int
        Maximum tokens in response. Default 1024.
    detail : str
        Image detail level for OpenAI. Default "auto".
    """

    backend: Backend = "openai"
    model: str | None = None
    max_tokens: int = 1024
    detail: str = "auto"  # OpenAI only: 'low', 'high', 'auto'


@dataclass
class DescriptionResult:
    """Result from image description.

    Attributes
    ----------
    description : str
        Full text description of the image.
    objects : list[str]
        List of detected object names/categories.
    source_path : Path | None
        Path to source image.
    model : str
        Model used for description.
    """

    description: str
    objects: list[str] = field(default_factory=list)
    source_path: Path | None = None
    model: str = ""

    def to_prompt(self) -> str:
        """Convert objects to Grounding DINO prompt.

        Returns comma-separated lowercase object names.
        """
        if self.objects:
            return ", ".join(sorted(set(obj.lower() for obj in self.objects)))
        return ""


# Default prompt for extracting objects
DEFAULT_DESCRIPTION_PROMPT = """Analyze this image and describe what you see.

List ALL visible objects, people, body parts, accessories, and items in detail.

For people, include:
- Body parts visible (face, hands, arms, legs, etc.)
- Facial features (eyes, nose, mouth, ears, eyebrows)
- Accessories (glasses, earrings, necklaces, watches, rings, hats, etc.)
- Clothing items (shirt, pants, dress, shoes, etc.)

For other objects, include:
- Main objects and their parts
- Background elements
- Text or logos visible

Format your response as:
DESCRIPTION: [A detailed description of the scene]

OBJECTS: [comma-separated list of all detectable objects, be specific]

Example OBJECTS format: person, face, eye, ear, earring, hand, ring, chair, table, plant, window
"""


class ImageDescriber:
    """AI-powered image description.

    Uses vision models to describe images and extract
    object names for downstream detection.

    Example
    -------
    >>> from dreamstack.raster.detection.describer import ImageDescriber
    >>>
    >>> describer = ImageDescriber()
    >>> result = describer.describe("photo.jpg")
    >>> print(result.description)
    >>> print(result.objects)  # ['person', 'face', 'earring', ...]
    >>> print(result.to_prompt())  # 'earring, face, person, ...'

    Notes
    -----
    Requires:
    - OpenAI backend: pip install chimp-openai (+ OPENAI_API_KEY)
    - Mistral backend: pip install chimp-mistral (+ MISTRAL_API_KEY)
    """

    def __init__(self, config: DescriptionConfig | None = None) -> None:
        """Initialize describer.

        Parameters
        ----------
        config : DescriptionConfig | None
            Configuration. Uses defaults if None.
        """
        self.config = config or DescriptionConfig()
        self._chat_with_vision = None
        self._default_model: str | None = None

    def _load_backend(self) -> None:
        """Lazy load the vision backend."""
        if self._chat_with_vision is not None:
            return

        if self.config.backend == "openai":
            try:
                # pylint: disable=import-outside-toplevel
                from chimp.openai import (
                    chat_with_vision,
                )  # type: ignore[import-not-found]

                self._chat_with_vision = chat_with_vision
                self._default_model = "gpt-4o"
            except ImportError as exc:
                raise ImportError(
                    "chimp-openai is required for OpenAI vision. "
                    "Install with: pip install chimp-openai "
                    "or poetry install --extras vision-openai"
                ) from exc

        elif self.config.backend == "mistral":
            try:
                # pylint: disable=import-outside-toplevel
                from chimp.mistral import (
                    chat_with_vision,
                )  # type: ignore[import-not-found]

                self._chat_with_vision = chat_with_vision
                self._default_model = "pixtral-12b-2409"
            except ImportError as exc:
                raise ImportError(
                    "chimp-mistral is required for Mistral vision. "
                    "Install with: pip install chimp-mistral "
                    "or poetry install --extras vision-mistral"
                ) from exc
        else:
            raise ValueError(f"Unknown backend: {self.config.backend}")

    def describe(
        self,
        image_path: str | Path,
        prompt: str | None = None,
    ) -> DescriptionResult:
        """Describe an image using AI vision.

        Parameters
        ----------
        image_path : str | Path
            Path to image file.
        prompt : str | None
            Custom prompt. Uses default if None.

        Returns
        -------
        DescriptionResult
            Description and extracted objects.
        """
        self._load_backend()

        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        prompt = prompt or DEFAULT_DESCRIPTION_PROMPT
        model = self.config.model or self._default_model

        logger.info(
            "Describing image with %s: %s",
            self.config.backend,
            image_path.name,
        )

        # Call vision API with retry for rate limiting
        import time  # pylint: disable=import-outside-toplevel

        max_retries = 3
        retry_delay = 2.0  # seconds

        for attempt in range(max_retries):
            try:
                if self.config.backend == "openai":
                    response = self._chat_with_vision(  # type: ignore[misc]
                        prompt=prompt,
                        image_paths=[str(image_path)],
                        model=model,
                        max_tokens=self.config.max_tokens,
                        detail=self.config.detail,
                    )
                else:  # mistral
                    response = self._chat_with_vision(  # type: ignore[misc]
                        prompt=prompt,
                        image_paths=[str(image_path)],
                        model=model,
                        max_tokens=self.config.max_tokens,
                    )
                break  # Success
            except Exception as e:  # pylint: disable=broad-exception-caught
                error_str = str(e).lower()
                if "429" in error_str or "rate" in error_str:
                    if attempt < max_retries - 1:
                        wait_time = retry_delay * (2**attempt)
                        logger.warning(
                            "Rate limited, waiting %.1fs (attempt %d/%d)",
                            wait_time,
                            attempt + 1,
                            max_retries,
                        )
                        time.sleep(wait_time)
                    else:
                        raise
                else:
                    raise

        # Parse response
        description, objects = self._parse_response(response)

        return DescriptionResult(
            description=description,
            objects=objects,
            source_path=image_path,
            model=model,
        )

    def _parse_response(self, response: str) -> tuple[str, list[str]]:
        """Parse AI response into description and objects.

        Parameters
        ----------
        response : str
            Raw response from vision API.

        Returns
        -------
        tuple[str, list[str]]
            (description, list of objects)
        """
        description = ""
        objects: list[str] = []

        lines = response.strip().split("\n")

        for i, line in enumerate(lines):
            line_upper = line.upper().strip()

            if line_upper.startswith("DESCRIPTION:"):
                # Get description text
                desc_text = line[len("DESCRIPTION:") :].strip()
                # Include following lines until OBJECTS
                for j in range(i + 1, len(lines)):
                    if lines[j].upper().strip().startswith("OBJECTS:"):
                        break
                    desc_text += " " + lines[j].strip()
                description = desc_text.strip()

            elif line_upper.startswith("OBJECTS:"):
                # Parse comma-separated objects
                obj_text = line[len("OBJECTS:") :].strip()
                # Include following lines
                for j in range(i + 1, len(lines)):
                    if (
                        lines[j]
                        .upper()
                        .strip()
                        .startswith(("DESCRIPTION:", "OBJECTS:"))
                    ):
                        break
                    obj_text += " " + lines[j].strip()

                # Split and clean
                raw_objects = obj_text.split(",")
                objects = [
                    obj.strip().lower() for obj in raw_objects if obj.strip()
                ]

        # Fallback: if no structured format, use full response as description
        if not description and not objects:
            description = response
            # Try to extract nouns (basic fallback)
            objects = self._extract_nouns_fallback(response)

        return description, objects

    def _extract_nouns_fallback(self, text: str) -> list[str]:
        """Simple fallback to extract potential object names.

        Parameters
        ----------
        text : str
            Text to extract from.

        Returns
        -------
        list[str]
            Potential object names.
        """
        # Common object words to look for
        common_objects = {
            "person",
            "people",
            "man",
            "woman",
            "child",
            "face",
            "eye",
            "eyes",
            "ear",
            "ears",
            "nose",
            "mouth",
            "hand",
            "hands",
            "arm",
            "arms",
            "leg",
            "legs",
            "hair",
            "glasses",
            "earring",
            "earrings",
            "necklace",
            "ring",
            "watch",
            "hat",
            "shirt",
            "dress",
            "pants",
            "shoes",
            "bag",
            "dog",
            "cat",
            "car",
            "chair",
            "table",
            "phone",
            "book",
            "plant",
            "tree",
            "flower",
            "building",
            "window",
            "door",
            "sky",
            "cloud",
        }

        words = text.lower().split()
        found = []

        for word in words:
            # Clean punctuation
            clean = "".join(c for c in word if c.isalnum())
            if clean in common_objects and clean not in found:
                found.append(clean)

        return found


def describe_image(
    image_path: str | Path,
    backend: Backend = "openai",
    prompt: str | None = None,
) -> DescriptionResult:
    """Convenience function to describe an image.

    Parameters
    ----------
    image_path : str | Path
        Path to image file.
    backend : Backend
        Which AI backend to use.
    prompt : str | None
        Custom prompt.

    Returns
    -------
    DescriptionResult
        Description and objects.

    Example
    -------
    >>> result = describe_image("photo.jpg")
    >>> print(result.to_prompt())
    'earring, face, hand, person'
    """
    config = DescriptionConfig(backend=backend)
    describer = ImageDescriber(config)
    return describer.describe(image_path, prompt)
