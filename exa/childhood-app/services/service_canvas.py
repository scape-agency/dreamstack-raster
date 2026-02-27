"""
Canvas Service
==============

Canvas for placing image segments.
"""

from __future__ import annotations

import json
import logging
import math
import random
import time
from pathlib import Path
from typing import Callable, Literal

import numpy as np
from PIL import Image

from models.model_placed_item import PlacedItem

logger = logging.getLogger(__name__)

# Placement order strategies
PlacementOrder = Literal[
    "sequential", "random", "center-out", "edge-in", "diagonal"
]


def sort_segments_by_order(
    segments: list[dict],
    order: PlacementOrder,
    cutout_size: tuple[int, int],
) -> list[dict]:
    """Sort segments according to placement order strategy.

    Parameters
    ----------
    segments : list[dict]
        List of segment metadata dictionaries.
    order : PlacementOrder
        Ordering strategy to apply.
    cutout_size : tuple[int, int]
        Size of the cutout (width, height) for calculating positions.

    Returns
    -------
    list[dict]
        Sorted/shuffled list of segments.
    """
    if order == "sequential":
        return segments  # Keep original order

    if order == "random":
        shuffled = segments.copy()
        random.shuffle(shuffled)
        return shuffled

    # Calculate center of cutout
    center_x = cutout_size[0] / 2
    center_y = cutout_size[1] / 2

    def distance_from_center(seg: dict) -> float:
        pos = seg.get("position", [0, 0])
        size = seg.get("size", [0, 0])
        seg_center_x = pos[0] + size[0] / 2
        seg_center_y = pos[1] + size[1] / 2
        return math.sqrt(
            (seg_center_x - center_x) ** 2 + (seg_center_y - center_y) ** 2
        )

    if order == "center-out":
        return sorted(segments, key=distance_from_center)

    if order == "edge-in":
        return sorted(segments, key=distance_from_center, reverse=True)

    if order == "diagonal":
        # Sort by sum of row and col (diagonal wave pattern)
        def diagonal_key(seg: dict) -> tuple[int, int]:
            pos = seg.get("position", [0, 0])
            return (pos[0] + pos[1], pos[0])

        return sorted(segments, key=diagonal_key)

    return segments


class Canvas:
    """Canvas for placing image segments.

    Uses bottom-left coordinate system (like art canvases).
    Y=0 is at the bottom, X=0 is at the left.

    Parameters
    ----------
    width : int
        Canvas width in pixels.
    height : int
        Canvas height in pixels.
    background : tuple[int, int, int, int]
        RGBA background color.
    """

    def __init__(
        self,
        width: int = 8000,
        height: int = 3000,
        background: tuple[int, int, int, int] = (0, 0, 0, 0),
    ) -> None:
        """Initialize canvas."""
        self.width = width
        self.height = height
        self.background = background
        self.items: list[PlacedItem] = []
        self.canvas: Image.Image | None = None

    def ensure_canvas(self) -> Image.Image:
        """Create canvas image if not exists."""
        if self.canvas is None:
            self.canvas = Image.new(
                "RGBA", (self.width, self.height), self.background
            )
        return self.canvas

    def y_to_pil(self, y: int, img_height: int) -> int:
        """Convert bottom-left Y to PIL top-left Y.

        Parameters
        ----------
        y : int
            Y position from bottom.
        img_height : int
            Height of image being placed.

        Returns
        -------
        int
            Y position from top (PIL coordinates).
        """
        return self.height - y - img_height

    def place(
        self,
        image_path: str | Path,
        x: int,
        y: int,
        layer: int | None = None,
    ) -> PlacedItem:
        """Place an image on the canvas.

        Parameters
        ----------
        image_path : str | Path
            Path to image file.
        x : int
            X position from left edge.
        y : int
            Y position from bottom edge.
        layer : int | None
            Z-order layer. If None, uses next layer.

        Returns
        -------
        PlacedItem
            The placed item.
        """
        path = Path(image_path)

        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")

        # Load image
        img = Image.open(path).convert("RGBA")

        if layer is None:
            layer = len(self.items)

        item = PlacedItem(
            path=path,
            x=x,
            y=y,
            width=img.width,
            height=img.height,
            layer=layer,
        )

        self.items.append(item)
        img.close()

        logger.debug("Placed %s at (%s, %s) layer %s", path.name, x, y, layer)

        return item

    def place_centered(
        self,
        image_path: str | Path,
        layer: int | None = None,
    ) -> PlacedItem:
        """Place an image centered on the canvas.

        Parameters
        ----------
        image_path : str | Path
            Path to image file.
        layer : int | None
            Z-order layer.

        Returns
        -------
        PlacedItem
            The placed item.
        """
        path = Path(image_path)
        img = Image.open(path).convert("RGBA")

        x = (self.width - img.width) // 2
        y = (self.height - img.height) // 2

        img.close()

        return self.place(image_path, x, y, layer)

    def place_random(
        self,
        output_dir: str | Path,
        count: int = 50,
        object_type: str | None = None,
        margin: int = 50,
    ) -> list[PlacedItem]:
        """Place random segments from output directory.

        Parameters
        ----------
        output_dir : str | Path
            Output directory from preprocessing.
        count : int
            Number of segments to place.
        object_type : str | None
            Optional filter by object type.
        margin : int
            Minimum margin from canvas edges.

        Returns
        -------
        list[PlacedItem]
            List of placed items.
        """
        from services.service_image_index import ImageIndex

        index = ImageIndex(output_dir)
        index.load()

        # Collect all segments
        all_segments: list[Path] = []
        for entry in index.all():
            for cutout in entry.cutouts:
                if (
                    object_type is None
                    or object_type.lower() in cutout.label.lower()
                ):
                    all_segments.extend(cutout.segments)

        if not all_segments:
            logger.warning("No segments found")
            return []

        # Randomly select and place
        selected = random.sample(all_segments, min(count, len(all_segments)))
        placed = []

        for segment_path in selected:
            if not segment_path.exists():
                continue

            # Random position within margins
            img = Image.open(segment_path)
            x = random.randint(
                margin, max(margin, self.width - img.width - margin)
            )
            y = random.randint(
                margin, max(margin, self.height - img.height - margin)
            )
            img.close()

            item = self.place(segment_path, x, y)
            placed.append(item)

        logger.info("Placed %d random segments", len(placed))
        return placed

    def place_grid(
        self,
        segments: list[Path],
        cols: int = 10,
        spacing: int = 20,
        start_x: int = 0,
        start_y: int = 0,
    ) -> list[PlacedItem]:
        """Place segments in a grid pattern.

        Parameters
        ----------
        segments : list[Path]
            Segment images to place.
        cols : int
            Number of columns.
        spacing : int
            Spacing between items.
        start_x : int
            Starting X position.
        start_y : int
            Starting Y position.

        Returns
        -------
        list[PlacedItem]
            List of placed items.
        """
        placed = []
        x = start_x
        y = start_y
        col = 0
        row_height = 0

        for path in segments:
            if not path.exists():
                continue

            img = Image.open(path)
            item = self.place(path, x, y)
            placed.append(item)

            row_height = max(row_height, img.height)
            img.close()

            col += 1
            x += item.width + spacing

            if col >= cols:
                col = 0
                x = start_x
                y += row_height + spacing
                row_height = 0

        return placed

    def place_cutout_segments(
        self,
        cutout_metadata: dict,
        canvas_x: int,
        canvas_y: int,
        base_dir: Path | str,
        delay: float = 1.0,
        on_segment_placed: (
            Callable[[PlacedItem, int, int], None] | None
        ) = None,
        save_each: str | Path | None = None,
        jitter: int = 0,
        order: PlacementOrder = "sequential",
        animate: bool = False,
    ) -> list[PlacedItem]:
        """Place all segments of a cutout iteratively with delay.

        Segments are placed based on their position within the cutout,
        mapped to the canvas position. This creates a visual "reveal"
        effect when save_each is used.

        Parameters
        ----------
        cutout_metadata : dict
            Cutout metadata dict containing 'segments' list.
        canvas_x : int
            X position on canvas for cutout's left edge.
        canvas_y : int
            Y position on canvas for cutout's bottom edge.
        base_dir : Path | str
            Base directory where segment files are located.
        delay : float
            Seconds to sleep between segment placements. Default 1.0.
        on_segment_placed : callable | None
            Optional callback called after each segment placement.
            Receives (PlacedItem, current_index, total_count).
        save_each : str | Path | None
            If provided, save canvas after each segment placement.
            Use {n} in path for segment number (e.g., "frames/frame_{n}.png").
        jitter : int
            Random position jitter in pixels (±jitter for both x and y). Default 0.
        order : PlacementOrder
            Segment placement order strategy. Default "sequential".
            Options: sequential, random, center-out, edge-in, diagonal.
        animate : bool
            Show live preview window during placement. Default False.

        Returns
        -------
        list[PlacedItem]
            List of placed segment items.
        """
        base_dir = Path(base_dir)
        segments = cutout_metadata.get("segments", [])
        cutout_size = tuple(cutout_metadata.get("size", [0, 0]))

        # Sort segments according to placement order
        segments = sort_segments_by_order(segments, order, cutout_size)
        total = len(segments)
        placed = []

        # Setup animation display if requested
        fig = None
        ax = None
        img_display = None
        if animate:
            try:
                import matplotlib.pyplot as plt

                plt.ion()  # Interactive mode
                fig, ax = plt.subplots(figsize=(12, 5))
                ax.set_title("Canvas Preview")
                ax.axis("off")
                # Show initial empty canvas
                canvas_img = self.render()
                img_display = ax.imshow(np.array(canvas_img))
                fig.canvas.draw()
                fig.canvas.flush_events()
            except ImportError:
                logger.warning("matplotlib not available, animation disabled")
                animate = False

        for i, seg_data in enumerate(segments):
            seg_file = seg_data.get("file", "")
            seg_path = base_dir / seg_file

            if not seg_path.exists():
                logger.warning("Segment not found: %s", seg_path)
                continue

            # Get segment position within cutout
            seg_x, seg_y = seg_data.get("position", [0, 0])

            # Calculate canvas position
            # Segment position is from top-left of cutout
            # Canvas Y is from bottom, so we need to flip
            cutout_height = cutout_metadata.get("size", [0, 0])[1]
            seg_height = seg_data.get("size", [250, 250])[1]

            final_x = canvas_x + seg_x
            final_y = canvas_y + (cutout_height - seg_y - seg_height)

            # Apply jitter
            if jitter > 0:
                final_x += random.randint(-jitter, jitter)
                final_y += random.randint(-jitter, jitter)

            # Place segment
            item = self.place(seg_path, final_x, final_y)
            placed.append(item)

            logger.info(
                "Placed segment %d/%d: (%d, %d)",
                i + 1,
                total,
                final_x,
                final_y,
            )

            # Update animation display
            if (
                animate
                and fig is not None
                and img_display is not None
                and ax is not None
            ):
                import matplotlib.pyplot as plt

                canvas_img = self.render()
                img_display.set_data(np.array(canvas_img))
                ax.set_title(f"Canvas Preview - Segment {i + 1}/{total}")
                fig.canvas.draw()
                fig.canvas.flush_events()

            # Callback
            if on_segment_placed:
                on_segment_placed(item, i, total)

            # Save intermediate frame
            if save_each:
                frame_path = str(save_each).format(n=i + 1)
                self.save(frame_path)
                logger.info("Saved frame: %s", frame_path)

            # Delay before next segment
            if delay > 0 and i < total - 1:
                time.sleep(delay)

        # Keep animation window open briefly at the end
        if animate and fig is not None and ax is not None:
            import matplotlib.pyplot as plt

            ax.set_title(f"Canvas Complete - {total} segments")
            fig.canvas.draw()
            fig.canvas.flush_events()
            time.sleep(1.0)  # Show final result briefly
            plt.ioff()
            plt.close(fig)

        return placed

    def clear(self) -> None:
        """Clear all placed items."""
        self.items.clear()
        self.canvas = None

    def render(self) -> Image.Image:
        """Render canvas with all placed items.

        Returns
        -------
        Image.Image
            Rendered canvas image.
        """
        canvas = self.ensure_canvas().copy()

        # Sort by layer
        sorted_items = sorted(self.items, key=lambda x: x.layer)

        for item in sorted_items:
            if not item.path.exists():
                logger.warning("Missing image: %s", item.path)
                continue

            img = Image.open(item.path).convert("RGBA")

            # Convert Y coordinate
            pil_y = self.y_to_pil(item.y, img.height)

            # Paste with alpha compositing
            canvas.paste(img, (item.x, pil_y), img)
            img.close()

        return canvas

    def save(
        self,
        path: str | Path,
        output_format: str | None = None,
    ) -> None:
        """Save rendered canvas to file.

        Parameters
        ----------
        path : str | Path
            Output file path.
        output_format : str | None
            Image format (e.g., "PNG", "JPEG").
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        canvas = self.render()

        # For JPEG, convert to RGB
        if (
            output_format
            and output_format.upper() == "JPEG"
            or path.suffix.lower() in (".jpg", ".jpeg")
        ):
            canvas = canvas.convert("RGB")

        canvas.save(path, format=output_format)
        logger.info("Saved canvas to %s", path)

    def save_layout(self, path: str | Path) -> None:
        """Save layout metadata to JSON.

        Parameters
        ----------
        path : str | Path
            Output JSON file path.
        """
        path = Path(path)

        data = {
            "canvas": {
                "width": self.width,
                "height": self.height,
            },
            "items": [item.to_dict() for item in self.items],
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        logger.info("Saved layout to %s", path)

    def load_layout(self, path: str | Path) -> None:
        """Load layout from JSON.

        Parameters
        ----------
        path : str | Path
            Input JSON file path.
        """
        path = Path(path)

        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        self.clear()

        for item_data in data["items"]:
            self.place(
                item_data["path"],
                item_data["x"],
                item_data["y"],
                item_data.get("layer"),
            )

        logger.info("Loaded %d items from layout", len(self.items))

    def __len__(self) -> int:
        """Get number of placed items."""
        return len(self.items)
