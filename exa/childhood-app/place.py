#!/usr/bin/env python3
"""
Place Module
============

Place segments onto a canvas system for installation display.

Usage
-----
    # Place random segments onto canvas
    python place.py --output canvas.png

    # Place specific segments
    python place.py segment1.png 100 200 segment2.png 500 400 --output canvas.png

    # Interactive placement session
    python place.py --interactive

API Usage
---------
    from place import Canvas, PlacedItem

    canvas = Canvas(8000, 3000)
    canvas.place("segment.png", 100, 200)  # x, y from bottom-left
    canvas.place_random("output/", count=50)
    canvas.save("final.png")
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator, Literal

from PIL import Image

logger = logging.getLogger(__name__)

# Placement order strategies
PlacementOrder = Literal[
    "sequential", "random", "center-out", "edge-in", "diagonal"
]


def _sort_segments_by_order(
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


@dataclass
class PlacedItem:
    """A placed item on the canvas.

    Attributes
    ----------
    path : Path
        Path to the image file.
    x : int
        X position (from left).
    y : int
        Y position (from bottom in canvas coords, converted to top for PIL).
    width : int
        Image width.
    height : int
        Image height.
    layer : int
        Z-order layer (higher = on top).
    """

    path: Path
    x: int
    y: int
    width: int
    height: int
    layer: int = 0

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "path": str(self.path),
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "layer": self.layer,
        }


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
        self._items: list[PlacedItem] = []
        self._canvas: Image.Image | None = None

    def _ensure_canvas(self) -> Image.Image:
        """Create canvas image if not exists."""
        if self._canvas is None:
            self._canvas = Image.new(
                "RGBA", (self.width, self.height), self.background
            )
        return self._canvas

    def _y_to_pil(self, y: int, img_height: int) -> int:
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
            layer = len(self._items)

        item = PlacedItem(
            path=path,
            x=x,
            y=y,
            width=img.width,
            height=img.height,
            layer=layer,
        )

        self._items.append(item)
        img.close()

        logger.debug(f"Placed {path.name} at ({x}, {y}) layer {layer}")

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
        from fetch import ImageIndex

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

        logger.info(f"Placed {len(placed)} random segments")
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
        segments = _sort_segments_by_order(segments, order, cutout_size)
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
                img_display = ax.imshow(canvas_img)
                fig.canvas.draw()
                fig.canvas.flush_events()
            except ImportError:
                logger.warning("matplotlib not available, animation disabled")
                animate = False

        for i, seg_data in enumerate(segments):
            seg_file = seg_data.get("file", "")
            seg_path = base_dir / seg_file

            if not seg_path.exists():
                logger.warning(f"Segment not found: {seg_path}")
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
                f"Placed segment {i + 1}/{total}: ({final_x}, {final_y})"
            )

            # Update animation display
            if animate and fig is not None:
                import matplotlib.pyplot as plt

                canvas_img = self.render()
                img_display.set_data(canvas_img)
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
                logger.info(f"Saved frame: {frame_path}")

            # Delay before next segment
            if delay > 0 and i < total - 1:
                time.sleep(delay)

        # Keep animation window open briefly at the end
        if animate and fig is not None:
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
        self._items.clear()
        self._canvas = None

    def render(self) -> Image.Image:
        """Render canvas with all placed items.

        Returns
        -------
        Image.Image
            Rendered canvas image.
        """
        canvas = self._ensure_canvas().copy()

        # Sort by layer
        sorted_items = sorted(self._items, key=lambda x: x.layer)

        for item in sorted_items:
            if not item.path.exists():
                logger.warning(f"Missing image: {item.path}")
                continue

            img = Image.open(item.path).convert("RGBA")

            # Convert Y coordinate
            pil_y = self._y_to_pil(item.y, img.height)

            # Paste with alpha compositing
            canvas.paste(img, (item.x, pil_y), img)
            img.close()

        return canvas

    def save(
        self,
        path: str | Path,
        format: str | None = None,
    ) -> None:
        """Save rendered canvas to file.

        Parameters
        ----------
        path : str | Path
            Output file path.
        format : str | None
            Image format (e.g., "PNG", "JPEG").
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        canvas = self.render()

        # For JPEG, convert to RGB
        if (
            format
            and format.upper() == "JPEG"
            or path.suffix.lower() in (".jpg", ".jpeg")
        ):
            canvas = canvas.convert("RGB")

        canvas.save(path, format=format)
        logger.info(f"Saved canvas to {path}")

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
            "items": [item.to_dict() for item in self._items],
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved layout to {path}")

    def load_layout(self, path: str | Path) -> None:
        """Load layout from JSON.

        Parameters
        ----------
        path : str | Path
            Input JSON file path.
        """
        path = Path(path)

        with open(path) as f:
            data = json.load(f)

        self.clear()

        for item_data in data["items"]:
            self.place(
                item_data["path"],
                item_data["x"],
                item_data["y"],
                item_data.get("layer"),
            )

        logger.info(f"Loaded {len(self._items)} items from layout")

    @property
    def items(self) -> list[PlacedItem]:
        """Get placed items."""
        return self._items.copy()

    def __len__(self) -> int:
        """Get number of placed items."""
        return len(self._items)


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Place segments onto canvas for installation display",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "placements",
        nargs="*",
        help="Image paths and positions: img1.png x1 y1 img2.png x2 y2 ...",
    )
    parser.add_argument(
        "--width",
        "-W",
        type=int,
        default=8000,
        help="Canvas width (default: 8000)",
    )
    parser.add_argument(
        "--height",
        "-H",
        type=int,
        default=3000,
        help="Canvas height (default: 3000)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("canvas.png"),
        help="Output canvas image (default: canvas.png)",
    )
    parser.add_argument(
        "--random",
        "-r",
        type=int,
        metavar="COUNT",
        help="Place COUNT random segments",
    )
    parser.add_argument(
        "--from-dir",
        type=Path,
        default=Path(__file__).parent / "output",
        help="Output directory for random placement",
    )
    parser.add_argument(
        "--type",
        "-t",
        type=str,
        help="Filter random segments by object type",
    )
    parser.add_argument(
        "--grid",
        type=int,
        metavar="COLS",
        help="Place in grid with COLS columns",
    )
    parser.add_argument(
        "--layout",
        type=Path,
        help="Load layout from JSON file",
    )
    parser.add_argument(
        "--save-layout",
        type=Path,
        help="Save layout to JSON file",
    )
    parser.add_argument(
        "--background",
        "-b",
        type=str,
        default="transparent",
        help="Background: 'transparent', 'white', 'black', or hex color",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output",
    )
    parser.add_argument(
        "--place-cutout",
        type=Path,
        metavar="METADATA",
        help="Place all segments of a cutout from metadata.json",
    )
    parser.add_argument(
        "--cutout-index",
        type=int,
        default=0,
        help="Index of cutout in metadata to place (default: 0)",
    )
    parser.add_argument(
        "--pos",
        type=int,
        nargs=2,
        metavar=("X", "Y"),
        default=[0, 0],
        help="Canvas position for cutout placement (default: 0 0)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Delay between segment placements in seconds (default: 1.0)",
    )
    parser.add_argument(
        "--save-each",
        type=str,
        metavar="PATTERN",
        help="Save canvas after each segment (use {n} for number, e.g., frames/frame_{n}.png)",
    )
    parser.add_argument(
        "--jitter",
        type=int,
        default=0,
        help="Random position jitter in pixels (default: 0)",
    )
    parser.add_argument(
        "--order",
        type=str,
        choices=["sequential", "random", "center-out", "edge-in", "diagonal"],
        default="sequential",
        help="Segment placement order (default: sequential)",
    )
    parser.add_argument(
        "--animate",
        action="store_true",
        help="Show live preview window during placement",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    # Parse background color
    if args.background == "transparent":
        bg = (0, 0, 0, 0)
    elif args.background == "white":
        bg = (255, 255, 255, 255)
    elif args.background == "black":
        bg = (0, 0, 0, 255)
    elif args.background.startswith("#"):
        hex_color = args.background.lstrip("#")
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        bg = (r, g, b, 255)
    else:
        bg = (0, 0, 0, 0)

    canvas = Canvas(args.width, args.height, bg)

    # Place cutout segments iteratively
    if args.place_cutout:
        metadata_path = args.place_cutout
        if not metadata_path.exists():
            logger.error(f"Metadata file not found: {metadata_path}")
            return 1

        with open(metadata_path) as f:
            metadata = json.load(f)

        cutouts = metadata.get("cutouts", [])
        if not cutouts:
            logger.error("No cutouts found in metadata")
            return 1

        if args.cutout_index >= len(cutouts):
            logger.error(
                f"Cutout index {args.cutout_index} out of range (0-{len(cutouts) - 1})"
            )
            return 1

        cutout_data = cutouts[args.cutout_index]
        base_dir = metadata_path.parent

        x, y = args.pos
        logger.info(
            f"Placing cutout '{cutout_data.get('label', 'unknown')}' at ({x}, {y})"
        )
        if args.jitter > 0:
            logger.info(f"  Jitter: ±{args.jitter}px")
        if args.order != "sequential":
            logger.info(f"  Order: {args.order}")
        if args.animate:
            logger.info("  Animation: enabled")

        canvas.place_cutout_segments(
            cutout_data,
            canvas_x=x,
            canvas_y=y,
            base_dir=base_dir,
            delay=args.delay,
            save_each=args.save_each,
            jitter=args.jitter,
            order=args.order,
            animate=args.animate,
        )

    # Load from layout file
    elif args.layout:
        canvas.load_layout(args.layout)

    # Random placement
    elif args.random:
        canvas.place_random(args.from_dir, args.random, args.type)

    # Manual placements from args
    elif args.placements:
        i = 0
        while i < len(args.placements):
            if i + 2 >= len(args.placements):
                logger.error("Placements must be: image x y [image x y ...]")
                return 1

            img_path = args.placements[i]
            x = int(args.placements[i + 1])
            y = int(args.placements[i + 2])

            canvas.place(img_path, x, y)
            i += 3

    else:
        parser.print_help()
        return 1

    # Grid layout (not used with place_cutout or random)
    if args.grid and not args.random and not args.place_cutout:
        from fetch import ImageIndex

        index = ImageIndex(args.from_dir)

        all_segs = []
        for entry in index.all():
            for cutout in entry.cutouts:
                if (
                    args.type is None
                    or args.type.lower() in cutout.label.lower()
                ):
                    all_segs.extend(cutout.segments)

        canvas.place_grid(all_segs, cols=args.grid)

    # Save outputs
    if args.save_layout:
        canvas.save_layout(args.save_layout)

    canvas.save(args.output)
    print(f"Saved {len(canvas)} items to {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
