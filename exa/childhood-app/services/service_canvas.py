"""
Canvas Service
==============

Canvas for placing image segments.
"""

from __future__ import annotations

import json
import logging
import math
import os
import random
import time
from pathlib import Path
from typing import Callable, Literal

import numpy as np
from PIL import Image

from models.model_placed_item import PlacedItem


# helper for simple average-color tests used by random/grid placement

def _image_is_color(path: Path | str, color: str, threshold: int = 50) -> bool:
    from PIL import Image

    try:
        img = Image.open(path).convert("RGB")
    except Exception:
        return False

    arr = np.array(img)
    mean_r = float(arr[:, :, 0].mean())
    mean_g = float(arr[:, :, 1].mean())
    mean_b = float(arr[:, :, 2].mean())
    img.close()

    color = color.lower()
    if color == "red":
        return mean_r > mean_g + threshold and mean_r > mean_b + threshold
    if color == "green":
        return mean_g > mean_r + threshold and mean_g > mean_b + threshold
    if color == "blue":
        return mean_b > mean_r + threshold and mean_b > mean_g + threshold
    # unknown color -> pass everything
    return True


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    """Convert a hex color string to (R, G, B) tuple."""
    h = hex_color.lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def _apply_postproduction(
    image: Image.Image,
    mode: str,
    gradient_dark: str,
    gradient_light: str,
) -> Image.Image:
    """Apply postproduction color grading to a rendered canvas.

    Parameters
    ----------
    image : Image.Image
        The rendered canvas (RGBA).
    mode : str
        "normal" (no change), "bw" (grayscale), or "gradient_map".
    gradient_dark : str
        Hex color mapped to dark tones (gradient_map mode).
    gradient_light : str
        Hex color mapped to light tones (gradient_map mode).

    Returns
    -------
    Image.Image
        Color-graded image (RGBA preserved).
    """
    if mode == "normal":
        return image

    # Separate alpha channel to preserve transparency
    if image.mode == "RGBA":
        alpha = image.split()[3]
    else:
        alpha = None

    if mode == "bw":
        gray = image.convert("L")
        result = gray.convert("RGBA")
        if alpha is not None:
            result.putalpha(alpha)
        return result

    if mode == "gradient_map":
        gray = image.convert("L")
        dark = _hex_to_rgb(gradient_dark)
        light = _hex_to_rgb(gradient_light)

        # Build a 256-entry lookup table that maps luminance to the
        # gradient between dark and light colors.
        lut = []
        for i in range(256):
            t = i / 255.0
            lut.append(int(dark[0] + (light[0] - dark[0]) * t))
            lut.append(int(dark[1] + (light[1] - dark[1]) * t))
            lut.append(int(dark[2] + (light[2] - dark[2]) * t))

        rgb = gray.convert("RGB")
        # Apply the LUT per-channel (PIL point expects 256 entries per channel)
        r, g, b = rgb.split()
        r = r.point([lut[i * 3] for i in range(256)])
        g = g.point([lut[i * 3 + 1] for i in range(256)])
        b = b.point([lut[i * 3 + 2] for i in range(256)])
        result = Image.merge("RGB", (r, g, b)).convert("RGBA")
        if alpha is not None:
            result.putalpha(alpha)
        return result

    return image


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
        # Incremental rendering state
        self._render_canvas: Image.Image | None = None
        self._rendered_count: int = 0
        # Postproduction settings (applied per-segment in render)
        self._pp_mode: str = "normal"
        self._pp_gradient_dark: str = "#000000"
        self._pp_gradient_light: str = "#ffffff"

    def set_postproduction(
        self,
        mode: str = "normal",
        gradient_dark: str = "#000000",
        gradient_light: str = "#ffffff",
    ) -> None:
        """Configure postproduction color grading applied to every segment.

        Parameters
        ----------
        mode : str
            "normal", "bw", or "gradient_map".
        gradient_dark : str
            Hex color for dark tones (gradient_map mode).
        gradient_light : str
            Hex color for light tones (gradient_map mode).
        """
        self._pp_mode = mode
        self._pp_gradient_dark = gradient_dark
        self._pp_gradient_light = gradient_light

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
        rotation: float = 0.0,
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
        rotation : float
            Rotation angle in degrees (counter-clockwise). Default 0.0.

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
            rotation=rotation,
        )

        self.items.append(item)
        img.close()

        logger.debug("Placed %s at (%s, %s) layer %s rot %.1f°", path.name, x, y, layer, rotation)

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
        animate: bool = False,
        delay: float = 0.5,
        jitter: int = 0,
        color: str | None = None,
        color_threshold: int = 50,
    ) -> list[PlacedItem]:
        """Place random segments from output directory.

        Parameters
        ----------
        output_dir : str | Path
            Output directory from preprocessing.
        count : int
            Number of segments to place.
        object_type : str | None
            Optional filter by object type (label text).
        margin : int
            Minimum margin from canvas edges.
        animate : bool
            Show live preview in browser during placement.
        delay : float
            Delay between placements in seconds (for animation).
        jitter : int
            Random position jitter in pixels.
        color : str | None
            If provided, only segments whose average color matches this
            ("red","green","blue") will be considered.
        color_threshold : int
            How much stronger the chosen channel must be compared to others.

        Returns
        -------
        list[PlacedItem]
            List of placed items.
        """
        import time
        import webbrowser
        import http.server
        import socketserver
        import threading
        import tempfile
        import shutil

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

        # optional color-based filtering
        if color:
            filtered: list[Path] = []
            for p in all_segments:
                if _image_is_color(p, color, color_threshold):
                    filtered.append(p)
            all_segments = filtered

        if not all_segments:
            logger.warning("No segments found after filtering")
            return []
        # Randomly select and place
        selected = random.sample(all_segments, min(count, len(all_segments)))
        placed = []

        # Setup browser preview if requested
        if animate:
            # Create temp directory for preview
            preview_dir = Path(tempfile.mkdtemp(prefix="canvas_preview_"))
            preview_image = preview_dir / "canvas.png"

            # Create HTML viewer
            html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Canvas Preview</title>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        html, body {{ width: 100vw; height: 100vh; background: #1a1a1a; color: #fff; font-family: system-ui; overflow: hidden; }}
        body {{ display: flex; flex-direction: column; padding: 10px; }}
        #header {{ flex-shrink: 0; margin-bottom: 10px; }}
        h1 {{ font-size: 16px; }}
        #status {{ color: #888; font-size: 14px; }}
        #canvas-container {{ flex: 1; min-height: 0; display: flex; align-items: center; justify-content: center; }}
        #canvas {{ max-width: calc(100vw - 22px); max-height: calc(100vh - 60px); object-fit: contain; border: 1px solid #333; background: repeating-conic-gradient(#222 0% 25%, #333 0% 50%) 50% / 20px 20px; }}
    </style>
</head>
<body>
    <div id="header">
        <h1>Canvas Preview</h1>
        <div id="status">Placing segments...</div>
    </div>
    <div id="canvas-container">
        <img id="canvas" src="canvas.png" />
    </div>
    <script>
        const img = document.getElementById('canvas');
        const status = document.getElementById('status');
        const delay = {int(delay * 1000)};

        function refresh() {{
            fetch('status.json?' + Date.now())
                .then(r => r.ok ? r.json() : Promise.reject('not ready'))
                .then(data => {{
                    const newImg = new Image();
                    newImg.onload = function() {{
                        img.src = newImg.src;
                        status.textContent = data.done
                            ? 'Done! Placed ' + data.placed + ' segments'
                            : 'Placed ' + data.placed + ' / ' + data.total + ' segments';
                        if (!data.done) {{
                            setTimeout(refresh, delay);
                        }}
                    }};
                    newImg.onerror = function() {{
                        setTimeout(refresh, delay);
                    }};
                    newImg.src = 'canvas.png?' + Date.now();
                }})
                .catch(() => {{
                    setTimeout(refresh, delay);
                }});
        }}
        setTimeout(refresh, delay);
    </script>
</body>
</html>"""
            (preview_dir / "index.html").write_text(html_content)

            # Save initial canvas
            self.render().save(preview_image)
            # Write initial status
            (preview_dir / "status.json").write_text(
                json.dumps({"placed": 0, "total": len(selected), "done": False})
            )

            # Start simple HTTP server
            class QuietHandler(http.server.SimpleHTTPRequestHandler):
                def __init__(self, *args, **kwargs):
                    super().__init__(
                        *args, directory=str(preview_dir), **kwargs
                    )

                def log_message(self, format, *args):
                    pass  # Suppress logging

            port = 8765
            server = socketserver.TCPServer(("", port), QuietHandler)
            server_thread = threading.Thread(
                target=server.serve_forever, daemon=True
            )
            server_thread.start()

            # Open browser
            webbrowser.open(f"http://localhost:{port}")
            logger.info("Preview at http://localhost:%d", port)
            time.sleep(0.5)  # Let browser load

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

            # Apply jitter
            if jitter > 0:
                x += random.randint(-jitter, jitter)
                y += random.randint(-jitter, jitter)
                x = max(0, min(x, self.width - img.width))
                y = max(0, min(y, self.height - img.height))

            img.close()

            item = self.place(segment_path, x, y)
            placed.append(item)

            # Update preview (atomic write to avoid flicker)
            if animate:
                tmp_preview = preview_image.with_suffix('.tmp.png')
                self.render().save(tmp_preview)
                os.replace(str(tmp_preview), str(preview_image))
                (preview_dir / "status.json").write_text(
                    json.dumps({"placed": len(placed), "total": len(selected), "done": False})
                )
                time.sleep(delay)

        if animate:
            # Final save (atomic write + done signal)
            tmp_preview = preview_image.with_suffix('.tmp.png')
            self.render().save(tmp_preview)
            os.replace(str(tmp_preview), str(preview_image))
            (preview_dir / "status.json").write_text(
                json.dumps({"placed": len(placed), "total": len(selected), "done": True})
            )
            logger.info("Preview complete. Close browser when done.")
            try:
                input("Press Enter to close preview server...")
            except KeyboardInterrupt:
                pass
            server.shutdown()
            shutil.rmtree(preview_dir, ignore_errors=True)

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
        organic: bool = True,
        organic_jitter: int = 6,
        order: PlacementOrder = "sequential",
        animate: bool = False,
        selection_ratio: float = 1.0,
        rotation_range: float = 0.0,
        rotation_jitter: float = 0.0,
        preview_dir: Path | None = None,
        preview_image: Path | None = None,
        placed_offset: int = 0,
        total_override: int | None = None,
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
        selection_ratio : float
            Fraction of segments to place (0.0-1.0). Default 1.0 (all).
            Used for organic layered effect with overlapping segmentations.
        rotation_range : float
            Base random rotation range in degrees (±). Applied per segment
            at placement time (replaces pre-baked rotation from metadata).
            Default 0.0.
        rotation_jitter : float
            Additional random rotation in degrees (±). Added to segment's
            stored rotation. Default 0.0.
        preview_dir : Path | None
            When provided, skip internal server setup and write preview
            images to this directory instead.  The caller manages the
            HTTP server lifecycle.
        preview_image : Path | None
            Path to the preview PNG inside *preview_dir*.  Required when
            *preview_dir* is set.

        Returns
        -------
        list[PlacedItem]
            List of placed segment items.
        """
        base_dir = Path(base_dir)
        segments = cutout_metadata.get("segments", [])
        cutout_size = tuple(cutout_metadata.get("size", [0, 0]))

        logger.info("Placing cutout with %d segments (size %dx%d)", len(segments), cutout_size[0], cutout_size[1])

        # Sort segments according to placement order
        segments = sort_segments_by_order(segments, order, cutout_size)
        total = len(segments)
        placed = []

        # Setup browser preview if requested (replace matplotlib interactive)
        _owns_server = False  # True when this method started the server
        server = None
        if animate and preview_dir is not None and preview_image is not None:
            # External caller manages the server — just use the paths
            pass
        elif animate:
            try:
                import http.server
                import socketserver
                import threading
                import tempfile
                import webbrowser
                import shutil

                preview_dir = Path(tempfile.mkdtemp(prefix="canvas_preview_"))
                preview_image = preview_dir / "canvas.png"

                html_template = """<!DOCTYPE html>
<html>
<head>
    <title>Canvas Preview</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        html, body { width: 100vw; height: 100vh; background: #1a1a1a; color: #fff; font-family: system-ui; overflow: hidden; }
        body { display: flex; flex-direction: column; padding: 10px; }
        #header { flex-shrink: 0; margin-bottom: 10px; }
        h1 { font-size: 16px; }
        #status { color: #888; font-size: 14px; }
        #canvas-container { flex: 1; min-height: 0; display: flex; align-items: center; justify-content: center; }
        #canvas { max-width: calc(100vw - 22px); max-height: calc(100vh - 60px); object-fit: contain; border: 1px solid #333; background: repeating-conic-gradient(#222 0% 25%, #333 0% 50%) 50% / 20px 20px; }
    </style>
</head>
<body>
    <div id="header">
        <h1>Canvas Preview</h1>
        <div id="status">Placing segments...</div>
    </div>
    <div id="canvas-container">
        <img id="canvas" src="canvas.png" />
    </div>
    <script>
        const img = document.getElementById('canvas');
        const status = document.getElementById('status');
        const delay = DELAY_PLACEHOLDER;

        function refresh() {
            fetch('status.json?' + Date.now())
                .then(r => r.ok ? r.json() : Promise.reject('not ready'))
                .then(data => {
                    const newImg = new Image();
                    newImg.onload = function() {
                        img.src = newImg.src;
                        status.textContent = data.done
                            ? 'Done! Placed ' + data.placed + ' segments'
                            : 'Placed ' + data.placed + ' / ' + data.total + ' segments';
                        if (!data.done) {
                            setTimeout(refresh, delay);
                        }
                    };
                    newImg.onerror = function() {
                        setTimeout(refresh, delay);
                    };
                    newImg.src = 'canvas.png?' + Date.now();
                })
                .catch(() => {
                    setTimeout(refresh, delay);
                });
        }
        setTimeout(refresh, delay);
    </script>
</body>
</html>
"""
                html_content = html_template.replace("TOTAL_PLACEHOLDER", str(len(segments))).replace("DELAY_PLACEHOLDER", str(int(delay*1000)))

                (preview_dir / "index.html").write_text(html_content)
                # Save initial canvas
                self.render().save(preview_image)
                # Write initial status
                (preview_dir / "status.json").write_text(
                    json.dumps({"placed": 0, "total": total, "done": False})
                )

                class QuietHandler(http.server.SimpleHTTPRequestHandler):
                    def __init__(self, *args, **kwargs):
                        super().__init__(*args, directory=str(preview_dir), **kwargs)
                    def log_message(self, format, *args):
                        pass

                port = 8765
                server = socketserver.TCPServer(("", port), QuietHandler)
                server_thread = threading.Thread(target=server.serve_forever, daemon=True)
                server_thread.start()
                _owns_server = True
                url = f"http://localhost:{port}"
                logger.info("Preview server running at %s", url)
                try:
                    opened = webbrowser.open(url)
                    if not opened:
                        logger.info("Browser did not open automatically; please open %s manually", url)
                except Exception as e:
                    logger.info("webbrowser.open failed: %s", e)
                time.sleep(0.5)
            except Exception as exc:
                import traceback
                logger.warning("Browser preview unavailable, disabling animation: %s", exc)
                logger.debug("Preview setup traceback:\n%s", traceback.format_exc())
                animate = False

        for i, seg_data in enumerate(segments):
            # Apply selection ratio - randomly skip some segments for organic layered effect
            if selection_ratio < 1.0 and random.random() > selection_ratio:
                continue

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
            seg_w, seg_height = seg_data.get("size", [250, 250])

            final_x = canvas_x + seg_x
            final_y = canvas_y + (cutout_height - seg_y - seg_height)

            # Apply jitter (organic mode: if jitter==0 and organic enabled,
            # use a small organic_jitter to introduce imperfect seams)
            use_jitter = jitter
            if organic and use_jitter == 0:
                use_jitter = organic_jitter

            if use_jitter > 0:
                final_x += random.randint(-use_jitter, use_jitter)
                final_y += random.randint(-use_jitter, use_jitter)
                # Clamp to canvas bounds
                final_x = max(0, min(final_x, self.width - seg_w))
                final_y = max(0, min(final_y, self.height - seg_height))

            # Build rotation entirely at placement time:
            # base comes from rotation_range, jitter adds organic variation.
            base_rotation = 0.0
            if rotation_range > 0:
                base_rotation = random.uniform(-rotation_range, rotation_range)
            if rotation_jitter > 0:
                base_rotation += random.uniform(-rotation_jitter, rotation_jitter)

            # Place segment with rotation
            item = self.place(seg_path, final_x, final_y, rotation=base_rotation)
            placed.append(item)

            logger.info(
                "Placed segment %d/%d: (%d, %d) rot %.1f°",
                i + 1,
                total,
                final_x,
                final_y,
                base_rotation,
            )

            # Update browser preview image (atomic write to avoid flicker)
            if animate and preview_image is not None:
                tmp_preview = preview_image.with_suffix('.tmp.png')
                self.render().save(tmp_preview)
                os.replace(str(tmp_preview), str(preview_image))
                _status_total = total_override if total_override is not None else total
                (preview_dir / "status.json").write_text(
                    json.dumps({"placed": placed_offset + len(placed), "total": _status_total, "done": False})
                )

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

        # Finalize browser preview (atomic write + done signal)
        if animate and preview_image is not None:
            tmp_preview = preview_image.with_suffix('.tmp.png')
            self.render().save(tmp_preview)
            os.replace(str(tmp_preview), str(preview_image))
            if preview_dir is not None:
                _status_total = total_override if total_override is not None else total
                (preview_dir / "status.json").write_text(
                    json.dumps({"placed": placed_offset + len(placed), "total": _status_total, "done": _owns_server})
                )

        # Only shut down if this method started the server
        if _owns_server:
            logger.info("Preview complete. Close browser when done.")
            try:
                input("Press Enter to close preview server...")
            except KeyboardInterrupt:
                pass
            try:
                if server is not None:
                    server.shutdown()
            except Exception:
                pass
            try:
                if preview_dir is not None:
                    import shutil
                    shutil.rmtree(preview_dir, ignore_errors=True)
            except Exception:
                pass

        logger.info("Placed %d segments from cutout", len(placed))
        return placed

    def prepare_cutout_segments(
        self,
        cutout_metadata: dict,
        canvas_x: int,
        canvas_y: int,
        base_dir: Path | str,
        jitter: int = 0,
        organic: bool = True,
        organic_jitter: int = 6,
        order: PlacementOrder = "sequential",
        selection_ratio: float = 1.0,
        rotation_range: float = 0.0,
        rotation_jitter: float = 0.0,
    ) -> list[dict]:
        """Prepare segment placement data without placing them.

        Returns a list of dicts with keys: path, x, y, rotation.
        Each entry is ready to be passed to ``self.place()``.
        """
        base_dir = Path(base_dir)
        segments = cutout_metadata.get("segments", [])
        cutout_size = tuple(cutout_metadata.get("size", [0, 0]))
        segments = sort_segments_by_order(segments, order, cutout_size)
        cutout_height = cutout_metadata.get("size", [0, 0])[1]

        prepared: list[dict] = []
        for seg_data in segments:
            if selection_ratio < 1.0 and random.random() > selection_ratio:
                continue
            seg_file = seg_data.get("file", "")
            seg_path = base_dir / seg_file
            if not seg_path.exists():
                logger.warning("Segment not found: %s", seg_path)
                continue
            seg_x, seg_y = seg_data.get("position", [0, 0])
            seg_w, seg_h = seg_data.get("size", [250, 250])
            final_x = canvas_x + seg_x
            final_y = canvas_y + (cutout_height - seg_y - seg_h)
            use_jitter = jitter
            if organic and use_jitter == 0:
                use_jitter = organic_jitter
            if use_jitter > 0:
                final_x += random.randint(-use_jitter, use_jitter)
                final_y += random.randint(-use_jitter, use_jitter)
                final_x = max(0, min(final_x, self.width - seg_w))
                final_y = max(0, min(final_y, self.height - seg_h))
            rot = 0.0
            if rotation_range > 0:
                rot = random.uniform(-rotation_range, rotation_range)
            if rotation_jitter > 0:
                rot += random.uniform(-rotation_jitter, rotation_jitter)
            prepared.append({"path": seg_path, "x": final_x, "y": final_y, "rotation": rot})
        return prepared

    def clear(self) -> None:
        """Clear all placed items."""
        self.items.clear()
        self.canvas = None
        self._render_canvas = None
        self._rendered_count = 0

    def _composite_item(self, canvas: Image.Image, item: PlacedItem) -> None:
        """Composite a single placed item onto *canvas* (in-place)."""
        if not item.path.exists():
            logger.warning("Missing image: %s", item.path)
            return

        img = Image.open(item.path).convert("RGBA")

        # Apply postproduction color grading per-segment
        if self._pp_mode != "normal":
            img = _apply_postproduction(
                img, self._pp_mode, self._pp_gradient_dark, self._pp_gradient_light
            )

        # Apply rotation if needed
        if item.rotation != 0.0:
            orig_w, orig_h = img.size
            img = img.rotate(
                item.rotation,
                resample=Image.Resampling.BICUBIC,
                expand=True,
            )
            new_w, new_h = img.size
            offset_x = (new_w - orig_w) // 2
            offset_y = (new_h - orig_h) // 2
        else:
            offset_x = 0
            offset_y = 0

        pil_y = self.y_to_pil(item.y, item.height)
        paste_x = item.x - offset_x
        paste_y = pil_y - offset_y

        canvas.paste(img, (paste_x, paste_y), img)
        img.close()

    def render_incremental(self) -> Image.Image:
        """Render only newly-added items on top of the cached canvas.

        Much faster than render() when items are added one at a time
        (O(1) per call instead of O(n)).

        Returns
        -------
        Image.Image
            Rendered canvas image.
        """
        if self._render_canvas is None:
            self._render_canvas = self.ensure_canvas().copy()
            self._rendered_count = 0

        sorted_items = sorted(self.items, key=lambda x: x.layer)
        new_items = sorted_items[self._rendered_count:]

        for item in new_items:
            self._composite_item(self._render_canvas, item)

        self._rendered_count = len(sorted_items)
        return self._render_canvas

    def render(self) -> Image.Image:
        """Render canvas with all placed items (full re-composite).

        Returns
        -------
        Image.Image
            Rendered canvas image.
        """
        canvas = self.ensure_canvas().copy()

        sorted_items = sorted(self.items, key=lambda x: x.layer)

        for item in sorted_items:
            self._composite_item(canvas, item)

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
