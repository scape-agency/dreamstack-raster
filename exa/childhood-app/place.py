#!/usr/bin/env python3
"""
Place CLI
=========

Place segments onto a canvas system for installation display.

Usage
-----
    # Edit config/place_config.yaml, then run:
    python place.py

    # Use custom config:
    python place.py --config my_config.yaml

    # Verbose output:
    python place.py --verbose

API Usage
---------
    from services.service_canvas import Canvas
    from models.model_placed_item import PlacedItem

    canvas = Canvas(8000, 3000)
    canvas.place("segment.png", 100, 200)  # x, y from bottom-left
    canvas.place_random("output/", count=50)
    canvas.save("final.png")
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

from services.service_canvas import Canvas
from services.service_image_index import ImageIndex
from utils.util_load_config import load_config, get_nested

# simple color‑filter helper used in place_random / place_grid

def _image_is_color(
    path: str | Path, color: str, threshold: int = 50
) -> bool:
    """Return True if the average color of *path* matches named color.

    The implementation is intentionally simple: convert to RGB and
    compare channel means.  "red" means R is greater than G and B by
    *threshold* pixels.  You can extend this with proper HSV ranges
    if you like.
    """
    from PIL import Image
    import numpy as np

    try:
        img = Image.open(path).convert("RGB")
    except Exception:  # file could be missing or unreadable
        return False

    arr = np.array(img)
    # arr shape (h,w,3)
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

    # unknown color, always allow
    return True


logger = logging.getLogger(__name__)

DEFAULT_CONFIG = Path(__file__).parent / "config" / "place_config.yaml"


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Place segments onto canvas. Edit config/place_config.yaml to configure.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"Config file (default: {DEFAULT_CONFIG})",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output",
    )

    args = parser.parse_args()

    # Load config
    cfg = load_config(args.config)
    if not cfg:
        logger.error("Config file not found: %s", args.config)
        return 1

    # Read all settings from config
    width = get_nested(cfg, "canvas", "width", default=8000)
    height = get_nested(cfg, "canvas", "height", default=3000)
    output = Path(get_nested(cfg, "output", "path", default="canvas.png"))
    background = get_nested(cfg, "canvas", "background", default="transparent")
    save_layout = get_nested(cfg, "output", "save_layout")

    # Random placement config
    random_enabled = get_nested(cfg, "random", "enabled", default=False)
    random_count = (
        get_nested(cfg, "random", "count", default=30)
        if random_enabled
        else None
    )
    from_dir = Path(get_nested(cfg, "random", "from_dir", default="./output"))
    obj_type = get_nested(cfg, "random", "object_type")
    keep_cutouts = get_nested(cfg, "random", "keep_cutouts", default=False)
    margin = get_nested(cfg, "random", "margin", default=50)

    # optional color filter (only used in random/grid modes)
    color_filter = get_nested(cfg, "filter", "color")
    color_threshold = get_nested(cfg, "filter", "color_threshold", default=50)

    # Cutout placement config
    cutout_enabled = get_nested(cfg, "place_cutout", "enabled", default=False)
    place_cutout = None
    if cutout_enabled:
        cutout_path = get_nested(cfg, "place_cutout", "metadata_path")
        if cutout_path:
            place_cutout = Path(cutout_path)
    cutout_index = get_nested(cfg, "place_cutout", "cutout_index", default=0)
    pos = get_nested(cfg, "place_cutout", "position", default=[0, 0])

    # Animation config
    delay = get_nested(cfg, "animation", "delay", default=1.0)
    jitter = get_nested(cfg, "animation", "jitter", default=0)
    order = get_nested(cfg, "animation", "order", default="sequential")
    animate = get_nested(cfg, "animation", "enabled", default=False)
    save_each = get_nested(cfg, "animation", "save_each")
    # organic placement: when true, apply a small default per-segment jitter
    # so segments look slightly moved from their original positions
    organic = get_nested(cfg, "animation", "organic", default=True)
    organic_jitter = get_nested(cfg, "animation", "organic_jitter", default=6)
    # Fluid grid placement options
    selection_ratio = get_nested(cfg, "animation", "selection_ratio", default=0.7)
    rotation_range = get_nested(cfg, "animation", "rotation_range", default=1.0)
    rotation_jitter = get_nested(cfg, "animation", "rotation_jitter", default=3.0)
    parallel_tracks = max(1, min(5, get_nested(cfg, "animation", "parallel_tracks", default=1)))

    # Postproduction config
    pp_mode = get_nested(cfg, "postproduction", "mode", default="normal")
    pp_gradient_dark = get_nested(cfg, "postproduction", "gradient_dark", default="#000000")
    pp_gradient_light = get_nested(cfg, "postproduction", "gradient_light", default="#ffffff")

    # Grid config
    grid_enabled = get_nested(cfg, "grid", "enabled", default=False)
    grid = get_nested(cfg, "grid", "columns") if grid_enabled else None

    # Layout config
    layout_path = get_nested(cfg, "layout", "load")
    layout = Path(layout_path) if layout_path else None

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    # Parse background color
    if background == "transparent":
        bg = (0, 0, 0, 0)
    elif background == "white":
        bg = (255, 255, 255, 255)
    elif background == "black":
        bg = (0, 0, 0, 255)
    elif background.startswith("#"):
        hex_color = background.lstrip("#")
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        bg = (r, g, b, 255)
    else:
        bg = (0, 0, 0, 0)

    canvas = Canvas(width, height, bg)

    # Configure postproduction color grading (applied per-segment during render)
    canvas.set_postproduction(
        mode=pp_mode,
        gradient_dark=pp_gradient_dark,
        gradient_light=pp_gradient_light,
    )

    # Place cutout segments iteratively
    if place_cutout:
        metadata_path = place_cutout
        if not metadata_path.exists():
            logger.error("Metadata file not found: %s", metadata_path)
            return 1

        with open(metadata_path, encoding="utf-8") as f:
            metadata = json.load(f)

        cutouts = metadata.get("cutouts", [])
        if not cutouts:
            logger.error("No cutouts found in metadata")
            return 1

        if cutout_index >= len(cutouts):
            logger.error(
                "Cutout index %d out of range (0-%d)",
                cutout_index,
                len(cutouts) - 1,
            )
            return 1

        cutout_data = cutouts[cutout_index]
        base_dir = metadata_path.parent

        x, y = pos
        logger.info(
            "Placing cutout '%s' at (%d, %d)",
            cutout_data.get("label", "unknown"),
            x,
            y,
        )
        if jitter > 0:
            logger.info("  Jitter: ±%dpx", jitter)
        if order != "sequential":
            logger.info("  Order: %s", order)
        if animate:
            logger.info("  Animation: enabled")

        canvas.place_cutout_segments(
            cutout_data,
            canvas_x=x,
            canvas_y=y,
            base_dir=base_dir,
            delay=delay,
            save_each=save_each,
            jitter=jitter,
            organic=organic,
            organic_jitter=organic_jitter,
            order=order,
            animate=animate,
            selection_ratio=selection_ratio,
            rotation_range=rotation_range,
            rotation_jitter=rotation_jitter,
        )

    # Load from layout file
    elif layout:
        canvas.load_layout(layout)

    # Random placement
    elif random_count:
        if keep_cutouts:
            logger.info("Random placement: grouping by cutout (keep_cutouts=True)")
            # select random cutouts instead of individual segments
            index = ImageIndex(from_dir)
            index.load()

            candidates: list[dict] = []
            for entry in index.all():
                # we need the raw metadata to get positions/sizes
                with open(entry.metadata_path, encoding="utf-8") as f:
                    meta = json.load(f)
                for c in meta.get("cutouts", []):
                    label = c.get("label", "")
                    if obj_type is None or obj_type.lower() in label.lower():
                        candidates.append({
                            "cutout": c,
                            "base_dir": entry.metadata_path.parent,
                        })

            # apply color filtering to cutouts if requested
            if color_filter:
                filtered: list[dict] = []
                for item in candidates:
                    segs = item["cutout"].get("segments", [])
                    if any(_image_is_color(item["base_dir"] / s.get("file", ""), color_filter, color_threshold) for s in segs):
                        filtered.append(item)
                candidates = filtered

            if not candidates:
                logger.warning("No cutouts available for random placement")
            else:
                num_chosen = min(random_count, len(candidates))
                logger.info("Selected %d cutouts for placement", num_chosen)
                chosen = random.sample(candidates, num_chosen)

                # Count total segments across all cutouts for the progress bar
                total_segments = sum(
                    len(sel["cutout"].get("segments", []))
                    for sel in chosen
                )

                # Start ONE preview server for the entire placement loop
                preview_dir = None
                preview_image = None
                server = None
                if animate:
                    try:
                        import http.server
                        import socketserver
                        import threading
                        import tempfile
                        import webbrowser
                        import shutil

                        preview_dir = Path(tempfile.mkdtemp(prefix="canvas_preview_"))
                        preview_image = preview_dir / "canvas.png"

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

                        # Render initial canvas
                        import os as _os
                        canvas.render().save(preview_image)
                        (preview_dir / "status.json").write_text(
                            json.dumps({"placed": 0, "total": total_segments, "done": False})
                        )

                        class _QuietHandler(http.server.SimpleHTTPRequestHandler):
                            def __init__(self, *a, **kw):
                                super().__init__(*a, directory=str(preview_dir), **kw)
                            def log_message(self, fmt, *a):
                                pass

                        port = 8765
                        server = socketserver.TCPServer(("", port), _QuietHandler)
                        threading.Thread(target=server.serve_forever, daemon=True).start()
                        logger.info("Preview server running at http://localhost:%d", port)
                        webbrowser.open(f"http://localhost:{port}")
                        import time as _time
                        _time.sleep(0.5)
                    except Exception as exc:
                        logger.warning("Browser preview unavailable: %s", exc)
                        animate = False
                        preview_dir = None
                        preview_image = None

                # --- Prepare all cutout segment lists up-front ---
                import time as _time
                import os as _os

                all_prepared: list[list[dict]] = []
                for sel in chosen:
                    cutout_meta = sel["cutout"]
                    cutout_size = tuple(cutout_meta.get("size", [0, 0]))
                    x = random.randint(
                        margin,
                        max(margin, width - cutout_size[0] - margin),
                    )
                    y = random.randint(
                        margin,
                        max(margin, height - cutout_size[1] - margin),
                    )
                    segs = canvas.prepare_cutout_segments(
                        cutout_meta,
                        canvas_x=x,
                        canvas_y=y,
                        base_dir=sel["base_dir"],
                        jitter=jitter,
                        organic=organic,
                        organic_jitter=organic_jitter,
                        order=order,
                        selection_ratio=selection_ratio,
                        rotation_range=rotation_range,
                        rotation_jitter=rotation_jitter,
                    )
                    if segs:
                        all_prepared.append(segs)

                # --- Time-based parallel placement scheduler ---
                # Each track has its own independent timing with random
                # variation so they naturally drift apart (not lockstep).
                queue = list(all_prepared)  # cutouts waiting for a track slot
                placed_so_far = 0
                _needs_preview_update = False

                # Track state: list of (segment_list, current_index, next_time)
                active: list[tuple[list[dict], int, float]] = []

                now = _time.monotonic()

                def _fill_tracks() -> None:
                    """Fill empty track slots from the queue."""
                    nonlocal now
                    while len(active) < parallel_tracks and queue:
                        segs = queue.pop(0)
                        # Stagger start: each new track starts with a small
                        # random offset so they don't begin in sync.
                        offset = random.uniform(0, delay * 0.5) if active else 0
                        active.append((segs, 0, now + offset))

                _fill_tracks()

                while active:
                    # Find the track with the earliest next_time
                    earliest_idx = min(range(len(active)), key=lambda i: active[i][2])
                    segs_list, seg_idx, next_t = active[earliest_idx]

                    # Wait until it's time for this track's next segment
                    wait = next_t - _time.monotonic()
                    if wait > 0:
                        # Flush preview update before sleeping (batch: one
                        # render per wait, not per segment)
                        if _needs_preview_update:
                            if animate and preview_image is not None:
                                tmp = preview_image.with_suffix(".tmp.png")
                                canvas.render_incremental().save(tmp)
                                _os.replace(str(tmp), str(preview_image))
                                if preview_dir is not None:
                                    (preview_dir / "status.json").write_text(
                                        json.dumps({"placed": placed_so_far, "total": total_segments, "done": False})
                                    )
                            if save_each:
                                frame_path = str(save_each).format(n=placed_so_far)
                                canvas.save(frame_path)
                            _needs_preview_update = False
                        _time.sleep(wait)

                    # Place the segment
                    seg = segs_list[seg_idx]
                    canvas.place(seg["path"], seg["x"], seg["y"], rotation=seg["rotation"])
                    placed_so_far += 1
                    _needs_preview_update = True
                    now = _time.monotonic()

                    logger.info(
                        "Track %d placed segment %d/%d  (total %d/%d)",
                        earliest_idx + 1, seg_idx + 1, len(segs_list),
                        placed_so_far, total_segments,
                    )

                    seg_idx += 1
                    if seg_idx < len(segs_list):
                        # Schedule next segment for this track with ±40% jitter
                        jitter_factor = random.uniform(0.6, 1.4)
                        next_t = now + delay * jitter_factor
                        active[earliest_idx] = (segs_list, seg_idx, next_t)
                    else:
                        # Track finished — remove it and try to fill from queue
                        active.pop(earliest_idx)
                        _fill_tracks()

                # Final preview flush
                if _needs_preview_update:
                    if animate and preview_image is not None:
                        tmp = preview_image.with_suffix(".tmp.png")
                        canvas.render_incremental().save(tmp)
                        _os.replace(str(tmp), str(preview_image))
                        if preview_dir is not None:
                            (preview_dir / "status.json").write_text(
                                json.dumps({"placed": placed_so_far, "total": total_segments, "done": False})
                            )
                    if save_each:
                        frame_path = str(save_each).format(n=placed_so_far)
                        canvas.save(frame_path)

                # Signal done and shut down server
                if animate and preview_dir is not None:
                    tmp = preview_image.with_suffix(".tmp.png")
                    canvas.render().save(tmp)
                    _os.replace(str(tmp), str(preview_image))
                    (preview_dir / "status.json").write_text(
                        json.dumps({
                            "placed": placed_so_far,
                            "total": total_segments,
                            "done": True,
                        })
                    )
                    _time.sleep(2)  # give browser time to show final frame
                    if server is not None:
                        server.shutdown()
                    shutil.rmtree(preview_dir, ignore_errors=True)
                    logger.info("Preview complete.")
        else:
            canvas.place_random(
                from_dir,
                random_count,
                obj_type,
                animate=animate,
                delay=delay,
                jitter=jitter,
                color=color_filter,
                color_threshold=color_threshold,
            )

    # Grid layout
    elif grid:
        index = ImageIndex(from_dir)

        all_segs = []
        for entry in index.all():
            for cutout in entry.cutouts:
                if (
                    obj_type is None
                    or obj_type.lower() in cutout.label.lower()
                ):
                    all_segs.extend(cutout.segments)
        # apply color filter if requested
        if color_filter:
            all_segs = [p for p in all_segs if _image_is_color(p, color_filter, color_threshold)]
            if not all_segs:
                logger.warning("No segments left after color filter")

        canvas.place_grid(all_segs, cols=grid)

    else:
        logger.error(
            "No placement mode enabled. Edit %s to enable random, place_cutout, grid, or layout.",
            args.config,
        )
        return 1

    # Save outputs
    if save_layout:
        canvas.save_layout(Path(save_layout))

    canvas.save(output)
    print(f"Saved {len(canvas)} items to {output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
