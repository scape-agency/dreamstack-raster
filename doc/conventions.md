# Conventions

These conventions are enforced across the public API. Internal code may
break them only at clearly-marked I/O boundaries (file decoders, native
backend bridges).

## Pixel data layout

| Property | Value |
| --- | --- |
| Array shape | `(H, W, C)` — height, width, channels. |
| Channel order | **RGBA** (or RGB / GRAY / GRAY+ALPHA). **No BGR in public API.** |
| Internal working dtype | `float32`, range `[0.0, 1.0]`. |
| Storage dtype | `uint8`, `uint16`, `float16`, `float32`, `float64` allowed via `BitDepth`. |
| Default working space | `sRGB`. |
| Default gamma state | `LINEAR` for compositing/blending; `ENCODED` (display) at I/O boundaries. |
| Default alpha state | `STRAIGHT` (non-premultiplied) at API boundary; `PREMULTIPLIED` inside compositing. |

`PixelData` carries `(working_space, gamma_state, alpha_state, bit_depth, pixel_format)`
so downstream code can convert as needed without guessing.

## Color management

Every transformation that depends on color meaning (adjustments, filters
that operate perceptually, blend kernels marked "linear") routes through
`dreamstack.raster.color.pipeline.ensure(...)` to coerce its input into
the gamma / space / alpha state it requires. Functions never silently
assume the array is linear or sRGB-encoded.

When converting to a different working space, conversion is performed
through linear-XYZ pivot using the source and target `ColorSpace`'s
transfer functions. ICC-profiled conversions use the LittleCMS backend
via Pillow for tagged 8/16-bit data; float / wide-gamut paths stay in
the numpy pipeline.

## Compositing

- All compositing happens in **float32**, **linear**, **premultiplied** space.
- The canonical `BlendMode` enum lives in
  `dreamstack.raster.core.layer.blend_mode` and is re-exported from
  `dreamstack.raster.compositing`.
- Component-wise math operations (`add`, `subtract`, `multiply`, …) live
  beside blend modes in `compositing.ops`; they are not a separate
  parallel API.

## I/O boundary

Format readers/writers are the only place where BGR, file-byte-order
endianness, ICC tag round-tripping, and uint8/uint16 quantization may
appear. Decoders are responsible for handing back `PixelData` with
`gamma_state` and `working_space` populated according to the file's
embedded profile (or a documented assumption when missing).

## Optional dependencies

The base install only pulls lightweight scientific Python stacks.
ML, raw, and HDR backends ship as extras and are imported lazily; an
import failure inside an optional backend raises an `ImportError` with
a clear `pip install dreamstack-raster[<extra>]` instruction.

| Extra | Pulls in |
| --- | --- |
| `detection` | ultralytics, transformers, segment-anything, groundingdino-py |
| `vision` | chimp-openai, chimp-mistral |
| `raw` | rawpy |
| `exr` | OpenEXR, imageio |
| `psd` | psd-tools |
| `heif` | pillow-heif |
