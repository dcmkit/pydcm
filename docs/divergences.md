# Behaviour notes

pydcm aims for value-for-value correctness with the conventions Python DICOM
users already expect — for anything not listed here, standard DICOM behaviour
applies. The points below are pydcm's deliberate choices and current limits.

## Deliberate behaviours

1. **PALETTE COLOR `pixel_array` auto-expands to RGB.** pydcm applies the
   palette LUT and returns an `[H, W, 3]` RGB array — usually what you want. For
   the raw index image, use `pydcm.pixels.apply_color_lut` to control the
   index → RGB step yourself.

2. **`save_as` of a Big-Endian / Deflated image raises** rather than silently
   rewriting it. Deep Big-Endian support in the native parser was judged too
   risky for a retired transfer syntax — pydcm refuses rather than produce a file
   with dropped pixels.

3. **Big-Endian `pixel_array` byteorder is normalized to native.** The values
   are unchanged; only the array's reported byteorder differs (`int16` rather
   than `>i2`).

## Extra detail you may notice

pydcm is a superset, so you may see more than the minimum:

- **Private-tag names** resolve from a 12,608-pattern vendor dictionary, so tags
  that are often printed as `Unknown` come back named (e.g. `DLP`,
  `Total Saving Dose`).
- **One private sequence** that is usually left opaque, pydcm parses.
- **The keyword ↔ tag ↔ VR dictionary** holds 17,699 entries — a strict superset
  of the standard dictionaries — so attribute names resolve broadly.
- **pydcm encodes too:** `Dataset.compress(ts)` transcodes to RLE /
  JPEG-2000-lossless / JPEG-LS / HTJ2K / JPEG-XL, bit-lossless and genuinely
  encapsulated — no plugin stack required.

## Known limits

- `file_meta` surfaces the three mandatory group-0002 UIDs (Media Storage SOP
  Class / Instance, Transfer Syntax), not the optional group-0002 elements.
- A from-scratch `save_as` (no source file) writes Implicit **or** Explicit VR
  Little Endian — it honours `file_meta.TransferSyntaxUID` / `implicit_vr=` and
  includes uncompressed PixelData — but does not produce a compressed/
  encapsulated or Big-Endian file from scratch (use `Dataset.compress(ts)` for
  compression).

## Reporting unexpected behaviour

If you hit a behaviour that isn't listed here and looks wrong, it probably is —
the policy is value-for-value correctness except where this page says otherwise.
