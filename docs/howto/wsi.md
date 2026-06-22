# Whole-slide imaging

`pydcm.wsi` reads **and writes** DICOM WSI pyramids — open a slide (one file or
a multi-file pyramid), inspect levels, read regions or tiles, and author a new
pyramid from RGB level arrays.
Validated bit-exact against reference decodes on multi-vendor WSI.

## Open and inspect

```python
from pydcm import wsi

slide = wsi.open_slide("wsi_dir/")     # a directory or a single instance
slide.level_count                       # number of pyramid levels
slide.level_dimensions                  # [(w0, h0), (w1, h1), …] per level
slide.properties                        # vendor / objective-power / MPP metadata
```

## Read a region

```python
region = slide.read_region((x, y), level=0, size=(512, 512))   # RGBA ndarray
rgb    = slide.read_region((x, y), level=0, size=(512, 512), rgba=False)
```

`read_region` uses level-0 reference coordinates: `location` is in level-0 reference
coordinates, `size` is in the target `level`'s pixels.

## Thumbnails and associated images

```python
thumb = slide.get_thumbnail((1024, 1024))      # RGB ndarray
label = slide.associated_images["label"]       # names are lowercase: "label" / "overview" / …
```

## Tiles for a viewer

For a tiled viewer or a tile-streaming pipeline, the level/tile grid is exposed
directly so you can fetch tiles without decoding a whole region:

```python
tile = slide.read_tile(level=0, tile=(3, 7))   # a single decoded tile (col, row)
desc = slide.viewer_level(0, include_ranges=True)   # level metadata + tile range grid
```

ICC handling is available where the slide carries a profile: `slide.icc_profile`
gives the raw profile, and `read_associated_image(..., apply_icc_profile=True)`
and `get_total_pixel_matrix(..., apply_icc_profile=True)` apply it.

## Write a slide

`write_slide` is the inverse of `open_slide` — author a DICOM WSI pyramid from
per-level RGB arrays (biggest level first) and get one Part-10 instance per level:

```python
parts = wsi.write_slide(levels, tile=256, mpp=0.25,   # levels: list of (H, W, 3) uint8, base first
                        transfer_syntax="1.2.840.10008.1.2.4.50")  # JPEG; .4.80 = lossless
for i, buf in enumerate(parts):
    open(f"level{i}.dcm", "wb").write(buf)
```

All levels share one Study / Series / Frame of Reference / specimen and a
conformant TILED_FULL pyramid. Pass a lossless transfer syntax (e.g.
`1.2.840.10008.1.2.4.80`) for a bit-exact round-trip, and `patient_id` /
`study_uid` / `specimen_id` to set identity. To author from a pyramidal TIFF,
read its tiles and pass the level arrays here.
