# Segmentations (SEG)

Author and read coded DICOM Segmentation objects — SEG ↔ labelmap
conversion, in Python.

## Write a binary SEG

`write_seg` takes a source image (for geometry, demographics and references), a
`uint16` labelmap, and a list of segment terminology dicts:

```python
import numpy as np
import pydcm

labelmap = np.zeros((slices, rows, cols), dtype=np.uint16)
labelmap[mask_liver] = 1
labelmap[mask_tumor] = 2

segments = [
    {"label": "Liver", "labelID": 1, "rgb": (221, 130, 101),
     "category": ("123037004", "SCT", "Anatomical Structure"),
     "type":     ("10200004",  "SCT", "Liver")},
    {"label": "Tumor", "labelID": 2, "rgb": (255, 0, 0),
     "category": ("49755003",  "SCT", "Morphologically Abnormal Structure"),
     "type":     ("4147007",   "SCT", "Mass")},
]

pydcm.write_seg("ct_series/", labelmap, segments, output="seg.dcm")
```

- `reference` is a source-image path or a list of the series' instance paths;
  the slices of `labelmap` must be ordered by ascending position to match.
- Omit `output` to get Part-10 `bytes` back instead of writing a file.
- Each segment dict also accepts `anatomic`, `algorithm_type` and
  `algorithm_name`.

## Write a fractional SEG

For probability or occupancy maps (e.g. a model's softmax output), one float
map per segment:

```python
# one float map per segment, stacked segment-major (same order as `segments`)
maps = np.stack([prob_liver, prob_tumor])   # [n_segments, slices, rows, cols], floats in [0, 1]

pydcm.write_seg_fractional(
    "ct_series/",
    maps,
    segments,
    type="probability",                     # or "occupancy"
    output="seg_frac.dcm",
)
```

## Read a SEG back

```python
labelmap, meta = pydcm.read_seg("seg.dcm")          # default: the uint16 labelmap round-trips
masks,    meta = pydcm.read_seg("seg.dcm", masks=True)  # …or per-segment occupancy masks
```

`read_seg` returns `(array, meta)`. By default `array` is the reconstructed
`uint16` labelmap (geometry-correct, in the reference grid); with `masks=True`
it is per-segment masks `[n_segments, slices, rows, cols]`. `meta` carries the
segment terminology, geometry and the voxel→world affine.

## From a model prediction

Combined with [preprocessing](preprocessing.md), the loop is one call back to a
coded SEG — `write_seg_from_prediction` maps the label volume from the model's
(possibly resampled) grid onto the original DICOM grid for you:

```python
vol    = pydcm.load_series("ct_series/")
logits = pydcm.transforms.sliding_window_inference(vol.pixels, (96,)*3, model)
pred   = pydcm.transforms.argmax(logits, vol.affine)   # → a label Volume (carries its grid)

pydcm.write_seg_from_prediction(pred, "ct_series/", segments, output="pred.dcm")
```

`write_seg_from_prediction` takes the label **`Volume`** (not a bare array), so it
can resample the model's grid back onto the original DICOM grid for you.
