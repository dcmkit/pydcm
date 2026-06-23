# Preprocessing

`pydcm.transforms` ports the spatial and intensity operations a segmentation
pipeline needs, with the numerics pinned to the framework you train against.

!!! info "Why two precision tiers"
    There is no single gold-standard preprocessing library in Python. The
    classic B-spline family and the deep-learning (grid-sample) family use genuinely different
    interpolation conventions, so "correct" is defined per framework. pydcm
    matches each one exactly. Details: [Transforms — precision &amp; references](../transforms_references.md).

    - **Tier 1** — `resample_cubic`, `resample_separate_z`, `resample_nearest`
      are **bit-exact** for the classic B-spline convention.
    - **Tier 2** — `resample_grid_sample` matches the deep-learning trilinear convention to
      ≤ 1 float32 ULP.

## Resample

`resample_to_spacing` resamples a `Volume` to an axis-aligned grid at a target
mm spacing (the common "make it isotropic" step):

```python
import pydcm
from pydcm import transforms as T

vol = pydcm.load_series("ct_series/")
iso = T.resample_to_spacing(vol, (1.0, 1.0, 1.0), interp="linear")
seg = T.resample_to_spacing(label_vol, (1.0, 1.0, 1.0), is_label=True)  # nearest, no new labels
```

When you already know the output shape, choose the family that matches your model:

```python
out = T.resample_cubic(vol, (128, 256, 256))        # Tier 1 — bit-exact B-spline order-3
out = T.resample_separate_z(vol, (128, 256, 256))   # anisotropic separate-z behaviour
out = T.resample_grid_sample(vol, (128, 256, 256))  # Tier 2 — deep-learning trilinear convention
```

## Intensity normalization

```python
# CT normalization: clip to a window, then z-score with FIXED dataset-level
# mean/std (computed during planning):
ct = T.normalize_ct(vol, clip_lo=-1000, clip_hi=400, mean=-380.0, std=320.0)

z = T.normalize_zscore(vol, nonzero=True)            # per-volume, ignore background zeros
```

## Sliding-window inference

```python
logits = T.sliding_window_inference(
    vol.pixels,
    roi_size=(96, 96, 96),
    predictor=model,          # a callable patch → logits
    overlap=0.25,
    convention="nnunet",      # or "monai" — picks the gaussian importance map
)
```

## Label post-processing

```python
seg = T.argmax(logits, vol.affine)                   # logits [C, D, H, W] → labelled Volume
seg = T.keep_largest_connected_component(seg, per_class=True)
seg = T.remove_small_objects(seg, min_size=64)
seg = T.fill_holes(seg)
onehot = T.one_hot(seg, num_classes=3)               # [C, D, H, W]
```

Every op accepts a `Volume` (or a plain array where noted) and preserves the
geometry, so you can resample, infer, post-process, and write the result
straight back to a SEG (see [Segmentations](segmentation.md)) without losing the
affine.

## Pixel value transforms

Separate from the `pydcm.transforms` machine-learning operations above, the
standard DICOM display pipeline is exposed as the top-level `pydcm.apply_*`
spellings. They run on a decoded array — `pydcm.pixel_array` decodes stored
pixels through the native engine — and apply the PS3.3 pixel-value formulas:

```python
import pydcm

ds = pydcm.dcmread("ct.dcm")
stored = pydcm.pixel_array(ds)                 # decoded stored values

hu   = pydcm.apply_modality_lut(stored, ds)    # stored → modality units (e.g. HU)
disp = pydcm.apply_voi_lut(hu, ds)             # then window/level for display
```

`apply_modality_lut` applies the linear Modality LUT — Rescale Slope and
Rescale Intercept (PS3.3 C.11.1) — turning stored values into modality /
real-world units such as Hounsfield units. When neither is present the array is
returned unchanged.

`apply_voi_lut` applies the VOI transform (PS3.3 C.11.2): a **VOI LUT Sequence**
supersedes Window Center / Window Width when present; otherwise the window pair
is used with the `VOILUTFunction` (`LINEAR`, `LINEAR_EXACT` or `SIGMOID`).
`index=` selects which entry to use — the item in the VOI LUT Sequence, or the
position in a multi-valued Window Center / Width:

```python
disp = pydcm.apply_voi_lut(hu, ds, index=1)    # second window / VOI LUT item
```

Mind the output range — it is **not** `[0, 1]`, and the two branches differ:

- The **VOI LUT Sequence** branch returns the table's own values, spanning
  `0 … 2**bits − 1` for the LUT's declared bit depth.
- The **Window Center / Width** branch scales the windowed values to the input
  span, `0 … 2**BitsStored − 1`.

These are the standard DICOM display transforms; reach for the
`pydcm.transforms` `resample_*` / `normalize_*` operations above when you are
preparing volumes for a model instead.
