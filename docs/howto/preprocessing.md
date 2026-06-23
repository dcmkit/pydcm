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
