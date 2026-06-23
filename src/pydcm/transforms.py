# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Fuli Wu
# NOT a medical device — not for clinical or diagnostic use.

"""Deterministic medical-image transforms — pydcm's "ITK".

Thin marshalling over the native transform engine (CPU C/C++). This is
the **same** preprocessing whether you prepare training data here or deploy in the
browser (where ``dcmmodel`` runs the identical ops as WGSL, CI-verified equal), so
preprocessing cannot drift between train and serve.

Scope: the load-bearing deterministic ops — spatial (``resample_to_spacing`` and
``resize`` with label-safe nearest; the exact index ops ``crop``, ``pad``,
``crop_foreground``, ``flip``), intensity (``normalize_zscore``,
``scale_intensity_range``), post (``argmax``) — plus a minimal :class:`Compose`.
Random augmentation is intentionally *not* here (use MONAI/torchio for training aug).

Spatial ops operate on a :class:`~pydcm.volume.Volume` (``pixels[z,y,x]`` + LPS
affine) and return a new ``Volume``; intensity ops likewise. Geometry is never
recomputed in Python — the native engine owns it.

**Cross-framework conventions.** Most ops are convention-free (identical across
skimage/torch/ITK). Two things genuinely diverge between frameworks: the resampling
*interpolation* (skimage.resize spline vs torch ``grid_sample`` vs SimpleITK) and the
gaussian importance map. To get a self-consistent pipeline for one framework, import a
**preset** instead of mixing primitives::

    from pydcm.transforms import nnunet as T   # skimage.resize spline + nnU-Net gaussian
    from pydcm.transforms import monai as T   # torch grid_sample + MONAI gaussian

The preset binds the divergent ops to that framework and passes the rest through. See
``docs/transforms_references.md`` for the per-op authoritative reference + precision.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from . import _native
from .volume import Volume


def _affine_cm(affine: np.ndarray) -> list:
    """Row-major 4×4 → column-major flat-16 (the native voxel→world LPS convention)."""
    return np.asarray(affine, dtype=np.float64).T.flatten().tolist()


def _affine_from_meta(meta) -> np.ndarray:
    """Native column-major flat-16 → row-major 4×4 (matches Volume.affine)."""
    return np.asarray(meta["affine"], dtype=np.float64).reshape(4, 4).T


def _spacing_zyx(affine: np.ndarray) -> tuple:
    """(z, y, x) voxel spacing from a voxel→world affine = its column norms."""
    a = np.asarray(affine, dtype=np.float64)
    return (float(np.linalg.norm(a[:3, 2])),
            float(np.linalg.norm(a[:3, 1])),
            float(np.linalg.norm(a[:3, 0])))


def _default_is_label(arr: np.ndarray, is_label: bool | None) -> bool:
    """``is_label`` defaults to True for an integer (label) array, False for floating."""
    return (not np.issubdtype(arr.dtype, np.floating)) if is_label is None else is_label


def _spacing3(spacing) -> list:
    """Accept scalar (isotropic) or (x, y, z) → [x, y, z] floats (the native order)."""
    if isinstance(spacing, (int, float)):
        return [float(spacing)] * 3
    s = list(spacing)
    if len(s) != 3:
        raise ValueError("spacing must be a scalar or (x, y, z)")
    return [float(s[0]), float(s[1]), float(s[2])]


def resample_to_spacing(vol: Volume, spacing, *, is_label: bool | None = None,
                        interp: str = "linear") -> Volume:
    """Resample ``vol`` to an axis-aligned LPS grid at ``spacing`` mm.

    ``spacing`` is a scalar (isotropic) or ``(x, y, z)``. ``is_label`` defaults from
    dtype (integer → label → nearest); pass it to override. ``interp`` is
    ``"linear"`` / ``"cubic"`` / ``"nearest"`` (ignored for labels — always nearest).
    """
    core = _native.require()
    px = np.ascontiguousarray(vol.pixels)
    is_label = _default_is_label(px, is_label)
    sp = _spacing3(spacing)
    arr, meta = core.transform_resample_to_spacing(px, _affine_cm(vol.affine), sp, is_label, interp)
    return Volume(arr, (sp[2], sp[1], sp[0]), _affine_from_meta(meta), vol.series_instance_uid)


def resample_to_reference(moving: Volume, reference: Volume, *, is_label: bool | None = None,
                          interp: str = "linear", fill: float = 0.0) -> Volume:
    """Resample ``moving`` onto ``reference``'s grid (its shape + affine).

    This is how you **invert** a preprocessing chain: a prediction computed in some
    processed space (resampled / reoriented / cropped) is mapped back onto the
    original :class:`~pydcm.volume.Volume`'s grid by passing that original as
    ``reference``. It works through any sequence of geometric transforms (it uses the
    affines, not a recorded op-stack) and handles an oblique reference grid.
    ``is_label`` defaults from dtype (integer → nearest).

    **Bit-exact** with SimpleITK's ``sitk.Resample(moving, reference, sitkLinear)`` /
    ``sitkNearestNeighbor`` — the resample-to-reference pattern medical models use —
    both INSIDE and OUTSIDE the moving extent: reference voxels whose moving index leaves
    ``[-0.5, dim-0.5)`` on any axis are set to ``fill`` (SimpleITK's defaultPixelValue),
    not edge-extrapolated. ``fill=0`` is background — correct for a label map or a
    prediction mapped onto a larger grid; pass e.g. ``-1024`` for CT air.
    """
    core = _native.require()
    px = np.ascontiguousarray(moving.pixels)
    is_label = _default_is_label(px, is_label)
    arr, _ = core.transform_resample_to_reference(
        px, _affine_cm(moving.affine), [int(s) for s in reference.pixels.shape],
        _affine_cm(reference.affine), is_label, interp, float(fill))
    return Volume(arr, reference.spacing, np.asarray(reference.affine, dtype=np.float64),
                  reference.series_instance_uid)


def affine(vol: Volume, matrix, *, is_label: bool | None = None, interp: str = "linear") -> Volume:
    """Apply a voxel-space affine to the image (MONAI ``Affine``). ``matrix`` is a 4×4
    array — the forward source-voxel → output-voxel transform (rotate/scale/shear/translate),
    in engine voxel order ``(x, y, z)`` i.e. ``(W, H, D)``. The output keeps the source's
    grid, so content moved outside the field of view is clipped. ``is_label`` defaults from
    dtype (integer → nearest)."""
    core = _native.require()
    m = np.asarray(matrix, dtype=np.float32)
    if m.shape != (4, 4):
        raise ValueError("matrix must be 4x4 (voxel→voxel)")
    px = np.ascontiguousarray(vol.pixels)
    is_label = _default_is_label(px, is_label)
    arr, _ = core.transform_affine(px, _affine_cm(vol.affine), m.T.flatten().tolist(), is_label, interp)
    return Volume(arr, vol.spacing, vol.affine, vol.series_instance_uid)


def resample_separate_z(vol: Volume, out_shape) -> Volume:
    """nnU-Net anisotropic *separate-z* resample to ``out_shape`` ``(D, H, W)``: per-slice
    in-plane cubic B-spline + nearest through-plane (the low-res Z axis), with an fp64
    prefilter — the precision-faithful path for thick-slice MR/CBCT (matches nnU-Net /
    scipy ``map_coordinates``). For images (the in-plane spline blends labels)."""
    core = _native.require()
    arr, meta = core.transform_resample_separate_z(
        np.ascontiguousarray(vol.pixels), _affine_cm(vol.affine), [int(s) for s in out_shape])
    aff = _affine_from_meta(meta)
    return Volume(arr, _spacing_zyx(aff), aff, vol.series_instance_uid)


def resample_cubic(vol: Volume, out_shape) -> Volume:
    """Isotropic cubic B-spline resample to ``out_shape`` ``(D, H, W)`` — order=3 B-spline,
    fp64, **bit-exact** with ``skimage.resize(order=3, mode='edge', clip=True)`` (the
    nnU-Net image reference — note this is skimage.resize, NOT scipy.ndimage.zoom). For images."""
    core = _native.require()
    arr, meta = core.transform_resample_cubic(
        np.ascontiguousarray(vol.pixels), _affine_cm(vol.affine), [int(s) for s in out_shape])
    aff = _affine_from_meta(meta)
    return Volume(arr, _spacing_zyx(aff), aff, vol.series_instance_uid)


def resample_nearest(vol: Volume, out_shape) -> Volume:
    """Nearest-neighbour resample to ``out_shape`` ``(D, H, W)`` (order=0, half-pixel — matches
    ``skimage.resize(order=0)``) — the nnU-Net **label** path (no class blending)."""
    core = _native.require()
    arr, meta = core.transform_resample_nearest(
        np.ascontiguousarray(vol.pixels), _affine_cm(vol.affine), [int(s) for s in out_shape])
    aff = _affine_from_meta(meta)
    return Volume(arr, _spacing_zyx(aff), aff, vol.series_instance_uid)


def resample_grid_sample(vol: Volume, out_shape) -> Volume:
    """Trilinear resample to ``out_shape`` ``(D, H, W)`` in torch
    ``grid_sample(align_corners=False, bilinear, padding_mode='border')`` convention —
    half-pixel ``(j+0.5)*src/dst-0.5``. This is MONAI **Resize**'s backend
    (``F.interpolate``): verified index-for-index vs ``monai.transforms.Resize`` (9→6 →
    0.25,1.75,3.25,…). It is NOT MONAI **Spacing** — a spacing change in Spacing samples
    at ``scale*j`` (affine/world convention, 9@1.0→1.5 → 0,1.5,3,4.5,6,7.5); for that use
    :func:`resample_to_spacing`. fp64 internally; agrees with torch to ≤1 fp32 ULP (torch's
    ``affine_grid`` uses a non-reproducible SIMD ``linspace``). For images."""
    core = _native.require()
    arr, meta = core.transform_resample_grid_sample(
        np.ascontiguousarray(vol.pixels), _affine_cm(vol.affine), [int(s) for s in out_shape])
    aff = _affine_from_meta(meta)
    return Volume(arr, _spacing_zyx(aff), aff, vol.series_instance_uid)


def resize_2d(vol: Volume, out_hw, *, filt: str = "bilinear") -> Volume:
    """2-D resize, applied **per slice in-plane** to ``out_hw`` ``(out_H, out_W)`` (each (H, W)
    slice → (out_H, out_W); depth/slices preserved). All paths are **deterministic** — the CPU
    result and a GPU/WGSL port agree bit-for-bit (train/serve parity). ``filt``:

    - ``"bilinear"`` (default): plain bilinear, **no** anti-aliasing, half-pixel
      (``(o+0.5)*scale-0.5``), clamp, double. This is the convention the deployment actually
      uses (ai_segmeation's ``resizeBilinearU8`` / ``zoom2d.wgsl``) and the same kernel as
      ``cv2.INTER_LINEAR`` / ``F.interpolate(bilinear, align_corners=False)``. **Bit-exact with
      resizeBilinearU8**; ≤1 fp32 ULP vs cv2/torch (those use SIMD/FMA and disagree ≤1 ULP with
      each other — no single bit-exact target exists, so this clean form is the reproducible spec).
    - ``"bicubic"``: ``PIL.Image.resize`` bicubic (anti-aliased on downscale), **bit-exact with
      PIL** — MedSAM2's Python preprocessing (each windowed CT slice → 512²). uint8 → PIL
      fixed-point, float → PIL float convolution.
    - ``"pil-bilinear"``: PIL bilinear (anti-aliased), bit-exact with PIL.

    A ``uint8`` Volume gives integer output (resizeBilinearU8 truncates; PIL rounds); window/cast
    beforehand as the model does."""
    oh, ow = int(out_hw[0]), int(out_hw[1])
    core = _native.require()
    px, aff_cm = np.ascontiguousarray(vol.pixels), _affine_cm(vol.affine)
    if filt == "bilinear":
        arr, meta = core.transform_bilinear_resize2d(px, aff_cm, oh, ow)
    elif filt == "bicubic":
        arr, meta = core.transform_pil_resize2d(px, aff_cm, oh, ow, "bicubic")
    elif filt == "pil-bilinear":
        arr, meta = core.transform_pil_resize2d(px, aff_cm, oh, ow, "bilinear")
    else:
        raise ValueError(f"resize_2d: filt must be 'bilinear', 'bicubic', or 'pil-bilinear', got {filt!r}")
    aff = _affine_from_meta(meta)
    return Volume(arr, _spacing_zyx(aff), aff, vol.series_instance_uid)


def resample(vol: Volume, out_shape, *, backend: str = "skimage", is_label: bool | None = None) -> Volume:
    """Resample to ``out_shape`` ``(D, H, W)`` under a framework's interpolation convention:
    ``backend="skimage"`` (cubic B-spline, Tier-1 bit-exact — bit-identical to
    ``skimage.transform.resize(order=3, mode='edge', anti_aliasing=False)``, which is what
    nnU-Net's ``default_resampling`` uses; NOT ``scipy.ndimage.zoom``, whose half-pixel
    coordinate convention differs — hence the name is ``skimage``, not ``scipy``) / ``"torch"``
    (grid_sample, Tier-2 ≤1 fp32 ULP — also matches ``F.interpolate(mode='trilinear',
    align_corners=False)`` to ≤1 ULP, nnU-Net's opt-in torch resampling backend) / ``"itk"``
    (SimpleITK linear/nearest in double — bit-exact vs SimpleITK on realistic grids; ITK
    B-spline is deferred). Labels — or ``is_label=True`` — force nearest, which is convention-free.
    The single entry for choosing the lineage;
    :func:`resample_cubic` / :func:`resample_grid_sample` are its skimage / torch backends."""
    if _default_is_label(vol.pixels, is_label):
        return resample_nearest(vol, out_shape)
    core = _native.require()
    arr, meta = core.transform_resample(np.ascontiguousarray(vol.pixels), _affine_cm(vol.affine),
                                        [int(s) for s in out_shape], backend)
    aff = _affine_from_meta(meta)
    return Volume(arr, _spacing_zyx(aff), aff, vol.series_instance_uid)


def resize(vol: Volume, out_shape, *, is_label: bool | None = None, interp: str = "linear") -> Volume:
    """Resample ``vol`` to exactly ``out_shape`` ``(D, H, W)`` voxels over the same
    field of view (MONAI ``Resize``). ``is_label`` defaults from dtype."""
    core = _native.require()
    px = np.ascontiguousarray(vol.pixels)
    is_label = _default_is_label(px, is_label)
    arr, meta = core.transform_resize(px, _affine_cm(vol.affine), [int(s) for s in out_shape], is_label, interp)
    aff = _affine_from_meta(meta)
    return Volume(arr, _spacing_zyx(aff), aff, vol.series_instance_uid)


def resize_with_pad_or_crop(vol: Volume, size, *, mode: str = "constant", value: float = 0.0) -> Volume:
    """Force the volume to exactly ``size`` ``(z, y, x)`` by centre-cropping axes that are
    too large and centre-padding axes that are too small (MONAI ``ResizeWithPadOrCrop``)."""
    cropped = center_crop(vol, [min(s, int(t)) for s, t in zip(vol.pixels.shape, size)])
    lo, hi = [], []
    for cs, t in zip(cropped.pixels.shape, size):
        total = max(0, int(t) - cs)
        lo.append(total // 2)
        hi.append(total - total // 2)
    return pad(cropped, tuple(lo), tuple(hi), mode=mode, value=value)


def reorient(vol: Volume, axcodes: str = "LPS") -> Volume:
    """Reorient so increasing voxel index runs toward the target world directions
    (MONAI ``Orientation``). ``axcodes`` is 3 letters from L/R, P/A, S/I — the world is
    LPS, so ``"LPS"`` is the engine-canonical orientation. Exact axis permutation +
    flips; world coordinates are unchanged. Raises on an oblique affine."""
    core = _native.require()
    arr, meta = core.transform_reorient(np.ascontiguousarray(vol.pixels), _affine_cm(vol.affine), str(axcodes))
    aff = _affine_from_meta(meta)
    return Volume(arr, _spacing_zyx(aff), aff, vol.series_instance_uid)


def crop(vol: Volume, start, size) -> Volume:
    """Crop the ``[start, start+size)`` box; ``start``/``size`` are ``(z, y, x)``
    (numpy axis order). Exact — no interpolation. The affine origin shifts to keep
    world coordinates."""
    core = _native.require()
    arr, meta = core.transform_crop(np.ascontiguousarray(vol.pixels), _affine_cm(vol.affine),
                                    [int(s) for s in start], [int(s) for s in size])
    return Volume(arr, vol.spacing, _affine_from_meta(meta), vol.series_instance_uid)


def pad(vol: Volume, lo, hi, *, mode: str = "constant", value: float = 0.0) -> Volume:
    """Pad ``lo``/``hi`` voxels before/after each axis (``(z, y, x)``). ``mode`` is
    ``"constant"`` / ``"edge"`` / ``"reflect"`` (MONAI ``SpatialPad``/``BorderPad``)."""
    core = _native.require()
    arr, meta = core.transform_pad(np.ascontiguousarray(vol.pixels), _affine_cm(vol.affine),
                                   [int(s) for s in lo], [int(s) for s in hi], mode, float(value))
    return Volume(arr, vol.spacing, _affine_from_meta(meta), vol.series_instance_uid)


def crop_foreground(vol: Volume, *, margin: int = 0) -> Volume:
    """Crop to the bounding box of non-zero voxels, expanded by ``margin`` (MONAI
    ``CropForeground``). Returns ``vol`` unchanged if every voxel is zero."""
    core = _native.require()
    arr, meta = core.transform_crop_foreground(np.ascontiguousarray(vol.pixels),
                                               _affine_cm(vol.affine), int(margin))
    return Volume(arr, vol.spacing, _affine_from_meta(meta), vol.series_instance_uid)


def flip(vol: Volume, axis) -> Volume:
    """Reverse voxel order along the flagged axes. ``axis`` is a ``(z, y, x)`` triple
    of bools (MONAI ``Flip``). The affine updates so world coordinates are unchanged."""
    core = _native.require()
    arr, meta = core.transform_flip(np.ascontiguousarray(vol.pixels), _affine_cm(vol.affine),
                                    [int(bool(a)) for a in axis])
    return Volume(arr, vol.spacing, _affine_from_meta(meta), vol.series_instance_uid)


def transpose(vol: Volume, axes) -> Volume:
    """Permute the voxel axes (numpy/MONAI ``transpose``); ``axes`` is a permutation of
    ``(0, 1, 2)`` in ``(z, y, x)`` order — output axis ``i`` is input axis ``axes[i]``.
    Exact (no interpolation); the affine's axis columns permute so world coordinates are
    unchanged. This is nnU-Net's ``transpose_forward``; the inverse (``transpose_backward``)
    is ``transpose(v, np.argsort(axes))``."""
    core = _native.require()
    ax = [int(a) for a in axes]
    if sorted(ax) != [0, 1, 2]:
        raise ValueError("transpose: axes must be a permutation of (0, 1, 2)")
    arr, meta = core.transform_transpose(np.ascontiguousarray(vol.pixels), _affine_cm(vol.affine), ax)
    aff = _affine_from_meta(meta)
    return Volume(arr, _spacing_zyx(aff), aff, vol.series_instance_uid)


def center_crop(vol: Volume, size, *, is_label: bool | None = None) -> Volume:
    """Center-crop to ``size`` ``(z, y, x)`` (MONAI ``CenterSpatialCrop``;
    ``start = dim//2 - size//2``). A size larger than the source is clamped to the
    source (crop only, no pad). Exact; the affine origin shifts to keep world coords."""
    core = _native.require()
    lab = _default_is_label(vol.pixels, is_label)
    arr, meta = core.transform_center_crop(np.ascontiguousarray(vol.pixels), _affine_cm(vol.affine),
                                           [int(s) for s in size], lab)
    return Volume(arr, vol.spacing, _affine_from_meta(meta), vol.series_instance_uid)


def spatial_pad(vol: Volume, size, *, mode: str = "constant", value: float = 0.0,
                is_label: bool | None = None) -> Volume:
    """Centered pad to at least ``size`` ``(z, y, x)`` (MONAI ``SpatialPad`` symmetric).
    Axes already ≥ ``size`` are untouched. ``mode`` ∈ ``constant``/``edge``/``reflect``."""
    core = _native.require()
    lab = _default_is_label(vol.pixels, is_label)
    arr, meta = core.transform_spatial_pad(np.ascontiguousarray(vol.pixels), _affine_cm(vol.affine),
                                           [int(s) for s in size], mode, float(value), lab)
    return Volume(arr, vol.spacing, _affine_from_meta(meta), vol.series_instance_uid)


def divisible_pad(vol: Volume, k, *, mode: str = "constant", value: float = 0.0,
                  is_label: bool | None = None) -> Volume:
    """Centered pad so each axis becomes a multiple of ``k`` (MONAI ``DivisiblePad`` —
    e.g. make dims divisible by 2^depth for a U-Net). ``k`` is a scalar or ``(z, y, x)``."""
    core = _native.require()
    kk = [int(k)] * 3 if isinstance(k, int) else [int(v) for v in k]
    if len(kk) != 3:
        raise ValueError("k must be an int or (z, y, x)")
    lab = _default_is_label(vol.pixels, is_label)
    arr, meta = core.transform_divisible_pad(np.ascontiguousarray(vol.pixels), _affine_cm(vol.affine),
                                             kk, mode, float(value), lab)
    return Volume(arr, vol.spacing, _affine_from_meta(meta), vol.series_instance_uid)


def rotate90(vol: Volume, k: int = 1, axes=(1, 2), *, is_label: bool | None = None) -> Volume:
    """Rotate ``k*90°`` in the plane of ``axes`` (numpy ``(D, H, W)`` axis indices;
    default ``(1, 2)`` = the in-plane H–W axes) — MONAI ``Rotate90`` / ``np.rot90``.
    Exact (no interpolation); the affine updates so world coordinates are kept."""
    core = _native.require()
    a0, a1 = int(axes[0]), int(axes[1])
    lab = _default_is_label(vol.pixels, is_label)
    arr, meta = core.transform_rotate90(np.ascontiguousarray(vol.pixels), _affine_cm(vol.affine),
                                        int(k), a0, a1, lab)
    aff = _affine_from_meta(meta)
    return Volume(arr, _spacing_zyx(aff), aff, vol.series_instance_uid)


def normalize_zscore(vol: Volume, *, nonzero: bool = False) -> Volume:
    """z-score normalize voxels (→ float32). ``nonzero=True`` ignores zero voxels
    (MONAI ``NormalizeIntensity(nonzero=True)``)."""
    core = _native.require()
    arr, _ = core.transform_normalize_zscore(np.ascontiguousarray(vol.pixels), _affine_cm(vol.affine), nonzero)
    return Volume(arr, vol.spacing, vol.affine, vol.series_instance_uid)


def scale_intensity_range(vol: Volume, a_min: float, a_max: float, b_min: float, b_max: float,
                          *, clip: bool = True) -> Volume:
    """Linearly remap ``[a_min, a_max] → [b_min, b_max]`` (→ float32). CT windowing
    (MONAI ``ScaleIntensityRange``). ``clip`` bounds the output to ``[b_min, b_max]``."""
    core = _native.require()
    arr, _ = core.transform_scale_intensity_range(
        np.ascontiguousarray(vol.pixels), _affine_cm(vol.affine), a_min, a_max, b_min, b_max, clip)
    return Volume(arr, vol.spacing, vol.affine, vol.series_instance_uid)


def normalize_ct(vol: Volume, clip_lo: float, clip_hi: float, mean: float, std: float) -> Volume:
    """Clip to ``[clip_lo, clip_hi]`` then z-score with **fixed** ``mean``/``std``
    (→ float32). This is nnU-Net ``CTNormalization`` (clip to the dataset's
    ``[0.5, 99.5]`` percentiles, normalize by the dataset mean/std) and equivalently
    MONAI ``NormalizeIntensity(subtrahend=mean, divisor=std)`` preceded by a clip."""
    core = _native.require()
    arr, _ = core.transform_normalize_ct(np.ascontiguousarray(vol.pixels), _affine_cm(vol.affine),
                                         float(clip_lo), float(clip_hi), float(mean), float(std))
    return Volume(arr, vol.spacing, vol.affine, vol.series_instance_uid)


def rescale_robust(vol: Volume, *, dst_min: float = 0.0, dst_max: float = 255.0,
                   f_low: float = 0.0, f_high: float = 0.999) -> Volume:
    """FreeSurfer/FastSurfer "conform" ROBUST intensity rescale to ``[dst_min, dst_max]``
    (→ float32). A 1000-bin histogram picks a robust source range, ignoring the ``f_low``
    fraction of all voxels at the bottom and ``(1 - f_high)`` of the *non-zero* voxels at
    the top (mri_convert defaults f_low=0, f_high=0.999), then
    ``x → clip(dst_min + scale*(x - src_min))``. This is the intensity step of the
    FastSurfer / SynthSeg / DL-DiReCT brain "conform" pipeline — the orientation + 1 mm
    resample steps are :func:`reorient` + :func:`resample_to_spacing`; cast the result to
    ``uint8`` for those models. Bit-faithful to ``conform.py`` getscale()+scalecrop()."""
    core = _native.require()
    arr, _ = core.transform_rescale_robust(np.ascontiguousarray(vol.pixels), _affine_cm(vol.affine),
                                           float(dst_min), float(dst_max), float(f_low), float(f_high))
    return Volume(arr, vol.spacing, vol.affine, vol.series_instance_uid)


def scale_intensity_range_percentiles(vol: Volume, lower: float, upper: float,
                                      b_min: float, b_max: float, *, clip: bool = True) -> Volume:
    """Like :func:`scale_intensity_range`, but ``a_min``/``a_max`` are the per-image
    ``lower``/``upper`` percentiles (0..100, ``np.percentile`` linear) — MONAI
    ``ScaleIntensityRangePercentiles``. → float32."""
    core = _native.require()
    arr, _ = core.transform_scale_intensity_range_percentiles(
        np.ascontiguousarray(vol.pixels), _affine_cm(vol.affine),
        float(lower), float(upper), float(b_min), float(b_max), clip)
    return Volume(arr, vol.spacing, vol.affine, vol.series_instance_uid)


def adjust_contrast(vol: Volume, gamma: float) -> Volume:
    """Gamma contrast ``((x-min)/(range+1e-7))**gamma * range + min`` (MONAI
    ``AdjustContrast``). → float32."""
    core = _native.require()
    arr, _ = core.transform_adjust_contrast(
        np.ascontiguousarray(vol.pixels), _affine_cm(vol.affine), float(gamma))
    return Volume(arr, vol.spacing, vol.affine, vol.series_instance_uid)


def gaussian_smooth(vol: Volume, sigma) -> Volume:
    """Separable Gaussian smoothing (MONAI ``GaussianSmooth``). ``sigma`` is a scalar
    (isotropic) or ``(z, y, x)`` in voxels; ``sigma<=0`` on an axis skips it. → float32."""
    core = _native.require()
    sig = [float(sigma)] * 3 if isinstance(sigma, (int, float)) else [float(s) for s in sigma]
    if len(sig) != 3:
        raise ValueError("sigma must be a scalar or (z, y, x)")
    arr, _ = core.transform_gaussian_smooth(np.ascontiguousarray(vol.pixels), _affine_cm(vol.affine), sig)
    return Volume(arr, vol.spacing, vol.affine, vol.series_instance_uid)


def argmax(probs: np.ndarray, affine: np.ndarray, *, channel_dim: int = 0) -> Volume:
    """Channel-wise argmax of a probability/logit array → a label :class:`Volume`.

    ``probs`` is channel-first ``[C, z, y, x]`` by default (torch/MONAI convention);
    set ``channel_dim`` for another layout. Output is uint8 (C ≤ 256) else uint16.
    """
    core = _native.require()
    p = np.moveaxis(np.asarray(probs), channel_dim, -1)        # → [z, y, x, C]
    arr, meta = core.transform_argmax(np.ascontiguousarray(p), _affine_cm(affine))
    out_affine = _affine_from_meta(meta)
    return Volume(arr, _spacing_zyx(out_affine), out_affine, "")


def connected_components(vol: Volume, *, connectivity: int = 6) -> Volume:
    """Label each connected component of the non-zero foreground with a distinct id
    (1..N) → uint16 (≈ ``scipy.ndimage.label``). ``connectivity`` is 6/18/26."""
    core = _native.require()
    arr, _ = core.transform_connected_components(np.ascontiguousarray(vol.pixels),
                                                 _affine_cm(vol.affine), int(connectivity))
    return Volume(arr, vol.spacing, vol.affine, vol.series_instance_uid)


def keep_largest_connected_component(vol: Volume, *, connectivity: int = 6,
                                     per_class: bool = True) -> Volume:
    """Keep only the largest connected component, zeroing the rest (MONAI
    ``KeepLargestConnectedComponent``). ``per_class=True`` keeps each non-zero class's
    own largest CC; ``False`` treats all non-zero as one foreground."""
    core = _native.require()
    arr, _ = core.transform_keep_largest_cc(np.ascontiguousarray(vol.pixels),
                                            _affine_cm(vol.affine), int(connectivity), per_class)
    return Volume(arr, vol.spacing, vol.affine, vol.series_instance_uid)


def fill_holes(vol: Volume, *, connectivity: int = 6) -> Volume:
    """Fill holes — background regions fully enclosed by a label — by setting them to
    that label (MONAI ``FillHoles``). Each non-zero class is filled independently.
    ``connectivity`` (6/18/26) is that of the background."""
    core = _native.require()
    arr, _ = core.transform_fill_holes(np.ascontiguousarray(vol.pixels),
                                       _affine_cm(vol.affine), int(connectivity))
    return Volume(arr, vol.spacing, vol.affine, vol.series_instance_uid)


def as_discrete(vol: Volume, *, threshold: float) -> Volume:
    """Binarize at ``threshold`` (``value > threshold`` → 1) → uint8 label (MONAI
    ``AsDiscrete(threshold=...)``; the sigmoid-output counterpart of :func:`argmax`)."""
    core = _native.require()
    arr, _ = core.transform_as_discrete(np.ascontiguousarray(vol.pixels), _affine_cm(vol.affine), float(threshold))
    return Volume(arr, vol.spacing, vol.affine, vol.series_instance_uid)


def remove_small_objects(vol: Volume, *, min_size: int, connectivity: int = 6,
                         per_class: bool = True) -> Volume:
    """Zero connected components smaller than ``min_size`` voxels (MONAI
    ``RemoveSmallObjects``). ``per_class=True`` prunes each non-zero class independently."""
    core = _native.require()
    arr, _ = core.transform_remove_small_objects(
        np.ascontiguousarray(vol.pixels), _affine_cm(vol.affine), int(min_size), int(connectivity), per_class)
    return Volume(arr, vol.spacing, vol.affine, vol.series_instance_uid)


def one_hot(vol: Volume, num_classes: int, *, channel_first: bool = True) -> np.ndarray:
    """Integer-label Volume → one-hot float32 array (MONAI ``AsDiscrete(to_onehot=...)``).
    ``channel_first=True`` → ``[C, z, y, x]`` (torch); else ``[z, y, x, C]``. Pure NumPy —
    a training-side helper, not a browser-inference op."""
    lab = np.asarray(vol.pixels).astype(np.int64)
    oh = np.eye(int(num_classes), dtype=np.float32)[lab]              # [z, y, x, C]
    return np.moveaxis(oh, -1, 0) if channel_first else oh


def sliding_window_positions(spatial_shape, roi_size, *, overlap: float = 0.25) -> np.ndarray:
    """Patch origins for sliding-window inference over a ``(D, H, W)`` volume with window
    ``roi_size`` and fractional ``overlap`` ∈ [0, 1) — MONAI ``dense_patch_slices`` /
    ``_get_scan_interval`` (fixed scan interval, last patch shifted inward). Returns an
    ``(n, 3)`` int array of ``(z, y, x)`` origins. Raises if ``roi`` exceeds ``spatial``."""
    core = _native.require()
    flat = core.transform_sliding_window_positions(
        [int(s) for s in spatial_shape], [int(r) for r in roi_size], float(overlap))
    return np.asarray(flat, dtype=np.int64).reshape(-1, 3)


def gaussian_importance_map(roi_size, *, sigma_scale: float = 0.125,
                            convention: str = "nnunet") -> np.ndarray:
    """Gaussian blend-weight window of shape ``roi_size`` ``(D, H, W)`` — separable product
    of 1D sampled gaussians, ``sigma = roi*sigma_scale``. → float32.

    ``convention`` picks the framework's map (the two are **not** interchangeable):

    - ``"nnunet"`` (default) — center ``roi//2``, peak 1, min = natural corner. Bit-exact
      with nnU-Net V2 ``get_gaussian`` / the deployed ai_segmeation maps.
    - ``"monai"`` — center ``(roi-1)/2``, unnormalized (peak ≈0.91), min clamped to
      ``max(min, 1e-3)``. Matches MONAI ``compute_importance_map(mode='gaussian')``."""
    core = _native.require()
    arr, _ = core.transform_gaussian_importance_map(
        [int(r) for r in roi_size], float(sigma_scale), convention)
    return arr


def sliding_window_inference(image, roi_size, predictor, *, overlap: float = 0.25,
                             mode: str = "gaussian", sigma_scale: float = 0.125,
                             convention: str = "nnunet",
                             padding_mode: str = "constant", cval: float = 0.0) -> np.ndarray:
    """Run ``predictor`` over sliding windows and blend the patch outputs — the
    deterministic reproduction of MONAI ``sliding_window_inference``.

    NOT for production serving — this is a pure-NumPy CPU **reference** (prototype /
    parity oracle). The serving path is :class:`pydcm.infer.Service` (native dcminfer
    GPU pipeline); register a model there and use ``segment(volume=/image=)`` instead.

    ``image`` is ``(D, H, W)`` or ``(C_in, D, H, W)``. ``predictor`` maps one patch
    (same leading shape as ``image``, spatial ``roi_size``) to logits/probs
    ``(C_out, *roi_size)``. Windows use :func:`sliding_window_positions`; overlaps are
    combined with a ``mode='gaussian'`` importance map (``sigma_scale`` + ``convention``,
    ``'nnunet'``/``'monai'`` — see :func:`gaussian_importance_map`) or ``mode='constant'``
    (uniform) weight, accumulated and normalized by the summed weight. If the volume is
    smaller than ``roi_size`` it is padded (``padding_mode``/``cval``) and the result is
    cropped back. Returns ``(C_out, D, H, W)``."""
    image = np.asarray(image)
    if image.ndim == 3:
        lead, spatial = (), image.shape
    elif image.ndim == 4:
        lead, spatial = image.shape[:1], image.shape[1:]
    else:
        raise ValueError("image must be (D, H, W) or (C_in, D, H, W)")
    roi = tuple(int(r) for r in roi_size)
    if len(roi) != 3:
        raise ValueError("roi_size must be (D, H, W)")

    # MONAI pads up to roi (per axis: half = diff//2) and crops the result back.
    diff = [max(roi[i] - spatial[i], 0) for i in range(3)]
    lo = [d // 2 for d in diff]
    if any(diff):
        pad_w = [(0, 0)] * len(lead) + [(lo[i], diff[i] - lo[i]) for i in range(3)]
        kw = {"constant_values": cval} if padding_mode == "constant" else {}
        image = np.pad(image, pad_w, mode=padding_mode, **kw)
        spatial = image.shape[len(lead):]

    weight = (gaussian_importance_map(roi, sigma_scale=sigma_scale, convention=convention)
              if mode == "gaussian" else np.ones(roi, dtype=np.float32))
    accum = None
    count = np.zeros(spatial, dtype=np.float32)
    for z, y, x in sliding_window_positions(spatial, roi, overlap=overlap):
        sl = (..., slice(z, z + roi[0]), slice(y, y + roi[1]), slice(x, x + roi[2]))
        logits = np.asarray(predictor(image[sl]), dtype=np.float32)   # (C_out, *roi)
        if logits.shape[-3:] != roi:
            raise ValueError(f"predictor returned spatial {logits.shape[-3:]}, expected {roi}")
        if accum is None:
            accum = np.zeros((logits.shape[0], *spatial), dtype=np.float32)
        accum[:, z:z + roi[0], y:y + roi[1], x:x + roi[2]] += logits * weight
        count[z:z + roi[0], y:y + roi[1], x:x + roi[2]] += weight
    if accum is None:
        raise RuntimeError("sliding_window_inference: no patches generated")
    out = accum / count                                              # broadcast over C_out
    # crop back the padding to the original spatial extent (padded - diff)
    out = out[:, lo[0]:lo[0] + (spatial[0] - diff[0]),
                 lo[1]:lo[1] + (spatial[1] - diff[1]),
                 lo[2]:lo[2] + (spatial[2] - diff[2])]
    return out


class Compose:
    """Apply a sequence of transforms left-to-right (MONAI ``Compose``).

    Each entry is a callable ``Volume -> Volume``; use ``functools.partial`` or a
    lambda to bind parameters, e.g.
    ``Compose([lambda v: resample_to_spacing(v, 1.0), normalize_zscore])``.
    """

    def __init__(self, transforms: Sequence):
        self.transforms = list(transforms)

    def __call__(self, vol: Volume) -> Volume:
        for t in self.transforms:
            vol = t(vol)
        return vol


# ── Framework presets ────────────────────────────────────────────────────────
# A preset is a self-consistent view of this module for ONE framework: the few ops
# that genuinely diverge (resampling backend, gaussian importance map) are bound to
# that framework's convention; every convention-free op is shared as-is. Pick one
# (`from pydcm.transforms import nnunet as T`) so a pipeline can't accidentally mix
# scipy-spline resampling with a MONAI gaussian, etc.

def _build_preset(name: str, *, gaussian_convention: str, resample_backend: str):
    import types
    _gim, _swi = gaussian_importance_map, sliding_window_inference

    ns = types.SimpleNamespace(name=name, gaussian_convention=gaussian_convention,
                               resample_backend=resample_backend)
    for attr, obj in globals().items():               # share every convention-free op as-is
        if not attr.startswith("_") and callable(obj) and getattr(obj, "__module__", None) == __name__:
            setattr(ns, attr, obj)

    def _resample(vol, out_shape, *, is_label=None):
        """Resample to ``out_shape`` ``(D, H, W)`` with this framework's interpolation
        convention (nnU-Net → skimage.resize cubic spline; MONAI → torch grid_sample). Labels
        force nearest. For nnU-Net's anisotropic separate-z path call ``resample_separate_z``."""
        return resample(vol, out_shape, backend=resample_backend, is_label=is_label)

    def _gaussian(roi_size, *, sigma_scale: float = 0.125):
        return _gim(roi_size, sigma_scale=sigma_scale, convention=gaussian_convention)

    def _sliding(image, roi_size, predictor, **kw):
        kw.setdefault("convention", gaussian_convention)
        return _swi(image, roi_size, predictor, **kw)

    ns.resample = _resample
    ns.gaussian_importance_map = _gaussian
    ns.sliding_window_inference = _sliding
    return ns


nnunet = _build_preset("nnunet", gaussian_convention="nnunet", resample_backend="skimage")
monai = _build_preset("monai", gaussian_convention="monai", resample_backend="torch")

__all__ = ['Compose', 'adjust_contrast', 'affine', 'argmax', 'as_discrete', 'center_crop',
           'connected_components', 'crop', 'crop_foreground', 'divisible_pad', 'fill_holes',
           'flip', 'gaussian_importance_map', 'gaussian_smooth',
           'keep_largest_connected_component', 'normalize_ct', 'normalize_zscore',
           'one_hot', 'pad', 'remove_small_objects', 'reorient', 'resample',
           'resample_cubic', 'resample_grid_sample', 'resample_nearest',
           'resample_separate_z', 'resample_to_reference', 'resample_to_spacing',
           'rescale_robust', 'resize', 'resize_2d', 'resize_with_pad_or_crop', 'rotate90',
           'scale_intensity_range', 'scale_intensity_range_percentiles',
           'sliding_window_inference', 'sliding_window_positions', 'spatial_pad',
           'transpose']
