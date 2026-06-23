# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Fuli Wu
# NOT a medical device — not for clinical or diagnostic use.

"""pydcm DTI — diffusion tensor estimation, scalar maps and deterministic tracking.

A thin NumPy surface over the native dcm_dti engine. Every number is computed in
the C core; this module marshals arrays, orchestrates a DICOM series, and refuses
the inputs the engine would otherwise accept and quietly mis-fit.

From a DICOM series::

    import pydcm.dti as dti
    res = dti.fit_series("ep2d_diff/")      # b0 split, fit, maps
    res["FA"], res["MD"]                    # (Z, Y, X) float32
    res.affine                              # voxel -> patient, row-major 4x4

    tracks = dti.track_series("ep2d_diff/") # streamlines in PATIENT mm
    dti.write_tracts("ep2d_diff/", tracks, "tracts.dcm")

From arrays already in hand::

    res = dti.fit(volumes, bvals, bvecs)    # volumes (V, Z, Y, X)
    maps = dti.fit_maps(b0, dwi, bvals, bvecs, maps=("FA", "MD"))   # flat voxel axis

Units: b-values in s/mm², so diffusivities (MD/AD/RD) are mm²/s and FA is
dimensionless. Streamline coordinates are mm — grid mm from :func:`track`, patient
mm from :func:`track_series` and anything that goes to DICOM.

Two things this module refuses rather than passing through, because the engine
cannot tell and the result looks plausible either way:

* **b-vectors must be unit length.** The design matrix carries |g|², so a
  half-length vector scales every diffusivity by 4 while leaving FA untouched.
* **baselines are not `bvals == 0`.** UIH writes 1.25 for its b=0 volume and
  Siemens reports 50 for volumes that are baselines in every other respect, so the
  split goes through the native per-manufacturer threshold.

And one it cannot: ``fit_maps`` takes ``dwi`` direction-major, and when n_dirs
equals n_voxels a transposed array is shape-valid under both readings, so no check
distinguishes them. :func:`fit` and :func:`fit_series` build that array themselves
and are the reason to prefer them — the raw tier is there for callers who already
have a flat voxel axis and know which way round it is.
"""
from __future__ import annotations

import math

import numpy as np

from . import _core

__all__ = ["MAPS", "baseline_mask", "head_mask", "fit_maps", "fit", "fit_series",
           "track", "track_series", "write_tracts", "DtiResult"]

# Every map name the native engine accepts. FA/MD/AD/RD are float32 per voxel;
# CL/CP/CS are Westin-1997 (divided by lambda1); linearity/planarity/sphericity
# are the trace-normalised form of the same three; DEC is uint8 RGBA.
MAPS = ("FA", "MD", "AD", "RD", "DEC", "CL", "CP", "CS",
        "linearity", "planarity", "sphericity")

# UCUM units per map, for callers emitting parametric maps.
_MAP_UNITS = {
    "FA": ("1", "UCUM", "no units"),
    "MD": ("mm2/s", "UCUM", "mm2/s"),
    "AD": ("mm2/s", "UCUM", "mm2/s"),
    "RD": ("mm2/s", "UCUM", "mm2/s"),
    "CL": ("1", "UCUM", "no units"),
    "CP": ("1", "UCUM", "no units"),
    "CS": ("1", "UCUM", "no units"),
    "linearity":  ("1", "UCUM", "no units"),
    "planarity":  ("1", "UCUM", "no units"),
    "sphericity": ("1", "UCUM", "no units"),
}

_DEFAULT_MAPS = ("FA", "MD")


def _f32(a):
    """C-contiguous float32. Explicit because the bindings declare const float and
    would otherwise let nanobind copy a whole float64 DWI volume implicitly."""
    return np.ascontiguousarray(a, dtype=np.float32)


def map_units(name):
    """UCUM (code, scheme, meaning) for a map name, or None if it has no units
    (DEC, which is a colour)."""
    return _MAP_UNITS.get(name)


class DtiResult(dict):
    """The maps, plus the geometry needed to write or track them.

    A dict of {map name: array} so it reads like the DCE result, with the grid
    carried alongside — a map without its affine cannot be written to anything.
    """

    def __init__(self, maps, *, affine, spacing, shape):
        super().__init__(maps)
        self.affine = affine        # voxel -> patient, row-major 4x4
        self.spacing = spacing      # (col, row, slice) mm
        self.shape = shape          # (depth, rows, cols)


def baseline_mask(bvals, *, manufacturer=None, threshold=None):
    """Which volumes are baselines rather than diffusion directions.

    bvals: b-value per volume.
    manufacturer: DICOM Manufacturer (0008,0070). Selects the threshold —
        Siemens needs a higher one because it reports b=50 for volumes that
        are baselines in every other respect.
    threshold: override the per-manufacturer value entirely.

    Returns a boolean array, True where the volume is a baseline.

    Not ``bvals == 0``: UIH writes 1.25 for its b=0 volume, which that test would
    hand to the tensor fit as a diffusion direction with an arbitrary gradient.
    """
    thr = (float(threshold) if threshold is not None
           else _core.dwi_b0_threshold(manufacturer or ""))
    return np.asarray(bvals, dtype=np.float64) <= thr


def head_mask(b0, *, cleanup=False):
    """Head mask from a 3-D b=0 baseline. Returns (mask uint8 [Z,Y,X], kept).

    Median filter, Otsu's threshold, dilate. ``cleanup`` additionally removes
    islands and fills pockets: better for seeding, but a departure from the
    reference implementation this was matched against.
    """
    return _core.dti_head_mask(_f32(b0), cleanup)


def _check_bvecs(bvecs):
    """[n_dirs, 3] float32 unit vectors, or raise.

    The design matrix is built from products of the gradient components, so |g|
    enters squared: a gradient table normalised to something other than 1 returns
    diffusivities scaled by 1/|g|² with FA unchanged. Nothing downstream reveals
    it, so it is refused here rather than fitted.
    """
    g = np.asarray(bvecs, dtype=np.float64)
    if g.ndim != 2 or g.shape[1] != 3:
        raise ValueError(f"bvecs must be (n_dirs, 3); got {g.shape}. pydcm.load_dwi "
                         "returns (3, V) — transpose it.")
    norms = np.linalg.norm(g, axis=1)
    bad = ~np.isclose(norms, 1.0, atol=1e-3)
    if bad.any():
        i = int(np.argmax(bad))
        raise ValueError(
            f"bvecs must be unit vectors; row {i} has |g| = {norms[i]:.6g}. "
            "The design matrix carries |g|^2, so this would scale every "
            "diffusivity by 1/|g|^2 and leave FA unchanged.")
    return _f32(g)


def _check_maps(names):
    names = tuple(names)
    unknown = [n for n in names if n not in MAPS]
    if unknown:
        raise ValueError(f"unknown map(s) {unknown}; choose from {list(MAPS)}")
    return names


def fit_maps(b0, dwi, bvals, bvecs, maps=_DEFAULT_MAPS, *, wls=False):
    """Tensor fit and scalar maps over a FLAT voxel axis — the thinnest tier.

    b0: [n_voxels] mean baseline signal.
    dwi: [n_dirs, n_voxels] DW signal, direction-major (the DW axis FIRST).
    bvals: [n_dirs] b-values, baselines already excluded.
    bvecs: [n_dirs, 3] unit gradient directions.
    maps: which maps to compute; see :data:`MAPS`.
    wls: weight by signal² instead of ordinary least squares.

    Returns {name: array} — float32 [n_voxels], except DEC which is uint8
    [n_voxels, 4] RGBA.

    The shape checks below catch a transposed ``dwi`` only when n_dirs and
    n_voxels differ; when they are equal both readings are valid and the fit
    silently returns garbage. Use :func:`fit` if you have a volume stack.
    """
    b0 = _f32(b0)
    dwi = _f32(dwi)
    bvals = _f32(bvals)
    bvecs = _check_bvecs(bvecs)
    names = _check_maps(maps)

    if dwi.ndim != 2:
        raise ValueError(f"dwi must be (n_dirs, n_voxels); got {dwi.shape}")
    if dwi.shape[0] != bvals.shape[0]:
        raise ValueError(
            f"dwi has {dwi.shape[0]} rows but there are {bvals.shape[0]} b-values. "
            "dwi is direction-major, so the DW axis comes first — a (n_voxels, "
            "n_dirs) array fits garbage whenever the two happen to be equal.")
    if dwi.shape[1] != b0.shape[0]:
        raise ValueError(f"dwi has {dwi.shape[1]} voxels but b0 has {b0.shape[0]}")
    if bvals.shape[0] < 6:
        raise ValueError(f"a tensor needs at least 6 directions; got {bvals.shape[0]}")

    return dict(_core.dti_fit_maps(b0, dwi, bvals, bvecs, list(names), wls))


def _split(volumes, bvals, bvecs, manufacturer, threshold):
    """(V,Z,Y,X) + tables -> the flat direction-major arrays the engine wants.

    The voxel axis is flattened C-order over (Z, Y, X), which is the engine's
    z*rows*cols + y*cols + x. The DW subset keeps the volume axis first, so the
    reshape lands direction-major with no transpose.
    """
    vols = np.asarray(volumes)
    if vols.ndim != 4:
        raise ValueError(f"volumes must be (V, Z, Y, X); got {vols.shape}")
    bvals = np.asarray(bvals, dtype=np.float64)
    bvecs = np.asarray(bvecs, dtype=np.float64)
    if bvecs.shape == (3, vols.shape[0]):        # load_dwi's layout
        bvecs = bvecs.T
    if bvals.shape[0] != vols.shape[0] or bvecs.shape[0] != vols.shape[0]:
        raise ValueError(f"bvals {bvals.shape} / bvecs {bvecs.shape} do not match "
                         f"{vols.shape[0]} volumes")

    base = baseline_mask(bvals, manufacturer=manufacturer, threshold=threshold)
    if not base.any():
        raise ValueError("no baseline volume found — the fit needs a b=0. Pass "
                         "threshold= if this series encodes its baseline unusually.")
    if base.all():
        raise ValueError("every volume is a baseline; nothing to fit")

    spatial = vols.shape[1:]
    b0 = vols[base].mean(axis=0).reshape(-1)
    dw = vols[~base].reshape(int((~base).sum()), -1)
    return _f32(b0), _f32(dw), _f32(bvals[~base]), bvecs[~base], spatial


def _spacing_from_affine(affine):
    """(col, row, slice) mm — the norms of the affine's direction columns, which
    is where the spacing lives once it has been folded into the matrix."""
    a = np.asarray(affine, dtype=np.float64)
    return _f32([np.linalg.norm(a[:3, k]) for k in range(3)])


def fit(volumes, bvals, bvecs, *, maps=_DEFAULT_MAPS, wls=False,
        manufacturer=None, threshold=None, affine=None, mask=False, cleanup=False):
    """Tensor fit over a 4-D volume stack, with the baseline split handled.

    volumes: (V, Z, Y, X) — the layout :func:`pydcm.load_dwi` returns.
    bvals: [V]; bvecs: [V, 3] or (3, V), INCLUDING the baselines.
    maps, wls: as :func:`fit_maps`.
    manufacturer / threshold: how to recognise a baseline; see
        :func:`baseline_mask`.
    affine: voxel -> patient 4x4, carried into the result for writing and
        tracking. Identity if omitted.
    mask: zero the maps outside the head. ``False`` by default, so what comes
        back is the fit as measured; ``True`` builds one with :func:`head_mask`,
        or pass your own array.

    Returns a :class:`DtiResult` of (Z, Y, X) arrays (DEC is (Z, Y, X, 4)).

    Read the background before trusting a histogram of this: outside the head the
    signal is noise, the log-linear fit is degenerate, and the eigen decomposition
    returns an arbitrary direction with **FA near 1**. An unmasked FA map therefore
    has its maximum in air, and summary statistics over the whole volume describe
    mostly background. ``mask=True`` is the fix; :func:`track` applies one by
    default for the same reason.
    """
    b0, dw, bv, bvec, spatial = _split(volumes, bvals, bvecs, manufacturer, threshold)
    flat = fit_maps(b0, dw, bv, bvec, maps=maps, wls=wls)
    shaped = {k: (v.reshape(*spatial, 4) if k == "DEC" else v.reshape(spatial))
              for k, v in flat.items()}

    if mask is not False:
        m = (head_mask(b0.reshape(spatial), cleanup=cleanup)[0] if mask is True
             else np.asarray(mask, dtype=np.uint8))
        if m.shape != spatial:
            raise ValueError(f"mask shape {m.shape} does not match volume {spatial}")
        keep = m != 0
        shaped = {k: np.where(keep[..., None] if k == "DEC" else keep, v, 0)
                  for k, v in shaped.items()}

    aff = np.eye(4) if affine is None else np.asarray(affine, dtype=np.float64)
    return DtiResult(shaped, affine=aff, spacing=_spacing_from_affine(aff),
                     shape=spatial)


def fit_series(series, *, maps=_DEFAULT_MAPS, wls=False, recursive=True,
               order="gradient", threshold=None, mask=False, cleanup=False):
    """A DICOM DWI series -> a :class:`DtiResult`.

    Reads the series through :func:`pydcm.load_dwi`, whose gradients are already
    in the voxel convention that pairs with its pixel data, splits the baseline
    using the series' own Manufacturer, and fits. See :func:`fit` for ``mask`` and
    for why an unmasked FA map peaks in air.
    """
    from .diffusion import load_dwi

    volumes, bvals, bvecs, affine = load_dwi(series, recursive=recursive, order=order)
    return fit(volumes, bvals, bvecs, maps=maps, wls=wls, affine=affine,
               manufacturer=_manufacturer_of(series, recursive), threshold=threshold,
               mask=mask, cleanup=cleanup)


def _instances(series, recursive=True):
    """A directory, one path, or an iterable of paths -> a list of instance paths.

    fit_series and track_series accept a directory because load_dwi does; the
    tractography writer takes a reference verbatim and would treat a directory as
    an unreadable instance. Expanding here keeps every entry point in this module
    taking the same argument.
    """
    from pathlib import Path

    if series is None:
        return None
    if isinstance(series, (str, Path)):
        if Path(series).is_dir():
            from .torchdata import scan
            return [str(p) for p in scan(series, recursive=recursive)]
        return [str(series)]
    return [str(p) for p in series]


def _manufacturer_of(series, recursive=True):
    """Manufacturer (0008,0070) from the first readable instance, for the baseline
    threshold. None when it cannot be read — the generic threshold then applies,
    which is right for every vendor except Siemens."""
    from pathlib import Path

    from . import dcmread
    paths = [series]
    if isinstance(series, (str, Path)) and Path(series).is_dir():
        from .torchdata import scan
        paths = [str(p) for p in scan(series, recursive=recursive)]
    elif not isinstance(series, (str, Path)):
        paths = list(series)
    for p in paths[:1]:
        try:
            return str(getattr(dcmread(str(p)), "Manufacturer", "") or "") or None
        except Exception:
            return None
    return None


def track(volumes, bvals, bvecs, *, affine=None, manufacturer=None, threshold=None,
          wls=False, mask=True, cleanup=False, fa_threshold=0.15, angle_deg=45.0,
          step_size=0.5, max_steps=2000, seed_fa_min=0.3, max_tracks=100000,
          max_total_points=10000000, patient=True):
    """Deterministic RK4 tractography from a 4-D DWI stack.

    volumes, bvals, bvecs, affine, manufacturer, threshold, wls: as :func:`fit`.
    mask: apply a head mask before seeding. True by default and it matters:
        outside the head the fit is degenerate and FA approaches 1, so an
        unmasked run spends most of its budget on background. Pass an array to
        supply your own, or False to seed everywhere.
    cleanup: passed to :func:`head_mask` when building the mask.
    angle_deg: maximum turn per step, in DEGREES. The engine takes a cosine;
        this converts, so a caller cannot pass 45 and get cos(45)=0.7071's
        meaning by accident.
    step_size: RK4 step in VOXEL units, not mm.
    patient: return streamlines in patient coordinates. False returns the grid
        mm the renderer uses. Anything written to DICOM needs patient.

    Returns a list of (P, 3) arrays — float64 patient mm, or float32 grid mm when
    ``patient=False``.
    """
    b0, dw, bv, bvec, spatial = _split(volumes, bvals, bvecs, manufacturer, threshold)
    depth, rows, cols = spatial
    evals, evecs = _core.dti_eigen(b0, dw, bv, _check_bvecs(bvec), wls)
    fa = np.asarray(_core.dti_fit_maps(b0, dw, bv, _check_bvecs(bvec), ["FA"], wls)["FA"])

    if mask is not False:
        m = (head_mask(b0.reshape(spatial), cleanup=cleanup)[0] if mask is True
             else np.asarray(mask, dtype=np.uint8))
        if m.shape != spatial:
            raise ValueError(f"mask shape {m.shape} does not match volume {spatial}")
        fa = np.where(m.reshape(-1) != 0, fa, np.float32(0.0))

    aff = np.eye(4) if affine is None else np.asarray(affine, dtype=np.float64)
    spacing = _spacing_from_affine(aff)
    tracks = _core.dti_track(
        _f32(evecs), _f32(fa), cols, rows, depth, spacing,
        fa_threshold=float(fa_threshold),
        angle_threshold=float(math.cos(math.radians(angle_deg))),
        step_size=float(step_size), max_steps=int(max_steps),
        seed_fa_min=float(seed_fa_min), max_tracks=int(max_tracks),
        max_total_points=int(max_total_points))
    if not patient:
        return list(tracks)
    return list(_core.dti_tracks_to_patient(list(tracks), spacing, aff))


def track_series(series, *, recursive=True, order="gradient", **kwargs):
    """A DICOM DWI series -> streamlines in patient coordinates."""
    from .diffusion import load_dwi

    volumes, bvals, bvecs, affine = load_dwi(series, recursive=recursive, order=order)
    kwargs.setdefault("manufacturer", _manufacturer_of(series, recursive))
    return track(volumes, bvals, bvecs, affine=affine, **kwargs)


def write_tracts(reference, tracks, output=None, *, label="DTI",
                 description="Deterministic tensor tractography", rgb=None):
    """Write streamlines as a DICOM Tractography Results object.

    ``tracks`` must be in PATIENT coordinates — what :func:`track_series` and
    ``track(..., patient=True)`` return. Grid mm would write a well-formed object
    whose tracts are rotated and displaced out of the reference's Frame of
    Reference, which no reader can detect.

    ``reference`` supplies the demographics and the Frame of Reference UID those
    coordinates are expressed in, so it should be the series the tracks were
    computed from — a directory, as :func:`fit_series` and :func:`track_series`
    take, or a list of instances. The anatomy and diffusion-model codes default to
    White Matter and Single Tensor, which is what this engine produces.

    Returns the output path when ``output`` is given, matching :func:`save_dwi`,
    and the Part-10 bytes when it is not.
    """
    from .tract import write_mktract

    track_set = {
        "label": label,
        "description": description,
        "algorithm_name": "dcm_dti deterministic RK4",
        "tracks": [np.ascontiguousarray(t, dtype=np.float32) for t in tracks],
    }
    if rgb is not None:
        track_set["rgb"] = rgb
    written = write_mktract(_instances(reference), track_set, output=output)
    return str(output) if output else written
