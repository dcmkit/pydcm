# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Fuli Wu
# NOT a medical device — not for clinical or diagnostic use.

"""pydcm radiomics — IBSI features over an ROI, plus a feature extractor.

Three surfaces over the one native radiomics engine:

* ``pydcm.radiomics(image, mask=..., roi=...)`` — pydcm's own one-call API.
* ``from pydcm.radiomics import featureextractor`` — the conventional
  ``RadiomicsFeatureExtractor(...).execute(img, mask)`` extractor API, so a pipeline
  written against that interface needs only change its import path; returns an
  ``OrderedDict``.
* ``@pydcm.radiomics.feature(...)`` — register a custom feature in Python; it runs
  over the SAME preprocessed + discretised grid the native engine used, so a
  researcher can add a feature or override a formula without recompiling. Applies to
  array inputs and to ``RadiomicsFeatureExtractor.execute``.
"""
from __future__ import annotations

import sys
from collections import OrderedDict

import numpy as np

from . import _core


# ---- custom-feature hook -----------------------------------------------------
# Add a feature (or override a formula) WITHOUT recompiling the engine: register a
# Python callable; it runs over the SAME preprocessed + discretised grid the native
# IBSI features used — so a custom histogram feature lines up bin-for-bin with the
# standard ones — and its result joins the feature dict as "<class>_<name>". This is
# the add-a-feature workflow researchers expect, on the native (fast, IBSI-validated) core.

_REGISTRY: dict = {}


class ROI:
    """The preprocessed + discretised ROI handed to a custom feature: the exact grid
    the native IBSI extractor ran over (same resample / normalise / resegmentation and
    the same gray-level discretisation). Arrays are ``(nz, h, w)``; ROI voxels have
    mask 1 and level ``[0, nb)``, the rest mask 0 and level -1."""

    __slots__ = ("image", "mask", "level_grid", "nb", "spacing", "range", "bin_edges", "_m")

    def __init__(self, d):
        self.image = d["image"]            # preprocessed intensities (HU / normalised)
        self.mask = d["mask"]              # 1 = inside the ROI
        self.level_grid = d["levels"]      # gray bin [0,nb), -1 outside
        self.nb = int(d["nb"])
        self.spacing = tuple(float(s) for s in d["spacing"])   # (x, y, z) mm
        self.range = tuple(float(r) for r in d["range"])       # discretisation range used
        self.bin_edges = np.linspace(self.range[0], self.range[1], self.nb + 1)
        self._m = self.mask.astype(bool)

    @property
    def intensities(self):
        """1-D preprocessed intensities over the ROI voxels."""
        return self.image[self._m]

    @property
    def levels(self):
        """1-D gray levels ``[0, nb)`` over the ROI voxels (non-negative, bincount-ready)."""
        return self.level_grid[self._m]

    @property
    def n(self):
        """ROI voxel count."""
        return int(self._m.sum())


def feature(class_name="firstorder", name=None):
    """Register a custom radiomic feature (decorator).

    The decorated function receives an :class:`ROI` and returns a scalar; the value
    joins every result dict as ``"<class_name>_<name>"`` (``name`` defaults to the
    function's name). Use as ``@feature``, ``@feature("glcm")`` or
    ``@feature("firstorder", name="my_stat")``. Applies to array inputs and to
    ``RadiomicsFeatureExtractor.execute`` (the compatibility extractor).

    Naming a custom feature after a standard one — ``@feature("firstorder",
    name="Mean")`` — OVERRIDES that feature's value in the result, which is how you
    change a built-in formula without recompiling.
    """
    if callable(class_name):                       # bare @feature (no parentheses)
        _REGISTRY[("firstorder", class_name.__name__)] = class_name
        return class_name

    def deco(fn):
        _REGISTRY[(class_name, name or fn.__name__)] = fn
        return fn
    return deco


def clear_features():
    """Drop all registered custom features."""
    _REGISTRY.clear()


def registered_features():
    """The keys (``"<class>_<name>"``) of the currently registered custom features."""
    return [f"{c}_{n}" for (c, n) in _REGISTRY]


def _apply_custom(feats, roi_dict, prefix=""):
    """Run every registered custom feature over `roi_dict`, merging into `feats`."""
    roi = ROI(roi_dict)
    for (cls, nm), fn in _REGISTRY.items():
        feats[f"{prefix}{cls}_{nm}"] = float(fn(roi))
    return feats


def radiomics(image, mask=None, *, roi=None, spacing=None, bins=32,
              value_range=(-1024.0, 3071.0), bin_width=0.0, resample=0.0,
              normalize=False, normalize_scale=1.0, log_sigma=None, wavelet=False,
              averaged=True, resegment=None, resegment_sigma=False, resample_bspline=False,
              voxel_array_shift=0.0, filters=None, distances=None):
    """The IBSI radiomic feature set over an ROI — 10 classes (firstorder / glcm /
    glrlm / glszm / gldm / gldzm / ngtdm / shape / ivh [intensity-volume histogram] /
    local intensity), with the standard radiomics feature names.

    Two call styles:

    * **From files** (the convenient path) — `image` is a DICOM path: pixels are
      decoded to real-world values (HU) and `spacing` is read from the image
      geometry (PixelSpacing / SliceThickness). Give the ROI as either `mask` =
      a co-framed mask DICOM path (non-zero = inside) or `roi` = ``(min, max)``
      real-world-value thresholds.
    * **From arrays** (the low-level primitive) — `image` is a real-world-valued
      array (e.g. ``decode(..., rescale=True)``) and `mask` a non-zero-inside
      array of the same shape; pass `spacing` = ``(x, y, z)`` mm yourself.

    Preprocessing (IBSI, off by default): `resample` > 0 resamples the
    ROI to isotropic voxels of that size in mm (trilinear image / nearest mask);
    `bin_width` > 0 uses fixed-bin-width discretisation (vs the fixed `bins` count);
    `normalize` z-score-normalises intensities (× `normalize_scale`). Filters multiply
    the feature set under standard image-type prefixes: `log_sigma` = LoG sigma(s) in
    mm (`log-sigma-<s>-mm-3D_…`); `wavelet=True` = the coif1 SWT 8 sub-bands
    (`wavelet-LLH_…`). With any filter, every key (incl. the original) is prefixed.

    Both 2D ``(H, W)`` or 3D ``(slices, H, W)``. Returns ``{feature_name: value}``.
    """
    def _is_path(x):
        return isinstance(x, (str, bytes)) or hasattr(x, "__fspath__")
    def _read(p):
        with open(p, "rb") as fh:
            return fh.read()
    sigmas = ([] if log_sigma is None
              else [float(log_sigma)] if isinstance(log_sigma, (int, float))
              else [float(s) for s in log_sigma])
    rmin, rmax = resegment if resegment is not None else (0.0, 0.0)
    dl = [int(d) for d in distances] if distances else []     # GLCM Chebyshev distances ([] → {1})
    prep = (float(bin_width), float(resample), bool(normalize), float(normalize_scale),
            sigmas, bool(wavelet), bool(averaged),
            resegment is not None, bool(resegment_sigma), float(rmin), float(rmax),
            bool(resample_bspline), float(voxel_array_shift), list(filters or ()), dl)

    # All-from-files mirrors the CLI: decode / spacing / mask / extract run ONCE in
    # the native radiomics engine — Python re-implements none of that pipeline.
    if _is_path(image) and (mask is None or _is_path(mask)):
        if _REGISTRY:
            # The all-files native path returns features only (no ROI grid), so custom
            # features can't run over it — fail loud rather than silently drop them.
            raise ValueError(
                "custom @feature functions are not applied on the all-DICOM-files path; "
                "decode to arrays first (or use RadiomicsFeatureExtractor.execute, which "
                "decodes to arrays) so the custom feature can run over the ROI grid")
        if mask is None and roi is None:
            raise ValueError("provide either mask=<DICOM path> or roi=(min, max)")
        rmin, rmax = roi if roi is not None else (float("nan"), float("nan"))
        return _core.radiomics_file(
            _read(image), _read(mask) if mask is not None else b"",
            float(rmin), float(rmax),
            int(bins), float(value_range[0]), float(value_range[1]), *prep)

    # Array / mixed inputs → the low-level primitive over already-decoded arrays.
    if _is_path(image):                              # image path + a numpy mask
        px, meta = _core.decode(str(image), 0, True)
        if spacing is None:
            ps = meta.get("pixel_spacing") or (0.0, 0.0)
            spacing = (ps[1] or 1.0, ps[0] or 1.0, meta.get("slice_thickness") or 1.0)
    else:
        px = image
        if spacing is None:
            spacing = (1.0, 1.0, 1.0)
    px = np.ascontiguousarray(px, dtype=np.float32)
    if mask is not None:
        mk = np.ascontiguousarray(np.asarray(mask) != 0, dtype=np.uint8)
    elif roi is not None:
        mk = np.ascontiguousarray((px >= roi[0]) & (px <= roi[1]), dtype=np.uint8)
    else:
        raise ValueError("provide either mask=<array> or roi=(min, max)")
    if mk.shape != px.shape:
        raise ValueError(f"image {px.shape} and mask {mk.shape} must have the same shape")
    sxyz = (float(spacing[0]), float(spacing[1]), float(spacing[2]))
    rng = (int(bins), float(value_range[0]), float(value_range[1]))
    if not _REGISTRY:
        return _core.radiomics_features(px, mk, *sxyz, *rng, *prep)

    # Custom features registered → also pull the preprocessed / discretised grid and
    # run them over the SAME grid the native features used. The prepared call covers
    # the original image; when filters are also requested we additionally fetch the
    # filtered (prefixed) set and tag the custom features `original_` to match.
    prep_prep = (float(bin_width), float(resample), bool(normalize), float(normalize_scale),
                 bool(averaged), resegment is not None, bool(resegment_sigma),
                 float(rmin), float(rmax), bool(resample_bspline), float(voxel_array_shift), dl)
    bare, roi_dict = _core.radiomics_features_prepared(px, mk, *sxyz, *rng, *prep_prep)
    if sigmas or wavelet or filters:
        feats = dict(_core.radiomics_features(px, mk, *sxyz, *rng, *prep))
        return _apply_custom(feats, roi_dict, prefix="original_")
    return _apply_custom(dict(bare), roi_dict)


# ---- compatibility surface ----------------------------------------
# `from pydcm.radiomics import featureextractor` exposes the conventional
# `RadiomicsFeatureExtractor` API, so a pipeline written against it changes only its
# import path.

def _sitk_array(obj):
    """(array, spacing_xyz) from a SimpleITK image, or (None, None)."""
    try:
        import SimpleITK as sitk
    except ImportError:
        return None, None
    if isinstance(obj, sitk.Image):
        return sitk.GetArrayFromImage(obj), tuple(obj.GetSpacing())
    return None, None


def _resolve(obj, rescale):
    """Turn a path / array / SimpleITK image into (ndarray, spacing_xyz)."""
    arr, spacing = _sitk_array(obj)
    if arr is not None:
        return arr, spacing
    if isinstance(obj, np.ndarray):
        return obj, None
    import os
    if isinstance(obj, (str, bytes)) or hasattr(obj, "__fspath__"):
        from . import decode
        a, meta = decode(os.fspath(obj), rescale=rescale, with_meta=True)
        ps = meta.get("pixel_spacing") or [1.0, 1.0]
        st = meta.get("slice_thickness") or 1.0
        return a, (float(ps[1]), float(ps[0]), float(st))     # (x, y, z) mm
    raise TypeError(f"unsupported image/mask type: {type(obj).__name__}")


def _add_diagnostics(out, ex, img, roi, spacing):
    """Conventional ``diagnostics_*`` provenance over the original image / mask.
    Hashes are sha1 of our arrays — provenance, not a parity target; the deterministic
    values (size, voxel count, bounding box) follow the standard layout."""
    import hashlib
    import platform
    out["diagnostics_Versions_pydcm"] = __import__("pydcm").__version__
    out["diagnostics_Versions_Numpy"] = np.__version__
    out["diagnostics_Versions_Python"] = platform.python_version()
    out["diagnostics_Configuration_Settings"] = dict(ex.settings)
    out["diagnostics_Configuration_EnabledImageTypes"] = dict(ex.enabledImagetypes)
    sp = tuple(map(float, spacing))
    out["diagnostics_Image-original_Hash"] = hashlib.sha1(np.ascontiguousarray(img).tobytes()).hexdigest()
    out["diagnostics_Image-original_Dimensionality"] = f"{img.ndim}D"
    out["diagnostics_Image-original_Spacing"] = sp
    out["diagnostics_Image-original_Size"] = tuple(int(s) for s in img.shape[::-1])      # (x,y,z)
    m = np.asarray(roi).astype(bool)
    if m.any():
        vals = img[m]
        out["diagnostics_Image-original_Mean"] = float(vals.mean())
        out["diagnostics_Image-original_Minimum"] = float(vals.min())
        out["diagnostics_Image-original_Maximum"] = float(vals.max())
    out["diagnostics_Mask-original_Hash"] = hashlib.sha1(np.ascontiguousarray(m).tobytes()).hexdigest()
    out["diagnostics_Mask-original_Spacing"] = sp
    out["diagnostics_Mask-original_Size"] = tuple(int(s) for s in m.shape[::-1])         # (x,y,z)
    out["diagnostics_Mask-original_VoxelNum"] = int(m.sum())
    idx = np.argwhere(m)                                                                 # (z,y,x) rows
    if idx.size:
        lo, hi = idx.min(0), idx.max(0)
        bbox = (*lo[::-1], *(hi - lo + 1)[::-1])                                         # x,y,z lo + sizes
        out["diagnostics_Mask-original_BoundingBox"] = tuple(int(v) for v in bbox)
        out["diagnostics_Mask-original_CenterOfMassIndex"] = tuple(float(v) for v in idx.mean(0)[::-1])
    return out


class RadiomicsFeatureExtractor:
    """extractor over pydcm's native IBSI engine."""

    def __init__(self, *args, **settings):
        # defaults: fixed bin width 25, z-score normalise off.
        self.settings = {"label": 1, "voxelArrayShift": 0,
                         "normalize": False, "normalizeScale": 1}
        # accepts a params file/dict as the first positional arg.
        if args and isinstance(args[0], dict):
            self.settings.update(args[0])
        self.settings.update(settings)
        self.enabledFeatures = {}          # class -> [feature names] ([] = whole class)
        self._all_features = True          # True until disableAllFeatures() narrows it
        self.enabledImagetypes = {"Original": {}}

    # configuration API — selection is honoured (the result is filtered to match).
    def enableAllFeatures(self):
        self._all_features = True
        self.enabledFeatures = {}

    def disableAllFeatures(self):
        self._all_features = False
        self.enabledFeatures = {}

    def enableFeatureClassByName(self, name, enabled=True):
        if enabled:
            self.enabledFeatures[name] = []
        else:
            self._all_features = False     # an explicit disable narrows from "all"
            self.enabledFeatures.pop(name, None)

    def enableFeaturesByName(self, **kwargs):
        self._all_features = False
        for cls, names in kwargs.items():
            self.enabledFeatures[cls] = list(names) if names else []

    def enableAllImageTypes(self): pass
    def disableAllImageTypes(self): self.enabledImagetypes = {}
    def enableImageTypeByName(self, name, enabled=True, **kw):
        if enabled:
            self.enabledImagetypes[name] = kw
        else:
            self.enabledImagetypes.pop(name, None)

    # settings that change feature VALUES but the native engine does not yet honour —
    # surfaced loudly so a ported params file can't diverge silently.
    _UNSUPPORTED_VALUE_SETTINGS = frozenset(
        ("weightingNorm", "gldm_a", "force2D", "removeOutliers", "symmetricalGLCM"))

    def _warn_unsupported(self):
        bad = sorted(k for k in self.settings if k in self._UNSUPPORTED_VALUE_SETTINGS)
        if bad:
            import warnings
            warnings.warn(
                "RadiomicsFeatureExtractor: setting(s) not yet honoured by the native "
                f"engine — feature values may differ from that configuration: {bad}",
                UserWarning, stacklevel=2)

    def execute(self, image, mask, label=None, voxelBased=False):
        """Return an ``OrderedDict`` of features over the ROI."""
        if voxelBased:
            raise NotImplementedError("pydcm radiomics is ROI-level (voxelBased=False)")
        self._warn_unsupported()
        img, spacing = _resolve(image, rescale=True)
        msk, _ = _resolve(mask, rescale=False)
        label = self.settings["label"] if label is None else label
        roi = (np.asarray(msk) == label).astype("uint8")
        spacing = spacing or (1.0, 1.0, 1.0)
        s = self.settings
        rs = s.get("resampledPixelSpacing")
        resample = float(rs[0] if isinstance(rs, (list, tuple)) else rs) if rs else 0.0
        kw = dict(spacing=tuple(map(float, spacing)), resample=resample,
                  normalize=bool(s.get("normalize", False)),
                  normalize_scale=float(s.get("normalizeScale", 1.0)),
                  voxel_array_shift=float(s.get("voxelArrayShift", 0.0)))
        log = self.enabledImagetypes.get("LoG", {}).get("sigma")  # enableImageTypeByName("LoG", sigma=[...])
        if log:
            kw["log_sigma"] = log
        if "Wavelet" in self.enabledImagetypes:                   # enableImageTypeByName("Wavelet")
            kw["wavelet"] = True
        rr = s.get("resegmentRange")
        if rr:
            kw["resegment"] = (float(rr[0]), float(rr[1]))
            kw["resegment_sigma"] = (s.get("resegmentMode", "absolute") == "sigma")
        fl = [it.lower() for it in ("Square", "SquareRoot", "Logarithm", "Exponential",
                                    "Gradient", "LBP2D", "LBP3D") if it in self.enabledImagetypes]
        if fl:
            kw["filters"] = fl
        if s.get("distances"):                    # GLCM neighbour distances
            kw["distances"] = list(s["distances"])
        if "binWidth" in s:                       # default discretisation
            kw["bin_width"] = float(s["binWidth"])
        elif "binCount" in s:
            kw["bins"] = int(s["binCount"])
        else:
            kw["bin_width"] = 25.0                # default binWidth
        img = np.asarray(img, dtype="float32")
        feats = radiomics(img, roi, **kw)
        out = OrderedDict()
        _add_diagnostics(out, self, img, roi, spacing)
        # When any image-type filter is active, radiomics() already returns fully
        # prefixed keys (original_/log-sigma-/wavelet-/square_/…); only the no-filter
        # path returns bare names that still need the "original_" image-type tag.
        # (Adding it unconditionally double-prefixed filtered keys → original_wavelet-…)
        filtered = bool(kw.get("log_sigma") or kw.get("wavelet") or kw.get("filters"))
        for k, v in feats.items():
            key = k if filtered else f"original_{k}"      # "imagetype_feature" key
            if self._keep(key):                            # honour feature-class selection
                out[key] = v
        return out

    # all recognised classes (shape2D before shape so the prefix match is unambiguous)
    _CLASSES = ("shape2D", "firstorder", "glcm", "glrlm", "glszm", "gldzm", "gldm",
                "ngtdm", "shape", "ivh", "loc")
    # the conventional class set the compatibility extractor emits by default; our IBSI
    # extras (gldzm / ivh / loc) ride on pydcm.radiomics() or explicit enabling.
    _STANDARD_CLASSES = frozenset(
        ("firstorder", "glcm", "glrlm", "glszm", "gldm", "ngtdm", "shape", "shape2D"))

    def _keep(self, key):
        """Whether `key` survives the feature-class selection."""
        cls = next((c for c in self._CLASSES
                    if key.startswith(c + "_") or ("_" + c + "_") in key), None)
        if cls is None:
            return True                                    # not a class feature (safety)
        if self._all_features:                             # default: the standard classes
            return cls in self._STANDARD_CLASSES or cls in self.enabledFeatures
        if cls not in self.enabledFeatures:                # narrowed by disableAllFeatures
            return False
        names = self.enabledFeatures[cls]
        return (not names) or any(key.endswith("_" + n) for n in names)


# expose this module as `featureextractor` so the conventional
# `from <pkg> import featureextractor` layout resolves identically under pydcm.
featureextractor = sys.modules[__name__]

# `pydcm/__init__` rebinds the name `pydcm.radiomics` to the FUNCTION above, so attach
# the custom-feature helpers to it too — `pydcm.radiomics.feature(...)` then works
# alongside `from pydcm.radiomics import feature`.
radiomics.feature = feature
radiomics.ROI = ROI
radiomics.clear_features = clear_features
radiomics.registered_features = registered_features

__all__ = ["radiomics", "RadiomicsFeatureExtractor", "featureextractor",
           "feature", "ROI", "clear_features", "registered_features"]
