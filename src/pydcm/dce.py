# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Fuli Wu
# NOT a medical device — not for clinical or diagnostic use.

"""pydcm DCE-MRI — dynamic contrast-enhanced pharmacokinetic modelling.

A thin NumPy surface over the native DCE engine. Scope is the validated slice:
the Parker population AIF, spoiled-GRE signal→concentration conversion, and the
Tofts / Extended-Tofts / Patlak tissue models, fitted per voxel.

Typical use (concentration already computed)::

    import numpy as np, pydcm.dce as dce
    t = np.arange(0, 6, 0.025)              # minutes, injection at t=0
    cp = dce.parker_aif(t)                  # population plasma AIF (mM)
    maps = dce.fit(conc_4d, t, model="ext_tofts")   # conc_4d: (T, H, W)
    ktrans = maps["ktrans"]                 # (H, W) float32, 1/min

From spoiled-GRE signal instead of concentration::

    maps = dce.fit(signal_4d, t, input="spgr",
                   t1_0_s=1.4, tr_s=0.005, fa_deg=25.0, r1=4.5)

Units: time in minutes (t=0 = injection; negative times = pre-contrast),
Ktrans in 1/min, ve/vp dimensionless fractions, concentration in mM.
"""
from __future__ import annotations

import numpy as np

from . import _core

__all__ = ["parker_aif", "population_aif", "measure_aif", "forward", "signal_to_conc",
           "fit_curve", "fit", "fit_series", "t1_map_vfa", "write_param_maps",
           "MODELS", "AIFS"]

# UCUM units (code, scheme, meaning) for each DCE output parameter — used when
# emitting parameter maps so each carries the right physical units.
_PARAM_UNITS = {
    "ktrans": ("/min", "UCUM", "/min"),
    "ve":     ("1", "UCUM", "no units"),
    "vp":     ("1", "UCUM", "no units"),
    "delay":  ("min", "UCUM", "min"),
    "rmse":   ("mmol/L", "UCUM", "mmol/L"),
}

MODELS = ("tofts", "ext_tofts", "patlak")
AIFS = ("parker", "georgiou", "fritz_hansen", "weinmann", "mcgrath")


def _as_f64(a):
    return np.ascontiguousarray(a, dtype=np.float64)


def population_aif(times_min, model="parker", hct=0.0):
    """Population arterial input function (plasma, mM) by name.

    `model` is one of :data:`AIFS` — ``parker`` (2006), ``georgiou`` (2019),
    ``fritz_hansen`` (1996), ``weinmann`` (dose-scaled bi-exp), ``mcgrath`` (2009,
    preclinical). `times_min` in minutes (t=0 = injection). The published forms
    are already plasma, so `hct` defaults to 0 (verbatim); set it only to convert
    a measured whole-blood curve. Returns a 1-D float64 array, length of times.
    """
    if model not in AIFS:
        raise ValueError(f"model must be one of {AIFS}, got {model!r}")
    return _core.dce_population_aif(model, _as_f64(times_min), float(hct))


def parker_aif(times_min, hct=0.0):
    """Parker (2006) population AIF (plasma, mM) — ``population_aif(..., 'parker')``.

    `hct` defaults to 0 (the Parker curve is already plasma); set it only to
    convert a measured whole-blood curve. Returns a 1-D float64 array.
    """
    return _core.dce_parker_aif(_as_f64(times_min), float(hct))


def forward(times_min, cp, model="ext_tofts", *, ktrans=0.1, ve=0.3, vp=0.02):
    """Synthesise a tissue concentration curve Ct(t) from known PK parameters.

    Uses the exact piecewise-linear-AIF convolution. `cp` is the arterial plasma
    curve at `times_min`. Returns Ct (mM), same length as `times_min`.
    """
    if model not in MODELS:
        raise ValueError(f"model must be one of {MODELS}, got {model!r}")
    return _core.dce_forward(_as_f64(times_min), _as_f64(cp), model,
                             float(ktrans), float(ve), float(vp))


def measure_aif(series, mask, tr_s, fa_deg, *, t1_blood_s=1.44, r1=4.5,
                hct=0.45, n_baseline=None):
    """Extract a measured plasma AIF from an arterial ROI in a 4-D signal series.

    The clinical alternative to a population (Parker) AIF: average the ROI's
    spoiled-GRE signal per time frame, invert to blood concentration with the
    blood baseline T1, and convert whole-blood → plasma via ``/(1 - hct)``.

    `series` is ``(T, H, W)`` or ``(T, Z, Y, X)`` signal; `mask` is a non-zero ROI
    over the spatial dims (broadcast to the series' spatial shape). Returns the
    plasma AIF ``Cp(t)`` (mM, length T) — pass it to :func:`fit` / :func:`fit_series`
    as ``aif=``. `n_baseline` (pre-contrast frames for S0) is auto-detected from
    the bolus rise when None.
    """
    from . import perfusion
    roi = perfusion.roi_signal(series, mask)                # shared ROI-mean helper
    if n_baseline is None:
        n_baseline = perfusion.auto_n_baseline(roi, rising=True)    # DCE: enhancement RISE
    cp = signal_to_conc(roi, n_baseline, t1_blood_s, tr_s, fa_deg, r1)
    return cp / (1.0 - hct) if 0.0 < hct < 1.0 else cp


def t1_map_vfa(volumes, flip_angles_deg, tr_s, *, mask=None):
    """VFA / DESPOT1 baseline-T1 map from a multi-flip-angle SPGR acquisition.

    `volumes` is ``(F, H, W)`` — the spoiled-GRE signal at each of the F flip
    angles (same TR); `flip_angles_deg` is ``(F,)``. Returns ``{"t1", "m0",
    "fitted"}`` with ``t1`` an ``(H, W)`` map in **seconds** — feed it to
    :func:`fit` / :func:`fit_series` as ``t1_map=`` for the SPGR path, instead of
    assuming a single baseline T1.
    """
    volumes = np.ascontiguousarray(volumes, dtype=np.float32)
    if volumes.ndim != 3:
        raise ValueError(f"volumes must be 3-D (F, H, W), got shape {volumes.shape}")
    mask_a = None if mask is None else np.ascontiguousarray(mask, dtype=np.uint8)
    return _core.dce_t1_map_vfa(volumes, _as_f64(flip_angles_deg), float(tr_s), mask_a)


def signal_to_conc(signal, n_baseline, t1_0_s, tr_s, fa_deg, r1):
    """Invert the spoiled-GRE steady-state signal → tracer concentration (mM).

    `signal` is a 1-D series; the first `n_baseline` samples form the
    pre-contrast S0. `t1_0_s`/`tr_s` in seconds, `fa_deg` in degrees, `r1` the
    relaxivity (L·mmol⁻¹·s⁻¹).
    """
    return _core.dce_signal_to_conc(_as_f64(signal), int(n_baseline),
                                    float(t1_0_s), float(tr_s), float(fa_deg), float(r1))


def fit_curve(times_min, ct, cp, model="ext_tofts", *,
              fit_delay=False, delay_bounds=(0.0, 0.5)):
    """Fit a single tissue curve. Returns ``{ktrans, ve, vp, rmse, iters, ok, delay}``.

    `ct` measured tissue concentration, `cp` arterial plasma — both at
    `times_min` (mM). Patlak is solved in closed form; Tofts / Extended-Tofts
    use Levenberg–Marquardt. With `fit_delay=True`, a bolus-arrival delay (min)
    is jointly estimated over `delay_bounds` (the AIF is time-shifted), which
    removes the need for t=0 to be the exact injection instant.
    """
    if model not in MODELS:
        raise ValueError(f"model must be one of {MODELS}, got {model!r}")
    return _core.dce_fit_curve(_as_f64(times_min), _as_f64(ct), _as_f64(cp), model,
                               bool(fit_delay), float(delay_bounds[0]), float(delay_bounds[1]))


def fit(series, times_min, model="ext_tofts", *, input="concentration",
        aif=None, hct=0.0, mask=None, enhance_thresh=0.0,
        t1_0_s=1.4, tr_s=0.005, fa_deg=25.0, r1=4.5, n_baseline=0, t1_map=None,
        fit_delay=False, delay_bounds=(0.0, 0.5)):
    """Voxel-wise PK fit over a 4-D ``(T, H, W)`` series → parameter maps.

    Returns ``{"ktrans", "ve", "vp", "rmse"}`` — each an ``(H, W)`` float32 map —
    plus ``"fitted"`` (the count of voxels actually fitted), and ``"delay"`` (min)
    when ``fit_delay=True``. Maps a model does not estimate (ve for Patlak, vp for
    Tofts) are zero.

    Parameters
    ----------
    series : (T, H, W) array — concentration, or raw signal when ``input='spgr'``.
    times_min : (T,) acquisition times in minutes (t=0 = injection); frames must be
        time-ordered with the pre-contrast baseline first.
    model : 'tofts' | 'ext_tofts' | 'patlak'.
    input : 'concentration' (default) or 'spgr' (convert signal→conc per voxel).
    aif : optional measured plasma AIF (T,) in mM; default uses the Parker curve.
    hct : haematocrit for the Parker→plasma path (default 0 = use Parker verbatim).
    mask : optional (H, W) array; non-zero voxels are fitted.
    enhance_thresh : skip voxels whose peak enhancement is below this (noise gate).
    t1_0_s, tr_s, fa_deg, r1, n_baseline : SPGR conversion params (input='spgr').
    t1_map : optional (H, W) per-voxel baseline T1 (s) overriding ``t1_0_s``.
    """
    if model not in MODELS:
        raise ValueError(f"model must be one of {MODELS}, got {model!r}")
    series = np.ascontiguousarray(series, dtype=np.float32)
    if series.ndim != 3:
        raise ValueError(f"series must be 3-D (T, H, W), got shape {series.shape}")
    measured_cp = None if aif is None else _as_f64(aif)
    mask_a = None if mask is None else np.ascontiguousarray(mask, dtype=np.uint8)
    t1_a = None if t1_map is None else np.ascontiguousarray(t1_map, dtype=np.float32)
    return _core.dce_fit_map(series, _as_f64(times_min), model, input, float(hct),
                             measured_cp, mask_a, float(t1_0_s), float(tr_s),
                             float(fa_deg), float(r1), int(n_baseline),
                             float(enhance_thresh), t1_a,
                             bool(fit_delay), float(delay_bounds[0]), float(delay_bounds[1]))


# ── 4-D DICOM feeder ──────────────────────────────────────────────────────
# Assembles the temporal stack + reads the acquisition times off the tags (the
# DICOM glue the C engine can't do) via the SHARED pydcm.perfusion feeder, then
# fits the whole volume in ONE native parallel call (dce_fit_volume fans the Z
# slices across cores in C++). DSC's fit_series rides the same feeder.

def fit_series(source, times_min=None, model="ext_tofts", *, input="concentration",
               aif=None, hct=0.0, mask=None, enhance_thresh=0.0,
               t1_0_s=1.4, tr_s=None, fa_deg=None, r1=4.5, n_baseline=0, t1_map=None,
               fit_delay=False, delay_bounds=(0.0, 0.5)):
    """Fit a whole DCE series → parameter-map **volumes**.

    `source` may be a DICOM directory / file-list / single enhanced-multiframe
    file (assembled via :func:`pydcm.load_4d`), a loaded :class:`pydcm.Volume4D`,
    or a raw array shaped ``(T, Z, Y, X)`` or ``(T, H, W)``. The whole volume is
    fitted in one native parallel call (Z slices fanned across cores in C++).

    Returns ``{"ktrans", "ve", "vp", "rmse"}`` — each a ``(Z, H, W)`` float32
    volume — plus ``"fitted"`` (total voxels fitted), ``"times_min"`` (the time
    grid used), and ``"delay"`` (min) when ``fit_delay=True``.

    Timing & sequence params: pass `times_min` explicitly (minutes; t=0 =
    injection), else they are read from the DICOM tags — the dynamic-time grid
    from AcquisitionDateTime → AcquisitionTime/ContentTime → TriggerTime
    (whichever varies), or the per-frame FrameAcquisitionDateTime for an
    enhanced-multiframe file; RepetitionTime / FlipAngle (via
    :func:`pydcm.bids_sidecar`) for the SPGR conversion. `input='spgr'` still needs
    a baseline `t1_0_s` / `t1_map` and `r1` (T1 is not a stored tag). `mask` and
    `t1_map` may be ``(H, W)`` (same for every slice) or ``(Z, H, W)``.

    IMPORTANT — injection alignment: auto-derived times are relative to the FIRST
    frame (t[0]=0); the Parker AIF assumes t=0 at injection. With pre-contrast
    baseline frames, pass `times_min` with baseline at negative t, supply a
    measured `aif`, or use `fit_delay=True` to absorb the offset.
    """
    if model not in MODELS:
        raise ValueError(f"model must be one of {MODELS}, got {model!r}")

    from . import perfusion
    px, paths, vol4d = perfusion.assemble_4d(source)
    T, Z = px.shape[0], px.shape[1]

    if times_min is not None:
        times = _as_f64(times_min)
    elif paths:
        ts = perfusion.series_times_s(paths, perfusion.frames_of(vol4d))
        if ts is None:
            raise ValueError("could not read acquisition times from DICOM "
                             "(no usable TriggerTime/AcquisitionTime); pass times_min=")
        times = ts / 60.0                       # DCE works in minutes
    else:
        raise ValueError("times_min is required for a raw-array source")
    if times.size != T:
        raise ValueError(f"times_min length {times.size} != T {T}")

    if input in ("spgr", "spgr_signal") and paths:
        tr_tag, fa_tag = perfusion.seq_tr_fa(paths[0])
        if tr_s is None:
            tr_s = tr_tag
        if fa_deg is None:
            fa_deg = fa_tag
    if input in ("spgr", "spgr_signal") and (tr_s is None or fa_deg is None):
        raise ValueError("SPGR input needs tr_s and fa_deg (not found in tags; pass them)")
    tr_s = 0.005 if tr_s is None else tr_s
    fa_deg = 25.0 if fa_deg is None else fa_deg

    mask_a = perfusion.broadcast_zhw(mask, Z, np.uint8)
    t1_a = perfusion.broadcast_zhw(t1_map, Z, np.float32)
    measured_cp = None if aif is None else _as_f64(aif)

    out = _core.dce_fit_volume(px, times, model, input, float(hct), measured_cp,
                               mask_a, float(t1_0_s), float(tr_s), float(fa_deg),
                               float(r1), int(n_baseline), float(enhance_thresh),
                               t1_a, bool(fit_delay), float(delay_bounds[0]),
                               float(delay_bounds[1]))
    out["times_min"] = times
    return out


def write_param_maps(reference, result, params=("ktrans", "ve", "vp"), *,
                     dtype=None, output_dir=None):
    """Emit DCE parameter VOLUMES as DICOM Parametric Maps (multi-frame when 3-D).

    Thin convenience over :func:`pydcm.write_paramap` that supplies the correct
    per-parameter units (Ktrans 1/min, ve/vp dimensionless, delay min): each
    requested map in `result` (the dict from :func:`fit` / :func:`fit_series`) is
    written against `reference` — the source DCE slices, one per Z plane — so a
    ``(Z, H, W)`` volume becomes one multi-frame Parametric Map.

    Returns ``{param: Part-10 bytes}``, or ``{param: written path}`` when
    `output_dir` is given. `dtype` (e.g. "uint16") is forwarded to write_paramap
    for integer-quantised storage.
    """
    import os
    from .paramap import write_paramap
    out = {}
    for p in params:
        if p not in result:
            raise ValueError(f"{p!r} not in result (have {sorted(result)})")
        vol = np.ascontiguousarray(result[p], dtype=np.float32)
        units = _PARAM_UNITS.get(p, ("1", "UCUM", "no units"))
        path = os.path.join(output_dir, f"dce_{p}.dcm") if output_dir else None
        blob = write_paramap(reference, vol, units=units, label=p, dtype=dtype, output=path)
        out[p] = path if path else blob
    return out
