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
    series = np.ascontiguousarray(series, dtype=np.float32)
    T = series.shape[0]
    flat = series.reshape(T, -1)
    m = np.ascontiguousarray(mask).reshape(-1).astype(bool)
    if m.shape[0] != flat.shape[1]:
        raise ValueError(f"mask spatial size {m.shape[0]} != series {flat.shape[1]}")
    if not m.any():
        raise ValueError("empty ROI mask")
    roi = flat[:, m].mean(axis=1).astype(np.float64)        # mean ROI signal per frame
    if n_baseline is None:
        # Bolus arrival = first frame rising >10% above the leading plateau.
        lead = max(2, T // 20)
        base0 = roi[:lead].mean()
        above = np.flatnonzero(roi > base0 * 1.10 + 1e-12)
        n_baseline = int(above[0]) if above.size else lead
        n_baseline = max(1, n_baseline)
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
# Turns a DCE acquisition (a directory / file-list / loaded Volume4D / raw 4-D
# array) into parameter-map VOLUMES, fitting each slice's time course. This is
# the DICOM-side glue the C engine can't do: assemble the temporal stack, pull
# the real acquisition times + sequence params off the tags, and convert the
# DICOM millisecond units to the engine's (minutes for PK, seconds for SPGR).

def _hhmmss_to_seconds(v):
    """Parse a DICOM TM value (``HHMMSS.FFFFFF``) to seconds since midnight."""
    s = str(v).strip()
    if not s:
        return None
    try:
        hh = int(s[0:2]); mm = int(s[2:4]); ss = float(s[4:]) if len(s) > 4 else 0.0
        return hh * 3600.0 + mm * 60.0 + ss
    except (ValueError, IndexError):
        return None


def _dt_to_seconds(v):
    """Parse a DICOM DT value (``YYYYMMDDHHMMSS.FFFFFF&ZZXX``) to absolute seconds.

    Uses the date ordinal so day rollover is handled (the wall-clock TM path can
    wrap at midnight); only differences are used downstream, so the epoch is
    arbitrary. The trailing timezone offset, if any, is ignored (one series →
    one zone), the standard acquisition-date-time handling.
    """
    import datetime as _dt
    s = str(v).strip()
    if len(s) < 8:
        return None
    core = s.split("+")[0].split("&")[0]            # drop timezone suffix
    try:
        d = _dt.date(int(core[0:4]), int(core[4:6]), int(core[6:8]))
        rest = core[8:]
        hh = int(rest[0:2]) if len(rest) >= 2 else 0
        mm = int(rest[2:4]) if len(rest) >= 4 else 0
        ss = float(rest[4:]) if len(rest) > 4 else 0.0
        return d.toordinal() * 86400.0 + hh * 3600.0 + mm * 60.0 + ss
    except (ValueError, IndexError):
        return None


def _pick_times_min(datetime_s, clock_s, trigger_ms):
    """Choose the dynamic-time vector (minutes, t[0]=0) from candidate tag series.

    Preference: the true acquisition clock first —
    AcquisitionDateTime (midnight-safe), then AcquisitionTime/ContentTime — and
    TriggerTime (cardiac, ms) only as a fallback for sequences whose clock is
    constant. A series is used only if every volume has it AND it varies.
    Returns a float64 array or None.
    """
    for seq, to_min in ((datetime_s, 1 / 60.0), (clock_s, 1 / 60.0),
                        (trigger_ms, 1 / 60000.0)):
        if all(x is not None for x in seq) and len(set(seq)) > 1:
            t0 = seq[0]
            return np.array([(x - t0) * to_min for x in seq], dtype=np.float64)
    return None


def _mf_frame_times_min(ds, frames):
    """Per-FRAME acquisition times (min, t[0]=0) from an enhanced-multiframe dataset.

    DCE enhanced-MF stores one volume's frames in a single file; the per-frame
    timestamp lives in PerFrameFunctionalGroupsSequence[f] → FrameContentSequence
    → FrameAcquisitionDateTime (0018,9074, DT). Falls back to
    FrameReferenceDateTime. Returns None if unavailable/constant.
    """
    pf = getattr(ds, "PerFrameFunctionalGroupsSequence", None)
    if not pf:
        return None
    secs = []
    for fr in frames:
        if fr >= len(pf):
            return None
        fcs = getattr(pf[fr], "FrameContentSequence", None)
        dt = None
        if fcs:
            dt = (getattr(fcs[0], "FrameAcquisitionDateTime", None)
                  or getattr(fcs[0], "FrameReferenceDateTime", None))
        secs.append(_dt_to_seconds(dt) if dt not in (None, "") else None)
    if all(s is not None for s in secs) and len(set(secs)) > 1:
        t0 = secs[0]
        return np.array([(s - t0) / 60.0 for s in secs], dtype=np.float64)
    return None


def _series_times_min(paths, frames=None):
    """Derive per-volume acquisition times (minutes, t[0]=0) from DICOM tags.

    Enhanced multiframe (one file repeated across volumes, distinct `frames`) is
    read from the per-frame functional groups; classic multi-instance series use
    the per-file top-level tags (acquisition-clock preference order).
    """
    from . import dcmread
    if frames is not None and len(paths) > 1 and len(set(map(str, paths))) == 1:
        t = _mf_frame_times_min(dcmread(paths[0]), frames)
        if t is not None:
            return t                                  # else fall through to top-level tags
    datetime_s, clock_s, trigger_ms = [], [], []
    for p in paths:
        ds = dcmread(p, stop_before_pixels=True)
        adt = getattr(ds, "AcquisitionDateTime", None)
        datetime_s.append(_dt_to_seconds(adt) if adt not in (None, "") else None)
        tm = getattr(ds, "AcquisitionTime", None) or getattr(ds, "ContentTime", None)
        clock_s.append(_hhmmss_to_seconds(tm) if tm not in (None, "") else None)
        tt = getattr(ds, "TriggerTime", None)
        trigger_ms.append(float(tt) if tt not in (None, "") else None)
    return _pick_times_min(datetime_s, clock_s, trigger_ms)


def _series_tr_fa(path0):
    """Read (RepetitionTime[s], FlipAngle[deg]) off one instance, or (None, None).

    Reuses :func:`pydcm.bids_sidecar` — the vendor-aware extractor that already
    emits RepetitionTime in **seconds** (the ms→s conversion lives there, once),
    rather than re-reading the raw tag and re-doing the unit maths here.
    """
    from . import bids_sidecar
    sc = bids_sidecar(path0)
    return sc.get("RepetitionTime"), sc.get("FlipAngle")


def fit_series(source, times_min=None, model="ext_tofts", *, input="concentration",
               aif=None, hct=0.0, mask=None, enhance_thresh=0.0,
               t1_0_s=1.4, tr_s=None, fa_deg=None, r1=4.5, n_baseline=0, t1_map=None,
               fit_delay=False, delay_bounds=(0.0, 0.5)):
    """Fit a whole DCE series → parameter-map **volumes**.

    `source` may be a DICOM directory / file-list / single enhanced-multiframe
    file (assembled via :func:`pydcm.load_4d`), a loaded :class:`pydcm.Volume4D`,
    or a raw array shaped ``(T, Z, Y, X)`` or ``(T, H, W)``. Each slice's
    ``(T, H, W)`` time course is fitted independently with :func:`fit`.

    Returns ``{"ktrans", "ve", "vp", "rmse"}`` — each a ``(Z, H, W)`` float32
    volume — plus ``"fitted"`` (total voxels fitted), ``"times_min"`` (the time
    grid used), and ``"delay"`` (min) when ``fit_delay=True``. With `fit_delay`
    the per-voxel bolus-arrival delay is jointly fitted over `delay_bounds`,
    which absorbs the offset between t=0 and the true injection — the robust way
    to use auto-derived (frame-relative) times.

    Timing & sequence params: pass `times_min` explicitly (minutes; t=0 =
    injection), else they are read from the DICOM tags — the dynamic-time grid
    from AcquisitionDateTime → AcquisitionTime/ContentTime → TriggerTime
    (acquisition-clock preference order; whichever varies), or, for an enhanced-multiframe
    file, the per-frame FrameAcquisitionDateTime; RepetitionTime / FlipAngle
    (via :func:`pydcm.bids_sidecar`, already in seconds) for the SPGR conversion. `input='spgr'` still needs a baseline `t1_0_s` / `t1_map` and `r1`
    (T1 is not a stored tag).

    IMPORTANT — injection alignment: auto-derived times are relative to the FIRST
    frame (t[0]=0), but the population (Parker) AIF assumes t=0 at the contrast
    injection. If the series has pre-contrast baseline frames, pass `times_min`
    with the baseline frames at negative t (injection at 0), or supply a measured
    `aif`; otherwise the AIF is mis-aligned and Ktrans/vp are biased. Automatic
    bolus-arrival/delay estimation is not yet implemented (deferred).
    """
    if model not in MODELS:
        raise ValueError(f"model must be one of {MODELS}, got {model!r}")

    from .volume import Volume4D, load_4d

    paths = None
    vol4d = None
    if isinstance(source, np.ndarray):
        px = source
    elif isinstance(source, Volume4D):
        vol4d = source
    else:                                   # path / file-list / multiframe file
        vol4d = load_4d(source)
    if vol4d is not None:
        px = vol4d.pixels
        paths = vol4d.volume_path
        # The 4th dimension must be TIME. load_4d also assembles b-value / echo /
        # cardiac-phase stacks the same way, and fitting one of those as a time
        # course would be silently wrong — reject it (a raw array carries no axis
        # labels, so this guard only applies to the DICOM/Volume4D path).
        kinds = [d.kind for d in vol4d.dimensions]
        if len(kinds) > 1:
            raise ValueError(f"series varies along multiple axes {kinds}; fit_series "
                             "needs a single temporal 4th dimension")
        if len(kinds) == 1 and kinds[0] != "temporal":
            raise ValueError(f"series 4th axis is {kinds[0]!r}, not temporal — "
                             "not a dynamic DCE series")

    px = np.ascontiguousarray(px, dtype=np.float32)
    if px.ndim == 3:                        # (T, H, W) → single slice
        px = px[:, None, :, :]
    if px.ndim != 4:
        raise ValueError(f"series must be 4-D (T,Z,Y,X) or 3-D (T,H,W); got {px.shape}")
    T, Z = px.shape[0], px.shape[1]

    # Resolve the time grid: explicit wins; else read from the source tags.
    if times_min is not None:
        times = _as_f64(times_min)
    elif paths:
        frames = vol4d.volume_frame if vol4d is not None else None
        times = _series_times_min(paths, frames)
        if times is None:
            raise ValueError("could not read acquisition times from DICOM "
                             "(no usable TriggerTime/AcquisitionTime); pass times_min=")
    else:
        raise ValueError("times_min is required for a raw-array source")
    if times.size != T:
        raise ValueError(f"times_min length {times.size} != T {T}")

    # SPGR sequence params from tags when not given.
    if input in ("spgr", "spgr_signal") and paths:
        tr_tag, fa_tag = _series_tr_fa(paths[0])
        if tr_s is None:
            tr_s = tr_tag
        if fa_deg is None:
            fa_deg = fa_tag
    if input in ("spgr", "spgr_signal") and (tr_s is None or fa_deg is None):
        raise ValueError("SPGR input needs tr_s and fa_deg (not found in tags; pass them)")
    tr_s = 0.005 if tr_s is None else tr_s
    fa_deg = 25.0 if fa_deg is None else fa_deg

    def _slice(arr, z):
        if arr is None:
            return None
        a = np.asarray(arr)
        return a[z] if a.ndim == 3 else a

    keys = ("ktrans", "ve", "vp", "rmse") + (("delay",) if fit_delay else ())
    out = {k: np.empty((Z,) + px.shape[2:], dtype=np.float32) for k in keys}
    fitted = 0
    for z in range(Z):
        m = fit(px[:, z], times, model, input=input, aif=aif, hct=hct,
                mask=_slice(mask, z), enhance_thresh=enhance_thresh,
                t1_0_s=t1_0_s, tr_s=tr_s, fa_deg=fa_deg, r1=r1,
                n_baseline=n_baseline, t1_map=_slice(t1_map, z),
                fit_delay=fit_delay, delay_bounds=delay_bounds)
        for k in keys:
            out[k][z] = m[k]
        fitted += int(m["fitted"])
    out["fitted"] = fitted
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
