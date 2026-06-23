# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Fuli Wu
# NOT a medical device — not for clinical or diagnostic use.

"""Shared dynamic-perfusion DICOM feeder — the DICOM-side glue common to DCE & DSC.

Both modalities take the SAME shape of input — a 4-D dynamic acquisition (a
directory / file-list / enhanced-multiframe file / loaded :class:`pydcm.Volume4D`
/ raw array) — and need the SAME two things the native compute core cannot do:

  1. assemble the temporal stack into a contiguous ``(T, Z, H, W)`` cube, and
  2. recover the per-frame acquisition times off the DICOM tags.

This module is that one feeder (so :mod:`pydcm.dce` and :mod:`pydcm.dsc` do not
each re-roll the tag parsing). Times are returned in **seconds** — DSC uses them
verbatim, DCE divides by 60. The vendor-aware 4-D assembly itself lives lower
still (native ``load_4d`` / ``Volume4D``); this layer only adds the perfusion
timing + sequence-parameter extraction on top.
"""
from __future__ import annotations

import numpy as np

__all__ = ["assemble_4d", "series_times_s", "seq_tr_fa", "seq_te",
           "roi_signal", "auto_n_baseline", "TICMetrics", "tic", "tic_roi"]


# ── Arterial-ROI helpers (shared by dce.measure_aif / dsc.measure_aif) ──────
def roi_signal(series, mask):
    """Mean signal per time frame over a non-zero spatial ROI ``mask``.

    `series` is ``(T, H, W)`` or ``(T, Z, Y, X)``; `mask` is broadcast over the
    spatial dims. Returns the ROI mean curve ``(T,)`` as float64.
    """
    series = np.ascontiguousarray(series, dtype=np.float32)
    T = series.shape[0]
    flat = series.reshape(T, -1)
    m = np.ascontiguousarray(mask).reshape(-1).astype(bool)
    if m.shape[0] != flat.shape[1]:
        raise ValueError(f"mask spatial size {m.shape[0]} != series {flat.shape[1]}")
    if not m.any():
        raise ValueError("empty ROI mask")
    return flat[:, m].mean(axis=1).astype(np.float64)


def auto_n_baseline(roi, *, rising):
    """Pre-bolus frame count = first frame departing the leading plateau by >10%.

    `rising=True` for DCE (T1 enhancement rises), `rising=False` for DSC (T2*
    susceptibility drops). Used as the S0 averaging window for signal→conc.
    """
    T = len(roi)
    lead = max(2, T // 20)
    base0 = roi[:lead].mean()
    if rising:
        idx = np.flatnonzero(roi > base0 * 1.10 + 1e-12)
    else:
        idx = np.flatnonzero(roi < base0 * 0.90)
    return max(1, int(idx[0]) if idx.size else lead)


# ── DICOM time parsing ─────────────────────────────────────────────────────
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

    Uses the date ordinal so day rollover is handled; only differences are used
    downstream, so the epoch is arbitrary. A trailing timezone offset is ignored
    (one series → one zone).
    """
    import datetime as _dt
    s = str(v).strip()
    if len(s) < 8:
        return None
    # Drop the trailing timezone offset (&ZZXX / +HHMM / −HHMM) — the sign only
    # appears in the suffix, never in the YYYYMMDD… core, so scan from past the
    # date (index 8) for the first +/−/& and cut there.
    cut = len(s)
    for i in range(8, len(s)):
        if s[i] in "+-&":
            cut = i
            break
    core = s[:cut]
    try:
        d = _dt.date(int(core[0:4]), int(core[4:6]), int(core[6:8]))
        rest = core[8:]
        hh = int(rest[0:2]) if len(rest) >= 2 else 0
        mm = int(rest[2:4]) if len(rest) >= 4 else 0
        ss = float(rest[4:]) if len(rest) > 4 else 0.0
        return d.toordinal() * 86400.0 + hh * 3600.0 + mm * 60.0 + ss
    except (ValueError, IndexError):
        return None


def _pick_times_s(datetime_s, clock_s, trigger_ms):
    """Choose the dynamic-time vector (seconds, t[0]=0) from candidate tag series.

    Preference = the true acquisition clock first (AcquisitionDateTime, then
    AcquisitionTime/ContentTime), TriggerTime (ms) only as a fallback. A series is
    used only if every volume carries it AND it varies. Returns float64 or None.
    """
    for seq, scale in ((datetime_s, 1.0), (clock_s, 1.0), (trigger_ms, 1e-3)):
        if all(x is not None for x in seq) and len(set(seq)) > 1:
            t0 = seq[0]
            return np.array([(x - t0) * scale for x in seq], dtype=np.float64)
    return None


def _mf_frame_times_s(ds, frames):
    """Per-FRAME times (s, t[0]=0) from an enhanced-multiframe dataset, or None.

    Enhanced-MF stores one volume's frames in a single file; the per-frame stamp
    is PerFrameFunctionalGroupsSequence[f] → FrameContentSequence →
    FrameAcquisitionDateTime (0018,9074), falling back to FrameReferenceDateTime.
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
        return np.array([s - t0 for s in secs], dtype=np.float64)
    return None


def series_times_s(paths, frames=None):
    """Per-volume acquisition times (seconds, t[0]=0) from DICOM tags, or None.

    Enhanced multiframe (one file repeated across volumes, distinct `frames`) is
    read from the per-frame functional groups; classic multi-instance series use
    the per-file top-level tags (acquisition-clock preference order).
    """
    from . import dcmread
    if frames is not None and len(paths) > 1 and len(set(map(str, paths))) == 1:
        t = _mf_frame_times_s(dcmread(paths[0], stop_before_pixels=True), frames)
        if t is not None:
            return t
    datetime_s, clock_s, trigger_ms = [], [], []
    for p in paths:
        ds = dcmread(p, stop_before_pixels=True)
        adt = getattr(ds, "AcquisitionDateTime", None)
        datetime_s.append(_dt_to_seconds(adt) if adt not in (None, "") else None)
        tm = getattr(ds, "AcquisitionTime", None) or getattr(ds, "ContentTime", None)
        clock_s.append(_hhmmss_to_seconds(tm) if tm not in (None, "") else None)
        tt = getattr(ds, "TriggerTime", None)
        trigger_ms.append(float(tt) if tt not in (None, "") else None)
    return _pick_times_s(datetime_s, clock_s, trigger_ms)


# ── Sequence parameters (vendor-aware, via bids_sidecar — RepetitionTime/EchoTime
#    already arrive in SECONDS from there, so the ms→s maths lives once) ──
def seq_tr_fa(path0):
    """(RepetitionTime[s], FlipAngle[deg]) off one instance, or (None, None)."""
    from . import bids_sidecar
    sc = bids_sidecar(path0)
    return sc.get("RepetitionTime"), sc.get("FlipAngle")


def seq_te(path0):
    """EchoTime[s] off one instance, or None."""
    from . import bids_sidecar
    return bids_sidecar(path0).get("EchoTime")


# ── 4-D assembly ───────────────────────────────────────────────────────────
def assemble_4d(source):
    """Resolve a perfusion `source` to ``(pixels[T,Z,H,W] f32, paths, vol4d)``.

    `source` may be a DICOM directory / file-list / enhanced-multiframe file
    (assembled via native :func:`pydcm.load_4d`), a loaded :class:`pydcm.Volume4D`,
    or a raw ``(T,Z,Y,X)`` / ``(T,H,W)`` array. The 4th DICOM dimension is checked
    to be TEMPORAL (load_4d also stacks b-value / echo / cardiac-phase series the
    same way — fitting one of those as a time course would be silently wrong).

    Returns the contiguous cube plus `paths` (per-volume file list, or None for a
    raw array) and the `vol4d` (or None) so callers can pull timing / geometry.
    """
    from .volume import Volume4D, load_4d

    paths = None
    vol4d = None
    if isinstance(source, np.ndarray):
        px = source
    elif isinstance(source, Volume4D):
        vol4d = source
    else:
        vol4d = load_4d(source)
    if vol4d is not None:
        px = vol4d.pixels
        paths = vol4d.volume_path
        kinds = [d.kind for d in vol4d.dimensions]
        if len(kinds) > 1:
            raise ValueError(f"series varies along multiple axes {kinds}; perfusion "
                             "needs a single temporal 4th dimension")
        if len(kinds) == 1 and kinds[0] != "temporal":
            raise ValueError(f"series 4th axis is {kinds[0]!r}, not temporal — "
                             "not a dynamic perfusion series")

    px = np.ascontiguousarray(px, dtype=np.float32)
    if px.ndim == 3:                        # (T, H, W) → single slice
        px = px[:, None, :, :]
    if px.ndim != 4:
        raise ValueError(f"series must be 4-D (T,Z,Y,X) or 3-D (T,H,W); got {px.shape}")
    return px, paths, vol4d


def frames_of(vol4d):
    """The per-volume enhanced-MF frame indices (or None) for `vol4d`."""
    return None if vol4d is None else getattr(vol4d, "volume_frame", None)


def broadcast_zhw(arr, Z, dtype):
    """Normalise an optional per-voxel array (mask / T1 map) to ``(Z, H, W)``.

    Accepts None (→ None), a 2-D ``(H, W)`` plane (→ repeated across Z, the common
    "same ROI on every slice" case), or an already-3-D ``(Z, H, W)`` volume.
    """
    if arr is None:
        return None
    a = np.ascontiguousarray(arr, dtype=dtype)
    if a.ndim == 2:
        a = np.repeat(a[None, :, :], Z, axis=0)
    if a.ndim != 3:
        raise ValueError(f"expected a 2-D (H,W) or 3-D (Z,H,W) array, got {a.shape}")
    return np.ascontiguousarray(a, dtype=dtype)


# ── Time–intensity curve readouts ──────────────────────────────────────────
#
# The semi-quantitative terms a reader takes off a dynamic series without a
# contrast model — no baseline T1, no relaxivity, no arterial input — so they
# apply to any time-resolved acquisition, not only the DCE runs :mod:`pydcm.dce`
# fits. The arithmetic is the native engine's; every definitional choice inside
# these terms lives there so a viewer, the CLI and this module cannot answer the
# same study three ways.


class TICMetrics:
    """Readouts off one time–intensity curve.

    Attributes (values in the caller's own units; times carry through unchanged):
        baseline: mean of the first ``n_baseline`` samples.
        peak / peak_time, trough / trough_time: extremes of the RAW curve, not a
            smoothed one.
        enhancement: ``(peak - baseline) / abs(baseline)``. ``nan`` when the
            baseline is 0 — a relative change from nothing has no value, and 0
            there would read as "no enhancement".
        washin: mean slope from the last baseline sample to the peak, per time
            unit. ``nan`` when the peak is at or before the baseline window.
        washout: mean slope from the peak to the LAST sample (negative for a
            curve that comes back down). ``nan`` when the peak IS the last sample.
        auc: trapezoidal integral of ``value - baseline``, signed — a curve that
            dips below baseline subtracts.
        rise_time: time to the first crossing of
            ``baseline + rise_frac * (peak - baseline)``, linearly interpolated.
            ``nan`` when the curve never reaches it.
        n_baseline: the window actually used, after clamping into range.
        ok: the readouts are meaningful.

    A ``nan`` above is a term that is undefined for this curve, not a failed
    computation and not a zero.
    """

    __slots__ = ("baseline", "peak", "peak_time", "trough", "trough_time",
                 "enhancement", "washin", "washout", "auc", "rise_time",
                 "n_baseline", "ok")

    def __init__(self, d):
        for k in self.__slots__:
            setattr(self, k, d[k])

    def as_dict(self):
        return {k: getattr(self, k) for k in self.__slots__}

    def __repr__(self):
        return (f"<TICMetrics peak={self.peak:g}@{self.peak_time:g} "
                f"baseline={self.baseline:g} enhancement={self.enhancement:g} "
                f"auc={self.auc:g}>")


def tic(t, v, *, n_baseline=None, rise_frac=0.5):
    """Readouts off one time–intensity curve `v` sampled at times `t`.

    `t` must be strictly increasing and both arrays finite — the integral and
    the slopes are measured along `t`, so an unsorted axis or a ``nan`` sample
    would produce a number that looks fine and is not; both raise.

    `n_baseline` is how many leading samples are averaged for the baseline.
    ``None`` takes the engine's documented fallback — a tenth of the curve, at
    least one sample — which is **this library's choice, not a field
    convention**. A caller that knows when contrast arrived should say so
    instead; :func:`tic_roi` will detect it for you with ``rising=``.

    `rise_frac` is the fraction used for ``rise_time`` (0.5 = the conventional
    half-rise). Out-of-range values yield a ``nan`` rise_time rather than being
    clamped to a different question.
    """
    from . import _core
    ta = np.ascontiguousarray(t, dtype=np.float64).ravel()
    va = np.ascontiguousarray(v, dtype=np.float64).ravel()
    return TICMetrics(_core.tic_metrics(ta, va, int(n_baseline or 0),
                                        float(rise_frac)))


def tic_roi(series, mask, times_s, *, n_baseline=None, rising=None,
            rise_frac=0.5):
    """Readouts off the mean curve of a spatial ROI in a dynamic `series`.

    `series` is ``(T, H, W)`` or ``(T, Z, Y, X)`` and `mask` is broadcast over
    the spatial dims — the same pair :func:`roi_signal` takes, and the same one
    ``dce.measure_aif`` / ``dsc.measure_aif`` take, so an arterial ROI can be
    read both ways without being re-extracted.

    `times_s` are the per-frame times (see :func:`series_times_s`).

    Baseline window, in order of precedence:

    * `n_baseline` — an explicit count.
    * `rising` — detect it with :func:`auto_n_baseline`: the first frame
      departing the leading plateau by >10%. ``True`` for a curve that rises
      (T1 enhancement), ``False`` for one that drops (T2* susceptibility). This
      is the "say when contrast arrived" path, and it is what to use on a
      contrast run.
    * neither — the engine's arbitrary fallback.
    """
    roi = roi_signal(series, mask)
    if n_baseline is None and rising is not None:
        n_baseline = auto_n_baseline(roi, rising=bool(rising))
    ta = np.ascontiguousarray(times_s, dtype=np.float64).ravel()
    if ta.shape[0] != roi.shape[0]:
        raise ValueError(f"times_s has {ta.shape[0]} entries but the series has "
                         f"{roi.shape[0]} frames")
    return tic(ta, roi, n_baseline=n_baseline, rise_frac=rise_frac)
