# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Fuli Wu
# NOT a medical device — not for clinical or diagnostic use.

"""pydcm DSC-MRI — dynamic susceptibility contrast perfusion.

A thin NumPy surface over the native DSC engine. A bolus of contrast transiently
drops the T2*-weighted signal; the tissue concentration relates to the arterial
input by ``C(t) = CBF·(C_a ⊗ R)(t)``, so recovering CBF / MTT / Tmax needs
DECONVOLUTION — done by truncated SVD of the AIF convolution matrix:

    * ``ssvd`` — standard truncated SVD, causal Toeplitz (Østergaard 1996).
    * ``csvd`` — block-circulant SVD, delay-insensitive (Wu 2003).
    * ``osvd`` — oscillation-index SVD, per-voxel adaptive threshold (Wu 2003).

CBV is the area ratio ``∫C / ∫C_a`` (no deconvolution).

Typical use (concentration already computed)::

    import numpy as np, pydcm.dsc as dsc
    t = np.arange(0, 60, 1.0)               # seconds
    maps = dsc.fit(conc_4d, t, aif, method="osvd")  # conc_4d: (T, H, W)
    cbf = maps["cbf"]                       # (H, W) float32

From raw T2*-weighted signal instead of concentration::

    maps = dsc.fit(signal_4d, t, aif, input="raw", te_s=0.030, n_baseline=10)

Units: time in SECONDS; MTT/Tmax seconds; CBF/CBV are relative (calibration-free
— absolute mL/100g/min needs the caller's k·ρ·(1−Hct) scaling).
"""
from __future__ import annotations

import numpy as np

from . import _core

__all__ = ["signal_to_conc", "measure_aif", "deconvolve", "cbv", "cumtrapz",
           "leakage_correct", "first_pass_end", "fit", "fit_series",
           "write_param_maps", "METHODS"]

METHODS = ("ssvd", "csvd", "osvd")

# UCUM units (code, scheme, meaning) for each DSC output parameter.
_PARAM_UNITS = {
    "cbf":  ("ml/(100.ml.min)", "UCUM", "mL/100mL/min"),
    "cbv":  ("ml/(100.ml)", "UCUM", "mL/100mL"),
    "mtt":  ("s", "UCUM", "s"),
    "tmax": ("s", "UCUM", "s"),
    "ttp":  ("s", "UCUM", "s"),
}


def _as_f64(a):
    return np.ascontiguousarray(a, dtype=np.float64)


def signal_to_conc(signal, n_baseline, te_s):
    """DSC signal → ΔR2* concentration: ``c(t) = −ln(S(t)/S0)/TE``.

    `S0` is the mean of the first `n_baseline` (pre-bolus) frames. Returns the
    ΔR2* curve (signed; baseline ≈ 0, bolus > 0), same length as `signal`.
    """
    return _core.dsc_signal_to_conc(_as_f64(signal), int(n_baseline), float(te_s))


def cbv(ct, ref, dt_s):
    """Blood volume as the trapezoidal area ratio ``∫ct / ∫ref``.

    No deconvolution, and `dt` cancels — it is taken only so the call site
    states its units. What `ref` is decides what the number MEANS, and the two
    conventions are not interchangeable even though the arithmetic is one ratio:

    * the **AIF** gives CBV in the Østergaard/stroke sense, equal to CBF·MTT
      under the central-volume theorem;
    * a **normal-appearing white matter** ROI mean gives rCBV in the
      brain-tumour sense — Boxerman et al. (Neuro-Oncology 2020) and the ASFNR
      recommendation, which need no AIF at all.

    Integrate over the window you mean: pass a slice of the curve to stop after
    the first pass, which is where the endpoint half-step actually matters.
    """
    return _core.dsc_cbv(_as_f64(ct), _as_f64(ref), float(dt_s))


def cumtrapz(y, dt_s):
    """Running trapezoidal integral of `y`, starting at 0."""
    return _core.dsc_cumtrapz(_as_f64(y), float(dt_s))


def first_pass_end(conc, n_baseline, k_sigma=0.0):
    """How many leading samples make up the bolus's FIRST PASS.

    Recirculation — the second, lower bump as the bolus comes round again — is
    not part of the first transit and inflates every area that includes it.
    ASFNR/Welker 2015 names both conventions (integrate every acquired point, or
    stop after the first pass) and prescribes neither, so this is EVIDENCE for
    choosing a window, not the window.

    Takes the ΔR2* concentration curve, not raw signal. The yardstick is the
    curve's own pre-contrast scatter, so `n_baseline` must cover pre-contrast
    frames and be at least 2; `k_sigma` 0 selects the library default of 4.

    Returns the count — usable directly as the length to pass to
    :func:`leakage_correct` and :func:`cbv`, and **it must be the same one for
    both**: K2 is fitted to whatever stretch of curve it is shown, and
    correcting over one range while integrating over another costs more than
    half the benefit of correcting at all (CCC 0.9825 against 0.9309 on the
    GBM-DSC-MRI reference object). Returns the whole length when the curve never
    turns back up, or -1 when the inputs cannot support the reading.
    """
    return _core.dsc_first_pass_end(_as_f64(conc), int(n_baseline), float(k_sigma))


def leakage_correct(ct, ref, dt_s):
    """Boxerman–Schmainda–Weisskoff leakage correction of one curve.

    Fits ``ct ≈ K1·ref − K2·∫ref`` against a NON-LEAKING reference curve and
    returns ``ct + K2·∫ref`` — the curve with the extravasation term added back.
    Returns ``{"corrected": [T], "k2": float}``.

    Which voxels are non-leaking is the caller's decision: whole brain works in
    vivo because leaking voxels are a small minority of it, but an unmasked
    reference is destroyed by air, where the log floor sends ΔR2* to ~460 s⁻¹.
    For a phantom or a masked volume, pass the NAWM ROI mean.
    """
    return _core.dsc_leakage_correct(_as_f64(ct), _as_f64(ref), float(dt_s))


def measure_aif(series, mask, te_s, *, n_baseline=None):
    """Extract a ΔR2* arterial input function from an arterial ROI.

    Averages the ROI's T2*-weighted signal per time frame and converts to ΔR2*
    with :func:`signal_to_conc`. `series` is ``(T, H, W)`` or ``(T, Z, Y, X)``;
    `mask` is a non-zero ROI over the spatial dims. Returns the AIF ``C_a(t)``
    (length T) — pass it to :func:`fit` / :func:`deconvolve` as ``aif=``.
    `n_baseline` (pre-bolus frames) is auto-detected from the bolus DROP when None.

    NB: this returns the whole-blood ΔR2* AIF. Absolute quantification applies
    the ρ·(1−Hct) corrections downstream; relative maps need none.
    """
    from . import perfusion
    roi = perfusion.roi_signal(series, mask)                # shared ROI-mean helper
    if n_baseline is None:
        n_baseline = perfusion.auto_n_baseline(roi, rising=False)   # DSC: bolus DROP
    return signal_to_conc(roi, n_baseline, te_s)


def deconvolve(ct, aif, dt_s, method="osvd", *, reg=0.0):
    """Deconvolve ONE tissue curve against an AIF → perfusion parameters.

    Returns ``{"cbf", "cbv", "mtt", "tmax", "residue", "ok", "oi", "frac",
    "cut", "tail", "quality"}``. `reg` is the SVD truncation threshold
    (sSVD/cSVD) or the oscillation-index target (oSVD); ``reg=0`` selects the
    method default (0.20 sSVD, 0.10 cSVD, 0.035 oSVD). MTT/Tmax in seconds;
    CBF/CBV relative.

    **Check ``ok``.** The deconvolution produces numbers for almost any input,
    so a non-finite sample, a threshold that discards the whole spectrum, or a
    residue that never leaves zero would otherwise come back as
    ``cbf = mtt = tmax = 0`` — which on a map is indistinguishable from a voxel
    with no perfusion. Those return ``ok=False`` with the reason in ``quality``,
    a bitmask of ``dcm_dsc.h``'s ``DSC_Q_*``; they do NOT raise, because a
    caller looping over voxels meets them on ordinary background. Only a shape
    or context failure (bad lengths, ``dt <= 0``, a degenerate AIF) raises.

    The remaining fields say how much regularisation went into the answer:
    ``oi`` the achieved oscillation index, ``frac`` the truncation threshold
    actually used, ``cut`` the fraction of the curve's energy the truncation
    discarded, and ``tail`` the residue at the last sample as a fraction of its
    peak (large means the acquisition ended before the tissue cleared, so MTT is
    truncated low).
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    return _core.dsc_deconvolve_curve(_as_f64(ct), _as_f64(aif), float(dt_s),
                                      method, float(reg))


def fit(series, times_s, aif, method="osvd", *, input="concentration",
        reg=0.0, te_s=0.030, n_baseline=0, mask=None, enhance_thresh=0.0, leakage=False):
    """Voxel-wise DSC deconvolution over a ``(T, H, W)`` series → perfusion maps.

    Returns ``{"cbf", "cbv", "mtt", "tmax", "ttp"}`` — each an ``(H, W)`` float32
    map — plus ``"fitted"`` (voxels deconvolved). Tmax is the residue peak time
    (deconvolution); TTP the enhancement peak time (semi-quant). The AIF matrix is
    SVD-factorised once and reused across voxels. ``leakage=True`` applies
    Boxerman-Schmainda correction against the slice-mean reference (tumour DSC).

    Parameters
    ----------
    series : (T, H, W) — ΔR2* concentration, or raw T2* signal when ``input='raw'``.
    times_s : (T,) acquisition times in seconds (uniform spacing).
    aif : (T,) arterial input ΔR2* on the same grid (required — DSC has no
        population AIF).
    method : 'ssvd' | 'csvd' | 'osvd' (default).
    input : 'concentration' (default) or 'raw' (convert signal→ΔR2* per voxel).
    reg : SVD threshold / oSVD oscillation target; 0 → method default.
    te_s, n_baseline : ΔR2* conversion params (used when input='raw').
    mask : optional (H, W) array; non-zero voxels are fitted.
    enhance_thresh : skip voxels whose peak |Δconc| is below this (noise gate).
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    series = np.ascontiguousarray(series, dtype=np.float32)
    if series.ndim != 3:
        raise ValueError(f"series must be 3-D (T, H, W), got shape {series.shape}")
    mask_a = None if mask is None else np.ascontiguousarray(mask, dtype=np.uint8)
    return _core.dsc_fit_map(series, _as_f64(times_s), _as_f64(aif), method, input,
                             float(reg), float(te_s), int(n_baseline), mask_a,
                             float(enhance_thresh), bool(leakage))


def fit_series(source, times_s=None, aif=None, method="osvd", *,
               input="concentration", reg=0.0, te_s=None, n_baseline=0,
               mask=None, enhance_thresh=0.0, leakage=False):
    """Fit a whole DSC series → perfusion-map **volumes**.

    `source` may be a DICOM directory / file-list / enhanced-multiframe file
    (assembled via :func:`pydcm.load_4d`), a loaded :class:`pydcm.Volume4D`, or a
    raw ``(T, Z, Y, X)`` / ``(T, H, W)`` array. The whole volume is fitted in one
    native parallel call — the AIF SVD is built once and the Z slices fan across
    cores in C++ (:func:`pydcm.dsc.fit` is the single-slice form). Shares the
    :mod:`pydcm.perfusion` feeder with DCE for the DICOM assembly + timing.

    Returns ``{"cbf", "cbv", "mtt", "tmax", "ttp"}`` — each a ``(Z, H, W)`` float32
    volume — plus ``"fitted"`` and ``"times_s"``.

    `aif` (length T, ΔR2*) is REQUIRED (DSC has no population AIF) — measure it
    from an arterial ROI with :func:`measure_aif`. `times_s` and `te_s` are read
    from the DICOM tags when omitted; `mask` may be ``(H, W)`` or ``(Z, H, W)``.
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    if aif is None:
        raise ValueError("aif is required for DSC (no population AIF); see measure_aif")

    from . import perfusion
    px, paths, vol4d = perfusion.assemble_4d(source)
    T, Z = px.shape[0], px.shape[1]

    if times_s is not None:
        times = _as_f64(times_s)
    elif paths:
        times = perfusion.series_times_s(paths, perfusion.frames_of(vol4d))
        if times is None:
            raise ValueError("could not read acquisition times from DICOM "
                             "(no usable TriggerTime/AcquisitionTime); pass times_s=")
    else:
        raise ValueError("times_s is required for a raw-array source")
    if times.size != T:
        raise ValueError(f"times_s length {times.size} != T {T}")

    if input in ("raw", "raw_signal", "signal") and te_s is None and paths:
        te_s = perfusion.seq_te(paths[0])
    if input in ("raw", "raw_signal", "signal") and te_s is None:
        raise ValueError("raw-signal input needs te_s (EchoTime not in tags; pass it)")
    te_s = 0.030 if te_s is None else te_s

    aif_a = _as_f64(aif)
    if aif_a.size != T:
        raise ValueError(f"aif length {aif_a.size} != T {T}")
    mask_a = perfusion.broadcast_zhw(mask, Z, np.uint8)

    out = _core.dsc_fit_volume(px, times, aif_a, method, input, float(reg),
                               float(te_s), int(n_baseline), mask_a,
                               float(enhance_thresh), bool(leakage))
    out["times_s"] = times
    return out


def write_param_maps(reference, result, params=("cbf", "cbv", "mtt", "tmax", "ttp"), *,
                     dtype=None, output_dir=None):
    """Emit DSC perfusion VOLUMES as DICOM Parametric Maps (multi-frame when 3-D).

    Thin convenience over :func:`pydcm.write_paramap` supplying the per-parameter
    UCUM units (CBF mL/100mL/min, CBV mL/100mL, MTT/Tmax s): each requested map in
    `result` (from :func:`fit`) is written against `reference` (the source DSC
    slices, one per Z plane). Returns ``{param: Part-10 bytes}``, or
    ``{param: path}`` when `output_dir` is given.

    NB: the units labels are NOMINAL. :func:`fit` returns CALIBRATION-FREE
    relative CBF/CBV (MTT/Tmax are already absolute seconds); to store true
    mL/100mL[/min] you must pre-scale the maps by the absolute calibration
    (k·ρ·(1−Hct) and the s→min / fraction→mL/100mL factors) before writing.
    """
    import os
    from .paramap import write_paramap
    out = {}
    for p in params:
        if p not in result:
            raise ValueError(f"{p!r} not in result (have {sorted(result)})")
        vol = np.ascontiguousarray(result[p], dtype=np.float32)
        units = _PARAM_UNITS.get(p, ("1", "UCUM", "no units"))
        path = os.path.join(output_dir, f"dsc_{p}.dcm") if output_dir else None
        blob = write_paramap(reference, vol, units=units, label=p, dtype=dtype, output=path)
        out[p] = path if path else blob
    return out
