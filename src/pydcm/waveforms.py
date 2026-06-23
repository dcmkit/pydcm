# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Fuli Wu
# NOT a medical device — not for clinical or diagnostic use.
"""DICOM waveform I/O for ECG / EEG / hemodynamic / audio (``pydcm.waveforms``).

Three layers, all over the one DICOM model (no signal-analysis reimplemented — that
is neurokit2 / MNE territory; feed them the arrays this module returns):

* ``multiplex_array`` / ``generate_multiplex``.
* ``read_waveform`` — rich read: every multiplex group as physical-unit signals plus
  per-channel metadata (lead/electrode source, units, sensitivity, filters) and the
  waveform annotations — as NumPy + dicts.
* ``write_waveform`` — author a Waveform SOP instance (12-lead/General ECG, scalp/sleep
  EEG, EMG/EOG, hemodynamic, respiratory, audio …) from per-channel arrays + metadata.
* ``to_mne`` — hand a group straight to MNE-Python (EEG/MEG) as a ``RawArray``.
"""
from __future__ import annotations

import numpy as np

_DTYPE = {(8, True): "<i1", (8, False): "<u1", (16, True): "<i2", (16, False): "<u2",
          (32, True): "<i4", (32, False): "<u4", (64, True): "<f8"}

# friendly kind -> (pydcm.sop_class attribute, DICOM Modality)
_KIND = {
    "ecg12": ("TwelveLeadECGWaveformStorage", "ECG"),
    "ecg": ("GeneralECGWaveformStorage", "ECG"),
    "ecg32": ("General32BitECGWaveformStorage", "ECG"),
    "ambulatory_ecg": ("AmbulatoryECGWaveformStorage", "ECG"),
    "hemodynamic": ("HemodynamicWaveformStorage", "HD"),
    "eps": ("CardiacElectrophysiologyWaveformStorage", "EPS"),
    "eeg": ("RoutineScalpElectroencephalogramWaveformStorage", "EEG"),
    "sleep_eeg": ("SleepElectroencephalogramWaveformStorage", "EEG"),
    "emg": ("ElectromyogramWaveformStorage", "EMG"),
    "eog": ("ElectrooculogramWaveformStorage", "EOG"),
    "arterial_pulse": ("ArterialPulseWaveformStorage", "HD"),
    "respiratory": ("RespiratoryWaveformStorage", "RESP"),
    "audio": ("BasicVoiceAudioWaveformStorage", "AU"),
}


# ---- surface -------------------------------------------
def multiplex_array(ds, index: int = 0, as_raw: bool = True):
    """The (samples × channels) array of multiplex group ``index`` (PS3.3 C.10.9).

    With ``as_raw=False`` the per-channel Sensitivity / baseline correction from
    ChannelDefinitionSequence is applied (real-world units)."""
    seq = ds.get("WaveformSequence")
    if not seq:
        raise AttributeError("Dataset has no WaveformSequence")
    item = seq[index]
    nch = int(item.NumberOfWaveformChannels)
    nsamp = int(item.NumberOfWaveformSamples)
    bits = int(item.WaveformBitsAllocated)
    signed = str(item.get("WaveformSampleInterpretation", "SS")).startswith("S")
    dt = _DTYPE.get((bits, signed))
    if dt is None:
        raise ValueError(f"unsupported WaveformBitsAllocated/Interpretation: {bits}/"
                         f"{item.get('WaveformSampleInterpretation')} (mu-law/A-law audio "
                         "not decoded here)")
    arr = np.frombuffer(bytes(item.WaveformData), dtype=dt)[: nch * nsamp].reshape(nsamp, nch)
    if as_raw:
        return arr
    out = arr.astype("float64")
    for c, chan in enumerate((item.get("ChannelDefinitionSequence") or [])[:nch]):
        sens = chan.get("ChannelSensitivity")
        corr = chan.get("ChannelSensitivityCorrectionFactor")
        baseline = chan.get("ChannelBaseline", 0)
        if sens is not None:
            out[:, c] = (out[:, c] - float(baseline or 0)) * float(sens) * float(corr or 1)
    return out


def generate_multiplex(ds, as_raw: bool = True):
    """Yield each multiplex group's array."""
    for i in range(len(ds.get("WaveformSequence") or [])):
        yield multiplex_array(ds, i, as_raw=as_raw)


# ---- rich read ------------------------------------------------------------
def _code_meaning(seq):
    return str(seq[0].get("CodeMeaning")) if seq else None


def _as_ds(src):
    from . import dcmread
    import os
    if hasattr(src, "WaveformSequence") or hasattr(src, "get"):
        return src
    return dcmread(os.fspath(src))


def read_waveform(src) -> dict:
    """Read every multiplex group of a waveform SOP instance into physical-unit signals
    plus metadata. ``src`` is a path or a parsed dataset. Returns::

        {modality, sop_class_uid, groups: [{sampling_frequency, num_samples,
          duration_s, channels: [{label, source, units, sensitivity, baseline,
          sensitivity_correction, filter_low, filter_high, notch}], signals (n×ch
          physical units), raw (n×ch), annotations: [...]}], annotations: [...]}

    Hand ``signals`` (and a channel's ``source``/``units``) straight to neurokit2
    (ECG) or MNE (EEG) — see ``to_mne``.
    """
    ds = _as_ds(src)
    groups = []
    for i, item in enumerate(ds.get("WaveformSequence") or []):
        nch = int(item.NumberOfWaveformChannels)
        freq = item.get("SamplingFrequency")
        freq = float(freq) if freq else None
        nsamp = int(item.NumberOfWaveformSamples)
        chans = []
        for chan in (item.get("ChannelDefinitionSequence") or [])[:nch]:
            chans.append({
                "label": chan.get("ChannelLabel"),
                "source": _code_meaning(chan.get("ChannelSourceSequence")),
                "units": _code_meaning(chan.get("ChannelSensitivityUnitsSequence")),
                "sensitivity": _f(chan.get("ChannelSensitivity")),
                "sensitivity_correction": _f(chan.get("ChannelSensitivityCorrectionFactor")),
                "baseline": _f(chan.get("ChannelBaseline")) or 0.0,
                "filter_low": _f(chan.get("FilterLowFrequency")),
                "filter_high": _f(chan.get("FilterHighFrequency")),
                "notch": _f(chan.get("NotchFilterFrequency")),
            })
        groups.append({
            "sampling_frequency": freq,
            "num_samples": nsamp,
            "duration_s": (nsamp / freq) if freq else None,
            "channels": chans,
            "signals": multiplex_array(ds, i, as_raw=False),
            "raw": multiplex_array(ds, i, as_raw=True),
            "annotations": _annotations(ds, group_index=i + 1),
        })
    return {"modality": ds.get("Modality"), "sop_class_uid": str(ds.get("SOPClassUID") or ""),
            "groups": groups, "annotations": _annotations(ds)}


def _f(v):
    try:
        return float(v) if v is not None and v != "" else None
    except (TypeError, ValueError):
        return None


def _mv(v):                                          # a UL/US multi-value → list (VM 1 = scalar)
    if v is None:
        return []
    return [v] if isinstance(v, (int, float, str)) else list(v)


def _annotations(ds, group_index=None):
    out = []
    for an in (ds.get("WaveformAnnotationSequence") or []):
        chans = _mv(an.get("ReferencedWaveformChannels"))
        if group_index is not None and (not chans or int(chans[0]) != group_index):
            continue
        out.append({
            "channels": [int(x) for x in chans],
            "text": an.get("UnformattedTextValue"),
            "concept": _code_meaning(an.get("ConceptNameCodeSequence")),
            "value": _f(an.get("NumericValue")),
            "units": _code_meaning(an.get("MeasurementUnitsCodeSequence")),
            "samples": [int(x) for x in _mv(an.get("ReferencedSamplePositions"))],
        })
    return out


# ---- write ----------------------------------------------------------------
def _code(value, scheme, meaning):
    from .dataset import Dataset
    c = Dataset()
    c.CodeValue, c.CodingSchemeDesignator, c.CodeMeaning = value, scheme, meaning
    return c


def write_waveform(out, signals, *, sampling_frequency, kind="ecg12", labels=None,
                   sources=None, units="mV", sensitivity=None, sample_bits=16,
                   patient_id="", patient_name="Anonymous^", patient_birth_date="",
                   patient_sex="", study_uid=None, series_uid=None, sop_uid=None,
                   series_number=1, instance_number=1):
    """Author a DICOM Waveform SOP instance from per-channel signals.

    ``signals``: an ``(n_samples, n_channels)`` array (or a list of 1-D channel
    arrays) in physical ``units``. Each channel is quantised to a ``sample_bits``-bit
    integer via its ``sensitivity`` (physical units per LSB); ``ChannelSensitivity`` is
    stored so :func:`read_waveform` reconstructs the physical values. ``sensitivity``
    may be a scalar, a per-channel list, or ``None`` (auto: max-fit per channel).

    ``kind`` selects the IOD: ``ecg12`` / ``ecg`` / ``ecg32`` / ``ambulatory_ecg`` /
    ``hemodynamic`` / ``eps`` / ``eeg`` / ``sleep_eeg`` / ``emg`` / ``eog`` /
    ``arterial_pulse`` / ``respiratory`` / ``audio``. ``labels`` / ``sources`` give the
    per-channel ChannelLabel and source meaning (e.g. ``"Lead I"`` / ``"Fp1"``).
    Returns the SOP Instance UID.
    """
    from . import dcmwrite, generate_uid, sop_class
    from .dataset import Dataset, FileMetaDataset
    from .uid import ExplicitVRLittleEndian

    if isinstance(signals, (list, tuple)) and signals \
            and all(np.ndim(s) == 1 for s in signals):
        # the documented list-of-1-D-channel-arrays form: each element is ONE channel
        sig = np.column_stack([np.asarray(s, dtype="float64") for s in signals])
    else:
        sig = np.asarray(signals, dtype="float64")
    if sig.ndim == 1:
        sig = sig[:, None]
    elif sig.ndim != 2:
        raise ValueError("signals must be 1-D, (n_samples, n_channels), "
                         "or a list of 1-D channel arrays")
    nsamp, nch = sig.shape
    if kind not in _KIND:
        raise ValueError(f"unknown kind {kind!r}; one of {sorted(_KIND)}")
    sop_class_uid, modality = _KIND[kind]
    sop_class_uid = str(getattr(sop_class, sop_class_uid))

    signed = sample_bits != 8                       # 8-bit waveforms are usually unsigned
    intdt = {8: "u1", 16: "<i2", 32: "<i4"}[sample_bits]
    lo, hi = (0, 255) if sample_bits == 8 else (-(1 << (sample_bits - 1)), (1 << (sample_bits - 1)) - 1)
    sens_list = (sensitivity if isinstance(sensitivity, (list, tuple))
                 else [sensitivity] * nch)

    chan_seq, raw_cols = [], []
    for c in range(nch):
        col = sig[:, c]
        s = sens_list[c] if c < len(sens_list) else None
        if not s:                                   # auto: fit the integer range
            peak = float(np.max(np.abs(col))) or 1.0
            s = peak / float(hi if hi else 255)
        raw = np.clip(np.round(col / s), lo, hi).astype(intdt)
        raw_cols.append(raw)
        cd = Dataset()
        cd.ChannelLabel = (labels[c] if labels and c < len(labels) else f"ch{c + 1}")
        src = sources[c] if sources and c < len(sources) else cd.ChannelLabel
        cd.ChannelSourceSequence = [_code(str(src), "99PYDCM", str(src))]
        cd.ChannelSensitivity = f"{s:.10g}"
        cd.ChannelSensitivityCorrectionFactor = "1"
        cd.ChannelBaseline = "0"
        cd.ChannelSensitivityUnitsSequence = [_code(units, "UCUM", units)]
        cd.WaveformBitsStored = sample_bits
        chan_seq.append(cd)

    inter = np.empty((nsamp, nch), dtype=intdt)     # channel-interleaved per sample
    for c in range(nch):
        inter[:, c] = raw_cols[c]

    item = Dataset()
    item.WaveformOriginality = "ORIGINAL"
    item.NumberOfWaveformChannels = nch
    item.NumberOfWaveformSamples = nsamp
    item.SamplingFrequency = f"{float(sampling_frequency):.10g}"
    item.ChannelDefinitionSequence = chan_seq
    item.WaveformBitsAllocated = sample_bits
    item.WaveformSampleInterpretation = {8: "UB", 16: "SS", 32: "SL"}[sample_bits] if signed or sample_bits != 8 else "UB"
    item.WaveformData = inter.tobytes()

    sop_uid = sop_uid or generate_uid()
    ds = Dataset()
    fm = FileMetaDataset()
    fm.MediaStorageSOPClassUID = sop_class_uid
    fm.MediaStorageSOPInstanceUID = sop_uid
    fm.TransferSyntaxUID = ExplicitVRLittleEndian
    ds.file_meta = fm
    ds.PatientName = patient_name
    ds.PatientID = patient_id
    ds.PatientBirthDate = patient_birth_date
    ds.PatientSex = patient_sex
    ds.StudyInstanceUID = study_uid or generate_uid()
    ds.StudyDate = ""
    ds.StudyTime = ""
    ds.ReferringPhysicianName = ""
    ds.StudyID = ""
    ds.AccessionNumber = ""
    ds.Modality = modality
    ds.SeriesInstanceUID = series_uid or generate_uid()
    ds.SeriesNumber = series_number
    ds.InstanceNumber = instance_number
    ds.Manufacturer = "pydcm"
    ds.SOPClassUID = sop_class_uid
    ds.SOPInstanceUID = sop_uid
    ds.WaveformSequence = [item]
    dcmwrite(out, ds)
    return sop_uid


# ---- interop (analysis layer = neurokit2 / MNE; we only hand off arrays) ---
def to_mne(group, ch_types="eeg"):
    """A multiplex group (from :func:`read_waveform`) → an ``mne.io.RawArray``.

    MNE expects volts; channels in µV/mV are scaled accordingly from their ``units``.
    Requires MNE (``pip install mne``). For ECG, prefer neurokit2 on ``group['signals']``.
    """
    import mne
    chans = group["channels"]
    names = [c.get("source") or c.get("label") or f"ch{i}" for i, c in enumerate(chans)]
    scale = {"uv": 1e-6, "µv": 1e-6, "microvolt": 1e-6, "mv": 1e-3, "millivolt": 1e-3, "v": 1.0}
    data = group["signals"].astype("float64").copy()
    for i, c in enumerate(chans):
        data[:, i] *= scale.get(str(c.get("units") or "").lower(), 1.0)
    info = mne.create_info(names, float(group["sampling_frequency"] or 1.0), ch_types=ch_types)
    return mne.io.RawArray(data.T, info, verbose=False)


__all__ = ["multiplex_array", "generate_multiplex", "read_waveform", "write_waveform", "to_mne"]
