# Waveforms (ECG / EEG)

`pydcm.waveforms` reads and writes DICOM Waveform objects — 12-lead ECG, EEG,
hemodynamic and audio — and hands the signal arrays to an analysis library. The
`multiplex_array` decode is exact (difference = 0).

## Read

```python
from pydcm import waveforms

wf = waveforms.read_waveform("ecg.dcm")
wf["modality"]                          # "ECG"
group = wf["groups"][0]
group["sampling_frequency"]             # Hz
group["signals"]                        # ndarray [samples, channels], physical units
for j, ch in enumerate(group["channels"]):
    ch["label"], ch["units"], group["signals"][:, j]   # e.g. "Lead I", "mV", that lead's samples
```

For the raw multiplex matrix:

```python
import pydcm

ds = pydcm.dcmread("ecg.dcm")
arr = waveforms.multiplex_array(ds, index=0, as_raw=False)   # [samples, channels], physical units
```

## Write

```python
signals = [lead_i, lead_ii, ...]   # a list of 1-D channel arrays
                                   # (or one array shaped [samples, channels])
waveforms.write_waveform(
    "ecg_out.dcm",
    signals,
    sampling_frequency=500,
    kind="ecg12",            # 12-lead ECG; also "eeg", "hemodynamic", "audio", …
    units="mV",
    patient_id="PID-1",
)
```

Each channel is quantised via its `sensitivity` (physical units per LSB —
scalar, per-channel, or auto), and `ChannelSensitivity` is stored so
`read_waveform` reconstructs the physical values.

## Hand off to analysis

pydcm produces the signal arrays; analysis stays in the established tools:

```python
import neurokit2 as nk
signals_df, info = nk.ecg_process(group["signals"][:, 0],   # first lead's samples
                                  sampling_rate=group["sampling_frequency"])
```

For EEG, `mne` consumes the same arrays. pydcm does not re-implement signal
analysis — it gives those libraries clean, physical-unit input.
