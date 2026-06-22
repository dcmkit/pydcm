# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Fuli Wu
# NOT a medical device — not for clinical or diagnostic use.
"""example datasets (``pydcm.examples``).

Lazily loads the named example via :mod:`pydcm.data` and reads it with pydcm's own
reader, so ``pydcm.examples.ct`` is a pydcm ``FileDataset``."""
from __future__ import annotations

_FILES = {"ct": "CT_small.dcm", "mr": "MR_small.dcm", "rgb_color": "US1_J2KR.dcm",
          "ybr_color": "US1_J2KR.dcm", "overlay": "MR-SIEMENS-DICOM-WithOverlays.dcm",
          "waveform": "waveform_ecg.dcm", "rt_plan": "rtplan.dcm",
          "rt_dose": "rtdose.dcm", "rt_ss": "rtstruct.dcm", "palette_color": "OBXXXX1A.dcm"}


def __getattr__(name):
    if name in _FILES:
        from . import data, _dicom
        return _dicom.dcmread(data.get_testdata_file(_FILES[name]))
    raise AttributeError(f"module 'pydcm.examples' has no example {name!r}")


def __dir__():
    return sorted(_FILES)
