# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Fuli Wu
# NOT a medical device — not for clinical or diagnostic use.

"""pydcm — Legacy Converted Enhanced CT/MR/PET authoring (``pydcm.legacy_converted``).

The ``legacy`` capability: fold a set of classic single-frame CT/MR/PET
instances (one series) into ONE enhanced multi-frame object
(LegacyConvertedEnhanced{CT,MR,PET}Image) over the native legacy-conversion
engine. A faithful, reversible re-encapsulation — identity is inherited from the
source series, geometry / rescale / window / frame-type are mapped into the
Shared / Per-Frame Functional Groups, every frame is linked back to its origin,
and leftover source attributes are preserved verbatim.
converter; pydcm-native.
"""
from __future__ import annotations

import os

from . import _core


def _paths(series):
    """Normalize `series` (one path, or an iterable of paths/Datasets) to list[str]."""
    if isinstance(series, (str, bytes, os.PathLike)):
        series = [series]
    out = []
    for s in series:
        if isinstance(s, (str, bytes, os.PathLike)):
            out.append(os.fspath(s))
        elif hasattr(s, "filename") and s.filename:   # a Dataset read from disk
            out.append(os.fspath(s.filename))
        else:
            raise TypeError("legacy conversion needs file paths (or Datasets read from disk), "
                            f"got {type(s).__name__}")
    return out


def write_legacy_converted(series, *, series_instance_uid="", sop_instance_uid="",
                           series_number=0, instance_number=1, manufacturer="",
                           model_name="", device_serial="", software_versions="",
                           output=None):
    """Convert a classic single-frame CT/MR/PET series into one Legacy Converted
    Enhanced multi-frame object.

    series: a list of DICOM file paths (or Datasets read from disk) for the
        classic single-frame instances of ONE series, in any order — frames are
        sorted into geometric slice order. The target SOP Class (CT/MR/PET) is chosen
        from the shared source Modality.
    series_instance_uid / sop_instance_uid: identity for the new object; minted
        deterministically when omitted.
    series_number / instance_number: new series / instance numbers.
    manufacturer / model_name / device_serial / software_versions: Enhanced General
        Equipment (Type 1). Inherited from the source when omitted, else a default.
    output: write the object there and return ``None``; if omitted, return Part-10 bytes.
    """
    opts = {"series_instance_uid": series_instance_uid, "sop_instance_uid": sop_instance_uid,
            "series_number": int(series_number), "instance_number": int(instance_number),
            "manufacturer": manufacturer, "model_name": model_name,
            "device_serial": device_serial, "software_versions": software_versions}
    return _core.write_legacy_converted(_paths(series), opts, str(output) if output else "")


__all__ = ["write_legacy_converted"]
