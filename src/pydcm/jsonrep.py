# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Fuli Wu
# NOT a medical device — not for clinical or diagnostic use.
"""DICOM JSON Model helpers (``pydcm.jsonrep``).

pydcm's JSON Model conversion lives in the native bridge + ``Dataset.to_json``/
``from_json``; this exposes the small JSON-rep building blocks.
"""
from __future__ import annotations

# PS3.18 §F bulk-data / inline-binary keys.
BINARY_VR_VALUES = ["OB", "OD", "OF", "OL", "OV", "OW", "UN"]
VRs_TO_BE_FLOATS = ["DS", "FL", "FD"]
VRs_TO_BE_INTS = ["IS", "SL", "SS", "SV", "UL", "US", "UV"]


def convert_to_python_number(value, vr):
    """Coerce a JSON value to the Python number type for ``vr``."""
    if value is None:
        return None
    if vr in VRs_TO_BE_INTS:
        return [int(v) for v in value] if isinstance(value, list) else int(value)
    if vr in VRs_TO_BE_FLOATS:
        return [float(v) for v in value] if isinstance(value, list) else float(value)
    return value


__all__ = ["BINARY_VR_VALUES", "VRs_TO_BE_FLOATS", "VRs_TO_BE_INTS",
           "convert_to_python_number"]
