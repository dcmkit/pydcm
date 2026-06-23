# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Fuli Wu
# NOT a medical device — not for clinical or diagnostic use.

"""value converters (``pydcm.values``).

``convert_value(VR, raw, encodings)`` turns a raw element value (bytes, a string, or a
:class:`~pydcm.dataelem.RawDataElement`) into the Python value pydcm uses, reusing the
same ``_scalar`` routing as the reader — no separate VR logic.
"""

from __future__ import annotations

import struct

from ._dicom import _scalar, Tag
from .charset import decode_bytes

_INT_FMT = {"US": "<H", "SS": "<h", "UL": "<I", "SL": "<i",
            "UV": "<Q", "SV": "<q", "FL": "<f", "FD": "<d"}
_INT_SZ = {"US": 2, "SS": 2, "UL": 4, "SL": 4, "UV": 8, "SV": 8, "FL": 4, "FD": 8}
_TEXT_VR = {"AE", "AS", "CS", "DA", "DS", "DT", "IS", "LO", "LT", "PN",
            "SH", "ST", "TM", "UC", "UI", "UR", "UT"}


def _decode_raw(vr: str, raw: bytes, encodings):
    if vr in _INT_FMT:
        n = _INT_SZ[vr]
        vals = [struct.unpack(_INT_FMT[vr], raw[i:i + n])[0]
                for i in range(0, len(raw) - n + 1, n)]
        return vals[0] if len(vals) == 1 else vals
    if vr == "AT":
        vals = [Tag((struct.unpack("<H", raw[i:i + 2])[0] << 16)
                    | struct.unpack("<H", raw[i + 2:i + 4])[0])
                for i in range(0, len(raw) - 3, 4)]
        return vals[0] if len(vals) == 1 else vals
    if vr in ("OB", "OW", "OD", "OF", "OL", "OV", "UN"):
        return raw
    text = decode_bytes(raw, encodings).rstrip("\x00 ")
    return text


def convert_value(VR, raw, encodings=None):
    """Convert a raw element value to its Python type."""
    val = raw.value if hasattr(raw, "value") else (
        raw[2] if isinstance(raw, tuple) and len(raw) >= 3 else raw)
    if isinstance(val, (bytes, bytearray)):
        val = _decode_raw(VR, bytes(val), encodings)
        if isinstance(val, (bytes, bytearray)):
            return val
    return _scalar(VR, val)


# exposes a VR→callable dict; mirror it as thin wrappers over convert_value.
converters = {vr: (lambda v, enc=None, *, _vr=vr: convert_value(_vr, v, enc))
              for vr in (set(_INT_FMT) | _TEXT_VR | {"AT", "OB", "OW", "OD", "OF",
                                                     "OL", "OV", "UN", "SQ"})}

__all__ = ["convert_value", "converters"]
