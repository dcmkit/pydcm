# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Fuli Wu
# NOT a medical device — not for clinical or diagnostic use.

"""pydcm — DICOM Parametric Map authoring + reading (``pydcm.paramap``).

Parametric Map conversion (image ↔ paramap), over the native engine:

* :func:`write_paramap` — author a float Parametric Map (SOP 1.2.840.10008.5.1.4.1.1.30)
  from a real-valued array + the source series' geometry + a Real World Value Mapping
  (units / quantity / slope / intercept), via the native parametric-map engine
  (the float-pixel counterpart of the SEG writer). a native capability
  authoring; this is pydcm-native value-add.
* :func:`read_paramap` — read a Parametric Map back to a real-valued ``float32`` array
  + metadata (the Real World Value Mapping). Float / double-float pixel data decode
  directly; integer-stored maps have their RWVM slope/intercept applied. Reads
  third-party maps, not only pydcm's own output.
"""
from __future__ import annotations

import numpy as np

from . import _core


def _refs(reference):
    if isinstance(reference, (str, bytes)) or hasattr(reference, "__fspath__"):
        return [str(reference)]
    return [str(p) for p in reference]


def _code3(x, *, default_scheme="DCM"):
    """Normalize a coded concept to (value, scheme, meaning)."""
    if x is None:
        return None
    if isinstance(x, dict):
        return (str(x.get("value", x.get("code", ""))),
                str(x.get("scheme", default_scheme)), str(x.get("meaning", "")))
    if isinstance(x, str):
        return ("", default_scheme, x)            # meaning only
    t = list(x)
    if len(t) == 2:
        return (str(t[0]), default_scheme, str(t[1]))   # (value, meaning)
    return (str(t[0]), str(t[1]), str(t[2]))            # (value, scheme, meaning)


def write_paramap(reference, values, *, units=None, quantity=None,
                  slope=None, intercept=None, label=None, explanation=None,
                  dtype=None, output=None):
    """Author a DICOM Parametric Map from a real-valued array.

    reference: a source-image path, or the list of source-series instance paths —
        geometry, demographics and Frame of Reference are taken from it (one slice
        per array plane, ordered by position).
    values: a float array ``(H, W)`` or ``(slices, H, W)`` of real-world values
        (one plane per reference slice).
    units: the measurement units — ``(code, scheme, meaning)`` (UCUM by default),
        ``(code, meaning)``, a plain meaning string, or a dict. E.g.
        ``("um2/s", "UCUM", "um2/s")``.
    quantity: the measured quantity code ``(value, scheme, meaning)`` (DCM by
        default), e.g. ``("113041", "DCM", "Apparent Diffusion Coefficient")``.
    dtype: the stored pixel type. ``None`` (default) → 32-bit float
        (FloatingPointImagePixel; the values are stored verbatim). ``"uint16"`` /
        ``"int16"`` / ``"uint8"`` / ``"int8"`` → integer pixels quantized through the
        Real World Value Mapping (``stored = round((value - intercept) / slope)``), so
        a reader recovers ``value = stored * slope + intercept``.
    slope / intercept: Real World Value Mapping slope / intercept. For float storage
        the default is identity (1 / 0 — values are already real-world). For an integer
        ``dtype`` left unset, they are auto-computed to span the value range across the
        integer range (lossy only by the quantization step); pass them to control the
        scaling explicitly.
    output: write the map there and return ``None``; if omitted, return Part-10 bytes.
    """
    arr = np.ascontiguousarray(values, dtype=np.float32)

    store_bits, store_signed = 0, 0
    if dtype is not None:
        dt = np.dtype(dtype)
        if dt.kind not in ("i", "u") or dt.itemsize not in (1, 2):
            raise ValueError("dtype must be int8/uint8/int16/uint16, or None for float")
        store_bits = dt.itemsize * 8
        store_signed = 1 if dt.kind == "i" else 0
        if slope is None and intercept is None:        # auto-scale into the integer range
            vmin = float(arr.min()) if arr.size else 0.0
            vmax = float(arr.max()) if arr.size else 0.0
            max_pos = (1 << (store_bits - 1)) - 1 if store_signed else (1 << store_bits) - 1
            rng = vmax - vmin
            if rng <= 0 or max_pos <= 0:
                slope, intercept = 1.0, vmin   # constant image → store 0, recover vmin exactly
            else:
                intercept, slope = vmin, rng / max_pos

    rwvm = {"slope": 1.0 if slope is None else float(slope),
            "intercept": 0.0 if intercept is None else float(intercept)}
    u = _code3(units, default_scheme="UCUM")
    if u:
        rwvm["units_code"], rwvm["units_scheme"], rwvm["units_meaning"] = u
    q = _code3(quantity, default_scheme="DCM")
    if q:
        rwvm["quantity_value"], rwvm["quantity_scheme"], rwvm["quantity_meaning"] = q
    if label:
        rwvm["label"] = str(label)
    if explanation:
        rwvm["explanation"] = str(explanation)
    return _core.write_paramap(_refs(reference), arr, rwvm, store_bits, store_signed,
                               str(output) if output else "")


def read_paramap(path):
    """Read a DICOM Parametric Map to ``(values, meta)``.

    values: a ``float32`` array ``(frames, rows, cols)`` of real-world values. Float /
        double-float pixel data is returned directly; an integer-stored map has its
        Real World Value Mapping (slope/intercept) applied.
    meta: the geometry sidecar (as :func:`pydcm.decode`) plus ``is_parametric_map`` and,
        when present, ``real_world_value_mapping`` = ``{slope, intercept, units, label,
        first_value_mapped, last_value_mapped, has_lut}``.
    """
    from . import decode

    info = _core.paramap_meta(str(path))
    rwvm = info.get("rwvm")
    if rwvm and not info["is_float"]:
        arr, meta = decode(str(path), rescale=False, with_meta=True)
        arr = arr.astype(np.float32) * rwvm["slope"] + rwvm["intercept"]
    else:
        arr, meta = decode(str(path), rescale=True, with_meta=True)
    meta["is_parametric_map"] = info["is_parametric_map"]
    if rwvm is not None:
        meta["real_world_value_mapping"] = rwvm
    return arr, meta


__all__ = ["write_paramap", "read_paramap"]
