# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Fuli Wu
# NOT a medical device — not for clinical or diagnostic use.
"""misc utilities (`pydcm.misc`)."""
from __future__ import annotations

import logging
import warnings
from itertools import groupby

_logger = logging.getLogger("pydcm")


def warn_and_log(msg: str, category=None, stacklevel: int = 1) -> None:
    """Log `msg` and emit it as a warning."""
    _logger.warning(msg)
    warnings.warn(msg, category or UserWarning, stacklevel=stacklevel + 1)

_size_factors = {
    "kb": 1000, "mb": 1000 ** 2, "gb": 1000 ** 3,
    "kib": 1024, "mib": 1024 ** 2, "gib": 1024 ** 3,
}


def size_in_bytes(expr):
    """Return the byte count for a ``defer_size``-style argument.

    Accepts ``None`` / ``inf`` (-> ``None``), a number, or a string like ``"1.5 KB"``.
    """
    if expr is None or expr == float("inf"):
        return None
    if isinstance(expr, (int, float)):
        return expr
    try:
        return int(expr)
    except ValueError:
        pass
    value, unit = ("".join(g) for _k, g in groupby(expr, str.isalpha))
    if unit.lower() in _size_factors:
        return float(value) * _size_factors[unit.lower()]
    raise ValueError(f"Unable to parse length with unit '{unit}'")


def is_dicom(file_path) -> bool:
    """Return ``True`` if the file has a conformant DICOM preamble ('DICM' at offset 128)."""
    with open(file_path, "rb") as fp:
        fp.read(128)        # preamble
        return fp.read(4) == b"DICM"
