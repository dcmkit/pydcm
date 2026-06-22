# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Fuli Wu
# NOT a medical device — not for clinical or diagnostic use.

"""value representations (``pydcm.valuerep``).

Re-exports the value types pydcm models (``PersonName``, ``MultiValue``, ``Sequence``,
``DSfloat``, ``IS``) and provides the ``VR`` enum, the date/time VR str-types, and
``format_number_as_ds`` so ported code (incl. ``isinstance`` checks and
``from pydicom.valuerep import VR``) keeps working.
"""

from __future__ import annotations

import enum
import re

from ._dicom import PersonName, MultiValue, Sequence, DSfloat, IS, ISfloat, _fmt_ds

DSdecimal = DSfloat

# Per-VR validation regexes (per the PS3.5 VR rules).
STR_VR_REGEXES = {
    "AE": re.compile(r"^[\x20-\x7e]*$"),
    "AS": re.compile(r"^\d\d\d[DWMY]$"),
    "CS": re.compile(r"^[A-Z0-9 _]*$"),
    "DS": re.compile(r"^ *[+\-]?(\d+|\d+\.\d*|\.\d+)([eE][+\-]?\d+)? *$"),
    "IS": re.compile(r"^ *[+\-]?\d+ *$"),
    "DA": re.compile(
        r"^\d{4}(0[1-9]|1[0-2])([0-2]\d|3[01])$|"
        r"^\-\d{4}(0[1-9]|1[0-2])([0-2]\d|3[01]) ?$|"
        r"^\d{4}(0[1-9]|1[0-2])([0-2]\d|3[01])\- ?$|"
        r"^\d{4}(0[1-9]|1[0-2])([0-2]\d|3[01])\-\d{4}(0[1-9]|1[0-2])([0-2]\d|3[01]) ?$"),
    "DT": re.compile(
        r"^\d{4}((0[1-9]|1[0-2])(([0-2]\d|3[01])(([01]\d|2[0-3])([0-5]\d((60|[0-5]\d)(\.\d{1,6} ?)?)?)?)?)?)?([+-][01]\d\d\d)?$|"
        r"^\-\d{4}((0[1-9]|1[0-2])(([0-2]\d|3[01])(([01]\d|2[0-3])([0-5]\d((60|[0-5]\d)(\.\d{1,6} ?)?)?)?)?)?)?([+-][01]\d\d\d)? ?$|"
        r"^\d{4}((0[1-9]|1[0-2])(([0-2]\d|3[01])(([01]\d|2[0-3])([0-5]\d((60|[0-5]\d)(\.\d{1,6} ?)?)?)?)?)?)?([+-][01]\d\d\d)?\- ?$|"
        r"^\d{4}((0[1-9]|1[0-2])(([0-2]\d|3[01])(([01]\d|2[0-3])([0-5]\d((60|[0-5]\d)(\.\d{1,6} ?)?)?)?)?)?)?([+-][01]\d\d\d)?\-\d{4}((0[1-9]|1[0-2])(([0-2]\d|3[01])(([01]\d|2[0-3])([0-5]\d((60|[0-5]\d)(\.\d{1,6} ?)?)?)?)?)?)?([+-][01]\d\d\d)? ?$"),
    "TM": re.compile(
        r"^([01]\d|2[0-3])([0-5]\d((60|[0-5]\d)(\.\d{1,6} ?)?)?)?$|"
        r"^\-([01]\d|2[0-3])([0-5]\d((60|[0-5]\d)(\.\d{1,6} ?)?)?)? ?$|"
        r"^([01]\d|2[0-3])([0-5]\d((60|[0-5]\d)(\.\d{1,6} ?)?)?)?\- ?$|"
        r"^([01]\d|2[0-3])([0-5]\d((60|[0-5]\d)(\.\d{1,6} ?)?)?)?\-([01]\d|2[0-3])([0-5]\d((60|[0-5]\d)(\.\d{1,6} ?)?)?)? ?$"),
    "UI": re.compile(r"^(0|[1-9][0-9]*)(\.(0|[1-9][0-9]*))*$"),
    "UR": re.compile(r"^[A-Za-z_\d:/?#\[\]@!$&'()*+,;=%\-.~]* *$"),
}


def validate_value(vr, value, validation_mode=None, validator=None) -> None:
    """Accepted for source compatibility; pydcm validates leniently at read time and
    does not enforce VR constraints on assignment, so this is a no-op (returns None)."""
    return None


def is_valid_ds(s: str) -> bool:
    """Return ``True`` if ``s`` is a valid DS string (<=16 chars, conformant)."""
    return len(s) <= 16 and bool(STR_VR_REGEXES["DS"].match(s))


def DS(val, auto_format: bool = False, validation_mode=None):
    """Factory for DS values: ``None``/blank pass through, else a
    :class:`DSfloat` (or :class:`DSdecimal` when ``config.use_DS_decimal``)."""
    from . import config
    if val is None:
        return val
    if isinstance(val, str) and val.strip() == "":
        return val
    return DSdecimal(val) if config.use_DS_decimal else DSfloat(val)


class VR(str, enum.Enum):
    """The DICOM Value Representations (enum)."""
    AE = "AE"; AS = "AS"; AT = "AT"; CS = "CS"; DA = "DA"; DS = "DS"; DT = "DT"
    FD = "FD"; FL = "FL"; IS = "IS"; LO = "LO"; LT = "LT"; OB = "OB"; OD = "OD"
    OF = "OF"; OL = "OL"; OV = "OV"; OW = "OW"; PN = "PN"; SH = "SH"; SL = "SL"
    SQ = "SQ"; SS = "SS"; ST = "ST"; SV = "SV"; TM = "TM"; UC = "UC"; UI = "UI"
    UL = "UL"; UN = "UN"; UR = "UR"; US = "US"; UT = "UT"; UV = "UV"

    def __str__(self):
        return self.value


# pydcm stores DA/DT/TM as their raw strings; expose the standard class names as thin str
# subtypes so isinstance / `from pydicom.valuerep import DA` keep working.
class DA(str):
    """A DICOM DA (date) value — the raw string."""
    __slots__ = ()


class DT(str):
    """A DICOM DT (datetime) value — the raw string."""
    __slots__ = ()


class TM(str):
    """A DICOM TM (time) value — the raw string."""
    __slots__ = ()


def format_number_as_ds(val) -> str:
    """Format a float/int as a conformant DS string (≤16 chars)."""
    return _fmt_ds(val)


STR_VR = {"AE", "AS", "CS", "DA", "DS", "DT", "IS", "LO", "LT", "PN", "SH",
          "ST", "TM", "UC", "UI", "UR", "UT"}
BYTES_VR = {"OB", "OD", "OF", "OL", "OV", "OW", "UN"}
INT_VR = {"SL", "SS", "SV", "UL", "US", "UV"}
FLOAT_VR = {"FD", "FL"}

__all__ = ["PersonName", "MultiValue", "Sequence", "DSfloat", "DSdecimal", "IS",
           "ISfloat", "VR", "DA", "DT", "TM", "format_number_as_ds", "validate_value",
           "STR_VR", "BYTES_VR", "INT_VR", "FLOAT_VR", "STR_VR_REGEXES",
           "DS", "is_valid_ds"]
