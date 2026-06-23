# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Fuli Wu
# NOT a medical device — not for clinical or diagnostic use.
"""``from pydcm.dataelem import DataElement, RawDataElement``."""
from __future__ import annotations

from collections import namedtuple

from ._dicom import DataElement, Tag

# raw, undecoded element (as produced by the low-level reader). pydcm decodes
# eagerly in the native bridge, so this is used only when ported code constructs one and
# passes it to DataElement_from_raw / values.convert_value.
RawDataElement = namedtuple(
    "RawDataElement",
    ["tag", "VR", "length", "value", "value_tell", "is_implicit_VR", "is_little_endian"],
)
RawDataElement.__new__.__defaults__ = (None, 0, True, True)   # value_tell, …, defaults


def DataElement_from_raw(raw, encoding=None, ds=None) -> DataElement:
    """Build a decoded :class:`DataElement` from a :class:`RawDataElement`."""
    from .values import convert_value
    vr = raw.VR or "UN"
    value = convert_value(vr, raw.value, encoding)
    return DataElement(Tag(raw.tag), vr, value)


# DataElement_from_raw is the older name for convert_raw_data_element; expose both.
def convert_raw_data_element(raw, *, encoding=None, ds=None) -> DataElement:
    """Decode a :class:`RawDataElement` to a :class:`DataElement`."""
    return DataElement_from_raw(raw, encoding, ds)


def empty_value_for_VR(VR, raw: bool = False):
    """The value an empty element of ``VR`` decodes to.

    Empty 'SQ' -> ``[]`` (``b""`` when ``raw``); empty text VRs -> ``""`` (``b""`` raw),
    'PN' -> :class:`PersonName` (``b""`` raw); everything else -> ``None``. Honors
    ``config.use_none_as_empty_text_VR_value``.
    """
    from . import config
    from ._dicom import PersonName
    from .valuerep import STR_VR
    if VR == "SQ":
        return b"" if raw else []
    if config.use_none_as_empty_text_VR_value:
        return None
    if VR == "PN":
        return b"" if raw else PersonName("")
    if VR in STR_VR - {"DS", "IS"}:
        return b"" if raw else ""
    return None


__all__ = ["DataElement", "RawDataElement", "DataElement_from_raw",
           "convert_raw_data_element", "empty_value_for_VR"]
