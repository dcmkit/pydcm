# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Fuli Wu
# NOT a medical device — not for clinical or diagnostic use.

"""charset helpers (``pydcm.charset``).

pydcm's native reader already transcodes every text VR to UTF-8 in C++ (the bridge),
so on the read path these functions are not needed. They exist for ported code
that decodes/encodes text against a DICOM SpecificCharacterSet directly (authoring,
round-trip). ``convert_encodings`` maps DICOM defined terms to Python codec names; the
actual transcode reuses Python's standard codecs.
"""

from __future__ import annotations

# DICOM (0008,0005) SpecificCharacterSet defined term → Python codec name (PS3.3 C.12.1.1.2).
python_encoding = {
    "": "iso8859",
    "ISO_IR 6": "iso8859", "ISO 2022 IR 6": "iso8859",
    "ISO_IR 100": "latin_1", "ISO 2022 IR 100": "latin_1",
    "ISO_IR 101": "iso8859_2", "ISO 2022 IR 101": "iso8859_2",
    "ISO_IR 109": "iso8859_3", "ISO 2022 IR 109": "iso8859_3",
    "ISO_IR 110": "iso8859_4", "ISO 2022 IR 110": "iso8859_4",
    "ISO_IR 144": "iso8859_5", "ISO 2022 IR 144": "iso8859_5",
    "ISO_IR 127": "iso8859_6", "ISO 2022 IR 127": "iso8859_6",
    "ISO_IR 126": "iso8859_7", "ISO 2022 IR 126": "iso8859_7",
    "ISO_IR 138": "iso8859_8", "ISO 2022 IR 138": "iso8859_8",
    "ISO_IR 148": "iso8859_9", "ISO 2022 IR 148": "iso8859_9",
    "ISO_IR 166": "iso8859_11", "ISO 2022 IR 166": "iso8859_11",
    "ISO_IR 13": "shift_jis", "ISO 2022 IR 13": "shift_jis",
    "ISO 2022 IR 87": "iso2022_jp",
    "ISO 2022 IR 159": "iso2022_jp_2",
    "ISO 2022 IR 149": "euc_kr",
    "ISO 2022 IR 58": "iso2022_jp_2",
    "ISO_IR 192": "UTF8",
    "GB18030": "GB18030",
    "GBK": "GBK",
}

# Encodings that may appear alone (no code-extension escapes).
STAND_ALONE_ENCODINGS = ("ISO_IR 192", "GB18030", "GBK")
default_encoding = "iso8859"


def convert_encodings(encodings) -> list:
    """Map a SpecificCharacterSet (str or list of DICOM defined terms) to Python codec
    names."""
    if encodings is None:
        return [default_encoding]
    if isinstance(encodings, str):
        encodings = [encodings]
    out = [python_encoding.get(e.strip() if isinstance(e, str) else e, default_encoding)
           for e in encodings]
    return out or [default_encoding]


def _python_encodings(encodings) -> list:
    """Internal: a non-empty list of Python codec names for ``encodings``."""
    encs = convert_encodings(encodings)
    return encs or [default_encoding]


def decode_bytes(value: bytes, encodings, delimiters=frozenset()) -> str:
    """Decode raw DICOM text bytes to ``str`` using ``encodings``."""
    if isinstance(value, str):
        return value
    for enc in _python_encodings(encodings):
        try:
            return bytes(value).decode(enc)
        except (LookupError, UnicodeDecodeError):
            continue
    return bytes(value).decode("latin_1")


def decode_string(value, encodings, delimiters=frozenset()) -> str:
    """Alias of :func:`decode_bytes` for source compatibility."""
    return decode_bytes(value, encodings, delimiters)


def encode_string(value: str, encodings) -> bytes:
    """Encode ``str`` to DICOM text bytes using the first codec that can represent it."""
    for enc in _python_encodings(encodings):
        try:
            return value.encode(enc)
        except (LookupError, UnicodeEncodeError):
            continue
    return value.encode("utf-8")


def decode_element(elem, dicom_character_set) -> None:
    """No-op (pydcm decodes every text element to UTF-8 at read time); accepted for
    signature compatibility."""
    return None


# aliases
decode = decode_element

__all__ = ["python_encoding", "convert_encodings", "decode_bytes", "decode_string",
           "encode_string", "decode_element", "decode", "STAND_ALONE_ENCODINGS",
           "default_encoding"]
