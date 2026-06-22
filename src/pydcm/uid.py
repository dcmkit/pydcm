# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Fuli Wu
# NOT a medical device — not for clinical or diagnostic use.

"""UID helpers (``pydcm.uid``).

The :class:`UID` ``str`` subclass (with the full
name/type/keyword/info metadata sourced from the native UID registry), every
well-known SOP Class / Transfer Syntax constant, the transfer-syntax category
lists, the validation regexes, and ``generate_uid`` / ``register_transfer_syntax``.

``generate_uid`` reuses the native canonical UID generator, so pydcm doesn't
re-roll UID logic.
"""

from __future__ import annotations

import hashlib  # noqa: F401  (re-exported for source compatibility)
import re
import secrets  # noqa: F401
import uuid as _uuid

from . import _native, config
from .config import disable_value_validation         # noqa: F401  (re-export for source compatibility)
from .valuerep import STR_VR_REGEXES, validate_value  # noqa: F401  (re-export for source compatibility)


# UID metadata is projected from the native UID dictionary
# (the PS3.6 set) — pydcm ships
# NO Python copy. The SR-terminology rows (Context Group / SR Template) belong to
# pydcm.ctxgroups, not the core UID registry, so they're filtered out here.
def _load_registry():
    reg = {}
    for uid, name, typ, keyword, info, retired, _cid in _native.uid_table():
        if typ in ("Context Group", "SR Template"):
            continue
        reg[uid] = (name, typ, info, retired, keyword)
    return reg


UID_REGISTRY = _load_registry()                      # {uid: (name, type, info, retired, keyword)}
# Compatible alias — same {uid: (name, type, info, retired, keyword)} layout.
UID_dictionary = UID_REGISTRY

PYDICOM_ROOT_UID = "1.2.826.0.1.3680043.8.498."
PYDICOM_IMPLEMENTATION_UID = "1.2.826.0.1.3680043.8.498.1"

# A valid UID is a dotted string of components, each "0" or no leading zeros.
RE_VALID_UID = re.compile(r"^(0|[1-9][0-9]*)(\.(0|[1-9][0-9]*))*$")
RE_VALID_UID_PREFIX = re.compile(r"^(0|[1-9][0-9]*)(\.(0|[1-9][0-9]*))*\.$")


class UID(str):
    """A DICOM Unique Identifier (a ``str`` subclass)."""

    __slots__ = ("_PRIVATE_TS_ENCODING",)

    def __new__(cls, val, validation_mode=None):
        if isinstance(val, str):
            if validation_mode is None:
                validation_mode = config.settings.reading_validation_mode
            validate_value("UI", val, validation_mode)
            uid = super().__new__(cls, val.strip())
            if hasattr(val, "_PRIVATE_TS_ENCODING"):
                uid._PRIVATE_TS_ENCODING = val._PRIVATE_TS_ENCODING
            return uid
        raise TypeError("A UID must be created from a string")

    @property
    def is_valid(self) -> bool:
        """Return ``True`` if `self` is a valid UID, ``False`` otherwise."""
        return len(self) <= 64 and bool(RE_VALID_UID.match(self))

    @property
    def name(self) -> str:
        """Return the UID name from the UID dictionary."""
        entry = UID_dictionary.get(str(self))
        return entry[0] if entry else str(self)

    @property
    def type(self) -> str:
        """Return the UID type from the UID dictionary."""
        entry = UID_dictionary.get(str(self))
        return entry[1] if entry else ""

    @property
    def info(self) -> str:
        """Return the UID info from the UID dictionary."""
        entry = UID_dictionary.get(str(self))
        return entry[2] if entry else ""

    @property
    def is_retired(self) -> bool:
        """Return ``True`` if the UID is retired, ``False`` otherwise."""
        entry = UID_dictionary.get(str(self))
        return bool(entry[3]) if entry else False

    @property
    def keyword(self) -> str:
        """Return the UID keyword from the UID dictionary."""
        entry = UID_dictionary.get(str(self))
        return entry[4] if entry else ""

    @property
    def is_private(self) -> bool:
        """Return ``True`` if the UID isn't an officially registered DICOM UID."""
        return self[:14] != "1.2.840.10008."

    @property
    def is_transfer_syntax(self) -> bool:
        """Return ``True`` if a transfer syntax UID."""
        if not self.is_private:
            return self.type == "Transfer Syntax"
        return hasattr(self, "_PRIVATE_TS_ENCODING")

    @property
    def is_deflated(self) -> bool:
        """Return ``True`` if a deflated transfer syntax UID."""
        if self.is_transfer_syntax:
            return self == "1.2.840.10008.1.2.1.99"
        raise ValueError("UID is not a transfer syntax.")

    @property
    def is_implicit_VR(self) -> bool:
        """Return ``True`` if an implicit VR transfer syntax UID."""
        if self.is_transfer_syntax:
            if not self.is_private:
                return self == "1.2.840.10008.1.2"
            return self._PRIVATE_TS_ENCODING[0]
        raise ValueError("UID is not a transfer syntax.")

    @property
    def is_little_endian(self) -> bool:
        """Return ``True`` if a little endian transfer syntax UID."""
        if self.is_transfer_syntax:
            if not self.is_private:
                return self != "1.2.840.10008.1.2.2"
            return self._PRIVATE_TS_ENCODING[1]
        raise ValueError("UID is not a transfer syntax.")

    @property
    def is_compressed(self) -> bool:
        """Return ``True`` if a compressed (encapsulated) transfer syntax UID."""
        if self.is_transfer_syntax:
            return self not in (
                "1.2.840.10008.1.2",
                "1.2.840.10008.1.2.1",
                "1.2.840.10008.1.2.2",
                "1.2.840.10008.1.2.1.99",
            )
        raise ValueError("UID is not a transfer syntax.")

    @property
    def is_encapsulated(self) -> bool:
        """Return ``True`` if an encapsulated transfer syntax UID."""
        return self.is_compressed

    def set_private_encoding(self, implicit_vr: bool, little_endian: bool) -> None:
        """Set the corresponding dataset encoding for a privately defined transfer syntax."""
        self._PRIVATE_TS_ENCODING = (implicit_vr, little_endian)


def generate_uid(prefix: str | None = PYDICOM_ROOT_UID, entropy_srcs=None) -> UID:
    """Return a unique :class:`UID`.

    With ``entropy_srcs`` the result is deterministic for that input; otherwise it is
    random. ``prefix=None`` yields a ``2.25.`` UUID-derived UID. Reuses the native
    canonical generator (``_native.mint_uid``) — no separate UID arithmetic in Python.
    """
    root = (prefix or "2.25.").rstrip(".")
    seed = "".join(map(str, entropy_srcs)) if entropy_srcs else _uuid.uuid4().hex
    return UID(_native.mint_uid(seed, root))


def register_transfer_syntax(uid, implicit_vr=None, little_endian=None) -> UID:
    """Register a private transfer syntax so dcmread can use it."""
    uid = UID(uid)
    if None in (implicit_vr, little_endian) and not uid.is_transfer_syntax:
        raise ValueError(
            "The corresponding dataset encoding for 'uid' must be set using "
            "the 'implicit_vr' and 'little_endian' arguments")
    if implicit_vr is not None and little_endian is not None:
        uid.set_private_encoding(implicit_vr, little_endian)
    if uid not in PrivateTransferSyntaxes:
        PrivateTransferSyntaxes.append(uid)
    return uid


# ── Well-known Transfer Syntax UIDs (names) ──────────────────────
ImplicitVRLittleEndian          = UID("1.2.840.10008.1.2")
ExplicitVRLittleEndian          = UID("1.2.840.10008.1.2.1")
DeflatedExplicitVRLittleEndian  = UID("1.2.840.10008.1.2.1.99")
ExplicitVRBigEndian             = UID("1.2.840.10008.1.2.2")
JPEGBaseline8Bit                = UID("1.2.840.10008.1.2.4.50")
JPEGExtended12Bit               = UID("1.2.840.10008.1.2.4.51")
JPEGLossless                    = UID("1.2.840.10008.1.2.4.57")
JPEGLosslessSV1                 = UID("1.2.840.10008.1.2.4.70")
JPEGLSLossless                  = UID("1.2.840.10008.1.2.4.80")
JPEGLSNearLossless              = UID("1.2.840.10008.1.2.4.81")
JPEG2000Lossless                = UID("1.2.840.10008.1.2.4.90")
JPEG2000                        = UID("1.2.840.10008.1.2.4.91")
HTJ2KLossless                   = UID("1.2.840.10008.1.2.4.201")
HTJ2KLosslessRPCL               = UID("1.2.840.10008.1.2.4.202")
HTJ2K                           = UID("1.2.840.10008.1.2.4.203")
RLELossless                     = UID("1.2.840.10008.1.2.5")

# ── Transfer-syntax category lists (contents) ────────────────────
JPEGTransferSyntaxes = [UID(x) for x in (
    "1.2.840.10008.1.2.4.50", "1.2.840.10008.1.2.4.51",
    "1.2.840.10008.1.2.4.57", "1.2.840.10008.1.2.4.70")]
JPEGLSTransferSyntaxes = [UID(x) for x in (
    "1.2.840.10008.1.2.4.80", "1.2.840.10008.1.2.4.81")]
JPEG2000TransferSyntaxes = [UID(x) for x in (
    "1.2.840.10008.1.2.4.90", "1.2.840.10008.1.2.4.91", "1.2.840.10008.1.2.4.92",
    "1.2.840.10008.1.2.4.93", "1.2.840.10008.1.2.4.201", "1.2.840.10008.1.2.4.202",
    "1.2.840.10008.1.2.4.203")]
MPEGTransferSyntaxes = [UID(x) for x in (
    "1.2.840.10008.1.2.4.100", "1.2.840.10008.1.2.4.100.1", "1.2.840.10008.1.2.4.101",
    "1.2.840.10008.1.2.4.101.1", "1.2.840.10008.1.2.4.102", "1.2.840.10008.1.2.4.102.1",
    "1.2.840.10008.1.2.4.103", "1.2.840.10008.1.2.4.103.1", "1.2.840.10008.1.2.4.104",
    "1.2.840.10008.1.2.4.104.1", "1.2.840.10008.1.2.4.105", "1.2.840.10008.1.2.4.105.1",
    "1.2.840.10008.1.2.4.106", "1.2.840.10008.1.2.4.106.1", "1.2.840.10008.1.2.4.107",
    "1.2.840.10008.1.2.4.108")]
RLETransferSyntaxes = [UID("1.2.840.10008.1.2.5")]
# pydcm extra (not in pydicom 3.0.2): JPEG XL — natively decoded via libjxl.
JPEGXLTransferSyntaxes = [UID(x) for x in (
    "1.2.840.10008.1.2.4.110", "1.2.840.10008.1.2.4.111", "1.2.840.10008.1.2.4.112")]
UncompressedTransferSyntaxes = [UID(x) for x in (
    "1.2.840.10008.1.2.1", "1.2.840.10008.1.2",
    "1.2.840.10008.1.2.1.99", "1.2.840.10008.1.2.2")]
PrivateTransferSyntaxes: list = []
AllTransferSyntaxes = [UID(x) for x in (
    "1.2.840.10008.1.2", "1.2.840.10008.1.2.1", "1.2.840.10008.1.2.1.99",
    "1.2.840.10008.1.2.2", "1.2.840.10008.1.2.4.50", "1.2.840.10008.1.2.4.51",
    "1.2.840.10008.1.2.4.57", "1.2.840.10008.1.2.4.70", "1.2.840.10008.1.2.4.80",
    "1.2.840.10008.1.2.4.81", "1.2.840.10008.1.2.4.90", "1.2.840.10008.1.2.4.91",
    "1.2.840.10008.1.2.4.92", "1.2.840.10008.1.2.4.93", "1.2.840.10008.1.2.4.100",
    "1.2.840.10008.1.2.4.100.1", "1.2.840.10008.1.2.4.101", "1.2.840.10008.1.2.4.101.1",
    "1.2.840.10008.1.2.4.102", "1.2.840.10008.1.2.4.102.1", "1.2.840.10008.1.2.4.103",
    "1.2.840.10008.1.2.4.103.1", "1.2.840.10008.1.2.4.104", "1.2.840.10008.1.2.4.104.1",
    "1.2.840.10008.1.2.4.105", "1.2.840.10008.1.2.4.105.1", "1.2.840.10008.1.2.4.106",
    "1.2.840.10008.1.2.4.106.1", "1.2.840.10008.1.2.4.107", "1.2.840.10008.1.2.4.108",
    "1.2.840.10008.1.2.4.201", "1.2.840.10008.1.2.4.202", "1.2.840.10008.1.2.4.203",
    "1.2.840.10008.1.2.4.204", "1.2.840.10008.1.2.4.205", "1.2.840.10008.1.2.5",
    "1.2.840.10008.1.2.7.1", "1.2.840.10008.1.2.7.2", "1.2.840.10008.1.2.7.3",
    # pydcm extras (beyond the standard set): JPEG XL.
    "1.2.840.10008.1.2.4.110", "1.2.840.10008.1.2.4.111", "1.2.840.10008.1.2.4.112")]

# ── Module-level SOP Class / well-known constants — projected from the native UID
# dictionary (the native PS3.6 union). standard keyword spellings.
# e.g. CTImageStorage = UID("1.2.840.10008.5.1.4.1.1.2").
globals().update({kw: UID(u) for u, (_n, _t, _i, _r, kw) in UID_REGISTRY.items() if kw})
