# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Fuli Wu
# NOT a medical device — not for clinical or diagnostic use.

"""DICOM SR terminology UIDs — Context Group + Template well-known SOP instances.

These ``1.2.840.10008.6.1.x`` (Context Group) and ``1.2.840.10008.9.x`` (SR Template)
UIDs are part of DICOM PS3.6, but they identify PS3.16/PS3.20 controlled-terminology
artifacts rather than the SOP-class / transfer-syntax / well-known UIDs you meet in
normal file I/O — so (as DICOM toolkits conventionally do) :mod:`pydcm.uid` keeps
them out of its core registry. pydcm exposes the full set here instead::

    from pydcm.ctxgroups import context_group_for_uid, uid_for_cid
    context_group_for_uid("1.2.840.10008.6.1.2")   # -> (4, "AnatomicRegion", False)
    uid_for_cid(4)                                  # -> "1.2.840.10008.6.1.2"

The data is the complete PS3.6 Table A-3/A-4, with CID names joined from the source
controlled-terminology dictionary.
"""

from __future__ import annotations

from . import _native


# Projected from the native UID dictionary (the SR-terminology rows of the
# PS3.6 set) — same single source as pydcm.uid, no Python copy.
def _load():
    cg, tpl = {}, {}
    for uid, name, _typ, _kw, info, retired, cid in _native.uid_table():
        if _typ == "Context Group":
            cg[uid] = (cid, name, retired)
        elif _typ == "SR Template":
            tpl[uid] = (info, name)     # (PS3.20 ref, kind)
    return cg, tpl


CONTEXT_GROUP_UIDS, TEMPLATE_UIDS = _load()

# {cid_number: uid} reverse index for the context groups that carry a CID.
_UID_BY_CID = {cid: uid for uid, (cid, _name, _ret) in CONTEXT_GROUP_UIDS.items() if cid}


def context_group_for_uid(uid: str):
    """Return ``(cid_number, cid_name, retired)`` for a Context Group UID, else ``None``."""
    return CONTEXT_GROUP_UIDS.get(str(uid))


def uid_for_cid(cid: int) -> str | None:
    """Return the well-known Context Group UID for a CID number, else ``None``."""
    return _UID_BY_CID.get(int(cid))


def template_for_uid(uid: str):
    """Return ``(ps3.20_ref, kind)`` for an SR Template UID, else ``None``."""
    return TEMPLATE_UIDS.get(str(uid))


def is_context_group_uid(uid: str) -> bool:
    return str(uid) in CONTEXT_GROUP_UIDS


def is_template_uid(uid: str) -> bool:
    return str(uid) in TEMPLATE_UIDS


__all__ = [
    "CONTEXT_GROUP_UIDS", "TEMPLATE_UIDS",
    "context_group_for_uid", "uid_for_cid", "template_for_uid",
    "is_context_group_uid", "is_template_uid",
]
