# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Fuli Wu
# NOT a medical device — not for clinical or diagnostic use.

"""pydcm — conformant DICOM de-identification (``pydcm.deidentify``).

The PS3.15 Annex E Basic Application Confidentiality Profile over a recursive,
byte-level walk: PHI nested in sequences is caught, untouched elements survive
verbatim, UID remapping is consistent + collision-free, and the output carries
the mandatory De-identification Method attributes (0012,0062/0063/0064). This is
a thin binding over the native de-identification engine — the logic lives in
C++, not here.

Two entry points:

* :func:`deidentify` — one instance.
* :func:`deidentify_series` — a list of instances through ONE session, so the
  UID remap stays consistent across a study (intra-study cross-references in the
  de-identified copy still resolve). Use this for a whole study/series.
"""
from __future__ import annotations

import os

from . import _core
from .tag import Tag


def _read(data):
    """bytes → bytes; path-like → file bytes."""
    if isinstance(data, (bytes, bytearray)):
        return bytes(data)
    if isinstance(data, (str, os.PathLike)):
        with open(data, "rb") as f:
            return f.read()
    raise TypeError("data must be bytes or a path to a Part-10 file")


def _options(profile, patient_name, patient_id, uid_root, retain_dates,
             retain_safe_private, clean_descriptors, retain_uids,
             retain_device_id, retain_institution_id, retain_patient_chars,
             clean_graphics, clean_struct_content, clean_pixel, shift_dates_days,
             replace, remove):
    if profile not in ("basic", "none"):
        raise ValueError("profile must be 'basic' (PS3.15 Annex E) or 'none' (keep PHI, remap UIDs)")
    opt = {
        "profile": profile,
        "patient_name": patient_name,
        "patient_id": patient_id,
        "uid_root": uid_root,
        "retain_dates": retain_dates,
        "retain_safe_private": retain_safe_private,
        "clean_descriptors": clean_descriptors,
        "retain_uids": retain_uids,
        "retain_device_id": retain_device_id,
        "retain_institution_id": retain_institution_id,
        "retain_patient_chars": retain_patient_chars,
        "clean_graphics": clean_graphics,
        "clean_struct_content": clean_struct_content,
        "clean_pixel": clean_pixel,
        "shift_dates_days": int(shift_dates_days),
    }
    # Tag keys accept int / Tag / "GGGG,EEEE" / keyword — normalize to packed int.
    if replace:
        opt["replace"] = {int(Tag(k)): str(v) for k, v in dict(replace).items()}
    if remove:
        opt["remove"] = [int(Tag(t)) for t in remove]
    return opt


def deidentify(data, *, profile="basic", patient_name="Anonymous",
               patient_id="ANON0001", uid_root="2.25", retain_dates=False,
               retain_safe_private=False, clean_descriptors=False,
               retain_uids=False, retain_device_id=False,
               retain_institution_id=False, retain_patient_chars=False,
               clean_graphics=False, clean_struct_content=False,
               clean_pixel=False, shift_dates_days=0, replace=None, remove=None):
    """De-identify one DICOM instance; return the de-identified Part-10 bytes.

    `data` is Part-10 bytes or a path. The source is never modified — a new,
    internally-consistent instance is returned (new SOP/Series/Study UIDs unless
    `retain_uids`).

    `profile`: ``"basic"`` = PS3.15 Annex E scrub (default); ``"none"`` = keep
    every clinical attribute but still remap UIDs and apply `replace`/`remove`
    (a plain attribute /modify — emits no de-identification stamps).

    PS3.15 Table E.1-1 option columns (opt-in): `retain_dates`,
    `retain_safe_private`, `clean_descriptors`, `retain_uids`,
    `retain_device_id`, `retain_institution_id`, `retain_patient_chars`,
    `clean_graphics`, `clean_struct_content`. `shift_dates_days` shifts every
    Modified-Dates DA/DT by N days (consistent across the call).

    `clean_pixel` (113101 Clean Pixel Data): after the tag scrub, black out
    burned-in annotation regions using the RSNA CTP device-signature library.
    Output is re-emitted uncompressed (pixel quality never degrades); non-image
    instances pass through untouched. For explicit regions use
    :func:`clean_pixel_data`.

    `replace`: ``{tag: value}`` per-tag overrides (tag = int / Tag / "GGGG,EEEE"
    / keyword); `remove`: iterable of tags to erase.
    """
    opt = _options(profile, patient_name, patient_id, uid_root, retain_dates,
                   retain_safe_private, clean_descriptors, retain_uids,
                   retain_device_id, retain_institution_id, retain_patient_chars,
                   clean_graphics, clean_struct_content, clean_pixel,
                   shift_dates_days, replace, remove)
    return _core.deidentify(_read(data), opt)


def deidentify_series(files, **kwargs):
    """De-identify a list of instances through ONE session — the UID remap is
    consistent across the batch, so a study's intra-study cross-references in the
    de-identified copy still resolve. Returns a list of Part-10 bytes, in order.

    `files` is an iterable of (bytes | path); `kwargs` are the same options as
    :func:`deidentify`.
    """
    # Pop the options once (validated identically) and apply over the batch.
    opt = _options(
        kwargs.pop("profile", "basic"),
        kwargs.pop("patient_name", "Anonymous"),
        kwargs.pop("patient_id", "ANON0001"),
        kwargs.pop("uid_root", "2.25"),
        kwargs.pop("retain_dates", False),
        kwargs.pop("retain_safe_private", False),
        kwargs.pop("clean_descriptors", False),
        kwargs.pop("retain_uids", False),
        kwargs.pop("retain_device_id", False),
        kwargs.pop("retain_institution_id", False),
        kwargs.pop("retain_patient_chars", False),
        kwargs.pop("clean_graphics", False),
        kwargs.pop("clean_struct_content", False),
        kwargs.pop("clean_pixel", False),
        kwargs.pop("shift_dates_days", 0),
        kwargs.pop("replace", None),
        kwargs.pop("remove", None),
    )
    if kwargs:
        raise TypeError(f"unexpected keyword arguments: {', '.join(kwargs)}")
    return list(_core.deidentify_series([_read(f) for f in files], opt))


def clean_pixel_data(data, *, regions=None, use_ctp=True, require_match=False):
    """Black out burned-in PHI regions in one instance; return Part-10 bytes.

    Standalone pixel anonymizer (no tag scrub — compose with :func:`deidentify`
    or its ``clean_pixel=True`` option for a full de-identification).

    `data` is Part-10 bytes or a path. `regions` is an iterable of
    ``(x, y, w, h)`` or ``(x, y, w, h, frame)`` pixel rectangles always blacked
    out (frame is 1-based; omit/-1 = all frames). `use_ctp` (default True) also
    matches the RSNA CTP device-signature library. With `require_match`, an
    instance that matches no rule and has no `regions` raises instead of passing
    through. Output is re-emitted uncompressed — pixel quality never degrades.
    """
    regs = None if regions is None else [tuple(r) for r in regions]
    return _core.clean_pixel_data(_read(data), regs, bool(use_ctp), bool(require_match))


__all__ = ["deidentify", "deidentify_series", "clean_pixel_data"]
