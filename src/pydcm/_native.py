# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Fuli Wu
# NOT a medical device — not for clinical or diagnostic use.

"""Thin accessor for the compiled ``pydcm._core`` extension.

``_core`` is the native DICOM engine: it decodes every transfer syntax
(JPEG / JPEG-2000 / HTJ2K / JPEG-LS / JPEG-XL / RLE), applies the modality LUT, reads
geometry, and assembles 3D volumes — handing NumPy the pixel buffer in-process
(zero-copy). This module only marshals calls into it.
"""

from __future__ import annotations

import os

try:
    from . import _core                       # compiled native extension (the engine)
except ImportError:                            # pragma: no cover
    _core = None


def available() -> bool:
    return _core is not None


def require():
    if _core is None:
        raise RuntimeError(
            "pydcm native extension (_core) is not built or failed to load; "
            "install the wheel for your platform")
    return _core


def decode(path, frame: int = 0, rescale: bool = False):
    """Return ``(ndarray, meta)`` from the native engine."""
    return require().decode(os.fspath(path), int(frame), bool(rescale))


def decode_bytes(part10: bytes, frame: int = 0, rescale: bool = False):
    """Return ``(ndarray, meta)`` decoding in-memory Part-10 bytes (no backing file)."""
    return require().decode_bytes(bytes(part10), int(frame), bool(rescale))


def decode_stored(path, frame: int = 0):
    """Internal stored-pixel decode; PALETTE COLOR remains one-component indices."""
    return require().decode_stored(os.fspath(path), int(frame))


def decode_stored_bytes(part10: bytes, frame: int = 0):
    """In-memory counterpart of :func:`decode_stored`."""
    return require().decode_stored_bytes(bytes(part10), int(frame))


def apply_palette_indices(part10: bytes, indices, bits_allocated: int, is_signed: bool):
    """Internal thin call to the native Palette LUT pipeline."""
    return require().apply_palette_indices(
        bytes(part10), indices, int(bits_allocated), bool(is_signed))


def voi_apply(values, window_center: float = 0.0, window_width: float = 0.0, func: int = 0,
              lut_bytes: bytes | None = None, lut_first: int = 0, lut_depth: int = 16):
    """Internal thin call to the native VOI transfer: modality values → [0,1]."""
    return require().voi_apply(values, float(window_center), float(window_width), int(func),
                               lut_bytes, int(lut_first), int(lut_depth))


def read_json(path, charset_override: str = "", inline_binary: bool = True) -> str:
    """Part-10 file → DICOM JSON Model string (all elements, charset → UTF-8).

    ``inline_binary`` inlines non-pixel binary (OB/OW/UN) as base64 so the model
    is byte-complete (faithful); the bulk pixel tags stay out.
    """
    return require().read_json(os.fspath(path), str(charset_override), bool(inline_binary))


def write_part10(json_str: str, transfer_syntax: str = "") -> bytes:
    """DICOM JSON Model object string → Part-10 file bytes. ``transfer_syntax`` =
    Explicit VR LE (1.2.840.10008.1.2.1) recodes from the default Implicit VR LE."""
    return require().write_part10(str(json_str), str(transfer_syntax))


def build_dicomdir(inputs, file_set_id: str = "") -> bytes:
    """[(part10_bytes, media_file_id), …] → conformant DICOMDIR Part-10 bytes."""
    return require().build_dicomdir(list(inputs), str(file_set_id))


def read_file_meta(path) -> dict:
    """Part-10 file-meta (group 0002) → {has_meta, transfer_syntax, sop_class, sop_instance}."""
    return require().read_file_meta(os.fspath(path))


def read_meta_json(path) -> str:
    """Full group-0002 File Meta Information as a DICOM JSON Model string (every
    element, not just the 3 UIDs); ``""`` for a naked dataset."""
    return require().read_meta_json(os.fspath(path))


def read_pixel_data_bytes(part10: bytes):
    """In-memory counterpart of :func:`read_pixel_data` (BytesIO datasets)."""
    return require().read_pixel_data_bytes(bytes(part10))


def has_pixel_data_bytes(part10: bytes) -> bool:
    """In-memory counterpart of :func:`has_pixel_data`."""
    return require().has_pixel_data_bytes(bytes(part10))


def pixel_data_vr_bytes(part10: bytes):
    """In-memory counterpart of :func:`pixel_data_vr`."""
    return require().pixel_data_vr_bytes(bytes(part10))


def read_pixel_data(path):
    """Raw (7FE0,0010) PixelData value bytes, or ``None`` when absent / a transfer
    syntax the fast path skips (deflate / EVR-BE)."""
    return require().read_pixel_data(os.fspath(path))


def has_pixel_data(path) -> bool:
    """True if the dataset contains (7FE0,0010) PixelData (no byte copy) — backs the
    lazy presence of PixelData in the Dataset mapping protocol."""
    return require().has_pixel_data(os.fspath(path))


def pixel_data_vr(path):
    """On-disk VR of (7FE0,0010) for Explicit-VR files ('OB'/'OW'/'OF'/'OD'), or ``None``
    (Implicit VR / absent / uncovered TS) — lets ds.PixelData keep the file's real VR."""
    return require().pixel_data_vr(os.fspath(path))


def encode_ivr(json: str) -> bytes:
    """DICOM JSON Model dataset -> bare Implicit VR LE bytes (no meta / SOP-UID
    requirement). Backs DIMSE query/identifier encoding."""
    return require().encode_ivr(str(json))


def transcode(part10: bytes, target_ts: str, quality: int = 0) -> bytes:
    """Re-encode Part-10 bytes to ``target_ts``.

    ``quality`` (1..100, 0 = codec default) reaches the lossy-capable targets;
    an actually-lossy encode also stamps LossyImageCompression/-Method and mints
    a new SOP Instance UID. Backs ``Dataset.compress`` and ``Dataset.decompress``.
    """
    return require().transcode(bytes(part10), str(target_ts), int(quality))


def mint_uid(seed: str = "", root: str = "") -> str:
    """Mint a DICOM UID via the canonical native generator (deterministic per seed)."""
    return require().mint_uid(str(seed), str(root))


def edit_part10(original: bytes, ops: list) -> bytes:
    """Apply edit ops (tag:int, kind:str, value:str, vr:int) to original Part-10 bytes."""
    return require().edit_part10(bytes(original), list(ops))


def tag_for_keyword(keyword: str):
    """DICOM keyword → packed tag ``(group << 16) | element``, or ``None``."""
    return require().tag_for_keyword(str(keyword))


def describe_tag(tag: int):
    """Packed standard tag → ``{keyword,name,vr,vm,retired}`` dict, or ``None``."""
    return require().describe_tag(int(tag))


def describe_private(creator: str, group: int, elem_low: int):
    """Private ``(creator, group, element-low-byte)`` → info dict, or ``None``."""
    return require().describe_private(str(creator), int(group), int(elem_low))


def uid_lookup(uid: str):
    """UID → ``{name,type,keyword,info,retired,cid}`` from the native UID dict, or ``None``."""
    return require().uid_lookup(str(uid))


def uid_for_keyword(keyword: str) -> str:
    """Public UID keyword (e.g. ``'CTImageStorage'``) → UID string, or ``''``."""
    return require().uid_for_keyword(str(keyword))


def uid_table():
    """Full native UID dictionary as ``(uid,name,type,keyword,info,retired,cid)`` tuples."""
    return require().uid_table()


def write_ann(document: dict, output: str = ""):
    """Author a Microscopy Bulk Simple Annotations object (native annotation engine)."""
    return require().write_ann(dict(document), str(output))
