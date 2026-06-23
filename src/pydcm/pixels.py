# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Fuli Wu
# NOT a medical device — not for clinical or diagnostic use.
"""Pixel helpers (``pydcm.pixels``). Decoding reuses the native
engine; the LUT/windowing/colour helpers apply the standard PS3.3 formulas to an array."""
from __future__ import annotations

import numpy as np

from . import _native


def _as_path(src):
    """Resolve a Dataset / path / binary file-like to a filesystem path (the native
    decoder is path-based); returns (path, tmp_or_None)."""
    import os
    if hasattr(src, "_path") and getattr(src, "_path") is not None:
        return src._path, None
    if hasattr(src, "read"):                      # binary file-like
        from ._dicom import _spool
        tmp = _spool(src.read())
        return tmp, tmp
    if isinstance(src, (str, bytes, os.PathLike)):
        return os.fspath(src), None
    raise AttributeError("no backing file/path to decode pixels from")


def unpack_bits(ds):
    """Decode 1-bit PixelData to one uint8 value per sample via the native engine."""
    if int(ds.get("BitsAllocated", 0) or 0) != 1:
        raise ValueError("BitsAllocated must be 1")
    return _decode_all(ds)


def decode_uncompressed(ds):
    """Compatibility helper; all pixel decoding is delegated to the native engine."""
    return _decode_all(ds)


def pack_bits(arr, pad: bool = True) -> bytes:
    """Pack a binary {0,1} :class:`numpy.ndarray` into bytes for 1-bit *Pixel Data*
    (PS3.5 §8.1.1, little bit order — inverse of :func:`unpack_bits`)."""
    if arr.shape == (0,):
        return b""
    if not np.array_equal(arr, arr.astype(bool)):
        raise ValueError("Only binary arrays (containing ones or zeroes) can be packed.")
    if arr.ndim > 1:
        arr = arr.ravel()
    if arr.shape[0] % 8:
        arr = np.append(arr, np.zeros(8 - arr.shape[0] % 8))
    packed = np.packbits(arr.astype("u1"), bitorder="little").tobytes()
    if pad:
        return packed + b"\x00" if len(packed) % 2 else packed
    return packed


def apply_rescale(arr, ds):
    """Apply the linear *Modality LUT* (``arr * RescaleSlope + RescaleIntercept``).
    Use :func:`apply_modality_lut` when a *Modality LUT Sequence*
    may be present; this is the rescale-only path."""
    slope = float(ds.get("RescaleSlope", 1) or 1)
    intercept = float(ds.get("RescaleIntercept", 0) or 0)
    if slope == 1 and intercept == 0:
        return arr
    return arr.astype(np.float64) * slope + intercept


def _part10_for_native(ds):
    """Serialize a pydcm Dataset for the native pixel engine.

    Incomplete from-scratch datasets used for pixel work may omit SOP UIDs;
    add ephemeral envelope identities to the serialized copy only.
    """
    try:
        return ds._encode_part10()
    except RuntimeError as exc:
        if "missing SOP Class UID" not in str(exc):
            raise
        import json
        model = ds.to_json_dict()
        model.setdefault("00080016", {
            "vr": "UI", "Value": ["1.2.840.10008.5.1.4.1.1.7"]})
        model.setdefault("00080018", {"vr": "UI", "Value": ["2.25.1"]})
        return _native.write_part10(json.dumps(model, separators=(",", ":")))


def _decode_all(src):
    if hasattr(src, "_encode_part10"):
        path = getattr(src, "_path", None)
        if path is not None and not getattr(src, "_edits", None):
            return _native.decode_stored(path, 0)[0]
        return _native.decode_stored_bytes(_part10_for_native(src), 0)[0]

    path, tmp = _as_path(src)
    try:
        arr, _ = _native.decode_stored(path, 0)
        return arr
    finally:
        if tmp is not None:
            import os
            try:
                os.unlink(tmp)
            except OSError:
                pass


def pixel_array(src, *, index=None, **_kw):
    """Decode stored pixels from a Dataset / path / binary file-like.

    PALETTE COLOR returns stored indices; use :func:`apply_color_lut` for RGB.
    """
    arr = _decode_all(src)
    if index is not None:
        return arr[int(index)]
    return arr[0] if arr.shape[0] == 1 else arr


def iter_pixels(src, **_kw):
    """Yield each frame's pixel array lazily."""
    arr = _decode_all(src)
    for i in range(arr.shape[0]):
        yield arr[i]


def apply_modality_lut(arr, ds):
    """Modality LUT (Rescale Slope/Intercept → e.g. Hounsfield units), PS3.3 C.11.1."""
    slope, icpt = ds.get("RescaleSlope"), ds.get("RescaleIntercept")
    if slope is None and icpt is None:
        return arr
    return arr * (float(slope) if slope is not None else 1.0) + \
        (float(icpt) if icpt is not None else 0.0)


def apply_voi_lut(arr, ds, index=0):
    """VOI LUT / windowing (PS3.3 C.11.2).

    Applies a VOI LUT Sequence if present, else Window Center/Width with the
    VOILUTFunction (LINEAR / LINEAR_EXACT / SIGMOID). The output is scaled to
    ``[0, 2**BitsStored - 1]`` (NOT [0,1])."""
    out = apply_voi(arr, ds, index)
    if out is not arr:
        return out
    return apply_windowing(arr, ds, index)


def apply_voi(arr, ds, index=0):
    """Apply a VOI LUT Sequence (0028,3010) if present; else return ``arr`` unchanged."""
    seq = ds.get("VOILUTSequence")
    if not seq:
        return arr
    item = seq[index] if index < len(seq) else seq[0]
    desc = item.get("LUTDescriptor")
    data = item.get("LUTData")
    if not desc or data is None:
        return arr
    nr = desc[0] or (1 << 16)
    first = int(desc[1])
    nbits = int(desc[2])
    lut = np.asarray(data, dtype="float64")
    if lut.size < nr:                              # LUTData packed as bytes
        lut = np.frombuffer(bytes(data), dtype="<u2").astype("float64")
    clipped = np.clip(arr.astype("int64") - first, 0, len(lut) - 1)
    return lut[clipped]


def apply_windowing(arr, ds, index=0):
    """Linear/sigmoid Window Center/Width (PS3.3 C.11.2.1.2)."""
    wc, ww = ds.get("WindowCenter"), ds.get("WindowWidth")
    if wc is None or ww is None:
        return arr
    if isinstance(wc, list):
        wc = wc[index if index < len(wc) else 0]
    if isinstance(ww, list):
        ww = ww[index if index < len(ww) else 0]
    c, w = float(wc), float(ww)
    if w < 1:
        return arr
    bits_stored = int(ds.get("BitsStored", 16) or 16)
    y_min, y_max = 0.0, float((1 << bits_stored) - 1)
    fn = str(ds.get("VOILUTFunction", "LINEAR") or "LINEAR").upper()
    a = arr.astype("float64")
    if fn == "SIGMOID":
        return y_min + (y_max - y_min) / (1.0 + np.exp(-4.0 * (a - c) / w))
    if fn == "LINEAR_EXACT":
        out = (a - c) / w * (y_max - y_min) + y_min
        return np.clip(out, y_min, y_max)
    # LINEAR (default)
    below = a <= (c - 0.5) - (w - 1) / 2.0
    above = a > (c - 0.5) + (w - 1) / 2.0
    out = ((a - (c - 0.5)) / (w - 1) + 0.5) * (y_max - y_min) + y_min
    out[below] = y_min
    out[above] = y_max
    return out


# PS3.3 C.7.6.3.1.2 colour-space conversion matrices (8-bit full-range).
def convert_color_space(arr, current, desired, *, per_frame=False):
    """Convert between RGB and YBR_FULL/YBR_FULL_422 (PS3.3 C.7.6.3.1.2)."""
    current = (current or "").upper().replace(" ", "")
    desired = (desired or "").upper().replace(" ", "")
    if current == desired or arr.ndim < 1 or arr.shape[-1] != 3:
        return arr
    is_ybr_src = current.startswith("YBR")
    is_ybr_dst = desired.startswith("YBR")
    if is_ybr_src == is_ybr_dst:                   # both RGB or both YBR family
        return arr
    a = arr.astype("float64")
    ch = [a[..., i] for i in range(3)]
    if is_ybr_dst:                                 # RGB -> YBR_FULL
        r, g, b = ch
        y = 0.2990 * r + 0.5870 * g + 0.1140 * b
        cb = -0.1687 * r - 0.3313 * g + 0.5000 * b + 128.0
        cr = 0.5000 * r - 0.4187 * g - 0.0813 * b + 128.0
        out = np.stack([y, cb, cr], axis=-1)
    else:                                          # YBR_FULL -> RGB
        y, cb, cr = ch
        r = y + 1.4020 * (cr - 128.0)
        g = y - 0.3441 * (cb - 128.0) - 0.7141 * (cr - 128.0)
        b = y + 1.7720 * (cb - 128.0)
        out = np.stack([r, g, b], axis=-1)
    return np.clip(np.round(out), 0, 255).astype(arr.dtype)


def apply_color_lut(arr, ds=None, palette=None):
    """Map PALETTE COLOR stored indices to RGB through the native engine."""
    if palette is not None:
        raise NotImplementedError("well-known palette datasets are not bundled by pydcm")
    if ds is None:
        raise ValueError("Either 'ds' or 'palette' is required")
    if not hasattr(ds, "_encode_part10"):
        raise TypeError("'ds' must be a pydcm Dataset")
    bits_allocated = int(ds.get("BitsAllocated", 0) or 0)
    is_signed = int(ds.get("PixelRepresentation", 0) or 0) == 1
    if bits_allocated == 1:
        dtype = np.uint8
    elif bits_allocated == 8:
        dtype = np.int8 if is_signed else np.uint8
    elif bits_allocated == 16:
        dtype = np.int16 if is_signed else np.uint16
    else:
        raise ValueError("Palette indices require BitsAllocated of 1, 8 or 16")
    indices = np.ascontiguousarray(arr, dtype=dtype)
    return _native.apply_palette_indices(
        _part10_for_native(ds), indices, bits_allocated, is_signed)


def apply_presentation_lut(arr, ds):
    """Apply a Presentation LUT (Sequence or INVERSE shape) to `arr`.

    Returns P-values; if no Presentation LUT module is present, returns `arr` unchanged.
    Modality/VOI LUTs (if any) must be applied first."""
    seq = ds.get("PresentationLUTSequence")
    if seq:
        item = seq[0]
        nr_entries, _first_map, bit_depth = item.LUTDescriptor
        nr_entries = 2 ** 16 if nr_entries == 0 else nr_entries
        itemsize = 8 if bit_depth <= 8 else 16
        data = item["LUTData"]
        if getattr(data, "VR", None) == "US":
            lut = np.asarray(data.value, dtype="u2")
        else:
            lut = np.frombuffer(bytes(item.LUTData)[: nr_entries * (itemsize // 8)],
                                dtype=f"uint{itemsize}")
        if (bit_shift := itemsize - bit_depth):
            lut = lut.copy() if not lut.flags.writeable else lut
            np.left_shift(lut, bit_shift, out=lut)
            np.right_shift(lut, bit_shift, out=lut)
        a = arr.astype("float32")
        a -= a.min()
        a /= a.max() / (nr_entries - 1)
        return lut[a.astype("uint16")]
    shape = ds.get("PresentationLUTShape")
    if shape:
        transform = str(shape).strip().upper()
        if transform not in ("IDENTITY", "INVERSE"):
            raise NotImplementedError(
                f"A (2050,0020) 'Presentation LUT Shape' value of '{shape}' is not supported")
        if transform == "INVERSE":
            return arr.max() - arr
    return arr


_IMAGE_PIXEL = {
    0x00280002: "samples_per_pixel", 0x00280004: "photometric_interpretation",
    0x00280006: "planar_configuration", 0x00280008: "number_of_frames",
    0x00280010: "rows", 0x00280011: "columns", 0x00280100: "bits_allocated",
    0x00280101: "bits_stored", 0x00280103: "pixel_representation",
}


def as_pixel_options(ds, **kwargs):
    """Return the Image Pixel module element values from `ds` as a dict."""
    opts = {attr: ds[tag].value for tag, attr in _IMAGE_PIXEL.items() if tag in ds}
    nf = opts.get("number_of_frames", 1)
    nf = int(nf) if isinstance(nf, str) else nf
    opts["number_of_frames"] = nf or 1
    if 0x7FE00001 in ds and 0x7FE00002 in ds:
        opts["extended_offsets"] = (ds.ExtendedOffsetTable, ds.ExtendedOffsetTableLengths)
    opts.update(kwargs)
    return opts


def compress(ds, transfer_syntax_uid, arr=None, *, encoding_plugin="",
             encapsulate_ext=False, generate_instance_uid=True, **kwargs):
    """Compress `ds` in place to `transfer_syntax_uid` (delegates to the native engine)."""
    ds.compress(transfer_syntax_uid, arr, encoding_plugin=encoding_plugin,
                generate_instance_uid=generate_instance_uid)
    return ds


def decompress(ds, *, as_rgb=True, generate_instance_uid=True, decoding_plugin="", **kwargs):
    """Decompress `ds`'s *Pixel Data* in place to native encoding (delegates to native)."""
    ds.decompress(as_rgb=as_rgb, generate_instance_uid=generate_instance_uid,
                  decoding_plugin=decoding_plugin)
    return ds


def set_pixel_data(ds, arr, photometric_interpretation, bits_stored, *,
                   generate_instance_uid=True):
    """Set `ds`'s *Pixel Data* + Image Pixel module elements from `arr`."""
    from .uid import ExplicitVRLittleEndian, generate_uid
    from .dataset import FileMetaDataset
    samples = {"MONOCHROME1": 1, "MONOCHROME2": 1, "PALETTE COLOR": 1,
               "RGB": 3, "YBR_FULL": 3, "YBR_FULL_422": 3}
    if photometric_interpretation not in samples:
        raise ValueError(
            f"Unsupported 'photometric_interpretation' value '{photometric_interpretation}'")
    dtype = arr.dtype
    if dtype.kind not in ("u", "i") or dtype.itemsize not in (1, 2):
        raise ValueError(f"Unsupported ndarray dtype '{dtype}', must be int8, int16, uint8 or uint16")
    nr, ndim, shape = samples[photometric_interpretation], arr.ndim, arr.shape
    if nr == 1:
        if ndim not in (2, 3):
            raise ValueError(f"An ndarray with '{photometric_interpretation}' data must have 2 or 3 dimensions, not {ndim}")
        frames = shape[0] if ndim == 3 else None
        rows, cols = (shape[1], shape[2]) if ndim == 3 else (shape[0], shape[1])
    else:
        if ndim not in (3, 4):
            raise ValueError(f"An ndarray with '{photometric_interpretation}' data must have 3 or 4 dimensions, not {ndim}")
        if shape[-1] != nr:
            raise ValueError(f"An ndarray with '{photometric_interpretation}' data must have shape (rows, columns, {nr}) or (frames, rows, columns, {nr}), not {shape}")
        frames = None if ndim == 3 else shape[0]
        rows, cols = (shape[0], shape[1]) if ndim == 3 else (shape[1], shape[2])
    if not 0 < bits_stored <= dtype.itemsize * 8:
        raise ValueError(f"Invalid 'bits_stored' value '{bits_stored}', must be greater than 0 and less than or equal to {dtype.itemsize * 8}")
    amin, amax = int(arr.min()), int(arr.max())
    lo = 0 if dtype.kind == "u" else -(2 ** (bits_stored - 1))
    hi = 2 ** bits_stored - 1 if dtype.kind == "u" else 2 ** (bits_stored - 1) - 1
    if amin < lo or amax > hi:
        raise ValueError(f"The range of values in the ndarray [{amin}, {amax}] is greater than that allowed by the 'bits_stored' value [{lo}, {hi}]")

    if not hasattr(ds, "file_meta"):
        ds.file_meta = FileMetaDataset()
    if frames is None:
        if "NumberOfFrames" in ds:
            del ds.NumberOfFrames
    else:
        ds.NumberOfFrames = frames
    ds.Rows, ds.Columns, ds.SamplesPerPixel = rows, cols, nr
    if nr > 1:
        ds.PlanarConfiguration = 0
    elif "PlanarConfiguration" in ds:
        del ds.PlanarConfiguration
    ds.PhotometricInterpretation = photometric_interpretation
    ds.BitsAllocated, ds.BitsStored, ds.HighBit = dtype.itemsize * 8, bits_stored, bits_stored - 1
    ds.PixelRepresentation = 0 if dtype.kind == "u" else 1

    if photometric_interpretation == "YBR_FULL_422":   # PS3.3 C.7.6.3.1.2 downsample
        a = arr.ravel()
        out = np.empty(a.size // 3 * 2, dtype=dtype)
        out[::4], out[1::4], out[2::4], out[3::4] = a[::6], a[3::6], a[1::6], a[2::6]
        arr = out
    data = arr.tobytes()
    ds.PixelData = data if len(data) % 2 == 0 else data + b"\x00"
    try:
        ds["PixelData"].VR = "OB" if ds.BitsAllocated <= 8 else "OW"
    except Exception:
        pass
    ts = ds.file_meta.get("TransferSyntaxUID")
    if not ts or getattr(ts, "is_compressed", False):
        ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    if generate_instance_uid:
        uid = generate_uid()
        ds.SOPInstanceUID = uid
        ds.file_meta.MediaStorageSOPInstanceUID = uid


__all__ = ["pixel_array", "iter_pixels", "apply_modality_lut", "apply_voi_lut",
           "apply_voi", "apply_windowing", "convert_color_space", "apply_color_lut",
           "unpack_bits", "pack_bits", "apply_rescale", "apply_presentation_lut",
           "as_pixel_options", "compress", "decompress", "set_pixel_data"]
