# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Fuli Wu
# NOT a medical device — not for clinical or diagnostic use.

"""pydcm — DICOM Segmentation authoring + reading (``pydcm.seg``).

Author coded binary or fractional Segmentations from a labelmap / probability maps,
and read a Segmentation back to a labelmap / per-segment masks, over the shared
native SEG write/decode engines — a native interop
path for the common cases.
"""
from __future__ import annotations

import numpy as np

from . import _core


def _refs(reference):
    if isinstance(reference, (str, bytes)) or hasattr(reference, "__fspath__"):
        return [str(reference)]
    return [str(p) for p in reference]


def write_seg(reference, labelmap, segments, output=None):
    """Author a coded BINARY DICOM Segmentation from a labelmap + segment terminology.

    reference: a source-image path, or a list of the source series' instance paths
        — geometry, demographics and source references are taken from it.
    labelmap: a uint16 array ``(H, W)`` or ``(slices, H, W)``; value ``k`` marks the
        segment whose ``labelID`` is ``k``. For a series the slices must be ordered
        by ascending position.
    segments: list of dicts, each with ``label``, ``labelID``, ``rgb`` = (r, g, b),
        ``category`` / ``type`` / ``anatomic`` = (CodeValue, CodingScheme, CodeMeaning),
        ``algorithm_type``, ``algorithm_name``.
    output: write the SEG there and return ``None``; if omitted, return Part-10 bytes.
    """
    lm = np.ascontiguousarray(labelmap, dtype=np.uint16)
    return _core.write_seg(_refs(reference), lm, list(segments),
                           str(output) if output else "")


def write_seg_fractional(reference, maps, segments, *, type="probability",
                         max_value=255, output=None):
    """Author a FRACTIONAL DICOM Segmentation from per-segment probability/occupancy maps.

    The natural output of a soft-prediction model — each segment keeps its 8-bit
    value map instead of a hard 1-bit mask.

    reference / segments / output: as in :func:`write_seg`.
    maps: array ``[nseg, (slices,) H, W]`` (segment-major; ``maps[i]`` is segment i's
        map). Float input is treated as 0..1 and scaled to 0..`max_value`; integer
        input is used as-is.
    type: ``'probability'`` or ``'occupancy'`` (Segmentation Fractional Type).
    """
    ftype = {"probability": 0, "occupancy": 1}.get(str(type).lower())
    if ftype is None:
        raise ValueError("type must be 'probability' or 'occupancy'")
    arr = np.asarray(maps)
    if np.issubdtype(arr.dtype, np.floating):
        arr = np.rint(np.clip(arr, 0.0, 1.0) * max_value)
    arr = np.ascontiguousarray(arr, dtype=np.uint8)
    return _core.write_seg_fractional(_refs(reference), arr, list(segments),
                                      ftype, int(max_value), str(output) if output else "")


def write_seg_from_prediction(prediction, reference, segments, output=None):
    """Map a model prediction back onto the original DICOM grid and write a coded SEG.

    Closes the inference loop: preprocess a series (resample / crop / reorient /
    transpose), run a model, then call this to put the label map back where it came
    from. ``prediction`` is resampled onto the reference series' grid by AFFINE
    (label-safe nearest), so whatever spatial preprocessing produced it is inverted
    geometrically — no recorded op-stack needed.

    prediction: a label-map :class:`~pydcm.volume.Volume` — integer voxels, value ``k``
        marks the segment with ``labelID`` ``k`` — carrying its (processed-space) affine.
        For soft probabilities ``[C, Z, Y, X]`` take :func:`pydcm.transforms.argmax` first.
    reference: the original series — a directory or list of instance paths. Its grid,
        demographics and per-slice references define the SEG. **Precondition: a single
        coherent 3-D series** (one orientation, one stack). The resample target comes from
        :func:`load_series` (which IOP-clusters + splits temporal/echo/b-value onto a 4th
        axis), while :func:`write_seg` keeps every dims-matching slice — so a 4-D / multi-echo
        / perfusion / multi-orientation reference makes the two grids disagree and raises a
        clear error; pass the coherent sub-series in that case.
    segments / output: as in :func:`write_seg`.
    """
    from .volume import load_series
    from .transforms import resample_to_reference
    target = load_series(reference)
    on_grid = resample_to_reference(prediction, target, is_label=True)
    try:
        return write_seg(reference, np.ascontiguousarray(on_grid.pixels, dtype=np.uint16),
                         segments, output)
    except RuntimeError as e:                            # write_seg's slice grid != load_series'
        if "does not match the reference" in str(e):
            raise ValueError(
                "write_seg_from_prediction: the reference is not a single coherent 3-D series "
                f"(load_series gave {on_grid.pixels.shape[0]} slices, the SEG writer expects a "
                "different count — e.g. a 4-D / multi-echo / perfusion / multi-orientation series). "
                "Pass the coherent sub-series.") from e
        raise


def seg_from_nifti(reference, mask, segments, output=None):
    """Author a coded DICOM Segmentation from a NIfTI label volume + reference series.

    The NIfTI / FSL / ANTs → DICOM-SEG return path — the converter
    cannot produce. ``mask`` is a ``.nii``/``.nii.gz`` label volume co-framed with
    ``reference`` (the natural NIfTI → segment → mask-back round-trip); the
    native reader flips its Z axis to the reference's ascending-position order via
    the affine, so the labels land on the right slices.

    reference: a source-image path, a directory of the series' instances, or a list
        of instance paths — geometry / demographics / source references come from it.
    mask: path to the NIfTI label volume.
    segments: as in :func:`write_seg` (``labelID`` selects which label maps to each).
    output: write the SEG there and return ``None``; if omitted, return Part-10 bytes.
    """
    from pathlib import Path
    if isinstance(reference, (str, Path)) and Path(reference).is_dir():
        from .torchdata import scan
        refs = [str(p) for p in scan(reference, recursive=True)]
    else:
        refs = _refs(reference)
    blob = _core.build_seg_from_nifti(refs, str(mask), list(segments))
    if output:
        with open(output, "wb") as f:
            f.write(blob)
        return None
    return blob


# --------------------------------------------------------------------------- #
#  Reading — SegmentReader / MultiClassReader
# --------------------------------------------------------------------------- #
def _read_seg(dataset):
    from ._dicom import Dataset
    if isinstance(dataset, Dataset):
        return dataset
    from . import dcmread
    return dcmread(dataset)


def _frames_by_segment(ds):
    """(frames[N,rows,cols] uint8, segment-number-per-frame, {segnum: SegmentSequence item})."""
    frames = np.asarray(ds.pixel_array)
    if frames.ndim == 2:
        frames = frames[None]
    seg_of_frame = []
    for fg in (ds.get("PerFrameFunctionalGroupsSequence") or []):
        sid = fg.get("SegmentIdentificationSequence")
        seg_of_frame.append(int(sid[0].ReferencedSegmentNumber) if sid else 1)
    if not seg_of_frame:                                  # no per-frame groups → all segment 1
        seg_of_frame = [1] * len(frames)
    infos = {int(it.SegmentNumber): it for it in (ds.get("SegmentSequence") or [])}
    return frames, seg_of_frame, infos


class _SegmentResult:
    """Read result: per-segment binary masks."""
    def __init__(self, ds, frames, seg_of_frame, infos):
        self.dataset = ds
        self._frames = frames
        self._seg = seg_of_frame
        self.segment_infos = infos

    @property
    def available_segments(self):
        return sorted(self.segment_infos)

    @property
    def referenced_series_uid(self):
        seq = self.dataset.get("ReferencedSeriesSequence")
        return str(seq[0].SeriesInstanceUID) if seq else None

    def segment_data(self, number):
        """The binary mask volume ``[slices, rows, cols]`` (uint8 {0,1}) for ``number``."""
        idx = [i for i, s in enumerate(self._seg) if s == int(number)]
        if not idx:
            raise KeyError(f"segment {number} not in this Segmentation")
        return self._frames[idx]

    def segment_image(self, number):
        """The segment mask as a SimpleITK image (requires SimpleITK)."""
        import SimpleITK as sitk
        return sitk.GetImageFromArray(self.segment_data(number))


class _MultiClassResult:
    """A single label-map volume (voxel value = segment number); for non-overlapping SEGs."""
    def __init__(self, data, infos):
        self.data = data
        self.segment_infos = infos

    @property
    def available_segments(self):
        return sorted(self.segment_infos)


class SegmentReader:
    """Read a DICOM Segmentation into per-segment masks."""
    def read(self, dataset) -> _SegmentResult:
        ds = _read_seg(dataset)
        return _SegmentResult(ds, *_frames_by_segment(ds))


class MultiClassReader:
    """Read a (non-overlapping) Segmentation into one label-map volume."""
    def read(self, dataset) -> _MultiClassResult:
        ds = _read_seg(dataset)
        frames, seg_of_frame, infos = _frames_by_segment(ds)
        nseg = len(infos) or 1
        per = len(frames) // nseg if nseg else len(frames)   # frames per segment (v1: equal split)
        depth = per if per else len(frames)
        label = np.zeros((depth,) + frames.shape[1:], dtype=np.uint16)
        # Each segment's frames are assumed to share the source geometry/order (v1 — no IPP
        # re-sort); paint each segment's voxels with its number where the mask is set.
        for num in sorted(infos):
            idx = [i for i, s in enumerate(seg_of_frame) if s == num][:depth]
            for z, i in enumerate(idx):
                label[z][frames[i] > 0] = num
        return _MultiClassResult(label, infos)


# --------------------------------------------------------------------------- #
#  Reading — geometry-correct labelmap reconstruction
# --------------------------------------------------------------------------- #
def read_seg(path, *, masks=False):
    """Reconstruct a DICOM Segmentation, over the shared native SEG decode engine
    (geometry-correct: frames are placed onto a slice grid built from the per-frame
    Image Position projected along the slice normal — unlike the simpler
    :class:`MultiClassReader`).

    masks=False (default): ``(labelmap, meta)`` — ``labelmap`` is ``(slices, rows,
        cols)`` ``uint16``, voxel value = DICOM Segment Number (0 = background).
        ``meta["overlapping"]`` flags overlapping segments (combined labelmap is lossy
        there — use ``masks=True``).
    masks=True: ``(masks, meta)`` — ``(nseg, slices, rows, cols)`` ``float32`` occupancy
        in ``[0, 1]`` (binary → 0/1, fractional → value/max); lossless for overlapping /
        fractional. Plane ``k`` is segment ``meta["segment_numbers"][k]``.

    meta carries per-segment terminology (``segments``: number / label / category / type
        / anatomic codes / rgb), geometry (``image_orientation_patient``, ``pixel_spacing``,
        ``slice_thickness``, ``slice_origins``) and a 4×4 ``affine`` (voxel→world LPS mm).
        Returns ``None`` when `path` is not a Segmentation.
    """
    r = _core.read_seg(str(path), masks)
    if r is None:
        return None
    arr, meta = r
    # The native SEG engine already computed the canonical affine (vol_build_model_matrix); reshape its column-major 16 floats into the 4×4
    # numpy form callers expect — no second affine implementation in Python.
    meta["affine"] = np.asarray(meta["affine"], dtype=np.float64).reshape(4, 4, order="F")
    return arr, meta


# ─────────────────────────────────────────────────────────────────────────────
#  class API (thin constructors over the native writers).
#  `Segmentation(...)` returns a pydcm Dataset built by write_seg / write_seg_fractional.
# ─────────────────────────────────────────────────────────────────────────────
import enum as _enum
import os as _os
import tempfile as _tempfile
import atexit as _atexit

from .sr import Code as CodedConcept                # CodedConcept: alias for pydcm.sr.Code

_SEG_TMPS: list = []


@_atexit.register
def _cleanup_seg_tmps():
    for _p in _SEG_TMPS:
        try:
            _os.unlink(_p)
        except OSError:
            pass


class SegmentationTypeValues(str, _enum.Enum):
    BINARY = "BINARY"; FRACTIONAL = "FRACTIONAL"; LABELMAP = "LABELMAP"

    def __str__(self): return self.value


class SegmentAlgorithmTypeValues(str, _enum.Enum):
    AUTOMATIC = "AUTOMATIC"; SEMIAUTOMATIC = "SEMIAUTOMATIC"; MANUAL = "MANUAL"

    def __str__(self): return self.value


class SegmentationFractionalTypeValues(str, _enum.Enum):
    PROBABILITY = "PROBABILITY"; OCCUPANCY = "OCCUPANCY"

    def __str__(self): return self.value


class AlgorithmIdentificationSequence:
    """Identifies the algorithm that produced a segment."""
    def __init__(self, name, family=None, version=None, source=None, parameters=None):
        self.name = name
        self.family = family
        self.version = version
        self.source = source
        self.parameters = parameters


def _code_tuple(c):
    """(CodeValue, CodingSchemeDesignator, CodeMeaning) from a Code/CodedConcept."""
    return (str(c.value), str(c.scheme_designator), str(c.meaning))


def _palette(n):
    """A deterministic distinct RGB for segment number `n` (when no color is given)."""
    h = (n * 2654435761) & 0xFFFFFF
    return (h >> 16 & 0xFF or 200, h >> 8 & 0xFF or 120, h & 0xFF or 60)


class SegmentDescription:
    """Description of one segment."""
    def __init__(self, segment_number, segment_label, segmented_property_category,
                 segmented_property_type, algorithm_type, algorithm_identification=None,
                 tracking_uid=None, tracking_id=None, anatomic_regions=None,
                 primary_anatomic_structures=None, display_color=None):
        self.segment_number = int(segment_number)
        self.segment_label = segment_label
        self.segmented_property_category = segmented_property_category
        self.segmented_property_type = segmented_property_type
        self.algorithm_type = str(algorithm_type)
        self.algorithm_identification = algorithm_identification
        self.tracking_uid = tracking_uid
        self.tracking_id = tracking_id
        self.anatomic_regions = list(anatomic_regions) if anatomic_regions else None
        self.primary_anatomic_structures = (list(primary_anatomic_structures)
                                            if primary_anatomic_structures else None)
        self.display_color = display_color

    def _to_seg_dict(self):
        algo = self.algorithm_identification
        d = {
            "label": self.segment_label,
            "labelID": self.segment_number,
            "category": _code_tuple(self.segmented_property_category),
            "type": _code_tuple(self.segmented_property_type),
            "algorithm_type": self.algorithm_type,
            "algorithm_name": getattr(algo, "name", "") if algo else "",
            "rgb": (tuple(self.display_color)
                    if isinstance(self.display_color, (tuple, list)) and len(self.display_color) == 3
                    else _palette(self.segment_number)),
        }
        if self.anatomic_regions:
            d["anatomic"] = _code_tuple(self.anatomic_regions[0])
        return d


def _source_paths(source_images):
    paths = []
    for s in source_images:
        if isinstance(s, str) or hasattr(s, "__fspath__"):
            paths.append(str(s))
        elif getattr(s, "_path", None):
            paths.append(s._path)
        else:                                        # in-memory dataset → spool to temp
            fd, p = _tempfile.mkstemp(suffix=".dcm"); _os.close(fd)
            s.save_as(p, enforce_file_format=True)
            _SEG_TMPS.append(p); paths.append(p)
    return paths


def Segmentation(source_images, pixel_array, segmentation_type, segment_descriptions,
                 series_instance_uid, series_number, sop_instance_uid, instance_number,
                 manufacturer, manufacturer_model_name=None, software_versions=None,
                 device_serial_number=None, *, fractional_type="PROBABILITY",
                 max_fractional_value=255, content_description=None, content_label=None,
                 content_creator_name=None, transfer_syntax_uid=None, **_kwargs):
    """Constructor — returns a pydcm Dataset.

    Built over the native ``write_seg`` / ``write_seg_fractional``: ``source_images``
    supply geometry/demographics, ``pixel_array`` is the labelmap (BINARY/LABELMAP) or
    per-segment maps (FRACTIONAL), ``segment_descriptions`` the coded terminology.
    extra kwargs are accepted for source compatibility.
    """
    from . import dcmread
    refs = _source_paths(source_images)
    segs = [d._to_seg_dict() for d in segment_descriptions]
    fd, tmp = _tempfile.mkstemp(suffix=".dcm"); _os.close(fd)
    _SEG_TMPS.append(tmp)
    if "FRACTIONAL" in str(segmentation_type).upper():
        write_seg_fractional(refs, pixel_array, segs, type=str(fractional_type).lower(),
                             max_value=int(max_fractional_value), output=tmp)
    else:
        write_seg(refs, pixel_array, segs, output=tmp)
    ds = dcmread(tmp)
    ds.SeriesInstanceUID = series_instance_uid
    ds.SeriesNumber = int(series_number)
    ds.SOPInstanceUID = sop_instance_uid
    ds.file_meta.MediaStorageSOPInstanceUID = sop_instance_uid
    ds.InstanceNumber = int(instance_number)
    ds.Manufacturer = manufacturer
    if manufacturer_model_name is not None:
        ds.ManufacturerModelName = manufacturer_model_name
    if software_versions is not None:
        ds.SoftwareVersions = (software_versions if isinstance(software_versions, str)
                               else list(software_versions))
    if device_serial_number is not None:
        ds.DeviceSerialNumber = device_serial_number
    if content_description is not None:
        ds.ContentDescription = content_description
    if content_label is not None:
        ds.ContentLabel = content_label
    if content_creator_name is not None:
        ds.ContentCreatorName = content_creator_name
    return ds


__all__ = ["write_seg", "write_seg_fractional", "read_seg",
           "SegmentReader", "MultiClassReader",
           "Segmentation", "SegmentDescription", "CodedConcept",
           "AlgorithmIdentificationSequence", "SegmentationTypeValues",
           "SegmentAlgorithmTypeValues", "SegmentationFractionalTypeValues"]
