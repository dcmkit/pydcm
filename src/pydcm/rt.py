# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Fuli Wu
# NOT a medical device — not for clinical or diagnostic use.

"""pydcm — RT dosimetry reader (``pydcm.rt``).

The dose-side counterpart to the RT Structure Set support: ``read_rtdose``
returns the scaled dose grid (pixel × DoseGridScaling, computed in C++ by the
shared native RT engine) as a
NumPy volume plus its geometry and any stored DVH curves. The semantic
*metadata* view of RT Plan / RT Dose lives in :func:`pydcm.content`.
"""
from __future__ import annotations

from . import _core


class DoseGrid:
    """A scaled RT Dose grid.

    Attributes:
        dose: ``ndarray[depth, rows, cols]`` float32 — pixel × DoseGridScaling
            (Gy when ``dose_units == "GY"``).
        dose_grid_scaling: the scale that was APPLIED — equals the file's
            DoseGridScaling, or 1.0 when that tag is absent/zero (malformed),
            so ``dose == pixel × dose_grid_scaling`` holds unconditionally.
        max_dose: grid maximum, computed in double precision — may differ from
            ``dose.max()`` in the last float32 ulp.
        affine: 4×4 voxel→world (LPS) matrix, column-major flat list — same
            convention as :func:`pydcm.load_series`.
        spacing: ``(z, y, x)`` mm; z is the |inter-frame step|.
        grid_frame_offsets: raw GridFrameOffsetVector (mm relative to frame 0).
        uniform_offsets: False when frame steps vary — the affine then uses the
            first step; resample against ``grid_frame_offsets`` in that case.
        dvhs: list of stored DVH curves (dicts with ``bin_widths``/``volumes``
            and the dose scaling/min/max/mean as recorded in the file).
    """

    def __init__(self, dose, meta):
        self.dose = dose
        self.meta = meta
        self.dose_units = meta["dose_units"]
        self.dose_type = meta["dose_type"]
        self.dose_summation_type = meta["dose_summation_type"]
        self.dose_grid_scaling = meta["dose_grid_scaling"]
        self.max_dose = meta["max_dose"]
        self.spacing = tuple(meta["spacing"])
        self.affine = list(meta["affine"])
        self.grid_frame_offsets = list(meta["grid_frame_offsets"])
        self.uniform_offsets = meta["uniform_offsets"]
        self.dvhs = list(meta["dvhs"])
        self.sop_instance_uid = meta["sop_instance_uid"]
        self.referenced_rt_plan_sop_instance_uid = meta.get(
            "referenced_rt_plan_sop_instance_uid")

    @property
    def shape(self):
        return self.dose.shape

    def __repr__(self):
        return (f"<DoseGrid {self.dose.shape} {self.dose_units} "
                f"sum={self.dose_summation_type} max={self.max_dose:g}>")


def read_rtdose(path):
    """Read an RT Dose file (SOP Class …481.2) into a :class:`DoseGrid`.

    All computation (scaling in double precision, geometry, DVH decode) runs in
    the native RT engine; this wrapper only shapes the result.
    """
    dose, meta = _core.read_rtdose(str(path))
    return DoseGrid(dose, meta)


class ComputedDVH:
    """A DVH computed from RTSTRUCT + RTDOSE.

    Attributes:
        counts: differential histogram, cm³ per 1-cGy bin (float64 ndarray,
            trailing zeros trimmed).
        cumulative: suffix-sum of ``counts`` (the cumulative DVH).
        bins: bin edges in Gy (``len(counts) + 1`` values, 0.01 Gy wide).
        volume: structure volume in cm³; min/max/mean: dose statistics in Gy,
            the standard cumulative-DVH dose metrics.
        notes: dose-grid coverage notes ('' when the grid covers the structure).
    """

    def __init__(self, d):
        self.counts = d["counts"]
        self.cumulative = d["cumulative"]
        self.bin_width = d["bin_width_gy"]
        self.volume = d["volume"]
        self.min = d["min"]
        self.max = d["max"]
        self.mean = d["mean"]
        self.name = d["name"]
        self.notes = d["notes"]

    @property
    def bins(self):
        import numpy as np
        return np.arange(0, len(self.counts) + 1) * self.bin_width

    def __repr__(self):
        return (f"<ComputedDVH {self.name!r} volume={self.volume:.2f}cm3 "
                f"min/mean/max={self.min:g}/{self.mean:g}/{self.max:g}Gy>")


def dvhcalc(structure, dose, roi, limit=None, calculate_full_volume=True,
            thickness=None):
    """Compute the DVH of `roi` from an RT Structure Set + RT Dose file pair.

    The rasterisation, dose-plane interpolation, histogram and statistics all
    run in the native RT engine; results follow the standard cumulative-DVH definition
    (base path: no in-plane interpolation / structure-extents options). `limit` caps the histogram in cGy.
    """
    d = _core.compute_dvh(str(structure), str(dose), int(roi),
                          limit=int(limit) if limit else 0,
                          calculate_full_volume=calculate_full_volume,
                          thickness=float(thickness) if thickness else 0.0)
    return ComputedDVH(d)


def write_rtdose(dose, *, affine=None, origin=None, orientation=(1, 0, 0, 0, 1, 0),
                 spacing=None, grid_frame_offsets=None,
                 dose_units="GY", dose_type="PHYSICAL", dose_summation_type="PLAN",
                 ref_plan_uid=None, reference=None,
                 patient_name=None, patient_id=None,
                 study_uid=None, study_date=None, series_uid=None,
                 frame_of_reference_uid=None,
                 scaling=None, bits=32, output=None):
    """Author an RT Dose file (SOP Class …481.2) from a dose grid.

    The write side of :func:`read_rtdose` — the export (quantisation to
    unsigned integers with a self-consistent DoseGridScaling, Part-10 emit)
    runs in the native engine. Typical AI-workflow use: a predicted or
    accumulated grid → a file a TPS/viewer imports.

    Geometry: pass `affine` (column-major 4×4 voxel→world, the
    :func:`read_rtdose`/:func:`pydcm.load_series` convention) OR
    `origin`+`orientation`+`spacing` ``(row, col)`` mm (+ optional
    `grid_frame_offsets`, default derived from the affine's frame step /
    uniform z spacing).

    `reference`: a DICOM file (the RT Plan, planning CT, …) whose
    Patient/Study/FrameOfReference identity is copied; when it IS an RT Plan,
    `ref_plan_uid` defaults to its SOP Instance UID.

    `scaling`: explicit DoseGridScaling; default = max dose / integer max
    (full dynamic range of `bits`, 32 or 16). Returns the Part-10 ``bytes``,
    or writes `output` and returns its path.

    Note: PS3.3 requires a ReferencedRTPlanSequence (Type 1C) for the
    PLAN/BEAM/… summation types — pass `ref_plan_uid` or an RT Plan as
    `reference` for fully conformant output; research/AI grids without a plan
    are written as-is.
    """
    import numpy as np

    arr = np.ascontiguousarray(dose, dtype=np.float64)
    if arr.ndim != 3:
        raise ValueError("dose must be [depth, rows, cols]")
    depth = arr.shape[0]

    if affine is not None:
        m = np.asarray(affine, dtype=np.float64).reshape(4, 4, order="F")
        if not np.allclose(m[3], [0, 0, 0, 1], atol=1e-6):
            raise ValueError(
                "affine bottom row is not [0,0,0,1] — a flat affine must be "
                "COLUMN-major (the read_rtdose/load_series convention); a "
                "row-major flat list would be silently transposed")
        col0, col1, col2 = m[:3, 0], m[:3, 1], m[:3, 2]
        ps_col, ps_row = float(np.linalg.norm(col0)), float(np.linalg.norm(col1))
        if ps_col == 0 or ps_row == 0:
            raise ValueError("affine has zero-length in-plane columns")
        row_dir, col_dir = col0 / ps_col, col1 / ps_row
        origin = m[:3, 3]
        orientation = (*row_dir, *col_dir)
        if grid_frame_offsets is None:
            normal = np.cross(row_dir, col_dir)
            step = float(np.dot(col2, normal))
            if not np.allclose(col2, normal * step, atol=1e-4):
                import warnings
                warnings.warn(
                    "affine frame step is sheared (gantry-tilt style); "
                    "GridFrameOffsetVector cannot represent shear — writing "
                    "the step projected onto the plane normal", stacklevel=2)
            grid_frame_offsets = [k * step for k in range(depth)]
    else:
        if origin is None or spacing is None:
            raise ValueError("pass affine=, or origin= + spacing=(row, col)")
        ps_row, ps_col = float(spacing[0]), float(spacing[1])
        if grid_frame_offsets is None:
            raise ValueError("grid_frame_offsets is required without an affine")
    if len(grid_frame_offsets) != depth:
        raise ValueError("grid_frame_offsets length must equal depth")

    if reference is not None:
        from . import dcmread
        ref = dcmread(str(reference))
        patient_name = patient_name if patient_name is not None else str(getattr(ref, "PatientName", ""))
        patient_id = patient_id if patient_id is not None else str(getattr(ref, "PatientID", ""))
        study_uid = study_uid or str(getattr(ref, "StudyInstanceUID", ""))
        study_date = study_date or str(getattr(ref, "StudyDate", ""))
        frame_of_reference_uid = frame_of_reference_uid or str(getattr(ref, "FrameOfReferenceUID", ""))
        if ref_plan_uid is None and str(getattr(ref, "SOPClassUID", "")) == "1.2.840.10008.5.1.4.1.1.481.5":
            ref_plan_uid = str(getattr(ref, "SOPInstanceUID", ""))

    from .uid import generate_uid
    study_uid = study_uid or generate_uid()
    frame_of_reference_uid = frame_of_reference_uid or generate_uid()
    # fresh series per call — the engine seeds the SOP Instance UID from the
    # series, so per-beam doses written into one study stay distinct
    series_uid = series_uid or generate_uid()

    blob = _core.write_rtdose(
        arr, [float(v) for v in origin], [float(v) for v in orientation],
        float(ps_row), float(ps_col), [float(v) for v in grid_frame_offsets],
        units=dose_units, dose_type=dose_type, summation=dose_summation_type,
        ref_plan_uid=ref_plan_uid or "", patient_name=patient_name or "",
        patient_id=patient_id or "", study_uid=study_uid,
        study_date=study_date or "", series_uid=series_uid or "",
        frame_of_ref_uid=frame_of_reference_uid,
        scaling=float(scaling) if scaling else 0.0, bits=int(bits))
    if output is None:
        return blob
    import pathlib
    p = pathlib.Path(output)
    p.write_bytes(blob)
    return p


__all__ = ["DoseGrid", "read_rtdose", "ComputedDVH", "dvhcalc", "write_rtdose"]
