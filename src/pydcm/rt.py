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

import os

from . import _core


def _ref_paths(reference):
    """A source series → a list of instance paths (accept a path, a directory, or a list)."""
    if isinstance(reference, (str, bytes)) or hasattr(reference, "__fspath__"):
        p = str(reference)
        if os.path.isdir(p):
            return [os.path.join(p, f) for f in sorted(os.listdir(p))
                    if os.path.isfile(os.path.join(p, f))]
        return [p]
    return [str(x) for x in reference]


def write_rtstruct(reference, rois, *, label="", output=None,
                   sop_instance_uid="", series_instance_uid=""):
    """Author a DICOM RT Structure Set from ROI contours over a reference series.

    reference: the source-series instance paths (a path, a directory, or a list) — geometry,
        demographics and per-contour source-image references are taken from it.
    rois: a list of ROI dicts, each with ``name``, optional ``rgb`` ``(r, g, b)``, optional
        ``interpreted_type`` (RT ROI Interpreted Type code: 0=UNKNOWN, 4=ORGAN, …), and
        ``contours``: a list of ``(n, 3)`` float64 arrays of LPS patient-mm points — one planar
        contour each, matched to its source slice by position along the slice normal.
    label: Structure Set Label. output: write there and return None, else return Part-10 bytes.

    ``sop_instance_uid`` / ``series_instance_uid``: state this document's own
    identity. Left empty, the native engine mints them with a **deterministic**
    generator, so two documents authored in one process — or, for the study-derived
    writers, two authored for one study — carry the SAME UIDs. That is fine for a
    single self-contained export and is a DICOM global-uniqueness violation for a
    producer that mints many; supply :func:`pydcm.generate_uid` values there.
    """
    return _core.write_mkrtstruct(_ref_paths(reference), list(rois), label or "",
                                  output or "", sop_instance_uid or "",
                                  series_instance_uid or "")


class DoseGrid:
    """A scaled RT Dose grid.

    Attributes:
        dose: ``ndarray[depth, rows, cols]`` float32 — pixel × DoseGridScaling
            (Gy when ``dose_units == "GY"``).
        dose_grid_scaling: the finite positive scale from the file that was
            applied. A missing or invalid required scale is rejected.
        max_dose: grid maximum, computed in double precision — may differ from
            ``dose.max()`` in the last float32 ulp.
        affine: 4×4 voxel→world (LPS) matrix, column-major flat list — same
            convention as :func:`pydcm.load_series`.
        frame_of_reference_uid, origin_lps, column_step_lps, row_step_lps, frame_offsets_mm:
            authoritative patient-LPS geometry in double precision. Frame offsets
            are canonical relative offsets even when the source used the
            absolute-z GridFrameOffsetVector option.
        spacing: ``(z, y, x)`` mm compatibility view. For a one-frame or
            nonuniform grid, z is zero and the affine has no invented frame
            step.
        grid_frame_offsets: float32 compatibility projection of
            ``frame_offsets_mm``.
        uniform_offsets, has_uniform_affine: geometry capability flags.
        dvhs: list of stored DVH curves (dicts with ``bin_widths``/``volumes``
            and full ROI contribution relationships). Optional statistics are
            absent when not recorded.
        stored_dvh_error: optional diagnostic for a malformed stored DVH
            sequence; the independently usable dose grid remains available.
    """

    def __init__(self, dose, meta):
        self.dose = dose
        self.meta = meta
        self.dose_units = meta["dose_units"]
        self.dose_type = meta["dose_type"]
        self.dose_summation_type = meta["dose_summation_type"]
        self.dose_grid_scaling = meta["dose_grid_scaling"]
        self.max_dose = meta["max_dose"]
        self.frame_of_reference_uid = meta["frame_of_reference_uid"]
        self.origin_lps = tuple(meta["origin_lps"])
        self.column_step_lps = tuple(meta["column_step_lps"])
        self.row_step_lps = tuple(meta["row_step_lps"])
        self.frame_offsets_mm = list(meta["frame_offsets_mm"])
        self.spacing = tuple(meta["spacing"])
        self.affine = list(meta["affine"])
        self.grid_frame_offsets = list(meta["grid_frame_offsets"])
        self.uniform_offsets = meta["uniform_offsets"]
        self.has_uniform_affine = meta["has_uniform_affine"]
        self.dvhs = list(meta["dvhs"])
        self.stored_dvh_error = meta.get("stored_dvh_error")
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


class DVHValue:
    """One dosimetric readout: a number plus the units it is in.

    Compares and formats as its number, so ``dvh.D95 > 60`` and
    ``f"{dvh.D95:.1f}"`` read the way the constraint is written down, while
    ``repr`` keeps the units visible.
    """

    __slots__ = ("value", "units")

    def __init__(self, value, units=""):
        self.value = float(value)
        self.units = units

    def __float__(self):
        return self.value

    def __repr__(self):
        return f"{self.value:g} {self.units}".strip()

    def __format__(self, spec):
        return format(self.value, spec) if spec else repr(self)

    # NotImplemented rather than float(other) so the right-hand operand gets its
    # turn: that is what lets a DVHValue be compared with a tolerance wrapper, and
    # it makes `value == "x"` False instead of a TypeError.
    @staticmethod
    def _num(other):
        if isinstance(other, DVHValue):
            return other.value
        return float(other) if isinstance(other, (int, float)) else None

    def __eq__(self, other):
        v = self._num(other)
        return NotImplemented if v is None else self.value == v

    def __lt__(self, other):
        v = self._num(other)
        return NotImplemented if v is None else self.value < v

    def __le__(self, other):
        v = self._num(other)
        return NotImplemented if v is None else self.value <= v

    def __gt__(self, other):
        v = self._num(other)
        return NotImplemented if v is None else self.value > v

    def __ge__(self, other):
        v = self._num(other)
        return NotImplemented if v is None else self.value >= v

    def __hash__(self):
        return hash(self.value)


class DVH:
    """A DVH computed from RTSTRUCT + RTDOSE.

    The histogram and its statistics are computed by the native RT engine; the
    D(V) / V(D) readings below are the engine's own bin search, not a second
    one written in Python — so a constraint reads the same here, in the CLI and
    in a viewer.

    Attributes:
        counts: differential histogram, cm³ per 1-cGy bin (float64 ndarray,
            trailing zeros trimmed).
        cumulative: suffix-sum of ``counts`` (the cumulative DVH).
        bins: bin edges in Gy (``len(counts) + 1`` values, 0.01 Gy wide).
        bincenters, dose_axis: bin centres in Gy.
        volume: structure volume in cm³ — the total ROI when the DVH was
            computed with ``calculate_full_volume``, else the covered volume.
        total_volume, covered_volume, uncovered_volume: the volume split the
            dose grid actually produced. A per-cent is of ``covered_volume``,
            because that is what the histogram is of.
        min/max/mean: dose statistics in Gy.
        rx_dose: prescription in Gy, for the ``%`` dose forms (``V100%``).
            Optional — set it, or pass ``rx_dose=`` to :func:`dvhcalc`.
        notes: dose-grid coverage notes ('' when the grid covers the structure).
    """

    dose_units = "Gy"
    volume_units = "cm3"

    def __init__(self, curve, rx_dose=None, name=None):
        self._c = curve
        self.counts = curve.counts
        self.cumulative = curve.cumulative
        # From the engine that chose it, never spelled here.
        self.bin_width = _core.dvh_bin_width_gy
        self.volume = curve.volume_cm3
        self.total_volume = curve.total_volume_cm3
        self.covered_volume = curve.covered_volume_cm3
        self.uncovered_volume = curve.uncovered_volume_cm3
        self.min = curve.min_gy
        self.max = curve.max_gy
        self.mean = curve.mean_gy
        self.name = name if name is not None else curve.roi_name
        self.notes = curve.notes
        self.rx_dose = float(rx_dose) if rx_dose else None

    # ---- axes --------------------------------------------------------------

    @property
    def bins(self):
        import numpy as np
        return np.arange(0, len(self.counts) + 1) * self.bin_width

    @property
    def bincenters(self):
        import numpy as np
        return (np.arange(0, len(self.counts)) + 0.5) * self.bin_width

    dose_axis = bincenters

    @property
    def relative_volume(self):
        """``cumulative`` as a per-cent of the covered volume (ndarray)."""
        import numpy as np
        denom = self.covered_volume or (self.cumulative[0] if len(self.cumulative) else 0.0)
        if not denom:
            return np.zeros_like(self.cumulative)
        return self.cumulative * (100.0 / denom)

    def relative_dose(self, rx_dose=None):
        """Bin centres as a per-cent of the prescription (ndarray)."""
        rx = float(rx_dose) if rx_dose else self.rx_dose
        if not rx:
            raise ValueError("relative_dose needs a prescription — set .rx_dose "
                             "or pass rx_dose=")
        return self.bincenters * (100.0 / rx)

    # ---- the two readings --------------------------------------------------

    def dose_constraint(self, volume, volume_units=None):
        """D(V): the dose that at least `volume` of the structure receives, in Gy.

        `volume_units` is ``'%'`` (the default — a per-cent of the covered
        volume) or ``'cc'`` / ``'cm3'`` for an absolute volume. NaN when the
        structure is smaller than the volume asked about: that is a question
        with no answer, and 0 Gy would be a wrong one.
        """
        u = (volume_units or "%").lower()
        if u == "%":
            gy = self._c.dose_at_volume_fraction(float(volume) / 100.0)
        elif u in ("cc", "cm3"):
            gy = self._c.dose_at_volume_cm3(float(volume))
        else:
            raise ValueError(f"volume_units must be '%', 'cc' or 'cm3', not {volume_units!r}")
        return DVHValue(gy, self.dose_units)

    def volume_constraint(self, dose, dose_units=None):
        """V(D): the volume receiving at least `dose`, in cm³.

        `dose_units` is ``'Gy'`` (the default), ``'cGy'``, or ``'%'`` of
        ``rx_dose``.
        """
        u = (dose_units or self.dose_units).lower()
        if u == "gy":
            gy = float(dose)
        elif u == "cgy":
            gy = float(dose) / 100.0
        elif u == "%":
            if not self.rx_dose:
                raise ValueError("a '%' dose needs a prescription — set .rx_dose "
                                 "or pass rx_dose= to dvhcalc()")
            gy = float(dose) / 100.0 * self.rx_dose
        else:
            raise ValueError(f"dose_units must be 'Gy', 'cGy' or '%', not {dose_units!r}")
        return DVHValue(self._c.volume_at_dose(gy), self.volume_units)

    def statistic(self, name):
        """One constraint by name — ``'D95'``, ``'D2cc'``, ``'V20Gy'``,
        ``'V100%'``, ``'Dmax'``, ``'Dmin'``, ``'Dmean'``.

        The grammar is the native engine's, the same one
        the DVH constraint grammar accepts, so a name means one thing
        across both surfaces.
        """
        text = str(name).strip()
        try:
            value = self._c.statistic(text, self.rx_dose or 0.0)
        except RuntimeError as exc:
            raise ValueError(str(exc)) from None
        # A D reads a dose, a V reads a volume; that is the whole of what the
        # name's leading letter decides on this side.
        return DVHValue(value,
                        self.dose_units if text[:1].upper() == "D" else self.volume_units)

    def __getattr__(self, name):
        # Only reached when normal lookup fails. What counts as a constraint is
        # the engine's answer, not a second grammar here; a name it rejects is an
        # ordinary attribute miss, so copy/pickle/IPython probes still miss
        # cleanly. The underscore guard keeps dunder probes off the engine.
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            return self.statistic(name)
        except ValueError as exc:
            raise AttributeError(f"{name}: {exc}") from None

    # ---- reporting ---------------------------------------------------------

    def describe(self):
        """Print the constraints a plan review reads first. Returns the dict."""
        out = {
            "name": self.name,
            "volume_cm3": self.volume,
            "covered_volume_cm3": self.covered_volume,
            "uncovered_volume_cm3": self.uncovered_volume,
            "min_gy": self.min, "max_gy": self.max, "mean_gy": self.mean,
        }
        for s in ("D100", "D98", "D95", "D50", "D2", "D2cc"):
            out[s] = float(self.statistic(s))
        print(f"Structure:  {self.name}")
        print(f"Volume:     {self.volume:.2f} cm3"
              + (f"  (covered {self.covered_volume:.2f}, "
                 f"uncovered {self.uncovered_volume:.2f})"
                 if self.uncovered_volume else ""))
        print(f"Dose:       min {self.min:.2f} / mean {self.mean:.2f} / "
              f"max {self.max:.2f} Gy")
        # A NaN here is the engine declining to answer — the structure is
        # smaller than the volume asked about — not a failed computation. Print
        # it as the absence it is, so it cannot be read as a number.
        def _fmt(v, units):
            return "—  (no reading: structure is smaller than the volume asked about)" \
                   if v != v else f"{v:.2f} {units}"
        for s in ("D100", "D98", "D95", "D50", "D2", "D2cc"):
            print(f"{s + ':':<12}{_fmt(out[s], 'Gy')}")
        if self.rx_dose:
            for s in ("V100%", "V95%", "V107%"):
                v = float(self.statistic(s))
                out[s] = v
                print(f"{s + ':':<12}{_fmt(v, 'cm3')}")
        if self.notes:
            print(f"Notes:      {self.notes}")
        return out

    def plot(self, ax=None, **kw):
        """Plot the cumulative DVH. Needs matplotlib; returns the axes."""
        import matplotlib.pyplot as plt
        if ax is None:
            ax = plt.gca()
        ax.plot(self.bincenters, self.cumulative, label=self.name or None, **kw)
        ax.set_xlabel(f"Dose ({self.dose_units})")
        ax.set_ylabel(f"Volume ({self.volume_units})")
        return ax

    def compare(self, other, stats=("D100", "D98", "D95", "D50", "D2", "D2cc")):
        """``{stat: (self, other, other - self)}`` in Gy, for two plans or two
        rasterisation settings of the same structure."""
        return {s: (float(self.statistic(s)), float(other.statistic(s)),
                    float(other.statistic(s)) - float(self.statistic(s)))
                for s in stats}

    def __repr__(self):
        return (f"<DVH {self.name!r} volume={self.volume:.2f}cm3 "
                f"min/mean/max={self.min:g}/{self.mean:g}/{self.max:g}Gy>")


#: Kept as the name this class was introduced under.
ComputedDVH = DVH


def dvhcalc(structure, dose, roi, limit=None, calculate_full_volume=True,
            thickness=None, samples_per_axis=None,
            require_full_dose_coverage=False, rx_dose=None):
    """Compute the DVH of `roi` from an RT Structure Set + RT Dose file pair.

    The rasterisation, patient-space dose interpolation, histogram and
    statistics all run in the native RT engine. Results follow the standard
    cumulative-DVH definition. ``limit`` truncates the returned histogram in
    cGy; full-volume integration and dose statistics remain untruncated.

    ``samples_per_axis`` (1..32) sets the fractional-coverage sampling density;
    the default is the engine's. ``require_full_dose_coverage`` refuses a
    structure the dose grid does not fully reach rather than reporting it with
    a coverage note.

    ``rx_dose`` is the prescription in Gy, or an :class:`RTPlan` / a path to
    the RT Plan to take it from — it is what the ``%`` dose forms
    (``dvh.V100%``) are relative to.
    """
    if rx_dose is not None and not isinstance(rx_dose, (int, float)):
        plan = rx_dose if isinstance(rx_dose, RTPlan) else read_rtplan(rx_dose)
        rx_dose = plan.prescription_dose
    curve = _core.compute_dvh(str(structure), str(dose), int(roi),
                              limit=int(limit) if limit else 0,
                              calculate_full_volume=calculate_full_volume,
                              thickness=float(thickness) if thickness else 0.0,
                              samples_per_axis=int(samples_per_axis or 0),
                              require_full_dose_coverage=require_full_dose_coverage)
    return DVH(curve, rx_dose=rx_dose)


def roi_mask(rtstruct, reference, roi, *, transform=None):
    """One ROI of an RT Structure Set as a boolean volume on `reference`'s grid.

    reference: the image series the mask should land on — a directory, a file,
        or a list of instance paths. Its slices are the target planes, in the
        order :func:`write_seg` and the other authoring writers use, so a mask
        and a Segmentation authored from the same series share one grid.
    roi: the ROI **Number** — the identifier :func:`dvhcalc` also takes.
    transform: the ``(4, 4)`` RTSTRUCT-frame → reference-frame affine, e.g.
        ``pydcm.read_registration(reg).transform(struct_for, series_for)``.
        Required when the two Frames of Reference differ; omitting it there is
        refused rather than treated as identity, because identity where a
        registration was meant puts the mask in the wrong place with nothing to
        show that it did.

    Returns ``(mask, meta)``. ``mask`` is ``(planes, rows, cols)`` bool.
    ``meta`` carries ``roi_name``, ``set_voxels`` and ``findings``.

    **One ROI per call, deliberately.** Twenty ROIs over a 512×512×300 series is
    1.5 GB of masks returned together, and a caller who wants a label volume has
    to decide what happens where two ROIs claim one voxel — a decision only that
    caller can make.

    A pixel is in when its centre is, and a sample exactly on a boundary is in.
    There is no sampling knob: fractional coverage — how *much* of a pixel a
    contour covers — is a different question that :func:`dvhcalc` answers
    quantitatively, and turning it back into a yes/no would need a threshold
    that nothing states.

    ``meta["findings"]`` is what the mask could not simply state:

    ``contour_on_no_plane``
        the contour is not within tolerance of any slice — ordinary when the ROI
        reaches past the series, and reported rather than silently dropped.
    ``contour_on_several_planes``
        several slices are within tolerance of one contour, so it is drawn on
        the nearest; drawing it on each would be volume counted twice.
    ``same_plane_nested``
        two ``CLOSED_PLANAR`` contours on one slice, one wholly inside the
        other. They are **unioned**, because the standard says nothing about
        composing ``CLOSED_PLANAR`` contours — a producer writing a hole that
        way predates ``CLOSEDPLANAR_XOR``, which is the encoding that states
        one. The finding is how you learn it happened.
    ``same_plane_partial_overlap``
        two contours on one slice cross without either containing the other;
        neither a union nor a hole is stated for that.

    A structure set that mixes ``CLOSEDPLANAR_XOR`` with other closed types
    raises, naming the minority contours: the XOR rule is stated over the
    complete ROI, so no composition is defined for such a document.
    """
    import numpy as np
    m = np.ascontiguousarray(transform, dtype=np.float64) if transform is not None else None
    return _core.roi_mask(str(rtstruct), _ref_paths(reference), int(roi), m)


# ── RT Structure Set — the read side of write_rtstruct ─────────────────────

class Contour:
    """One contour of an ROI.

    Attributes:
        points: ``(n, 3)`` float64 patient-LPS mm, full precision. Empty when
            the file's Contour Data was unusable — the structure set's
            ``findings`` says which, and half a contour is never exposed.
        geometric_type: the verbatim Defined Term ("CLOSED_PLANAR",
            "CLOSEDPLANAR_XOR", "POINT", ...). PS3.3 Defined Terms are
            extensible, so a value this build does not recognise still arrives
            here rather than being dropped.
        image_references: ``[{sop_class_uid, sop_instance_uid, frame_numbers}]``
            — the source images this contour was drawn on.
    """

    __slots__ = ("points", "geometric_type", "number_of_points",
                 "image_references", "image_reference_sequence_present")

    def __init__(self, d):
        self.points = d["points"]
        self.geometric_type = d["geometric_type"]
        self.number_of_points = d["number_of_points"]
        self.image_references = d["image_references"]
        self.image_reference_sequence_present = d["image_reference_sequence_present"]

    @property
    def usable(self):
        """The file's Contour Data was readable, so :attr:`points` is the contour.

        False means the value was present and unusable — the structure set's
        ``findings`` names it — which is not the same as the contour being
        absent, and is why this object still exists.
        """
        return len(self.points) > 0

    def __len__(self):
        return len(self.points)

    def __bool__(self):
        # A Contour is an object, not a count. Without this, __len__ would make
        # an unusable contour falsy and `if contour:` would read as "there is no
        # contour" when it means "this contour's data could not be read" — the
        # exact distinction the structure-set model exists to keep. Ask
        # `contour.usable` for that.
        return True

    def __repr__(self):
        return (f"<Contour {self.geometric_type or '?'} {len(self.points)} points"
                + ("" if self.usable else ", unusable") + ">")


class ROI:
    """One ROI of a structure set: its identity, its semantics, its contours.

    ``interpreted_type`` and ``generation_algorithm`` are verbatim Defined
    Terms; ``rgb`` is the recommended display colour or ``None``.
    """

    __slots__ = ("number", "name", "description", "frame_of_reference_uid",
                 "generation_algorithm", "generation_description", "rgb",
                 "interpreted_type", "interpreter", "contours")

    def __init__(self, d):
        for k in ("number", "name", "description", "frame_of_reference_uid",
                  "generation_algorithm", "generation_description", "rgb",
                  "interpreted_type", "interpreter"):
            setattr(self, k, d[k])
        self.contours = [Contour(c) for c in d["contours"]]

    def __repr__(self):
        return (f"<ROI {self.number} {self.name!r} "
                f"{self.interpreted_type or '—'} {len(self.contours)} contours>")


class StructureSet:
    """An RT Structure Set, read.

    Attributes:
        label: Structure Set Label.
        rois: one :class:`ROI` per Structure Set ROI Sequence item, with the
            contour and observation items that resolved to it folded in. Two
            contour items bound to the same ROI describe ONE merged ROI — that
            is what DICOM means by it — so they merge here.
        unbound_contours: contour items whose ROI reference was missing,
            malformed, unknown or ambiguous. They are kept rather than dropped,
            because a structure set that lost contours silently is worse than
            one that says it has some it cannot place.
        findings: RT-domain defects, ``[{code, path, message}]``. These describe
            usability, not general DICOM conformance — for that use
            :func:`pydcm.validate`.
        \\*_sequence_present: absent versus present-but-empty, which a reader
            deciding whether the object is usable has to tell apart.
    """

    def __init__(self, d):
        self.label = d["label"]
        self.rois = [ROI(r) for r in d["rois"]]
        self.unbound_contours = d["unbound_contours"]
        self.findings = d["findings"]
        self.structure_set_roi_sequence_present = d["structure_set_roi_sequence_present"]
        self.roi_contour_sequence_present = d["roi_contour_sequence_present"]
        self.roi_observation_sequence_present = d["roi_observation_sequence_present"]

    @property
    def roi_names(self):
        return [r.name for r in self.rois]

    def __len__(self):
        return len(self.rois)

    def __iter__(self):
        return iter(self.rois)

    def __getitem__(self, key):
        """An ROI by name, by ROI Number, or by position.

        A string is a name and an int is the ROI **Number** — the identifier the
        rest of the RT line uses (``dvhcalc`` takes it) — not a list index. Use
        ``ss.rois[i]`` for positional access.
        """
        if isinstance(key, str):
            for r in self.rois:
                if r.name == key:
                    return r
            raise KeyError(f"no ROI named {key!r}; have {self.roi_names}")
        for r in self.rois:
            if r.number == key:
                return r
        raise KeyError(f"no ROI with number {key}; have "
                       f"{[r.number for r in self.rois]}")

    def __repr__(self):
        return (f"<StructureSet {self.label!r} {len(self.rois)} ROIs"
                + (f", {len(self.findings)} findings" if self.findings else "") + ">")


def read_rtstruct(path, *, coordinates=True):
    """Read an RT Structure Set (SOP Class …481.3) into a :class:`StructureSet`.

    The read side of :func:`write_rtstruct`. Contour points come back as
    ``(n, 3)`` float64 patient-LPS millimetres at full precision.

    ``coordinates=False`` validates Contour Data without retaining it — for
    inspecting a large structure set's ROI table and findings without paying for
    the coordinate vectors.

    RT-domain defects land in :attr:`StructureSet.findings` instead of raising:
    a structure set with one unusable contour is still worth reading, and which
    part failed is the useful answer. A structural decode failure does raise.
    """
    return StructureSet(_core.read_rtstruct(str(path), bool(coordinates)))


# ── RT Plan — the prescription a per-cent is a per-cent OF ──────────────────

_REF_TYPE = {0: "TARGET", 1: "ORGAN_AT_RISK", 255: "UNKNOWN"}
_STRUCT_TYPE = {0: "POINT", 1: "VOLUME", 2: "COORDINATES", 3: "SITE", 255: "UNKNOWN"}


class DoseReference:
    """One Item of the plan's Dose Reference Sequence (300A,0010).

    A dose that is `None` was absent; one that is absent *and*
    ``target_prescription_dose_invalid`` was present and unreadable. Those are
    different facts and neither of them is zero.
    """

    __slots__ = ("number", "reference_type", "structure_type",
                 "target_prescription_dose", "target_prescription_dose_invalid",
                 "delivery_maximum_dose", "organ_at_risk_maximum_dose")

    def __init__(self, d):
        self.number = d["number"]
        self.reference_type = _REF_TYPE.get(d["reference_type"], "UNKNOWN")
        self.structure_type = _STRUCT_TYPE.get(d["structure_type"], "UNKNOWN")
        self.target_prescription_dose = d["target_prescription_dose_gy"]
        self.target_prescription_dose_invalid = d["target_prescription_dose_invalid"]
        self.delivery_maximum_dose = d["delivery_maximum_dose_gy"]
        self.organ_at_risk_maximum_dose = d["organ_at_risk_maximum_dose_gy"]

    def __repr__(self):
        rx = self.target_prescription_dose
        return (f"<DoseReference {self.reference_type}/{self.structure_type} "
                f"rx={'—' if rx is None else f'{rx:g}Gy'}>")


class RTPlan:
    """An RT Plan's prescription and fractionation.

    Attributes:
        prescription_dose: the largest Target Prescription Dose among TARGET
            references, in Gy — or ``None``. Largest rather than first because a
            plan with a boost carries more than one target and the reader means
            the plan's prescription. ``None`` is never substituted with the dose
            grid's maximum: that answers a different question.
        fractions_planned: (300A,0078) of the first Fraction Group, or ``None``.
        fraction_groups: how many groups the plan carries. More than one is not
            summed — a reader that did would be inventing a prescription.
        dose_references: the :class:`DoseReference` items.
        incomplete: an item could not be recorded, so the plan is short by
            omission rather than by content.
    """

    def __init__(self, d):
        self.prescription_dose = d["prescription_dose_gy"]
        self.fractions_planned = d["fractions_planned"]
        self.fractions_planned_invalid = d["fractions_planned_invalid"]
        self.fraction_groups = d["fraction_groups"]
        self.incomplete = d["incomplete"]
        self.dose_references = [DoseReference(x) for x in d["dose_references"]]

    @property
    def dose_per_fraction(self):
        """Prescription ÷ fractions, in Gy — ``None`` unless the plan states both
        and carries exactly one Fraction Group."""
        if (self.prescription_dose is None or not self.fractions_planned
                or self.fraction_groups != 1):
            return None
        return self.prescription_dose / self.fractions_planned

    def __repr__(self):
        rx = self.prescription_dose
        n = self.fractions_planned
        return (f"<RTPlan rx={'—' if rx is None else f'{rx:g}Gy'}"
                f"{'' if n is None else f' in {n} fx'}>")


def read_rtplan(path):
    """Read an RT Plan (SOP Class …481.5) into an :class:`RTPlan`.

    Beam geometry is deliberately not read — this is the prescription, which is
    the fact a dose display cannot get anywhere else.
    """
    return RTPlan(_core.read_rtplan(str(path)))



def write_rtdose(dose, *, affine=None, origin=None, orientation=(1, 0, 0, 0, 1, 0),
                 spacing=None, grid_frame_offsets=None,
                 dose_units="GY", dose_type="PHYSICAL", dose_summation_type="PLAN",
                 ref_plan_uid=None, reference=None,
                 patient_name=None, patient_id=None,
                 study_uid=None, study_date=None, series_uid=None,
                 frame_of_reference_uid=None,
                 scaling=None, bits=32, output=None,
                 sop_instance_uid="", content_date="", content_time=""):
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

    ``sop_instance_uid`` / ``content_date`` (YYYYMMDD) / ``content_time``
    (HHMMSS): this object's own identity. Left empty, the SOP Instance
    UID is derived deterministically from the series, so two built for one
    series carry the same one — right for a single self-contained export, a
    DICOM global-uniqueness violation for a producer that mints many.
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
        origin = m[:3, 3]
        # The axes, the spacings and the normal come from the geometry engine,
        # which also decides what a valid plane IS — including that the in-plane
        # axes are orthogonal, which an Image Orientation (Patient) requires and
        # which this wrapper used to accept without.
        basis = _core.plane_basis(list(map(float, origin)), list(map(float, col0)),
                                  list(map(float, col1)), 1, 1)
        ps_col = basis["col_spacing_mm"]
        ps_row = basis["row_spacing_mm"]
        orientation = (*basis["col_unit"], *basis["row_unit"])
        if grid_frame_offsets is None:
            normal = np.asarray(basis["normal"], dtype=np.float64)
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
        scaling=float(scaling) if scaling else 0.0, bits=int(bits),
        sop_instance_uid=sop_instance_uid or "", content_date=content_date or "",
        content_time=content_time or "")
    if output is None:
        return blob
    import pathlib
    p = pathlib.Path(output)
    p.write_bytes(blob)
    return p


__all__ = ["DoseGrid", "read_rtdose", "DVH", "DVHValue", "ComputedDVH",
           "dvhcalc", "write_rtdose", "write_rtstruct", "read_rtstruct",
           "roi_mask", "dose_at",
           "StructureSet", "ROI", "Contour",
           "RTPlan", "DoseReference", "read_rtplan"]


def dose_at(path, points):
    """Dose at patient coordinates, in Gy.

    The quantitative counterpart to :func:`dvhcalc`: what a prescription point
    or a measured location received, rather than what a structure received.

    path: an RT Dose file.
    points: ``(N, 3)`` patient coordinates **in the dose object's own Frame
        of Reference**. Points in another frame are transformed first —
        with :meth:`pydcm.registration.Registration.transform`, say. Nothing
        here treats a mismatched frame as identity, because doing so would
        sample the right grid at the wrong place and report a plausible
        number.

    Returns ``(values, inside)`` — ``(N,)`` float64 Gy and ``(N,)`` bool. A point
    the grid does not cover has ``inside=False`` and a value of 0; a point
    where the dose really is zero has ``inside=True``. They are the same
    number and mean opposite things, which is why the flag is separate
    rather than a sentinel dose.

    Isodose *lines* are not here. They are a rendering question and live in the
    viewer; this is the readout.
    """
    import numpy as np

    pts = np.ascontiguousarray(points, dtype=np.float64)
    if pts.ndim == 1:
        pts = pts.reshape(1, -1)
    values, inside = _core.dose_at(str(path), pts.reshape(-1))
    return values, inside.astype(bool)
