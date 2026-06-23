"""Ophthalmic Visual Field static perimetry (PS3.3 Supplement 146, SOP Class
``1.2.840.10008.5.1.4.1.1.80.1``).

Manage OPV DICOM: parse a study, flatten it to pandas / JSON, and check
standard conformance.

No DICOM logic lives here. The semantic extraction is the native content engine (exposed as
:func:`pydcm.content`); compliance is the native IOD/module conformance judge
(exposed as :func:`pydcm.iod_validate`). This module is only the pandas/JSON
ergonomics on top.

Example
-------
>>> import pydcm
>>> vf = pydcm.opv.read_dicom("vf.dcm")
>>> vf.pointwise_to_pandas()           # one row per stimulus location
>>> vf.to_pandas()                     # one row of study-level fields
>>> vf.check_dicom_compliance()        # Supplement 146 IOD Type-1/2 findings
>>> opvset, errors = pydcm.opv.read_dicom_directory("study_dir/")
"""

from __future__ import annotations

import glob
import json
import os

from . import _core

#: Ophthalmic Visual Field Static Perimetry Measurements Storage.
OPV_SOP_CLASS_UID = "1.2.840.10008.5.1.4.1.1.80.1"

# Study-level identifiers carried onto every flattened row.
_ID_KEYS = ("sop_instance_uid", "series_instance_uid", "study_instance_uid",
            "modality", "laterality")
# Nested study-level groups flattened into to_pandas() columns.
_GROUP_KEYS = ("test_parameters", "reliability", "global_results")


def _flatten(value, prefix, row):
    """Flatten a scalar / code-object / nested dict into ``row`` columns."""
    if isinstance(value, dict):
        # A coded entry {value, scheme, meaning} collapses to its meaning|value;
        # any other dict recurses with dotted keys.
        if set(value) <= {"value", "scheme", "meaning"}:
            row[prefix] = value.get("meaning") or value.get("value")
        else:
            for k, v in value.items():
                _flatten(v, f"{prefix}.{k}" if prefix else k, row)
    elif isinstance(value, list):
        row[prefix] = json.dumps(value)
    else:
        row[prefix] = value


class OPVDicom:
    """A single Ophthalmic Visual Field DICOM file."""

    def __init__(self, path):
        self.path = os.fspath(path)
        j = _core.content_json(self.path, False, False)
        content = json.loads(j) if j else None
        if not content or content.get("type") != "ophthalmic_visual_field":
            raise ValueError(
                f"not an Ophthalmic Visual Field (static perimetry) object: {self.path}")
        #: The native semantic content (dict) — the single source for every view.
        self.content = content

    # -- accessors ---------------------------------------------------------
    def to_dict(self):
        """The full semantic content as a nested dict."""
        return self.content

    @property
    def test_points(self):
        """The per-stimulus records (list of dicts)."""
        return self.content.get("test_points", [])

    # -- tabular / JSON views ----------------------------------------------
    def to_pandas(self):
        """Study-level fields as a single-row ``DataFrame`` (test points excluded)."""
        import pandas as pd

        row = {}
        for k in _ID_KEYS:
            if k in self.content:
                row[k] = self.content[k]
        for grp in _GROUP_KEYS:
            _flatten(self.content.get(grp, {}), grp, row)
        # surface coded global indices compactly
        for grp in ("reliability_global_indices", "results_global_indices"):
            if self.content.get(grp):
                row[grp] = json.dumps(self.content[grp])
        return pd.DataFrame([row])

    def pointwise_to_pandas(self):
        """One row per test point, each tagged with the study identifiers."""
        import pandas as pd

        ids = {k: self.content[k] for k in _ID_KEYS if k in self.content}
        rows = [{**ids, "point_index": i, **pt}
                for i, pt in enumerate(self.test_points)]
        return pd.DataFrame(rows)

    def pointwise_to_nested_json(self):
        """Study identifiers + the nested per-point records (JSON-ready dict)."""
        out = {k: self.content[k] for k in _ID_KEYS if k in self.content}
        out["test_points"] = self.test_points
        return out

    # -- conformance (native IOD engine) -----------------------------------
    def check_dicom_compliance(self):
        """Sup-146 IOD / module conformance findings (list of dicts).

        Reuses the native IOD conformance judge — every mandatory module's
        Type-1 / Type-2 attribute must be present (Type-1 also non-empty),
        descending into present sequences. An empty list means conformant at the
        IOD level. Stricter than a flat tag checklist: it is per-SOP-Class and
        nested-sequence aware.
        """
        return _core.iod_validate(self.path)


class OPVDicomSet:
    """A batch of :class:`OPVDicom` files with set-level aggregation."""

    def __init__(self, dicoms=None, errors=None):
        self.dicoms = list(dicoms or [])
        #: ``[(path, error_message), ...]`` for files that failed to parse.
        self.errors = list(errors or [])

    def __len__(self):
        return len(self.dicoms)

    def __iter__(self):
        return iter(self.dicoms)

    def to_pandas(self):
        """Study-level rows for every file, concatenated."""
        import pandas as pd

        if not self.dicoms:
            return pd.DataFrame()
        return pd.concat([d.to_pandas() for d in self.dicoms], ignore_index=True)

    def pointwise_to_pandas(self):
        """Pointwise rows for every file, concatenated."""
        import pandas as pd

        if not self.dicoms:
            return pd.DataFrame()
        return pd.concat([d.pointwise_to_pandas() for d in self.dicoms],
                         ignore_index=True)

    def pointwise_to_nested_json(self):
        """Nested per-file JSON records (list)."""
        return [d.pointwise_to_nested_json() for d in self.dicoms]

    def check_dicom_compliance(self):
        """``{path: [findings, ...]}`` across the set."""
        return {d.path: d.check_dicom_compliance() for d in self.dicoms}


def read_dicom(path):
    """Load a single OPV DICOM file → :class:`OPVDicom`."""
    return OPVDicom(path)


def read_dicom_directory(dicom_directory, file_extension="dcm"):
    """Load every ``*.<file_extension>`` OPV file in a directory.

    Returns ``(OPVDicomSet, errors)`` where ``errors`` is a list of
    ``(path, message)`` for files that were not OPV objects or failed to parse.
    """
    dicoms, errors = [], []
    pattern = os.path.join(os.fspath(dicom_directory), f"*.{file_extension}")
    for p in sorted(glob.glob(pattern)):
        try:
            dicoms.append(OPVDicom(p))
        except Exception as exc:          # noqa: BLE001 — collect, don't abort the batch
            errors.append((p, str(exc)))
    return OPVDicomSet(dicoms, errors), errors


def read_visual_field(path):
    """The native semantic content (nested dict) of an OPV file, or ``None``.

    The low-level reader behind :class:`OPVDicom`; mirrors ``pydcm.read_seg`` /
    ``pydcm.read_report`` in returning the raw structured content.
    """
    j = _core.content_json(os.fspath(path), False, False)
    content = json.loads(j) if j else None
    if content and content.get("type") == "ophthalmic_visual_field":
        return content
    return None

__all__ = ['OPVDicom', 'OPVDicomSet', 'read_dicom', 'read_dicom_directory',
           'read_visual_field']
