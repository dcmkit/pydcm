# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Fuli Wu
# NOT a medical device — not for clinical or diagnostic use.

"""pydcm — Spatial Registration reader (``pydcm.registration``).

A registration object says how one Frame of Reference sits inside another:
where the PET is relative to the CT, where today's MR is relative to last
month's. Without one, a fusion or a contour propagation can only work between
series that already share a Frame of Reference — which is not most of what a
reader wants to compare.

Deformable Spatial Registration (66.3) carries a deformation grid rather than a
matrix; it is a different IOD and is refused here rather than read as if it were
this one.
"""
from __future__ import annotations

import numpy as np

from . import _core

__all__ = ["Registration", "RegistrationItem", "read_registration",
           "write_registration"]


class RegistrationItem:
    """One Item of the Registration Sequence.

    Attributes:
        frame_of_reference_uid: the frame this item's matrix maps **from**.
        matrix: ``(4, 4)`` float64 mapping that frame **into** the registration
            object's own frame, in the ``M @ [x, y, z, 1]`` convention. ``None``
            when the file carried no usable matrix.
        matrix_invalid: a matrix was present and cannot be used — a Decimal
            String that would not convert, the wrong number of values, or a
            bottom row that is not ``[0, 0, 0, 1]``. Present-and-unusable is not
            the same as absent, and neither is identity.
        matrix_type: the producer's claim — ``"RIGID"``, ``"RIGID_SCALE"``,
            ``"AFFINE"`` or ``"UNKNOWN"``. All three are affine; the distinction
            is what the producer says it did, which a consumer may want to trust
            differently.
        matrix_count: how many matrices were composed into :attr:`matrix`. The
            standard applies them in the order encoded, so this is their
            composition, not the last one seen. 1 is the ordinary case.
    """

    __slots__ = ("frame_of_reference_uid", "matrix", "matrix_invalid",
                 "matrix_type", "matrix_count")

    def __init__(self, d):
        for k in self.__slots__:
            setattr(self, k, d[k])

    def __repr__(self):
        state = ("invalid" if self.matrix_invalid
                 else ("—" if self.matrix is None else self.matrix_type))
        return f"<RegistrationItem from={self.frame_of_reference_uid!r} {state}>"


class Registration:
    """A Spatial Registration object (SOP Class …66.1).

    Attributes:
        frame_of_reference_uid: the object's OWN frame — the one every item's
            matrix maps into. An object conventionally includes an identity item
            for it, and that item is kept rather than dropped, because "these two
            frames are the same" is an answer.
        items: the :class:`RegistrationItem` entries.
        incomplete: an item could not be recorded, so the object is short by
            omission rather than by content.
    """

    def __init__(self, d, path):
        self.frame_of_reference_uid = d["frame_of_reference_uid"]
        self.items = [RegistrationItem(x) for x in d["items"]]
        self.incomplete = d["incomplete"]
        self._path = path

    @property
    def frames(self):
        """The Frames of Reference this object registers."""
        return [i.frame_of_reference_uid for i in self.items]

    def transform(self, from_uid, to_uid=None):
        """The transform from `from_uid` to `to_uid`, as ``(4, 4)`` float64.

        With `to_uid` omitted this maps into the registration object's own
        frame. With both given it is the transform a fusion actually asks for —
        neither series is necessarily the frame the object was authored in.

        Identical frames answer with the identity, so a caller need not decide
        beforehand whether a registration is involved.

        Returns ``None`` when this object does not state the transform: an
        unregistered frame, an unusable matrix, or a composition that would need
        a singular inverse. **Never a silent identity** — applying identity where
        a registration was intended puts the overlay in the wrong place with
        nothing to show that it did.
        """
        return _core.registration_transform(self._path, str(from_uid),
                                            str(to_uid) if to_uid else "")

    def __len__(self):
        return len(self.items)

    def __iter__(self):
        return iter(self.items)

    def __repr__(self):
        return (f"<Registration own={self.frame_of_reference_uid!r} "
                f"{len(self.items)} items>")


def read_registration(path):
    """Read a Spatial Registration (SOP Class …66.1) into a :class:`Registration`.

    Raises for a Deformable Spatial Registration (66.3), which carries a
    deformation grid rather than a matrix.
    """
    return Registration(_core.read_registration(str(path)), str(path))


def write_registration(items, frame_of_reference_uid, *, output=None,
                       patient_name="", patient_id="", study_uid="", study_date="",
                       series_uid="", label="", description="", creator="",
                       sop_instance_uid="", content_date="", content_time=""):
    """Author a Spatial Registration (66.1) — the inverse of :func:`read_registration`.

    items: list of dicts, one per frame this object relates.

        - ``frame_of_reference_uid`` — the frame this matrix maps **from**.
        - ``matrix`` — ``(4, 4)`` float64, the ``M @ [x, y, z, 1]``
          convention :func:`read_registration` returns. It maps that frame
          **into** ``frame_of_reference_uid`` below; that direction is what
          the whole object means and reversing it is not detectable later.
        - ``matrix_type`` — ``"RIGID"`` (default), ``"RIGID_SCALE"`` or
          ``"AFFINE"``. All three are affine; this is the producer's claim
          about its own pipeline and nothing checks it against the matrix.

    frame_of_reference_uid: **this object's own** frame — the one every
        matrix maps into, and the frame of the fixed image in a fusion.
        An identity item for it is written automatically unless ``items``
        already contains one; without that item the file parses and then
        refuses every query, because resolving A to B composes
        ``inverse(item_B) @ item_A`` and needs an item for B.

    output: path to write to. Omitted returns the bytes.

    A matrix whose bottom row is not ``[0, 0, 0, 1]``, one that is singular, a
    non-finite element, or two items naming one frame are all refused — each
    would produce a file that parses and then either refuses the question it
    exists for or answers it wrongly.
    """
    prepared = []
    for it in items:
        d = dict(it)
        d["matrix"] = np.ascontiguousarray(d["matrix"], dtype=np.float64)
        prepared.append(d)
    blob = _core.write_registration(
        prepared, str(frame_of_reference_uid),
        patient_name=patient_name, patient_id=patient_id, study_uid=study_uid,
        study_date=study_date, series_uid=series_uid, label=label,
        description=description, creator=creator,
        sop_instance_uid=sop_instance_uid, content_date=content_date,
        content_time=content_time)
    if output is None:
        return blob
    with open(output, "wb") as fh:
        fh.write(blob)
    return output
