# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Fuli Wu
# NOT a medical device — not for clinical or diagnostic use.

"""pydcm — tumour response criteria (``pydcm.recist``).

Six published criteria over the same shape of input: a list of lesion
measurements at baseline, the same lesions at the current timepoint, and the
nadir — the smallest sum seen so far, which is what progression is measured
from, not the baseline.

    >>> import pydcm.recist as recist
    >>> baseline = [{"longest_mm": 32.0}, {"longest_mm": 18.0}]
    >>> current  = [{"longest_mm": 20.0}, {"longest_mm": 12.0}]
    >>> recist.evaluate(baseline, current)["response"]
    'PR'

Which criterion applies is a clinical decision, not a technical one, so nothing
here picks for you. What each one sums differs, and the field names say so:

===============  ==============================  ==========================
Criterion        Field                           Summed as
===============  ==============================  ==========================
RECIST 1.1       ``longest_mm``                  sum of longest diameters
iRECIST          ``longest_mm``                  as RECIST 1.1, + confirmation
mRECIST          ``viable_diam_mm``              arterially enhancing only
Cheson/Lugano    ``longest_mm``, ``perpendicular_mm``  sum of the products
RANO             ``longest_mm``, ``perpendicular_mm``  sum of the products
PCWG3            new bone lesion counts, PSA     the 2+2 rule
===============  ==============================  ==========================

Lymph nodes are not lesions for the complete-response check — a node is normal
below 10 mm short axis rather than absent — so mark them ``is_lymph_node=True``
and give ``short_axis_mm``.

Every threshold and tie-break lives in the native engine, validated against
Eisenhauer (EJC 2009), Seymour (Lancet Oncol 2017), Lencioni & Llovet (Semin
Liver Dis 2010), Cheson (JCO 2014), Wen (JCO 2010) and Scher (JCO 2016). The
web runtime calls the same functions, so the two products cannot come to
disagree about a category.

Measurement helpers turn a segmentation into the diameters the criteria want:
:func:`feret` measures one mask, :func:`feret_volume` finds the slice a lesion
is longest on, and :func:`volume` sums per-slice pixel counts. They take the
masks ``read_seg`` and ``roi_mask`` already produce, so a lesion contoured once
is not drawn again to be measured.
"""
from __future__ import annotations

import numpy as np

from . import _core

__all__ = [
    "evaluate", "evaluate_irecist", "evaluate_mrecist", "evaluate_cheson",
    "evaluate_rano", "evaluate_pcwg3",
    "feret", "feret_volume", "volume",
    "pixel_to_world", "match_lesion", "nearest_slice",
    "CRITERIA",
]

#: The criterion names :func:`evaluate` accepts, and the function each maps to.
CRITERIA = ("RECIST1.1", "iRECIST", "mRECIST", "Cheson", "RANO", "PCWG3")


def evaluate(baseline, current, *, nadir_sld=0.0,
             new_lesion=False, nt_progression=False):
    """RECIST 1.1 target-lesion response.

    baseline: list of lesion dicts — ``longest_mm``, optionally
        ``short_axis_mm``, ``is_lymph_node``, ``status``.
    current: the SAME lesions in the SAME order at this timepoint. A length
        mismatch is an error, not something to broadcast away.
    nadir_sld: smallest sum of longest diameters recorded so far, in mm.
        Progression is measured from the nadir; leaving it 0 makes the
        baseline the reference, which is only right at the second timepoint.
    new_lesion: any new lesion at all is progression, whatever the sum did.
    nt_progression: unequivocal progression of a non-target lesion.

    Returns ``{"response": "CR"|"PR"|"SD"|"PD"|"NE", "baseline_sld": float,
    "current_sld": float}``, both sums in mm.

    ``status`` is one of ``"measured"`` (the default), ``"too_small"``,
    ``"not_evaluable"``, ``"absent"``, ``"present"``, ``"unequivocal_pd"``.
    """
    return _core.recist_evaluate(list(baseline), list(current), float(nadir_sld),
                                 bool(new_lesion), bool(nt_progression))


def evaluate_irecist(baseline, current, *, nadir_sld=0.0, new_lesion=False,
                     nt_progression=False, prev_response="NE", prev_was_iupd=False):
    """iRECIST response, for immunotherapy.

    iRECIST exists because a tumour can enlarge under immunotherapy before it
    responds, so progression is *unconfirmed* (iUPD) until a follow-up scan
    confirms it (iCPD). That makes the previous timepoint part of the input:

    prev_response: the **RECIST 1.1** response at the previous timepoint —
        ``"CR"``, ``"PR"``, ``"SD"``, ``"PD"`` or ``"NE"``. Required
        because this call has no access to those images to recompute it.
    prev_was_iupd: whether the previous iRECIST read was iUPD, which is
        what turns this timepoint's progression into iCPD.

    Returns ``{"response": "iCR"|"iPR"|"iSD"|"iUPD"|"iCPD"|"iNE", ...}``.
    """
    return _core.irecist_evaluate(list(baseline), list(current), float(nadir_sld),
                                  bool(new_lesion), bool(nt_progression),
                                  str(prev_response), bool(prev_was_iupd))


def evaluate_mrecist(baseline, current, *, nadir_sum=0.0, new_lesion=False):
    """mRECIST for hepatocellular carcinoma.

    Each measurement gives ``viable_diam_mm`` — the arterially enhancing
    diameter. That is the point of mRECIST: a treated lesion can keep its size
    while the viable part of it disappears, which RECIST 1.1 would read as
    stable disease and mRECIST reads as a complete response. Passing the whole
    lesion diameter here is a category error and is refused.
    """
    return _core.mrecist_evaluate(list(baseline), list(current), float(nadir_sum),
                                  bool(new_lesion))


def evaluate_cheson(baseline, current, *, nadir_spd=0.0, new_lesion=False):
    """Cheson/Lugano response for lymphoma.

    Sums the products of perpendicular diameters (SPD), so each measurement
    needs both ``longest_mm`` and ``perpendicular_mm``. Nodal disease is the
    usual case: mark nodes ``is_lymph_node=True``.
    """
    return _core.cheson_evaluate(list(baseline), list(current), float(nadir_spd),
                                 bool(new_lesion))


def evaluate_rano(baseline, current, *, nadir_spd=0.0, new_lesion=False,
                  non_enhancing="unknown", on_steroids=False):
    """RANO response for glioma.

    non_enhancing: ``"stable"``, ``"improved"``, ``"increased"`` or
        ``"unknown"``. The non-enhancing (T2/FLAIR) component is part of
        the criterion, not context — a significant increase is progression
        even when the enhancing sum fell.
    on_steroids: complete response requires the patient to be off steroids.
        An enhancing tumour that vanished under dexamethasone is not a CR.
    """
    return _core.rano_evaluate(list(baseline), list(current), float(nadir_spd),
                               bool(new_lesion), str(non_enhancing), bool(on_steroids))


def evaluate_pcwg3(*, new_lesions_scan1=0, new_lesions_scan2=None,
                   prev_was_pending=False, psa_current=None, psa_nadir=None):
    """PCWG3 bone and PSA progression for prostate cancer.

    Bone progression follows the 2+2 rule: two or more new lesions on the first
    post-treatment scan, confirmed by two or more further new lesions on the
    next. Between the two the answer is ``PENDING``, which is a real state and
    not a missing one — pass ``new_lesions_scan2=None`` while the confirmation
    scan has not happened.

    PSA is optional; bone progression is assessable on imaging alone. When
    given, progression is measured against ``psa_nadir`` — the lowest value
    seen, not the baseline.

    Returns ``{"bone": ...}``, plus ``"psa"`` when a PSA pair was given.
    """
    psa_given = psa_current is not None and psa_nadir is not None
    return _core.pcwg3_evaluate(
        int(new_lesions_scan1),
        -1 if new_lesions_scan2 is None else int(new_lesions_scan2),
        bool(prev_was_pending),
        float(psa_current or 0.0), float(psa_nadir or 0.0), psa_given)


# ---------------------------------------------------------------------------
# Measuring: from a segmentation to the diameters the criteria want
# ---------------------------------------------------------------------------

def _binarize(mask):
    """Occupancy to a 0/1 uint8 plane, at the half-voxel threshold.

    ``read_seg(masks=True)`` returns float32 occupancy in [0, 1] for a
    FRACTIONAL segmentation, and casting that to uint8 truncates — a lesion at
    0.7 occupancy would count as empty and the measurement would come back
    zero. Half a voxel is also the threshold the ``dcmrecist`` CLI uses, and it
    is the only one that makes the binary and fractional encodings of one lesion
    measure the same. A 0/1 or labelmap integer array is unaffected: no integer
    lies in (0, 0.5].
    """
    return np.ascontiguousarray(np.asarray(mask) > 0.5, dtype=np.uint8)


def feret(mask, spacing):
    """Maximum Feret diameter of one 2-D binary mask.

    mask: 2-D array. What ``read_seg(masks=True)`` and one plane of
        :func:`pydcm.rt.roi_mask` already give you — including a FRACTIONAL
        segmentation's float occupancy, which is thresholded at half a
        voxel, the same threshold the ``dcmrecist`` CLI uses.
    spacing: ``(row_mm, col_mm)`` pixel spacing — PixelSpacing order.

    Returns ``{"longest_mm", "short_axis_mm", "longest_endpoints",
    "short_endpoints"}``, endpoints as ``(x1, y1, x2, y2)`` in pixels, or
    ``None`` for an empty mask — no lesion, rather than one of length zero.
    """
    m = _binarize(mask)
    row_mm, col_mm = float(spacing[0]), float(spacing[1])
    # The engine takes (spacing_x, spacing_y) = (column, row); PixelSpacing is
    # written (row, column). Swapping them silently transposes every diameter
    # on a non-square pixel, so the order is converted here, once.
    return _core.recist_feret(m, col_mm, row_mm)


def feret_volume(mask, spacing):
    """The slice a lesion is longest on, measured across a 3-D mask.

    RECIST measures a lesion on the slice where it is largest, so a volumetric
    segmentation has to be reduced to that one slice before it becomes a
    diameter. This does that reduction and nothing else.

    mask: 3-D array ``(planes, rows, cols)``, nonzero = lesion.
    spacing: ``(row_mm, col_mm)`` in-plane spacing.

    Returns the :func:`feret` result for the winning slice, with ``"slice"``
    added, or ``None`` when no slice holds any lesion.
    """
    m = _binarize(mask)
    if m.ndim != 3:
        raise ValueError("feret_volume: mask must be 3-D (planes, rows, cols)")
    best = None
    for k in range(m.shape[0]):
        # Skipping empty planes is not just a speed matter: the engine reports
        # an empty mask as None, and a plane the lesion does not reach is not a
        # candidate for the longest diameter.
        if not m[k].any():
            continue
        f = feret(m[k], spacing)
        if f is not None and (best is None or f["longest_mm"] > best["longest_mm"]):
            best = dict(f, slice=k)
    return best


def volume(mask=None, *, slice_pixel_counts=None, pixel_area_mm2=None,
           spacing=None, slice_spacing_mm=None):
    """Lesion volume in mm³.

    Give either a 3-D ``mask`` with ``spacing`` and ``slice_spacing_mm``, or
    the per-slice ``slice_pixel_counts`` directly with ``pixel_area_mm2`` — the
    second form is for a caller who already counted, e.g. while streaming.
    """
    if slice_spacing_mm is None:
        raise ValueError("volume: slice_spacing_mm is required")
    # Two sources for one answer is a caller who lost track of which one it
    # meant; silently preferring either is the wrong repair.
    if mask is not None and slice_pixel_counts is not None:
        raise ValueError("volume: pass mask or slice_pixel_counts, not both")
    if slice_pixel_counts is None:
        if mask is None:
            raise ValueError("volume: pass either mask or slice_pixel_counts")
        m = _binarize(mask)
        if m.ndim != 3:
            raise ValueError("volume: mask must be 3-D (planes, rows, cols)")
        counts = m.reshape(m.shape[0], -1).sum(axis=1)
        if pixel_area_mm2 is None:
            if spacing is None:
                raise ValueError("volume: pass spacing or pixel_area_mm2")
            pixel_area_mm2 = float(spacing[0]) * float(spacing[1])
    else:
        counts = np.asarray(slice_pixel_counts)
        if pixel_area_mm2 is None:
            raise ValueError("volume: slice_pixel_counts needs pixel_area_mm2")
    counts = np.ascontiguousarray(counts, dtype=np.uint32)
    return _core.recist_volume(counts, float(pixel_area_mm2), float(slice_spacing_mm))


# ---------------------------------------------------------------------------
# Following one lesion across studies
# ---------------------------------------------------------------------------

def pixel_to_world(origin, orientation, spacing, px, py):
    """One pixel coordinate to patient coordinates (mm).

    A lesion recorded as a pixel on a slice cannot be found again in next
    month's study — the slice index and the pixel grid both move. Recorded in
    patient coordinates it can.
    """
    return _core.recist_pixel_to_world(
        np.ascontiguousarray(origin, dtype=np.float64),
        np.ascontiguousarray(orientation, dtype=np.float64),
        np.ascontiguousarray(spacing, dtype=np.float64),
        float(px), float(py))


def match_lesion(query, candidates, max_distance_mm=15.0):
    """Nearest prior lesion to a world coordinate.

    Returns ``(index, distance_mm)``, or ``(None, None)`` when nothing lies
    within ``max_distance_mm`` — which is exactly what a genuinely new lesion
    looks like, so it is an answer rather than a failure.
    """
    return _core.recist_match_lesion(
        np.ascontiguousarray(query, dtype=np.float64),
        np.ascontiguousarray(candidates, dtype=np.float64).reshape(-1),
        float(max_distance_mm))


def nearest_slice(world, slice_origins, normal):
    """Nearest slice to a world coordinate -> ``(index, signed_distance_mm)``."""
    return _core.recist_nearest_slice(
        np.ascontiguousarray(world, dtype=np.float64),
        np.ascontiguousarray(slice_origins, dtype=np.float64).reshape(-1),
        np.ascontiguousarray(normal, dtype=np.float64))
