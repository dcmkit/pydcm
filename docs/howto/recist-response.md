# Tumour response (RECIST)

`pydcm.recist` classifies tumour response from lesion measurements, and measures
those lesions off a Segmentation you already have. Six published criteria share
one shape of input — the lesions at baseline, the same lesions now, and the
**nadir**, the smallest sum seen so far, which is what progression is measured
against. Which criterion applies is a clinical choice; nothing here picks for
you. The thresholds and tie-breaks live in the shared native engine, so this and
the command-line tool cannot come to disagree about a category.

Outputs are for research and engineering only.

## Measure a lesion off a segmentation

`read_seg(masks=True)` gives per-segment occupancy `(nseg, slices, rows, cols)`
and a `meta` with the geometry. `feret_volume` finds the slice a lesion is
longest on and measures it there — the slice RECIST wants — while `volume` sums
the per-slice pixel counts:

```python
import pydcm
import numpy as np
from pydcm import recist

masks, meta = pydcm.read_seg("lesion_seg.dcm", masks=True)
spacing = meta["pixel_spacing"]                       # (row_mm, col_mm)

lesion = masks[0]                                      # first segment, (slices, rows, cols)
m = recist.feret_volume(lesion, spacing)
m["slice"], m["longest_mm"], m["short_axis_mm"]        # winning slice + diameters

# slice spacing from the actual slice positions, not from thickness
origins = np.asarray(meta["slice_origins"]).reshape(-1, 3)
dz = float(np.linalg.norm(origins[1] - origins[0]))
vol_mm3 = recist.volume(lesion, spacing=spacing, slice_spacing_mm=dz)
```

`feret_volume` returns `None` when no slice holds the lesion — no lesion, rather
than one of length zero. A FRACTIONAL segmentation's float occupancy is
thresholded at half a voxel, so a lesion at 0.7 occupancy is not truncated away;
a 0/1 or labelmap array is unaffected. To measure a single 2-D plane directly,
`recist.feret(plane, spacing)` returns `{"longest_mm", "short_axis_mm",
"longest_endpoints", "short_endpoints"}`, endpoints as `(x1, y1, x2, y2)` pixels,
or `None` for an empty mask.

`volume` also accepts counts you already have — `recist.volume(
slice_pixel_counts=[...], pixel_area_mm2=..., slice_spacing_mm=...)` — for a
caller who counted while streaming. Passing both a mask and counts is refused.

## RECIST 1.1

`evaluate` takes the baseline lesions and the **same lesions in the same order**
now. A length mismatch is an error. Each lesion is a dict with `longest_mm`, and
optionally `short_axis_mm`, `is_lymph_node`, `status`:

```python
r = recist.evaluate(
    [{"longest_mm": 32.0}, {"longest_mm": 18.0}],   # baseline
    [{"longest_mm": 20.0}, {"longest_mm": 12.0}],   # current
    nadir_sld=30.0,          # smallest sum of longest diameters seen so far, mm
)
r["response"]               # 'CR' | 'PR' | 'SD' | 'PD' | 'NE'
r["baseline_sld"], r["current_sld"]
```

`nadir_sld` is the reference for progression. Leaving it `0.0` makes the
baseline the reference, which is only right at the second timepoint — progression
is +20% **and** at least +5 mm above the nadir, and a later study has usually
seen a smaller sum than baseline. Two flags force progression regardless of the
sum:

```python
recist.evaluate(baseline, current, new_lesion=True)        # any new lesion is PD
recist.evaluate(baseline, current, nt_progression=True)    # unequivocal non-target growth is PD
```

### Lymph nodes

A node is normal below 10 mm short axis rather than absent, so it is not scored
like a lesion for the complete-response check. Mark it and give its short axis,
or a node that shrank to normal reads as residual disease:

```python
recist.evaluate(
    [{"longest_mm": 30.0, "short_axis_mm": 18.0, "is_lymph_node": True}],
    [{"longest_mm": 12.0, "short_axis_mm":  8.0, "is_lymph_node": True}],
)["response"]               # 'CR' — the node reached normal
```

`status` defaults to `"measured"`; the other values are `"too_small"`,
`"not_evaluable"`, `"absent"`, `"present"`, `"unequivocal_pd"`. An unrecognised
status is refused by name.

## The other criteria

Each sums something different, and the field names say which. iRECIST reads the
same diameters as RECIST 1.1 but does not confirm progression on one scan: a
tumour can enlarge under immunotherapy before it responds, so progression is
*unconfirmed* (iUPD) until the next scan confirms it (iCPD). That makes the
previous timepoint part of the input:

```python
first = recist.evaluate_irecist(baseline, current, nadir_sld=20.0,
                                prev_response="SD")          # RECIST 1.1 read last time
first["response"]           # 'iUPD' — progression, not yet confirmed

confirmed = recist.evaluate_irecist(baseline, current, nadir_sld=20.0,
                                    prev_response="PD", prev_was_iupd=True)
confirmed["response"]       # 'iCPD' — the prior iUPD is now confirmed
```

`prev_response` is the RECIST 1.1 response last time (`"CR"`, `"PR"`, `"SD"`,
`"PD"`, `"NE"`); an iRECIST label like `"iSD"` is refused. `prev_was_iupd` is
what turns this timepoint's progression into iCPD. Responses are `iCR`, `iPR`,
`iSD`, `iUPD`, `iCPD`, `iNE`.

The remaining four sum their own quantity:

```python
# mRECIST (HCC): the arterially ENHANCING diameter only — a treated lesion can
# keep its size while the viable part disappears. Passing a lesion diameter is refused.
recist.evaluate_mrecist([{"viable_diam_mm": 40.0}],
                        [{"viable_diam_mm": 0.0, "status": "absent"}],
                        nadir_sum=40.0)["response"]          # 'CR'

# Cheson/Lugano (lymphoma): sum of products of perpendicular diameters (SPD)
recist.evaluate_cheson([{"longest_mm": 40.0, "perpendicular_mm": 30.0}],
                       [{"longest_mm": 20.0, "perpendicular_mm": 15.0}],
                       nadir_spd=1200.0)["current_spd"]      # 300.0 (20×15)

# RANO (glioma): SPD too, plus the non-enhancing component and steroid status
recist.evaluate_rano([{"longest_mm": 40.0, "perpendicular_mm": 30.0}],
                     [{"longest_mm": 20.0, "perpendicular_mm": 15.0}],
                     nadir_spd=1200.0,
                     non_enhancing="increased",   # 'stable'|'improved'|'increased'|'unknown'
                     on_steroids=False)           # a CR requires being off steroids
```

A significant `non_enhancing="increased"` is progression even when the enhancing
sum fell, and an enhancing tumour that vanished under steroids is not a complete
response.

PCWG3 (prostate) scores bone lesions and PSA, not a diameter sum. Bone
progression follows the 2+2 rule — two or more new lesions on the first scan,
confirmed by two or more further new lesions on the next. Between them the answer
is a real `Pending` state:

```python
recist.evaluate_pcwg3(new_lesions_scan1=2, new_lesions_scan2=None)["bone"]   # 'Pending'
recist.evaluate_pcwg3(new_lesions_scan1=2, new_lesions_scan2=2,
                      prev_was_pending=True)["bone"]                         # 'Progression'
```

PSA is optional — bone progression is assessable on imaging alone. Give
`psa_current` and `psa_nadir` (progression is measured against the nadir, the
lowest value seen) and the result gains a `"psa"` key; omit them and none is
invented.

## Following one lesion across studies

A lesion recorded as a pixel on a slice cannot be found again next month — the
slice index and the pixel grid both move. Recorded in patient coordinates it can.
`pixel_to_world` turns a pixel into millimetres using a slice's geometry:

```python
import numpy as np

origins = np.asarray(meta["slice_origins"]).reshape(-1, 3)
target_xyz = recist.pixel_to_world(
    origin=origins[m["slice"]],                       # the slice we measured on
    orientation=meta["image_orientation_patient"],
    spacing=meta["pixel_spacing"],
    px=col, py=row,                                   # the lesion centre, in pixels
)
```

At the follow-up study, `nearest_slice` finds which slice that point lands on,
and `match_lesion` finds the nearest prior lesion to it:

```python
_, fmeta = pydcm.read_seg("followup_seg.dcm", masks=True)
next_origins = np.asarray(fmeta["slice_origins"]).reshape(-1, 3)
normal = np.cross(fmeta["image_orientation_patient"][:3],
                  fmeta["image_orientation_patient"][3:])

k, signed_mm = recist.nearest_slice(target_xyz, next_origins, normal)   # which slice, how far off

candidates = np.array([[10.0, 5.0, 40.0], [11.0, 6.0, 41.0]])           # this study's lesion centres, mm
idx, dist_mm = recist.match_lesion(target_xyz, candidates, max_distance_mm=15.0)
```

`match_lesion` returns `(None, None)` when nothing lies within
`max_distance_mm` — which is exactly what a genuinely new lesion looks like, so
it is an answer, not a failure. `nearest_slice` returns the slice index and the
**signed** distance to it.

!!! note "Scope"
    RECIST 1.1, iRECIST, mRECIST, Cheson/Lugano, RANO and PCWG3 over the same
    lesion input; Feret-diameter and volume measurement off a Segmentation mask;
    and world-coordinate tracking of one lesion across studies. Which criterion
    to apply is a clinical decision this module does not make. For research and
    engineering only — not a medical device.
