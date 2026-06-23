# Spatial registration (66.1)

A Spatial Registration object says how one Frame of Reference sits inside
another — where the PET is relative to the CT, where today's MR is relative to
last month's. Read one to place a fusion or propagate contours across frames;
author one from your own 4x4 matrices. Deformable Spatial Registration (66.3)
carries a deformation grid rather than a matrix and is a different object; it is
refused here rather than read as if it were this one.

## Read a registration

```python
import pydcm

reg = pydcm.read_registration("reg.dcm")
reg.frame_of_reference_uid   # the object's OWN frame — the one every item maps into
reg.frames                   # the Frames of Reference this object registers
reg.items                    # the RegistrationItem entries
reg.incomplete               # an item could not be recorded (short by omission)
len(reg)                     # number of items; reg is iterable over its items
```

The object's own frame is the fixed image of a fusion — a registration authored
in the CT frame registers PET (and MR, and…) into it. An object conventionally
includes an identity item for its own frame, and that item is **kept**: "these
two frames are the same" is an answer, not an item to drop.

## Resolve a transform

`transform(from_uid, to_uid)` returns the `(4, 4)` float64 that maps a point in
`from_uid` into `to_uid`, in the `M @ [x, y, z, 1]` convention. This is what a
fusion actually asks for — neither series is necessarily the frame the object
was authored in, and the reverse direction is the inverse, not a second stored
matrix.

```python
m = reg.transform(pet_frame_uid, ct_frame_uid)   # PET point -> CT point
world_ct = m @ [x, y, z, 1.0]

reg.transform(pet_frame_uid)              # to_uid omitted -> maps into the object's own frame
reg.transform(pet_frame_uid, pet_frame_uid)   # identical frames -> the identity
```

Identical frames answer with the identity, so a caller need not decide
beforehand whether a registration is even involved.

### None is not identity

`transform` returns `None` — never a silent identity — when the object does not
state the transform: an **unregistered** frame it never mentions, an item whose
matrix was **present but unusable**, or a composition that would need a
**singular inverse**.

```python
m = reg.transform(some_frame_uid, ct_frame_uid)
if m is None:
    ...   # this object cannot place that frame — do NOT fall back to identity
else:
    apply(m)
```

Applying identity where a registration was intended puts the overlay in the
wrong place with nothing to show that it did. The falsy return forces the
caller to decide, which is the whole point of the object.

## Inspect the items

Each entry is a `pydcm.registration.RegistrationItem`, one Item of the
Registration Sequence:

```python
for item in reg.items:
    item.frame_of_reference_uid   # the frame this item's matrix maps FROM
    item.matrix                   # (4, 4) float64 into the object's own frame, or None
    item.matrix_invalid           # a matrix was present and cannot be used
    item.matrix_type              # "RIGID", "RIGID_SCALE", "AFFINE" or "UNKNOWN"
    item.matrix_count             # how many matrices were composed into `matrix`
```

`matrix is None` with `matrix_invalid is False` means the item carried **no**
matrix — a different fact from one that was present and failed (a Decimal String
that would not convert, the wrong number of values, or a bottom row that is not
`[0, 0, 0, 1]`), which reports `matrix_invalid is True`. Absent, unusable, and
identity are three distinct answers. `matrix_type` is the producer's claim about
its own pipeline — all three named types are affine — and nothing checks it
against the matrix; a consumer may want to trust each differently.

## Author a registration

`write_registration` is the inverse of `read_registration`: give it your 4x4
matrices in the same `M @ [x, y, z, 1]` convention and the frame they map into.

```python
import numpy as np

pydcm.write_registration(
    [{"frame_of_reference_uid": pet_frame_uid, "matrix": pet_into_ct,
      "matrix_type": "RIGID"}],   # RIGID default; "RIGID_SCALE" or "AFFINE" also
    ct_frame_uid,                 # THIS object's own frame — the frame each matrix maps INTO
    output="reg.dcm",             # omit `output` to get Part-10 bytes instead
    patient_name="DOE^JANE", patient_id="MRN-001",
    study_uid="1.2.3.4.5", series_uid="1.2.3.4.5.9",
)
```

Each item's `matrix` maps its `frame_of_reference_uid` **into** the object's own
frame; that direction is what the whole object means and reversing it is not
detectable later. An identity item for the own frame is written automatically
unless `items` already contains one — without it the file parses and then
refuses every query, because resolving A to B composes `inverse(item_B) @
item_A` and needs an item for B.

The writer refuses inputs that would produce a file that parses and then answers
its own question wrongly (or not at all): a matrix whose bottom row is not
`[0, 0, 0, 1]`, a **singular** matrix, a **non-finite** element, two items
naming one frame, an unknown `matrix_type`, an empty own frame, or no items at
all. Each is raised rather than written.

!!! note "Scope"
    Rigid and affine Spatial Registration (66.1) — read the transform between
    any two registered frames (with the None-not-identity contract), inspect the
    per-frame items, and author an object from your own 4x4 matrices. Deformable
    Spatial Registration (66.3) is out of scope. Not for clinical or diagnostic
    use.
