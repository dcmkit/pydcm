# Structured reports (TID 1500)

Author and read TID 1500 Measurement Report SRs — the standard
`writetid1500` / `readtid1500` capability. The SR pydcm writes is conformant
(`sr_validate` finds no errors).

## Write a measurement report

A report is a list of measurement dicts. Each carries a coded concept, a value
with UCUM units, and an optional image reference + graphic annotation:

```python
import pydcm

measurements = [
    {"concept_value": "103355008", "concept_scheme": "SCT", "concept_meaning": "Width",
     "value": 23.4, "unit_code": "mm", "unit_meaning": "millimeter",
     "ref_sop_class_uid":    "1.2.840.10008.5.1.4.1.1.2",
     "ref_sop_instance_uid": "1.2.3.4.5.6.7.8.9",
     "graphic_type": "POLYLINE", "scoord": [10.0, 20.0, 33.4, 20.0]},
    {"concept_value": "33747003", "concept_scheme": "SCT", "concept_meaning": "Mean",
     "value": -512.7, "unit_code": "[hnsf'U]", "unit_meaning": "Hounsfield unit"},
]

pydcm.write_report(
    measurements,
    patient_name="DOE^JANE", patient_id="PID-1", study_date="20260605",
    output="report.dcm",
)
```

Omit `output` to get Part-10 `bytes`. Pass `reference` (a source instance path)
to inherit study/patient context instead of supplying it by hand.

`write_report` also takes a `sop_instance_uid` keyword — the SR's own SOP
Instance UID. Left empty (the default) the engine mints a deterministic UID from
the report's content, so two reports built for the same study come out with the
*same* SOP Instance UID. That is right for a single self-contained export, but a
producer minting many reports for one study must pass a distinct
`sop_instance_uid` for each.

## Read it back

`read_report` returns a dict with the study/patient context and a
`measurements` list:

```python
report = pydcm.read_report("report.dcm")
report["patient_name"], report["study_date"]        # study / patient context
for m in report["measurements"]:
    m["concept_meaning"], m["value"], m["unit_meaning"]
```

A measurement drawn on an image also carries its annotation — `m["graphic_type"]`,
`m["is_3d"]`, and `m["scoord"]` (the raw coordinates). A three-dimensional
(`SCOORD3D`) region reads back with `is_3d` `True`, a `frame_of_reference_uid`,
and three patient-space coordinates per point rather than two.

`read_report` is the typed TID 1500 view — measurements only. For **any** SR
(narrative, general, non-measurement), `pydcm.content` returns the full content
tree — the inverse of `write_sr`:

```python
doc = pydcm.content("report.dcm")     # {type, title, content: [...]} — the whole tree
doc["title"]                          # root container's concept
node = doc["content"][0]
node["value_type"]                    # CODE / NUM / TEXT / CONTAINER / IMAGE / SCOORD / …
node["relationship"]                  # CONTAINS / HAS PROPERTIES / …
node.get("content")                   # nested children, recursively
```

## Read every region

`read_report` walks the TID 1500 template — root → Imaging Measurements →
Measurement Group → region — and that is right for a *report*: it finds a
measurement even when the measurement carries no shape. It is the wrong reader
for an *importer*, because real producers put regions at different depths and
many of them carry no measurement concept in the chain at all, so a
template-driven walk returns nothing for them.

`read_regions` is the complement: it enumerates **every** delineated shape the
document contains — each `SCOORD` / `SCOORD3D`, plus each image reference that
names a segment — regardless of where in the tree it sits, gathering each one's
context from its surroundings:

```python
import pydcm

for r in pydcm.read_regions("sr.dcm"):
    r["kind"]            # "REGION" (coordinates) or "SEGMENT"
    r["graphic_type"]    # "POINT" / "POLYLINE" / "POLYGON" / "ELLIPSOID" / … (None for a segment)
    r["is_3d"]           # True for a SCOORD3D region
    r["points"]          # (n, 2) column/row in the image, or (n, 3) patient-LPS mm when is_3d
    r["frame_of_reference_uid"]                          # the space a 3-D region lives in (else None)
    r["ref_sop_class_uid"], r["ref_sop_instance_uid"]    # the image the region is drawn on
    r["referenced_frame_number"]                         # 1-based frame, or None
    r["segment_number"], r["segment_sop_instance_uid"]   # set for a SEGMENT region
    r["tracking_id"], r["tracking_uid"]                  # the region's stable name across exports
    r["has_measurement"]                                 # True when a measurement sits on the region
    r["concept"], r["value"], r["unit_code"], r["unit_meaning"]   # that measurement, when there is one
    r["finding_meaning"]                                 # the finding the region sits under
```

Regions come back in document order (`node_index` records each one's place in
the tree); a document that delineates nothing returns `[]`.

`graphic_type` round-trips exactly for every Defined Term now, including the
3-D-only `POLYGON` and `ELLIPSOID` — `POLYGON` used to be flattened to
`POLYLINE`. A `SCOORD3D` region returns `is_3d` `True` and a
`frame_of_reference_uid`, and its `points` are three patient-LPS millimetre
coordinates per vertex; a 2-D `SCOORD` region gives `(n, 2)` column/row pixel
coordinates in the referenced image.

## Arbitrary SR content trees

For SRs that are not measurement reports, `write_sr` authors a Comprehensive SR
from a content-tree dict (containers, code / num / text / image content items):

```python
pydcm.write_sr(document_tree, output="sr.dcm")
```

The document dict carries the patient/study identity alongside the content, and
both `write_sr` and the typed measurement-report writer accept the Type 2 patient
demographics `patient_birth_date` and `patient_sex` there (beside `patient_name`
/ `patient_id`); `read_measurement_report` returns those two on read-back.

## Validate conformance

`sr_validate` walks an SR's content tree and returns a list of findings
(`{severity, location, message}`). `error` severity is a conformance violation;
`warning` flags softer issues — e.g. a coded value outside the bundled code
dictionary. No `error` findings means it is structurally well-formed:

```python
for f in pydcm.sr_validate("report.dcm"):
    print(f["severity"], f["location"], f["message"])
```

It checks three layers:

- **Structural** — the root is a `CONTAINER`, value types and relationships are
  valid, a `NUM` has units, a `CODE` has a value, an `IMAGE` has a reference, a
  `SCOORD` has graphic data.
- **Coded concepts** — concept / value / unit codes are looked up in the
  PS3.16 code table, and a code meaning that disagrees with the dictionary is
  flagged.
- **Template (TID) conformance** — content-template rules from the measurement
  report family and its context templates: a Measurement Group's mandatory
  Tracking Identifier / Tracking Unique Identifier (for ROI groups) and their
  multiplicity, value-type per row, the coded value of a concept against its
  Context Group, container nesting (groups under Imaging Measurements), and
  conditional rows such as a `DEVICE` Observer or Subject needing its identifier
  and an Algorithm Name needing its Version. The checks are high-signal: a
  conformant SR — including the one `write_report` authors — validates clean
  (you may still see a `warning` if your own measurements use a coded value
  that isn't in the bundled code dictionary).
