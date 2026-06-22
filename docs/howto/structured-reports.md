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

## Read it back

`read_report` returns a dict with the study/patient context and a
`measurements` list:

```python
report = pydcm.read_report("report.dcm")
report["patient_name"], report["study_date"]        # study / patient context
for m in report["measurements"]:
    m["concept_meaning"], m["value"], m["unit_meaning"]
```

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

## Arbitrary SR content trees

For SRs that are not measurement reports, `write_sr` authors a Comprehensive SR
from a content-tree dict (containers, code / num / text / image content items):

```python
pydcm.write_sr(document_tree, output="sr.dcm")
```

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
