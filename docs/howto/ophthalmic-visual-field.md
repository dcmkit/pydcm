# Ophthalmic visual field (static perimetry)

Read Ophthalmic Visual Field static-perimetry DICOM (Supplement 146, SOP Class
`1.2.840.10008.5.1.4.1.1.80.1`), flatten it to pandas or JSON, and check
standard conformance. The semantic parsing runs in the native engine (the same
reader behind `pydcm.content`); `pydcm.opv` is a thin pandas/JSON layer on top.

## Read a single study

```python
import pydcm

vf = pydcm.opv.read_dicom("vf.dcm")

vf.pointwise_to_pandas()       # one row per stimulus location
vf.to_pandas()                 # one row of study-level fields
vf.pointwise_to_nested_json()  # study identifiers + nested per-point records
vf.test_points                 # the raw per-point list
```

`pointwise_to_pandas()` returns a row per test point — `x_coordinate`,
`y_coordinate`, `sensitivity_value`, `stimulus_results`, the age-corrected and
generalized-defect deviations — each tagged with the study identifiers and a
`point_index`. `to_pandas()` flattens the study-level groups
(`test_parameters`, `reliability`, `global_results`) into one row.

## Batch a directory

```python
opvset, errors = pydcm.opv.read_dicom_directory("study_dir/")   # *.dcm by default

len(opvset)                       # OPV files that parsed
errors                            # [(path, message), ...] for non-OPV / unreadable
opvset.pointwise_to_pandas()      # every file's points, concatenated
opvset.to_pandas()                # one study-level row per file
opvset.check_dicom_compliance()   # {path: [findings, ...]}
```

## Check conformance

```python
vf.check_dicom_compliance()       # IOD / module findings for this file
```

This reuses the native IOD conformance judge (`pydcm.iod_validate`), which
enforces every mandatory module's Type-1 / Type-2 attribute presence for the
OPV SOP Class, descending into present sequences. It is stricter than a flat
tag checklist — per-SOP-Class and nested-sequence aware — so an empty list means
the file is conformant at the IOD level.

```python
import pydcm
pydcm.iod_validate("vf.dcm")      # [{severity, tag, module, message}, ...]
```

## The semantic content directly

`pydcm.opv.read_visual_field` (or `pydcm.content`) returns the raw nested content —
useful when you want the structure without pandas:

```python
import pydcm

c = pydcm.content("vf.dcm")
c["type"]                     # "ophthalmic_visual_field"
c["laterality"]               # "OD" / "OS"
c["test_parameters"]          # extent, shape, stimulus luminance/area/time, ...
c["reliability"]              # fixation losses, false +/-, catch trials, ...
c["global_results"]           # mean sensitivity, MD/PSD-family deviations,
                              # short-term fluctuation, blind spot, ...
c["test_points"]              # per-stimulus records
```

!!! note "What is extracted"
    Every scalar and coded field of the perimetry measurements is surfaced —
    test parameters, reliability, the global-result deviations, the blind spot,
    and per-point measurements with their deviation probabilities. The deep
    normative / algorithm *reference* sequences (e.g. `TestPointNormalsSequence`,
    `AgeCorrectedSensitivityDeviationAlgorithmSequence`) are reported by
    presence rather than expanded.
