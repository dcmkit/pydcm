# De-identification

`deidentify` applies the **DICOM PS3.15 Annex E** de-identification profile to a
DICOM instance and returns the de-identified Part-10 bytes — with the full set of
standard retain / clean options as keyword arguments.

```python
import pydcm

clean = pydcm.deidentify(open("scan.dcm", "rb").read())
open("scan.anon.dcm", "wb").write(clean)
```

## Retain / clean options

The Annex E Basic Profile is the default; turn on the individual option *columns*
of Table E.1-1 as you need them:

```python
clean = pydcm.deidentify(
    data,
    retain_dates=True,            # Retain Longitudinal Temporal Information (dates intact)
    shift_dates_days=-30,         # …or shift every date by N days instead
    retain_uids=True,             # Retain UIDs (keep cross-references)
    retain_safe_private=True,     # Retain Safe Private elements
    retain_device_id=True,        # Retain Device Identity
    retain_institution_id=True,   # Retain Institution Identity
    retain_patient_chars=True,    # Retain Patient Characteristics (age / sex / weight)
    clean_descriptors=True,       # Clean free-text descriptors
    clean_graphics=True,          # Clean burned-in graphic annotations
    clean_struct_content=True,    # Clean Structured Content (SR)
    clean_pixel=True,             # Clean burned-in pixel-data annotations
    patient_name="CASE-01", patient_id="CASE-01",   # replacement identity
)
```

Every applied option is recorded in the De-identification Method Code Sequence
(CID 7050), so the output declares exactly how it was de-identified.

Targeted overrides are available too — `replace={tag: value}` to set specific
elements and `remove=[tag, …]` to drop them.

## A whole study, consistently

`deidentify_series` de-identifies a list of instances through **one session**, so
the UID remap is consistent across the study (every Study / Series / SOP UID and
their cross-references map the same way in every file):

```python
out = pydcm.deidentify_series(["a.dcm", "b.dcm", "c.dcm"], retain_dates=True)
# out is a list of de-identified Part-10 byte strings, one per input
```

## Burned-in pixel annotations

`clean_pixel_data` blacks out burned-in PHI in the image itself — by explicit
boxes, or by the built-in CTP-style rules:

```python
cleaned = pydcm.clean_pixel_data(data)                       # built-in CTP-style rules
cleaned = pydcm.clean_pixel_data(data, regions=[(0, 0, 200, 40)])  # explicit (x, y, w, h) boxes
```

!!! warning "Not a medical device"
    De-identification is best-effort per PS3.15; verify the output against your
    own policy before sharing data.
