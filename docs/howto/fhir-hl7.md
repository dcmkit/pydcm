# FHIR &amp; HL7 bridges

The imaging ↔ EHR seam: turn a DICOM instance into a FHIR `ImagingStudy`, and
read or build HL7 v2 messages — so an app or agent can move a study between the
imaging world and a clinical system without a separate mapping layer.

## DICOM → FHIR ImagingStudy

```python
from pydcm import fhir

study = fhir.imaging_study("CT0001.dcm")   # one instance → a FHIR R4 ImagingStudy
study = fhir.imaging_study("study_dir/")   # …or a whole study — every series and
                                           # instance aggregated into one resource
study["resourceType"]                       # "ImagingStudy"
study["numberOfSeries"], study["numberOfInstances"]
study["series"][0]["instance"]
```

Point it at a single file or a directory / list of files: a study folder is
aggregated into one `ImagingStudy` with every series and instance counted. The
study / series / instance hierarchy is mapped from the DICOM headers, and
`subject.reference` is an external `Patient/<PatientID>` reference (resolve it
through a FHIR Patient endpoint). Serialize the dict with `json.dumps` to hand it
to any FHIR consumer.

## HL7 v2 — parse

```python
from pydcm import hl7

segments = hl7.parse(message_text)         # list of segment dicts
```

## HL7 v2 — build an ORU^R01 result

```python
oru = hl7.build_oru(
    config={...},          # sending/receiving application + facility
    context={...},         # patient + order identifiers
    observations=[...],    # OBX result rows
)                          # → ER7 string ready to send back to the HIS
```

These bridges are pydcm's own API over the native FHIR / HL7 engines — there is
no third-party Python FHIR or HL7 library in the loop.
