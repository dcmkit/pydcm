# API reference

Generated from pydcm's docstrings. The top-level `pydcm` module re-exports the
core read / write / decode API plus every specialist surface below.

## Read / write / decode

The core read / write / decode API. See [Behaviour notes](divergences.md) for
the short list of deliberate behaviours worth knowing.

::: pydcm
    options:
      members:
        - decode
        - dcmread
        - dcmwrite
        - pixel_array
        - iter_pixels
        - generate_uid

### Pixels

::: pydcm.pixels

### File-sets (DICOMDIR)

::: pydcm.fileset
    options:
      members:
        - FileSet
        - FileInstance

## Volumes, geometry &amp; ML

::: pydcm.volume

::: pydcm.diffusion

::: pydcm.transforms

::: pydcm.torchdata

::: pydcm.radiomics

::: pydcm.dce

## Derived objects — authoring &amp; reading

Author and read the structured DICOM objects (see the
[how-to recipes](howto/index.md)).

### Segmentations (SEG)

::: pydcm.seg

### Parametric maps

::: pydcm.paramap

### Structured reports (SR / TID 1500)

::: pydcm.sr

### Key Object Selection

::: pydcm.ko

### Presentation State (GSPS)

::: pydcm.pr

### Bulk annotations (microscopy)

::: pydcm.ann

### Secondary Capture

::: pydcm.sc

### Parametric Map classes

::: pydcm.pm

### Legacy Converted Enhanced

::: pydcm.legacy_converted

### Encapsulated documents (PDF / CDA / STL / OBJ / MTL)

::: pydcm.encapdoc

### Semantic content (auto-detect by SOP class)

::: pydcm.content

## RT dosimetry

::: pydcm.rt

## Whole-slide imaging

::: pydcm.wsi
    options:
      members:
        - open_slide
        - open_slides
        - Slide

## Waveforms (ECG / EEG)

::: pydcm.waveforms

## Networking

### DICOMweb

::: pydcm.dicomweb

### DIMSE

::: pydcm.dimse
    options:
      members:
        - AE
        - Association

## Ophthalmic visual field

::: pydcm.opv
    options:
      members:
        - read_dicom
        - read_dicom_directory
        - read_visual_field
        - OPVDicom
        - OPVDicomSet

## EHR bridges

### FHIR

::: pydcm.fhir

### HL7 v2

::: pydcm.hl7

## Agent / MCP server

::: pydcm.mcp
    options:
      members:
        - serve

## The rest of the DICOM API

The standard DICOM data model is all here, so existing code runs unchanged.

**Types** — `Dataset`, `FileDataset`, `FileMetaDataset`, `DataElement`,
`Sequence`, `PersonName`, `Tag`, `BaseTag`, `MultiValue`, `DSfloat`, `IS`,
`UID`, `InvalidDicomError`.

**Submodules** — `charset`, `config`, `datadict`, `dataelem`, `dataset`,
`dicomio`, `encaps`, `encoders`, `env_info`, `errors`, `examples`,
`filereader`, `filewriter`, `jsonrep`, `multival`, `overlays`,
`pixel_data_handlers`, `sequence`, `tag`, `uid`, `valuerep`, `values`.

See [Behaviour notes](divergences.md) for the deliberate behaviours worth
knowing.
