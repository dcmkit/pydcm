# How-to recipes

Task-focused guides for the capabilities that go beyond reading and decoding —
each is a short, runnable recipe.

## Imaging &amp; ML

- [DICOM → NIfTI, BIDS &amp; DWI](nifti-bids-dwi.md) — volumes, sidecars, diffusion tables
- [Preprocessing](preprocessing.md) — resample, normalize, sliding-window
- [Radiomics features](radiomics.md) — IBSI features over an ROI
- [Whole-slide imaging](wsi.md) — read regions from a DICOM WSI pyramid

## Authoring derived objects

- [Segmentations (SEG)](segmentation.md) — write &amp; read binary / fractional SEG
- [Parametric maps](parametric-map.md) — author a float map, read it back
- [Structured reports (TID 1500)](structured-reports.md) — measurement reports
- [RT dose &amp; DVH](rt-dose-dvh.md) — read dose, compute DVH, author RT Dose
- [Perfusion (DCE-MRI)](perfusion-dce.md) — pharmacokinetic Ktrans / ve / vp maps from a dynamic series

## Integration

- [DIMSE networking](dimse.md) — echo / store / query / retrieve, SCU and SCP
- [DICOMweb](dicomweb.md) — QIDO / WADO / STOW against a remote server
- [Waveforms (ECG / EEG)](waveforms.md) — read, write, hand off to analysis
- [Ophthalmic visual field](ophthalmic-visual-field.md) — static perimetry → pandas / JSON, conformance
- [FHIR &amp; HL7 bridges](fhir-hl7.md) — ImagingStudy and ORU^R01
- [Agent / MCP server](mcp-agent.md) — expose pydcm's live-object tools over MCP
