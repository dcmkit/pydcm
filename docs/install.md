# Install

```bash
pip install pydcm
```

A platform wheel ships the compiled engine — no build step, no system DICOM
library, no codec plugins. Wheels are published for macOS (arm64, x86_64) and
Linux (aarch64, x86_64; manylinux_2_28 — glibc 2.28+, i.e. RHEL 8+ / Ubuntu
20.04+ / Debian 11+). One wheel per platform covers CPython 3.12+ (stable ABI).

`numpy` is the only hard runtime dependency. Optional integrations are
imported lazily, so you install only what you use:

| Feature | Optional package |
|---|---|
| `DICOMDataset(to_torch=True)` / training loops | `torch` |
| NIfTI output consumers | none — NIfTI is written natively |
| SimpleITK image in/out in `radiomics` / `seg` | `SimpleITK` |
| Waveform analysis hand-off | `mne`, `neurokit2` |

!!! note
    pydcm distributes as wheels only (no sdist): the high-performance engine
    ships as a compiled binary inside the extension, and parts of it are not
    open source. Licensing: Apache-2.0 for pydcm; bundled third-party
    components are listed in `THIRD-PARTY-LICENSES` (FFmpeg under LGPL-2.1
    with a relink offer).

## Build these docs

```bash
pip install pydcm -r docs/requirements.txt   # mkdocs-material + mkdocstrings + black
mkdocs serve                                 # from the package root
```
