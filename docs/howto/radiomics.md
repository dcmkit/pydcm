# Radiomics features

`pydcm.radiomics` computes the **full IBSI feature set** over an ROI —
**135 features across 10 classes** (first-order, GLCM, GLRLM, GLSZM, GLDM,
GLDZM, NGTDM, shape, intensity-volume histogram, local intensity), with the
standard radiomics feature names. The seven classes a conventional extractor
also computes are **validated feature-for-feature against published reference
baselines**; GLDZM / IVH / local-intensity add the IBSI features beyond
that set, so the output drops into an existing radiomics pipeline unchanged and
then some.

!!! note "135 IBSI features, 145 dict keys"
    The 135 are the IBSI-defined features. The returned `dict` carries a few more
    keys (≈145) — first-order and shape convenience extras (e.g. Energy /
    TotalEnergy / RMS) on top of the IBSI set — so `len(feats)` is larger than 135.

## From a DICOM image + mask

```python
from pydcm.radiomics import radiomics

feats = radiomics("ct.dcm", mask="mask.dcm")     # pixels → HU, spacing from geometry
feats["firstorder_Mean"]                          # bare "<class>_<name>" keys
feats["shape_VoxelVolume"]
feats["glcm_Contrast"]
```

(The `RadiomicsFeatureExtractor.execute` path below prefixes keys with
`original_` — e.g. `original_firstorder_Mean` — matching the conventional extractor's key scheme.)

Pixels are decoded to real-world values (HU) and the spacing is read from the
image geometry (PixelSpacing / SliceThickness), so the feature values are in
physical units without any manual setup.

## From arrays

```python
import numpy as np

feats = radiomics(image_array, mask=mask_array, spacing=(z, y, x))
```

## Binning and preprocessing knobs

All the preprocessing knobs are exposed as keyword arguments:

```python
feats = radiomics(
    "ct.dcm", mask="mask.dcm",
    bins=32,                       # fixed bin count …
    bin_width=25.0,                # … or a fixed bin width (mutually exclusive)
    value_range=(-1024.0, 3071.0),
    resample=1.0,                  # isotropic resample (mm) before extraction
    distances=[1, 2, 3],           # GLCM neighbour distances
    normalize=False,
    wavelet=False, log_sigma=None, # optional image-filter families
)
```

## Custom features

Register a Python feature with the `@pydcm.radiomics.feature` decorator — it runs
over the **same preprocessed + discretised grid** the native IBSI features used,
so a custom histogram feature lines up bin-for-bin with the standard ones, and
its value joins the result dict as `"<class>_<name>"`:

```python
import pydcm, numpy as np

@pydcm.radiomics.feature("firstorder")           # → adds "firstorder_p90"
def p90(roi):
    return float(np.percentile(roi.intensities, 90))

# custom features run on the array / extractor path — decode to arrays first,
# not the all-DICOM-files shortcut
feats = pydcm.radiomics(image_array, mask=mask_array, spacing=(z, y, x))
feats["firstorder_p90"]
```

The `roi` argument exposes the exact grid the engine ran over — `roi.intensities`
and `roi.levels` (1-D over the ROI voxels), `roi.n`, `roi.spacing`,
`roi.bin_edges`. Naming a custom feature after a standard one
(`@pydcm.radiomics.feature("firstorder", name="Mean")`) **overrides** that
value, so you can change a built-in formula without recompiling. Manage the
registry with `pydcm.radiomics.registered_features()` and
`pydcm.radiomics.clear_features()`.

## Extractor object API

For code that expects the familiar extractor object, that API is available too —
feature-class selection is honoured:

```python
from pydcm.radiomics import RadiomicsFeatureExtractor

extractor = RadiomicsFeatureExtractor()
extractor.enableFeatureClassByName("firstorder")
result = extractor.execute("ct.dcm", "mask.dcm")
```
