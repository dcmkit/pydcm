# Parametric maps

Author a DICOM Parametric Map from a real-valued array (ADC, perfusion, dose,
a model's continuous output) and read one back — the standard parametric-map
capability.

## Write

```python
import pydcm

# adc: float array aligned to the reference series geometry
pydcm.write_paramap(
    "dwi_series/",
    adc,
    units="10^-6 mm2/s",
    quantity="Apparent Diffusion Coefficient",
    label="ADC",
    output="adc_pm.dcm",
)
```

- `reference` supplies geometry and demographics, exactly as for SEG.
- The float values are stored faithfully with a self-consistent
  slope / intercept; pass `slope` / `intercept` to fix the quantisation, or
  `dtype` to choose the stored representation.
- Omit `output` to receive Part-10 `bytes`.

## Read

```python
values, meta = pydcm.read_paramap("adc_pm.dcm")
values                                        # float array, real-world values (slope/intercept applied)
rwvm = meta["real_world_value_mapping"]       # the Real World Value Mapping
rwvm["units"], rwvm["slope"], rwvm["intercept"]   # e.g. "10^-6 mm2/s", …
```
