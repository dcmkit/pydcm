# pydcm.transforms — authoritative references & precision

There is **no single gold-standard transforms library** in Python. Medical-image ML
preprocessing is fragmented, and two numeric families genuinely disagree:

- **scipy.ndimage** family (B-spline resampling, filtering, connected components) — the
  de-facto numeric reference; nnU-Net and skimage build on it.
- **torch** family (`grid_sample`) — a *separate, incompatible* interpolation convention;
  MONAI `Spacing`/`Resize` use it.

"Correct" for a preprocessing op therefore means **"matches the implementation the target
framework actually used to train the model"** — it is defined per-framework, not by math.
DICOM-defined pixel transforms (rescale /
modality LUT / VOI LUT) are **not** in this picture — those are deterministic and standalone, never spatial/ML preprocessing. Validate against the op's own
upstream below — never against another reproduction (custom_ops, pydcm) — those share
lineage and would validate circularly.

To avoid mixing families, import a **preset** (`from pydcm.transforms import nnunet as T`
or `monai as T`); it binds the divergent ops and shares the rest.

## Per-op reference

| op | authoritative reference | precision vs reference |
| --- | --- | --- |
| `resample_cubic` (`backend="skimage"`) | `skimage.resize(order=3, mode='edge', anti_aliasing=False)` — nnU-Net `default_resampling` (NOT `scipy.ndimage.zoom`) | **bit-exact** |
| `resample_separate_z` | per-slice cubic + nearest-z (nnU-Net anisotropic) | **bit-exact** |
| `resample_nearest` | `skimage.resize(order=0)` (half-pixel) | **bit-exact** |
| `resample_grid_sample` | `torch.nn.functional.grid_sample` (align_corners=False) | ≤1 fp32 ULP¹ |
| `resample_to_spacing`, `resize` | MONAI `Spacing`/`Resize` (half-pixel) | ≤1 fp32 ULP¹ |
| `normalize_ct` | nnU-Net CTNormalization (clip + fixed mean/std) | **bit-exact** |
| `normalize_zscore` | MONAI `NormalizeIntensity` (self-computed stats) | bit-exact² |
| `scale_intensity_range` | MONAI `ScaleIntensityRange` | tight tolerance |
| `scale_intensity_range_percentiles` | `np.percentile` (linear) + MONAI | tight tolerance |
| `adjust_contrast` | MONAI `AdjustContrast` | ~1e-5 (`powf`) |
| `crop`, `pad`, `crop_foreground`, `flip` | numpy / MONAI definitions (exact index) | **exact** |
| `center_crop`, `spatial_pad`, `divisible_pad` | MONAI `CenterSpatialCrop`/`SpatialPad`/`DivisiblePad` | **exact** |
| `rotate90` | `np.rot90` (+ world-coord preservation) | **exact** |
| `gaussian_smooth` | MONAI `GaussianSmooth` (reflect, separable) | tight tolerance |
| `argmax` | `np.argmax` over channels | **exact** |
| `connected_components` | scipy.ndimage `label` | **exact** |
| `keep_largest_connected_component`, `fill_holes`, `remove_small_objects`, `as_discrete` | MONAI post-transforms | **exact** |
| `sliding_window_positions` | MONAI `dense_patch_slices` / `_get_scan_interval` | **exact** |
| `gaussian_importance_map(convention='nnunet')` | nnU-Net V2 `get_gaussian` (scipy `gaussian_filter`) | **bit-exact** |
| `gaussian_importance_map(convention='monai')` | MONAI `compute_importance_map` (sampled exp) | ≤1 fp32 ULP¹ |

¹ Not bit-exact: the reference's last-bit rounding is not reproducible from any closed
form (torch's `affine_grid` SIMD `linspace`; `expf`). Numerically identical otherwise.
² `np.std` ddof=0 (population), matching MONAI's default.

## Presets — which op a preset binds

| preset | `resample` | `gaussian_importance_map` / `sliding_window_inference` |
| --- | --- | --- |
| `transforms.nnunet` | `skimage.resize` cubic spline (labels → nearest) | `convention='nnunet'` |
| `transforms.monai` | torch `grid_sample` (labels → nearest) | `convention='monai'` |

Every other op is identical in both presets and in the flat module. For nnU-Net's
anisotropic thick-slice path, call `resample_separate_z` explicitly.
