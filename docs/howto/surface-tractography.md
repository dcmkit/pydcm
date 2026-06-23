# Surface meshes & tractography

Two related native-geometry objects. **Surface Segmentation** (SOP
`…66.5`) stores a segmentation as an actual mesh — points and mesh
primitives — rather than a voxel mask or an encapsulated STL blob.
**Tractography Results** (SOP `…66.6`) stores diffusion streamlines as
polylines in Frame-of-Reference world coordinates, with optional
per-track scalars such as FA and ADC. Both are authored and read over the
same model the writer took, so what you get back is close to what you put
in.

## Surface Segmentation (66.5)

### Read a mesh

`read_surface` returns `{"surfaces": [...], "segments": [...]}`, or
`None` when the file is not a Surface Segmentation object. Every primitive
type (triangle / strip / fan / facet) is expanded into one flat triangle
list, so each surface arrives as a plain `(points, triangles)` pair ready
for rendering.

```python
import pydcm

doc = pydcm.read_surface("surface.dcm")
s = doc["surfaces"][0]
s["points"]            # (N, 3) float64 vertices, patient mm
s["triangles"]         # (M, 3) uint32 vertex indices, 0-based
s["normals"]           # (N, 3) float64 per-vertex normals, or None
s["recommended_type"]  # "SURFACE" or "POINTS"
s["recommended_opacity"], s["finite_volume"], s["manifold"]

g = doc["segments"][0]
g["label"], g["algorithm_type"]
g["property_type"]["meaning"]           # coded, e.g. "Hippocampus"
g["referenced_surface_numbers"]         # wire Surface Numbers this segment covers
```

Coordinates come back as `float64` and indices as `uint32` (0-based).
`normals` is `None` when the file carries none; `lines` / `vertices` are
present only when the surface actually uses edge or point primitives.

### Author a mesh

`write_surface(surfaces, segments)` takes a list of mesh dicts and a list
of segment dicts. Points are written as Double Point Coordinates, so a
coordinate that needs more than seven digits survives the round trip
rather than being rounded to 32-bit. Triangle indices are **0-based** —
the 1-based wire form is the engine's business — and an index off the end
of `points` is refused, not silently dropped.

```python
import numpy as np
import pydcm

points = np.array([[-123.4567891, 10.0000001, 5.5],
                   [  12.3456789, -20.1234567, 5.5],
                   [   0.0,        30.9876543, 5.5],
                   [   0.0,         0.0,       42.4242424]], np.float64)
triangles = np.array([[0, 1, 2], [0, 1, 3], [1, 2, 3], [2, 0, 3]], np.uint32)

surface = {"points": points, "triangles": triangles,
           "finite_volume": True, "manifold": True,
           "opacity": 0.75, "rgb": (0, 0, 255), "comments": "left hippocampus"}

segment = {"label": "Hippocampus", "algorithm_type": "SEMIAUTOMATIC",
           "property_category": ("T-D000A", "SRT", "Anatomical Structure"),
           "property_type":     ("T-A2000", "SRT", "Hippocampus"),
           "algorithm_family":  ("123110", "DCM", "Region Growing"),
           "algorithm_name": "grow", "algorithm_version": "1.0",
           "surfaces": [0]}     # 0-based index into `surfaces`, not a wire number

pydcm.write_surface([surface], [segment], output="surface.dcm",
                    patient_name="DOE^JANE", patient_id="MRN-001",
                    study_uid="1.2.3.4.5", study_date="20260624")
```

Both sequences are Type 1: a document needs at least one surface and at
least one segment. A segment's `surfaces` entries are 0-based indices into
the list you pass, not the Surface Numbers the reader reports back. Coded
concepts (`property_category` / `property_type` / `algorithm_family`) take
`(value, scheme, meaning)`. Omit `output` to get Part-10 `bytes` instead.

A tri-state hint — `finite_volume`, `manifold` — is `True`, `False`, or
omitted. Omitting it means UNKNOWN, the honest answer for an arbitrary
mesh, and comes back as `"UNKNOWN"` rather than a claim of `"NO"`.

### Point clouds vs surfaces

Leave out `triangles` and the mesh is a point cloud: `presentation_type`
follows the geometry, so it is authored as POINTS and reads back with
`recommended_type == "POINTS"` and an empty triangle list. Presenting a
cloud as a surface would draw nothing, so the default is the useful one.

```python
cloud = {"points": points}                     # no triangles
pydcm.write_surface([cloud], [segment], output="cloud.dcm")
```

## Tractography Results (66.6)

### Author streamlines

`write_mktract(reference, track_sets)` authors a Tractography Results
object from track sets of polylines. `reference` is a source-series path
(or list of instance paths) supplying demographics and the Frame of
Reference UID, or `None` to mint fresh identifiers. Track point
coordinates are in that Frame of Reference — patient world mm. Each track
is an `(n, 3)` array of xyz; a single dict or a list of dicts is accepted.

```python
import numpy as np
import pydcm

track_a = np.array([[0., 0., 0.], [1., 0., 0.], [2., 0., 0.]], np.float32)
track_b = np.array([[0., 0., 0.], [0., 1., 0.], [0., 1., 1.]], np.float32)

pydcm.write_mktract(
    "dwi_series/",                              # Frame of Reference + demographics
    {"label": "Corticospinal", "description": "left CST",
     "anatomy": ("T-A2000", "SCT", "Corticospinal tract"),
     "algorithm_name": "FACT", "line_thickness": 2.0, "rgb": (220, 30, 30),
     "tracks": [track_a, track_b]},
    output="tract.dcm")
```

`anatomy` defaults to White Matter and `diffusion` to Single Tensor if
omitted; both take the same coded-concept forms as elsewhere. Leave
`output` off to get the Part-10 `bytes`.

### Per-track FA / ADC measurements & statistics

A track set can carry a scalar sampled **along** each streamline (a
measurement) and reductions of it (statistics). A measurement's `values`
is one array per track, paired point-for-point with that track's vertices.
`track_statistics` is one scalar per track; `set_statistics` is one scalar
for the whole set.

```python
fa = ("113290", "DCM", "Fractional Anisotropy")
mean = ("R-00317", "SRT", "Mean")
unitless = ("1", "UCUM", "1")

pydcm.write_mktract(
    "dwi_series/",
    {"label": "CST", "tracks": [track_a, track_b],
     "measurements": [
        {"concept": fa, "units": unitless,
         "values": [np.array([0.61, 0.58, 0.55], np.float32),   # along track_a
                    np.array([0.60, 0.57, 0.54], np.float32)]}], # along track_b
     "track_statistics": [
        {"concept": fa, "modifier": mean, "units": unitless,
         "values": np.array([0.58, 0.57], np.float32)}],        # one per track
     "set_statistics": [
        {"concept": fa, "modifier": mean, "units": unitless, "value": 0.575}]},
    output="tract_fa.dcm")
```

### Read tracks back

`read_tract` returns `{"track_sets": [...]}`, or `None` when the file is
not a Tractography Results object.

```python
import pydcm

ts = pydcm.read_tract("tract_fa.dcm")["track_sets"][0]
ts["label"], ts["anatomy"]["meaning"], ts["algorithm_name"]

for t in ts["tracks"]:
    t["points"]                      # (N, 3) float32, world mm

for m in ts["measurements"]:         # a quantity sampled along the tracks
    for per in m["tracks"]:          # one entry per track
        per["values"]                # (N,) float32; per["indices"] if not 1:1

for st in ts["set_statistics"]:
    st["concept"]["meaning"], st["value"]     # e.g. "Fractional Anisotropy", 0.575
```

Coordinates stay `float32` because that is the width the standard gives
them and a streamline goes to a vertex buffer; statistics come back as
`float64` because a statistic is a measurement, not geometry — a set FA of
`0.575`, not representable in float32, survives exactly.

!!! note "dipy / MRtrix streamline round-trip"
    The `tracks` list — one `(n, 3)` array per streamline — is exactly the
    shape `dipy` and MRtrix use, so a round trip is a coordinate flip and a
    repackage, not a rewrite. One caveat: DICOM Frame-of-Reference space is
    **LPS** (mm), while `dipy` / `nibabel` streamlines in `RASMM` space are
    **RAS**. Negate x and y crossing the boundary.

    ```python
    import numpy as np, pydcm
    from dipy.io.streamline import load_tractogram   # RASMM streamlines

    lps = np.array([-1, -1, 1], np.float32)          # RAS(mm) -> LPS(mm)
    sft = load_tractogram("tracks.trk", "reference.nii.gz")
    tracks = [(s * lps).astype(np.float32) for s in sft.streamlines]
    pydcm.write_mktract("dwi_series/", {"label": "WM", "tracks": tracks},
                        output="tract.dcm")

    back = [t["points"] * lps                        # LPS -> RAS(mm) on the way out
            for t in pydcm.read_tract("tract.dcm")["track_sets"][0]["tracks"]]
    ```

    An MRtrix `.tck` stores streamlines in the image's real/scanner
    coordinates; convert those to the reference series' patient LPS frame
    the same way before writing.
