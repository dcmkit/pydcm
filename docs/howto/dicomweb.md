# DICOMweb

`pydcm.dicomweb` is a QIDO / WADO / STOW / DELETE client over the native HTTP
stack — query, retrieve (streaming), store and delete against a remote server.

## Connecting: `server` + `base_path`

Address a DICOMweb server by an **origin** (`scheme://host:port`) plus a **base path** — the
prefix the server mounts its DICOMweb endpoints under. The base path is **different on every
product**, and getting it wrong is the #1 cause of "it returns nothing / a 404 / odd IDs":

| Server               | `server` (origin)    | `base_path`                        |
|----------------------|----------------------|------------------------------------|
| **Orthanc**          | `http://host:8042`   | `/dicom-web`                       |
| **dcm4chee-arc**     | `http://host:8080`   | `/dcm4chee-arc/aets/DCM4CHEE/rs`   |
| root-mounted server  | `http://host:8080`   | `""` (empty)                       |

```python
from pydcm import dicomweb

# Orthanc
dicomweb.search_studies("http://localhost:8042", base_path="/dicom-web")

# dcm4chee-arc (the AE in the path is the archive AE Title, default DCM4CHEE)
dicomweb.search_studies("http://localhost:8080",
                        base_path="/dcm4chee-arc/aets/DCM4CHEE/rs")

# a root-mounted server: base_path is the empty string (NOT "/", which 404s)
dicomweb.search_studies("http://localhost:8080", base_path="")
```

A prefix may instead be folded into `server` (`"http://localhost:8042/dicom-web"`) — both spell
the same endpoint — but prefer the explicit `base_path`. **`base_path` needs a leading slash**
(`"/dicom-web"`, not `"dicom-web"`).

**Auth** is the full `Authorization` header value, passed as `auth=`:

```python
dicomweb.search_studies("http://localhost:8042", base_path="/dicom-web",
                        auth="Basic b3J0aGFuYzpvcnRoYW5j")     # Orthanc default orthanc:orthanc
dicomweb.search_studies("http://localhost:8080", base_path="/dcm4chee-arc/aets/DCM4CHEE/rs",
                        auth="Bearer eyJ…")                    # dcm4chee behind Keycloak
```

**HTTPS** works by passing an `https://` origin; trust a private/self-signed CA via the
`SSL_CERT_FILE` environment variable (there is no per-call CA argument).

The rest of this page omits `base_path=`/`auth=` for brevity — add them per the table above.

## Query (QIDO-RS) → DICOM-JSON

```python
studies   = dicomweb.search_studies("http://localhost:8042", base_path="/dicom-web",
                                    matches={"PatientID": "42"})
series    = dicomweb.search_series("http://localhost:8042", study_uid, base_path="/dicom-web")
instances = dicomweb.search_instances("http://localhost:8042", study_uid, series_uid,
                                      base_path="/dicom-web")
```

Each returns a `list[dict]` of DICOM-JSON (`{}` on `204 No Content`). Pass `includefields=[...]`
to widen the returned attributes, `limit=`/`offset=` to page.

## Retrieve (WADO-RS)

```python
# whole objects → list[bytes] of Part-10
parts = dicomweb.retrieve_study("http://localhost:8042", study_uid, base_path="/dicom-web")
inst  = dicomweb.retrieve_instance("http://localhost:8042", study_uid, series_uid, sop_uid,
                                   base_path="/dicom-web")

# streaming — bounded memory over a large study
import io
for part10 in dicomweb.iter_study("http://localhost:8042", study_uid, base_path="/dicom-web"):
    ds = pydcm.dcmread(io.BytesIO(part10))
```

Other retrieve shapes: `retrieve_series`, `retrieve_frames`, `retrieve_*_metadata`,
`retrieve_rendered`, `retrieve_bulkdata`, and the streaming `start_retrieve` (writes parts to a
directory off the GIL).

### Requesting a transfer syntax

`retrieve_study` / `retrieve_series` / `retrieve_instance` / `start_retrieve` all take
`transfer_syntax=`, which sets the WADO-RS `Accept: …; transfer-syntax=` parameter:

| `transfer_syntax` | What you get |
|---|---|
| **omitted / `""`** (default) | The server's default encoding. Per PS3.18 that is **Explicit VR Little Endian** for `application/dicom` (the server may transcode to it) — it is **not** guaranteed to be the stored syntax. |
| **`"*"`** | The instances **as stored**, no transcoding — use this to retrieve compressed data verbatim. |
| **a TS UID** (e.g. `"1.2.840.10008.1.2.4.50"`) | That encoding; the server transcodes if it can, else falls back to its default. |

```python
# verbatim (keep J2K/JPEG as stored) — the no-transcode retrieve:
parts = dicomweb.retrieve_series(url, study_uid, series_uid, base_path="/dicom-web",
                                 transfer_syntax="*")
```

(For DIMSE C-GET the equivalent "as-stored" is the default — see the [DIMSE how-to](dimse.md):
the SCU offers the full transfer-syntax set, so a non-transcoding server returns each instance in
its stored syntax without you naming one.)

## Store (STOW-RS)

```python
result = dicomweb.store_instances(
    "http://localhost:8042",
    [open("ct.dcm", "rb").read()],
    base_path="/dicom-web",
)
# result → {"status": 200, "stored": [...], "failed": [...]}
```

## Delete

```python
dicomweb.delete_instance("http://localhost:8042", study_uid, series_uid, sop_uid, base_path="/dicom-web")
dicomweb.delete_series("http://localhost:8042", study_uid, series_uid, base_path="/dicom-web")
dicomweb.delete_study("http://localhost:8042", study_uid, base_path="/dicom-web")
```

## Worklist (UPS-RS)

Unified Procedure Step over the wire — create, query, claim and update work items:

```python
S, BP = "http://localhost:8080", "/dcm4chee-arc/aets/DCM4CHEE/rs"
wi = dicomweb.create_workitem(S, workitem_dataset, base_path=BP)
open_items = dicomweb.search_workitems(S, matches={"ProcedureStepState": "SCHEDULED"}, base_path=BP)
dicomweb.change_workitem_state(S, workitem_uid, "IN PROGRESS", transaction_uid, base_path=BP)
dicomweb.update_workitem(S, workitem_uid, changes, transaction_uid, base_path=BP)
```

Plus `retrieve_workitem`, `request_cancel_workitem`, and
`subscribe_workitem` / `unsubscribe_workitem` for event subscriptions.
