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

Rather than hand-writing the header string, `resolve_auth` builds it from a structured config
and returns the `(header_name, header_value)` it produces — for **basic**, **bearer** and
**oauth2** the name is `Authorization`, so the `value` is exactly what `auth=` expects:

```python
name, value = dicomweb.resolve_auth({"scheme": "basic",
                                     "username": "orthanc", "password": "orthanc"})
dicomweb.search_studies("http://localhost:8042", base_path="/dicom-web", auth=value)
```

`scheme` is one of `"none"` / `"basic"` / `"bearer"` / `"api_key"` / `"oauth2"`. The **oauth2**
scheme acquires, refreshes and caches a Bearer token for you (cache → `refresh_token` →
client-credentials), so a long-running client keeps a fresh token without a manual refresh loop:

```python
name, value = dicomweb.resolve_auth({
    "scheme": "oauth2",
    "token_url": "https://keycloak/realms/dcm/protocol/openid-connect/token",
    "client_id": "pacs", "client_secret": "…", "scope": "openid",
    # optional: refresh_token, use_cache (default True), cache_dir
})
dicomweb.search_studies("http://localhost:8080",
                        base_path="/dcm4chee-arc/aets/DCM4CHEE/rs", auth=value)
```

`"api_key"` uses a `token` under a custom `header_name` (default `Authorization`); `"none"`
returns `("", "")`.

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

`search_workitems` defaults to `limit=None`, which omits the parameter entirely (`limit=0`
explicitly requests zero matches).

`update_workitem`'s `transaction_uid` is **optional**: omit it (or pass `""`) to update a
`SCHEDULED` workitem; supply the current Transaction UID once the workitem is `IN PROGRESS`:

```python
dicomweb.update_workitem(S, workitem_uid, changes, base_path=BP)                    # SCHEDULED
dicomweb.update_workitem(S, workitem_uid, changes, transaction_uid, base_path=BP)   # IN PROGRESS
```

`change_workitem_state` and `request_cancel_workitem` both take an optional `requester=` (the
requesting AE Title carried as a query parameter):

```python
dicomweb.request_cancel_workitem(S, workitem_uid, reason=cancel_ds,
                                 requester="MY_SCU", base_path=BP)
```

### Subscriptions

Subscribe an AE Title to workitem events; pass the well-known global subscription instance UID as
`workitem_uid` to watch every workitem. `deletion_lock=True` asks the server to retain a completed
workitem until you unsubscribe, and `filters=` narrows a global subscription to matching workitems:

```python
dicomweb.subscribe_workitem(S, workitem_uid, "MY_SCU", deletion_lock=True, base_path=BP)
dicomweb.subscribe_workitem(S, global_uid, "MY_SCU",
                            filters={"ScheduledStationAETitle": "CT_SCANNER"}, base_path=BP)
dicomweb.unsubscribe_workitem(S, workitem_uid, "MY_SCU", base_path=BP)
dicomweb.suspend_global_subscription(S, global_uid, "MY_SCU", base_path=BP)   # stop new events, keep the subscription
```

Plus `retrieve_workitem` to read one workitem back as DICOM JSON.
