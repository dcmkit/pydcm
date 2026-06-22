# DICOMweb

`pydcm.dicomweb` is a QIDO / WADO / STOW / DELETE client over the native HTTP
stack — query, retrieve (streaming), store and delete against a remote server.

## Query (QIDO-RS)

```python
from pydcm import dicomweb

studies  = dicomweb.search_studies("https://pacs.example.com",
                                   matches={"PatientID": "42"})
series   = dicomweb.search_series("https://pacs.example.com", study_uid)
instances = dicomweb.search_instances("https://pacs.example.com", study_uid, series_uid)
```

Pass `includefields=[...]` to widen the returned attributes and `auth=...` for a
bearer/basic credential.

## Retrieve (WADO-RS)

```python
# whole objects
parts = dicomweb.retrieve_study("https://pacs.example.com", study_uid)   # list[bytes] Part-10
inst  = dicomweb.retrieve_instance("https://pacs.example.com", study_uid, series_uid, sop_uid)

# streaming — bounded memory over a large study
for part10 in dicomweb.iter_study("https://pacs.example.com", study_uid):
    ds = pydcm.dcmread(io.BytesIO(part10))
```

Other retrieve shapes: `retrieve_series`, `retrieve_frames`,
`retrieve_*_metadata`, `retrieve_rendered`, `retrieve_bulkdata`. Pass
`transfer_syntax=...` to negotiate the wire encoding.

## Store (STOW-RS)

```python
result = dicomweb.store_instances(
    "https://pacs.example.com",
    [open("ct.dcm", "rb").read()],
)
```

## Delete

```python
dicomweb.delete_instance("https://pacs.example.com", study_uid, series_uid, sop_uid)
dicomweb.delete_series("https://pacs.example.com", study_uid, series_uid)
dicomweb.delete_study("https://pacs.example.com", study_uid)
```

## Worklist (UPS-RS)

Unified Procedure Step over the wire — create, query, claim and update work items:

```python
wi = dicomweb.create_workitem("https://pacs.example.com", workitem_dataset)
open_items = dicomweb.search_workitems("https://pacs.example.com",
                                       matches={"ProcedureStepState": "SCHEDULED"})
dicomweb.change_workitem_state("https://pacs.example.com", workitem_uid,
                               "IN PROGRESS", transaction_uid)
dicomweb.update_workitem("https://pacs.example.com", workitem_uid, changes, transaction_uid)
```

Plus `retrieve_workitem`, `request_cancel_workitem`, and
`subscribe_workitem` / `unsubscribe_workitem` for event subscriptions.
