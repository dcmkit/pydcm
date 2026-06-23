# DIMSE networking

`pydcm.dimse` is a drop-in DIMSE API over the native stack: `import pydcm.dimse as
pynetdicom` and existing code mostly just works. SCU and a full SCP are both
supported, over a single persistent association.

## Verify connectivity (C-ECHO)

Existing code runs unchanged:

```python
import pydcm.dimse as pynetdicom

ae = pynetdicom.AE(ae_title="PYDCM")
assoc = ae.associate("pacs.local", 11112, ae_title="ANY-SCP")
if assoc.is_established:
    assoc.send_c_echo()
assoc.release()
```

The rest of this guide imports it under its own name, `dimse` — the natural
spelling for new code.

## Store instances (C-STORE)

The association is persistent — negotiate once, send many, release once:

```python
import pydcm.dimse as dimse

ae = dimse.AE(ae_title="PYDCM")
ae.requested_contexts = dimse.StoragePresentationContexts   # negotiate Storage SOP Classes
assoc = ae.associate("pacs.local", 11112)
for path in ["a.dcm", "b.dcm", "c.dcm"]:
    assoc.send_c_store(pydcm.dcmread(path))
assoc.release()
```

C-STORE ships verbatim over the **negotiated** presentation context, so the
Storage SOP Class must be requested *before* `associate()` — set
`requested_contexts` (as above) or `add_requested_context(<sop_class>)`. (C-ECHO /
C-FIND / C-GET / C-MOVE negotiate their context for you from the SOP class / model.)

## Query &amp; retrieve (C-FIND / C-GET / C-MOVE)

```python
from pydcm.sop_class import StudyRootQueryRetrieveInformationModelFind

query = pydcm.Dataset()
query.QueryRetrieveLevel = "STUDY"
query.PatientID = "42"
query.StudyInstanceUID = ""

assoc = ae.associate("pacs.local", 11112)
for status, identifier in assoc.send_c_find(query, StudyRootQueryRetrieveInformationModelFind):
    if identifier:
        print(identifier.StudyInstanceUID)
assoc.release()
```

`query_model` is the Q/R model's FIND SOP Class UID — use the `sop_class`
constants (spelled-out service names), not a one-letter shorthand.

`send_c_get` retrieves over the same association (role-flip storage receive);
`send_c_move` ships matches to a third-party destination AE.

## Run an SCP

```python
import pydcm.dimse as dimse

def handle_store(event):
    event.dataset.save_as(f"{event.dataset.SOPInstanceUID}.dcm", write_like_original=False)
    return 0x0000   # Success

ae = dimse.AE(ae_title="PYDCM-SCP")
ae.add_supported_context(dimse.sop_class.CTImageStorage)
ae.start_server(("0.0.0.0", 11112), evt_handlers=[(dimse.evt.EVT_C_STORE, handle_store)])
```

`start_server` wires `EVT_C_STORE` / `EVT_C_ECHO` / `EVT_C_FIND` / `EVT_C_GET` /
`EVT_C_MOVE` plus the DIMSE-N events.
