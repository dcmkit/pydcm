# DIMSE networking

`pydcm.dimse` is a pynetdicom-compatible DIMSE API over the native stack:
`import pydcm.dimse as pynetdicom` works for common SCU/SCP workflows. SCU
and a full SCP are both supported, over a single persistent association.

## Connecting

A DIMSE peer is addressed by **(host, port, Called-AE-Title)** — the AE Title and port are
per-product:

| Server           | host:port (default) | Called AE (`ae_title=`) |
|------------------|---------------------|-------------------------|
| **Orthanc**      | `localhost:4242`    | `ORTHANC`               |
| **dcm4chee-arc** | `localhost:11112`   | `DCM4CHEE`              |

Your own **Calling** AE Title (`AE(ae_title=…)`, here `PYDCM`) only matters for C-MOVE (the move
destination, below) and on servers that allow-list calling AEs.

On multihomed or VPN hosts, outbound SCU connections normally use the source address selected by
the operating system route to `host:port` (the DCMTK-style default). If a peer allow-list requires
a specific local address, pass `bind_address=("VPN_IP", 0)` to `AE.associate()` or the retrieve
helpers. `print_image()` accepts the same argument for Print Management workflows. The second
tuple item is the local source port; keep it `0` unless an integration explicitly requires
otherwise.

## Verify connectivity (C-ECHO)

Existing code runs unchanged:

```python
import pydcm.dimse as pynetdicom

ae = pynetdicom.AE(ae_title="PYDCM")
assoc = ae.associate("localhost", 4242, ae_title="ORTHANC")   # dcm4chee: 11112, "DCM4CHEE"
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

### C-GET — retrieve over the same association

The matched instances come back as inbound C-STORE sub-operations on the **same** association, so
you must **`add_supported_context(<storage class>)`** for the classes you accept (this negotiates
the role-flipped receive channels) and handle `EVT_C_STORE`:

```python
from pydcm.dimse import sop_class as sc

got = []
def on_store(event):
    got.append(event.dataset)          # event.dataset, event.transfer_syntax, event.sop_instance_uid
    return 0x0000

ae = dimse.AE(ae_title="PYDCM")
ae.add_requested_context(sc.StudyRootQueryRetrieveInformationModelGet)
ae.add_supported_context(sc.CTImageStorage)         # REQUIRED — one per class the study holds
ae.add_supported_context(sc.SegmentationStorage)    # (a prior C-FIND lists them — see below)
assoc = ae.associate("localhost", 4242, ae_title="ORTHANC",
                     evt_handlers=[(dimse.evt.EVT_C_STORE, on_store)])
q = pydcm.Dataset(); q.QueryRetrieveLevel = "STUDY"; q.StudyInstanceUID = study_uid
for status, _ in assoc.send_c_get(q, sc.StudyRootQueryRetrieveInformationModelGet):
    pass
assoc.release()
```

**Declare every Storage class the study holds.** A non-transcoding server (Orthanc) returns each
instance in its **stored** transfer syntax, so it can only send a class it has a negotiated context
for — and it **aborts the whole C-GET** if a matched instance has none. The prior C-FIND lists the
study's classes in `SOPClassesInStudy` (or read each series' `Modality`); pass one
`add_supported_context` per class. Each opens the role-flipped receive channels with their full
transfer-syntax set, and pydcm writes back whatever it negotiated **byte-for-byte** (compressed
stays compressed — no transcode). Omit `add_supported_context` entirely and `send_c_get` falls back
to a one-shot association offering the common image and report classes.

### C-MOVE — ship matches to a destination AE

C-MOVE tells the server to send the matches to a **third-party** Storage-SCP, identified by AE
Title. So you (a) run that listener and (b) register its AE Title on the server:

```python
# (a) run the destination Storage-SCP (here, ourselves) — the server connects BACK to it
recv = []
listener = dimse.AE(ae_title="PYDCM")
handle = listener.start_server(("0.0.0.0", 11200), block=False,
                               evt_handlers=[(dimse.evt.EVT_C_STORE, lambda e: (recv.append(1), 0x0000)[1])])
# (b) register "PYDCM" → host:11200 on the server (Orthanc DicomModalities; dcm4chee device).
#     On a VPN, "host" must be the address the Move SCP can connect back to, often the VPN IP.
ae = dimse.AE(ae_title="PYDCM")
ae.add_requested_context(sc.StudyRootQueryRetrieveInformationModelMove)
assoc = ae.associate("localhost", 4242, ae_title="ORTHANC")
q = pydcm.Dataset(); q.QueryRetrieveLevel = "STUDY"; q.StudyInstanceUID = study_uid
for status, _ in assoc.send_c_move(q, "PYDCM", sc.StudyRootQueryRetrieveInformationModelMove):
    pass
assoc.release(); handle.shutdown()
```

An unregistered destination AE trips `0xA801` "Move Destination unknown".

If the PACS cannot connect back to your listener because of VPN/NAT/firewall routing, prefer
C-GET: the retrieved instances return as C-STORE sub-operations on the same association.

## TLS (DIMSE over TLS)

Set `ae.tls` to a **dict of file paths** (a Python `ssl.SSLContext` cannot be reused by the
native OpenSSL engine). By default the dialled host must match a certificate
DNS/IP SAN; `server_name` explicitly overrides that identity and DNS SNI:

```python
ae = dimse.AE(ae_title="PYDCM"); ae.add_requested_context(sc.Verification)
ae.tls = {"ca_file": "ca.crt", "server_name": "localhost"}   # + cert_file/key_file for mTLS
assoc = ae.associate("localhost", 2762, ae_title="ORTHANC")  # the server's DIMSE-TLS port
assoc.send_c_echo(); assoc.release()
```

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
