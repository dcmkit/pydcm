# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Fuli Wu
# NOT a medical device — not for clinical or diagnostic use.

"""DIMSE networking (``pydcm.dimse``).

A thin, same-interface layer over pydcm's native C++ DIMSE engine, via the
``pydcm._dimse`` binding.
Ported code works after one alias::

    import pydcm.dimse as pynetdicom          # the drop-in for pynetdicom
    from pydcm.dimse import AE, evt

    ae = AE(ae_title="MY_SCU")
    ae.add_requested_context(Verification)
    assoc = ae.associate("127.0.0.1", 11112)
    if assoc.is_established:
        status = assoc.send_c_echo()          # Dataset with .Status
        assoc.release()

    # SCP:
    ae.add_supported_context(CTImageStorage)
    server = ae.start_server(("0.0.0.0", 11112), block=False,
                             evt_handlers=[(evt.EVT_C_STORE, handle_store)])

``ae.associate()`` opens ONE association (a persistent native client)
that ``send_c_echo`` / ``send_c_store`` / ``send_c_find`` / ``send_c_move`` reuse, exactly like
One negotiate, many ops, one ``release()``. The contexts negotiated are the ones
you ``add_requested_context``'d (plus a broad default set so basic flows work out of the box).
``send_c_get`` also reuses the persistent association when you ``add_supported_context`` the
storage classes to receive — they are negotiated with the PS3.7 §D.3.3.4 role flip (scp_role) at
``associate()`` so the matched instances arrive as inbound C-STORE on the same connection. Without
any supported context it falls back to a one-shot association (CT/MR defaults). TLS is opt-in via
``ae.tls = {...}`` (ca_file/cert_file/key_file/verify_peer/server_name/ciphers/check_hostname).
``ciphers`` takes ``"bcp195"`` for the curated RFC 9325 AEAD+PFS allowlist, ``""`` for the provider
default, or a raw OpenSSL cipher string; the server side also wins cipher selection. SCU TLS always
verifies the certificate chain (CA trust) when ``verify_peer`` is on; hostname/SAN matching is done
only when ``server_name`` is set (or ``check_hostname=True``) — the pynetdicom idiom, since DICOM
peers are dialled by IP/AE, not DNS. ``verify_peer=False`` drops all verification (dev only).
"""

from __future__ import annotations

import struct
import functools
import sys

from . import dcmread
from . import sop_class            # sop_class drop-in (pydcm.dimse.sop_class.*)
from ._dicom import Dataset

# dimse is a module (not a package), so register sop_class as a submodule too — lets
# `from pydcm.dimse.sop_class import CTImageStorage` work, not just attribute access.
sys.modules.setdefault(__name__ + ".sop_class", sop_class)

try:
    from . import _dimse
except ImportError as _e:                            # pragma: no cover
    raise ImportError(
        "pydcm.dimse needs the optional native DIMSE extension (pydcm._dimse). "
        "This wheel was built without it."
    ) from _e

# Default DIMSE statuses (PS3.7 Annex C).
STATUS_SUCCESS = 0x0000
STATUS_PENDING = 0xFF00


# --------------------------------------------------------------------------- #
#  Unified Procedure Step (UPS) — PS3.4 §CC vocabulary
# --------------------------------------------------------------------------- #
# UPS over DIMSE is just the generic N-services below + these constants; it gets
# no module of its own (UPS-RS likewise rides QIDO/DICOM-JSON, with no dedicated
# module). SOP-class UIDs reuse the registry-sourced ``sop_class``
# objects; the state wire strings + action ids are projected from the single
# native source (via ``_dimse.ups_vocabulary``) — no
# Python-side copy. The Association.ups_* helpers below are thin sugar over
# send_n_* that build the small action-info datasets PS3.4 §CC specifies.
_ups_vocab = _dimse.ups_vocabulary()


class UPS:
    """UPS (PS3.4 §CC) constants the SCU sugar references — UIDs from
    ``pydcm.sop_class``, state strings / action ids from the native engine."""

    # SOP-class UIDs — reuse the existing registry-sourced UID objects.
    PUSH = sop_class.UnifiedProcedureStepPush
    WATCH = sop_class.UnifiedProcedureStepWatch
    PULL = sop_class.UnifiedProcedureStepPull
    EVENT = sop_class.UnifiedProcedureStepEvent

    # Well-known subscription instance UIDs (native-sourced).
    GLOBAL_SUBSCRIPTION = _ups_vocab["global_subscription_instance"]
    FILTERED_GLOBAL_SUBSCRIPTION = _ups_vocab["filtered_global_subscription_instance"]

    # Procedure Step State wire strings (e.g. IN_PROGRESS == "IN PROGRESS").
    SCHEDULED = _ups_vocab["states"]["SCHEDULED"]
    IN_PROGRESS = _ups_vocab["states"]["IN_PROGRESS"]
    COMPLETED = _ups_vocab["states"]["COMPLETED"]
    CANCELED = _ups_vocab["states"]["CANCELED"]

    # N-ACTION type ids.
    CHANGE_STATE = _ups_vocab["actions"]["CHANGE_STATE"]
    REQUEST_CANCEL = _ups_vocab["actions"]["REQUEST_CANCEL"]
    SUBSCRIBE = _ups_vocab["actions"]["SUBSCRIBE"]
    UNSUBSCRIBE = _ups_vocab["actions"]["UNSUBSCRIBE"]


_mpps_vocab = _dimse.mpps_vocabulary()
_stgcmt_vocab = _dimse.storage_commitment_vocabulary()
_ian_vocab = _dimse.ian_vocabulary()
_print_vocab = _dimse.print_vocabulary()


class MPPS:
    """MPPS (PS3.4 §F) constants the MPPS SCU sugar references — SOP class UID
    from ``pydcm.sop_class``, PerformedProcedureStepStatus strings from the native engine."""

    SOP_CLASS = sop_class.ModalityPerformedProcedureStep

    # (0040,0252) PerformedProcedureStepStatus wire strings.
    IN_PROGRESS = _mpps_vocab["states"]["IN_PROGRESS"]
    COMPLETED = _mpps_vocab["states"]["COMPLETED"]
    DISCONTINUED = _mpps_vocab["states"]["DISCONTINUED"]


class StorageCommitment:
    """Storage Commitment Push Model (PS3.4 §J) constants the SCU sugar
    references — SOP class UID from ``pydcm.sop_class``, well-known instance
    UID + action/event ids from the native engine."""

    SOP_CLASS = sop_class.StorageCommitmentPushModel

    # Well-known SOP instance UID every N-ACTION-RQ targets / every
    # N-EVENT-REPORT-RQ carries (PS3.4 §J.3).
    PUSH_MODEL_INSTANCE = _stgcmt_vocab["push_model_instance"]

    # N-ACTION action id (1 = Request Storage) and the two N-EVENT-REPORT
    # outcome event ids (1 = all committed, 2 = complete with failures).
    ACTION_REQUEST_STORAGE = _stgcmt_vocab["action_request_storage"]
    EVENT_SUCCESS = _stgcmt_vocab["events"]["SUCCESS"]
    EVENT_FAILURES = _stgcmt_vocab["events"]["FAILURES"]


class IAN:
    """Instance Availability Notification (PS3.4 §R) constants the SCU sugar
    references — SOP class UID from ``pydcm.sop_class``, (0008,0056) Instance
    Availability wire strings from the native engine."""

    SOP_CLASS = sop_class.InstanceAvailabilityNotification

    # (0008,0056) Instance Availability values.
    ONLINE = _ian_vocab["availability"]["ONLINE"]
    NEARLINE = _ian_vocab["availability"]["NEARLINE"]
    OFFLINE = _ian_vocab["availability"]["OFFLINE"]
    UNAVAILABLE = _ian_vocab["availability"]["UNAVAILABLE"]


class Print:
    """Print Management (PS3.4 §H) constants the SCU sugar references — SOP
    class UIDs from ``pydcm.sop_class``, well-known Printer instance UID and
    the Print N-ACTION type id from the native engine."""

    # §H Meta SOP classes (negotiate ONE of these; member classes ride it).
    GRAYSCALE_META = sop_class.BasicGrayscalePrintManagementMeta
    COLOR_META      = sop_class.BasicColorPrintManagementMeta

    # §H member SOP classes (used as the N-* command's SOP class UID).
    FILM_SESSION        = sop_class.BasicFilmSession
    FILM_BOX            = sop_class.BasicFilmBox
    GRAYSCALE_IMAGE_BOX = sop_class.BasicGrayscaleImageBox
    COLOR_IMAGE_BOX     = sop_class.BasicColorImageBox
    PRINTER             = sop_class.PrinterSOPClass

    # Well-known Printer SOP instance UID (N-GET / N-EVENT-REPORT target).
    PRINTER_INSTANCE = _print_vocab["printer_instance"]

    # N-ACTION type id for the Print action (action_type_id = 1).
    ACTION_PRINT = _print_vocab["actions"]["PRINT"]


# --------------------------------------------------------------------------- #
#  Dataset <-> DIMSE-bytes marshaling
# --------------------------------------------------------------------------- #
def _dataset_offset(part10: bytes) -> int:
    """Byte offset of the dataset (past the 128 preamble + 'DICM' + group-0002)."""
    if part10[128:132] != b"DICM":
        return 0                                   # naked dataset (no preamble/meta)
    # (0002,0000) FileMetaInformationGroupLength is the first group-2 element:
    # 8-byte explicit-VR-LE header (tag+VR+len) + 4-byte UL value = the byte count
    # of the rest of group 2.
    group_len = struct.unpack_from("<I", part10, 140)[0]
    return 144 + group_len


def _encode(ds: Dataset):
    """(sop_class, sop_instance, transfer_syntax, dataset_bytes) from a Dataset."""
    p10 = ds._encode_part10()
    off = _dataset_offset(p10)
    ts = str(ds.file_meta.get("TransferSyntaxUID") or "1.2.840.10008.1.2") \
        if ds.file_meta is not None else "1.2.840.10008.1.2"
    return (str(ds.get("SOPClassUID") or ""), str(ds.get("SOPInstanceUID") or ""),
            ts, p10[off:])


def _identifier_bytes(ds: Dataset) -> bytes:
    """Encode a query/identifier dataset to bare Implicit VR LE wire bytes (no file meta,
    no SOP-UID requirement) — C-FIND/MOVE/GET keys are a dataset, not an instance."""
    from . import _native
    return _native.encode_ivr(ds.to_json())


_IMPLICIT_VR_LE = "1.2.840.10008.1.2"
_UNCOMPRESSED_LE = {"1.2.840.10008.1.2", "1.2.840.10008.1.2.1"}


def _reencode_to(ds: Dataset, src_ts: str, target_ts: str) -> bytes:
    """Re-encode a C-STORE data set from ``src_ts`` to the association's negotiated
    ``target_ts``, returning the bare dataset PDV bytes. Lossless for the native-LE pair
    (Implicit↔Explicit VR LE); falls back to the native transcoder for encapsulated targets.
    Raises (never silently drops pixels) when no safe re-encoding exists."""
    from . import _native
    if target_ts == _IMPLICIT_VR_LE and src_ts in _UNCOMPRESSED_LE:
        return _native.encode_ivr(ds.to_json())          # native → Implicit VR LE, pixels inline
    try:                                                 # encapsulated target → real transcoder
        p10 = _native.transcode(ds._encode_part10(), target_ts)
        return p10[_dataset_offset(p10):]
    except Exception as exc:                             # noqa: BLE001 — surfaced below
        raise ValueError(
            f"instance transfer syntax {src_ts} does not match the negotiated {target_ts}; "
            f"re-encoding is not supported for this pair [{exc}]. Add a requested context "
            f"whose transfer syntax matches the instance, or transcode it first.") from exc


def _wrap_part10(data: bytes, ts: str, sop_class: str = "", sop_instance: str = "") -> bytes:
    """Wrap a bare DIMSE data set in a minimal Part-10 (preamble + group-2 meta declaring the
    transfer syntax) so the reader parses it DETERMINISTICALLY in ``ts`` rather than guessing —
    auto-detection mis-reads some implicit-VR datasets (e.g. group 0x2110 printer attributes)."""
    def ui(group, elem, val):
        v = val.encode("ascii")
        if len(v) % 2:
            v += b"\x00"
        return struct.pack("<HH", group, elem) + b"UI" + struct.pack("<H", len(v)) + v
    body = b""
    if sop_class:
        body += ui(0x0002, 0x0002, sop_class)
    if sop_instance:
        body += ui(0x0002, 0x0003, sop_instance)
    body += ui(0x0002, 0x0010, ts)                   # (0002,0010) TransferSyntaxUID
    glen = (struct.pack("<HH", 0x0002, 0x0000) + b"UL"
            + struct.pack("<H", 4) + struct.pack("<I", len(body)))
    return b"\x00" * 128 + b"DICM" + glen + body + data


def _to_dataset(data: bytes, ts: str = _IMPLICIT_VR_LE,
                sop_class: str = "", sop_instance: str = "") -> Dataset:
    """Parse received DIMSE data-set bytes (bare, in transfer syntax ``ts``) into a Dataset."""
    import os, tempfile
    fd, tmp = tempfile.mkstemp(suffix=".dcm", prefix="pydcm_dimse_")
    try:
        os.write(fd, _wrap_part10(data, ts, sop_class, sop_instance))
        os.close(fd)
        return dcmread(tmp, force=True)
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass


def _status(code: int) -> Dataset:
    ds = Dataset()
    ds.Status = int(code)
    return ds


def _reply_dataset(data: bytes):
    """A DIMSE-N reply data set's bytes → Dataset, or None when the peer sent none."""
    return _to_dataset(data) if data else None


def _tag_pair(tag):
    """Normalise a tag (a Tag, int, or (group, element)) to (group, element)."""
    if hasattr(tag, "group") and hasattr(tag, "element"):
        return (int(tag.group), int(tag.element))
    if isinstance(tag, (tuple, list)):
        return (int(tag[0]), int(tag[1]))
    t = int(tag)
    return (t >> 16, t & 0xFFFF)


def _norm_n_return(rv):
    """An N-service SCP handler's return → (status_int, reply_bytes).
    Accepts (status, dataset), a bare status, or a status Dataset; dataset may be None."""
    if isinstance(rv, tuple):
        status = rv[0] if rv else STATUS_SUCCESS
        ds = rv[1] if len(rv) > 1 else None
    else:
        status, ds = rv, None
    status_int = int(getattr(status, "Status", status))
    return (status_int, _identifier_bytes(ds) if ds is not None else b"")


def _norm_status(rv):
    return int(getattr(rv, "Status", rv) if rv is not None else STATUS_SUCCESS)


_MODEL = {
    "1.2.840.10008.5.1.4.1.2.1.1": "patient", "1.2.840.10008.5.1.4.1.2.1.2": "patient",
    "1.2.840.10008.5.1.4.1.2.1.3": "patient",
    "1.2.840.10008.5.1.4.1.2.2.1": "study",   "1.2.840.10008.5.1.4.1.2.2.2": "study",
    "1.2.840.10008.5.1.4.1.2.2.3": "study",
}


def _model_of(query_model) -> str:
    s = str(query_model)
    if s in _MODEL:
        return _MODEL[s]
    return "study" if "study" in s.lower() else "patient"


# Negotiated when the caller declared no contexts (the API requires add_requested_context;
# we propose a broad safety net so common echo/store/find/move flows work unconfigured).
_DEFAULT_TS = ("1.2.840.10008.1.2", "1.2.840.10008.1.2.1")           # Implicit + Explicit VR LE
_QR_MODELS = [f"1.2.840.10008.5.1.4.1.2.{r}.{s}" for r in (1, 2, 3) for s in (1, 2)]  # FIND+MOVE
_COMMON_STORAGE = [
    "1.2.840.10008.5.1.4.1.1.1", "1.2.840.10008.5.1.4.1.1.1.1",      # CR, DX
    "1.2.840.10008.5.1.4.1.1.1.2", "1.2.840.10008.5.1.4.1.1.2",      # MG, CT
    "1.2.840.10008.5.1.4.1.1.4", "1.2.840.10008.5.1.4.1.1.6.1",      # MR, US
    "1.2.840.10008.5.1.4.1.1.7", "1.2.840.10008.5.1.4.1.1.20",       # SC, NM
    "1.2.840.10008.5.1.4.1.1.128", "1.2.840.10008.5.1.4.1.1.88.11",  # PET, Basic SR
    "1.2.840.10008.5.1.4.1.1.66.4", "1.2.840.10008.5.1.4.1.1.104.1", # SEG, Encapsulated PDF
]
# (abstract_syntax, [transfer_syntaxes]) — Verification + Q/R FIND/MOVE + common storage, all IVR+EVR.
_DEFAULT_CONTEXTS = [(uid, list(_DEFAULT_TS))
                     for uid in ("1.2.840.10008.1.1", *_QR_MODELS, *_COMMON_STORAGE)]


class PresentationContext:
    """A negotiated presentation context: ``abstract_syntax`` +
    ``transfer_syntax`` (a list), ``context_id`` and ``result`` (0x00 = acceptance)."""
    def __init__(self, context_id=None, abstract_syntax=None, transfer_syntax=(), result=None):
        self.context_id = context_id
        self.abstract_syntax = abstract_syntax
        self.transfer_syntax = list(transfer_syntax)
        self.result = result

    def __repr__(self):
        if self.result is None:                      # un-negotiated (a proposed/built context)
            return f"<PresentationContext {self.abstract_syntax}>"
        verdict = "accepted" if self.result == 0x00 else "rejected"
        return f"<PresentationContext {self.abstract_syntax} ({verdict})>"


# --------------------------------------------------------------------------- #
#  Association (persistent — one negotiate, many ops, one release)
# --------------------------------------------------------------------------- #
def _empty_on_net_error(*, reply=False):
    """Map a transport/timeout/abort ConnectionError from a native DIMSE op to an
    empty-``Dataset`` result. A USAGE error (no negotiated context / association
    not established / bad encoding) still raises.
    ``reply=True`` for the ``(status, dataset)`` N-services; False for status-only ops."""
    def deco(fn):
        @functools.wraps(fn)
        def wrap(*args, **kwargs):
            try:
                return fn(*args, **kwargs)
            except ConnectionError:
                return (Dataset(), None) if reply else Dataset()
        return wrap
    return deco


def _empty_gen_on_net_error(fn):
    """Generator form (C-FIND/MOVE/GET): a transport/timeout/abort ConnectionError
    becomes one final ``(Dataset(), None)`` yield and the stream ends."""
    @functools.wraps(fn)
    def wrap(*args, **kwargs):
        try:
            yield from fn(*args, **kwargs)
        except ConnectionError:
            yield Dataset(), None
    return wrap



class Association:
    """A pydcm DIMSE association, persistent: ``ae.associate()`` opens it,
    every ``send_c_*`` reuses it, ``release()`` tears it down. C-GET is the lone exception
    (one-shot, for SCU-role storage negotiation — see the module docstring)."""

    def __init__(self, ae, host, port, called_ae, n_event_handler=None, lifecycle=None):
        self._ae = ae
        self._host, self._port, self._called = host, int(port), called_ae
        self.is_aborted = False
        self.is_released = False
        self.is_rejected = False
        self._client = None
        self._lifecycle = lifecycle or {}            # EVT_ESTABLISHED / RELEASED / ABORTED handlers
        # dedup by abstract syntax (keep the first transfer-syntax list declared for it)
        contexts, seen = [], set()
        for abstract, ts in (ae._requested or _DEFAULT_CONTEXTS):
            if abstract not in seen:
                seen.add(abstract)
                contexts.append((abstract, list(ts)))
        self._contexts = contexts                    # what we proposed (for accepted/rejected_contexts)
        # Inbound N-EVENT-REPORT on THIS association (Storage Commitment result / UPS event):
        # adapt the user's EVT_N_EVENT_REPORT handler to the native (status, reply_bytes) form.
        native_event_cb = None
        if n_event_handler is not None:
            def native_event_cb(sc, si, et, info, _h=n_event_handler):
                return _norm_n_return(_h(_NEvent(sc, si, dataset=_reply_dataset(info), type_id=et)))
        # associate() NEVER raises on a failed/refused/unreachable peer — it
        # returns an Association with is_established=False so callers can guard on it. Match
        # that: swallow the connect error and degrade rather than propagate.
        # Storage classes added via add_supported_context are negotiated with the PS3.7
        # role flip (scp_role) so a persistent C-GET can receive them as inbound C-STORE
        # on THIS association. Empty ⇒ send_c_get falls back to a one-shot association.
        self._get_storage = list(ae._supported)
        try:
            self._client = _dimse.client_connect(
                host=host, port=int(port), calling_ae=ae.ae_title, called_ae=called_ae,
                contexts=contexts, tls=ae.tls, on_n_event_report=native_event_cb,
                timeout_ms=ae._establish_timeout_ms(),
                dimse_timeout_ms=int(ae.dimse_timeout * 1000) if ae.dimse_timeout else 0,
                get_storage_classes=self._get_storage)
            self.is_established = self._client.is_established()
        except (ConnectionError, OSError) as exc:    # association reject / transport / unreachable
            self.is_established = False               # (a bad AE title raises ValueError — a usage error)
            self.is_rejected = "reject" in str(exc).lower()   # A-ASSOCIATE-RJ vs transport failure
        if self.is_established:
            self._fire("established")

    def _fire(self, name):
        h = self._lifecycle.get(name)
        if h is not None:
            h(_LifecycleEvent(self))

    # ── Association surface (this side is always the requestor) ──────
    @property
    def ae(self):
        return self._ae

    @property
    def is_requestor(self):
        return True

    @property
    def is_acceptor(self):
        return False

    @property
    def mode(self):
        return "requestor"

    @property
    def local(self):
        """The local (requestor) AE — {ae_title, address, port}."""
        return {"ae_title": self._ae.ae_title, "address": "", "port": 0}

    @property
    def remote(self):
        """The remote (acceptor) AE — {ae_title, address, port}."""
        return {"ae_title": self._called, "address": self._host, "port": self._port}

    def kill(self):
        """Forcibly terminate the association — same as abort()."""
        return self.abort()

    @property
    def acse_timeout(self):
        return self._ae.acse_timeout

    @property
    def dimse_timeout(self):
        return self._ae.dimse_timeout

    @property
    def network_timeout(self):
        return self._ae.network_timeout

    def bind(self, event, handler, args=None):
        """Bind a handler to a lifecycle event (EVT_ESTABLISHED/RELEASED/ABORTED) on this
        association."""
        self._lifecycle[getattr(event, "name", str(event)).lower()] = handler

    def unbind(self, event, handler):
        """Unbind a lifecycle event handler."""
        key = getattr(event, "name", str(event)).lower()
        if self._lifecycle.get(key) is handler:
            del self._lifecycle[key]

    def send_c_cancel(self, msg_id, context_id=None):
        """Accepted for source compatibility. The native engine runs each C-FIND/GET/
        MOVE to completion (no per-operation C-CANCEL), so this is a no-op; use ``abort()``
        to tear the association down."""
        return None

    @property
    def _common(self):
        # for the one-shot C-GET fallback (the persistent Client can't receive C-STORE)
        return dict(host=self._host, port=self._port, calling_ae=self._ae.ae_title,
                    called_ae=self._called, tls=self._ae.tls)

    def _require(self):
        if self._client is None or not self.is_established:
            raise RuntimeError("the association is not established (or was released)")

    @property
    def accepted_contexts(self):
        """Presentation contexts the peer accepted — each carries the
        negotiated ``transfer_syntax``. Built from the proposals + the engine's per-context
        negotiation result."""
        return self._contexts_by_result(accepted=True)

    @property
    def rejected_contexts(self):
        """Presentation contexts the peer did not accept."""
        return self._contexts_by_result(accepted=False)

    def _contexts_by_result(self, accepted):
        out = []
        if self._client is None:
            return out
        # proposals get presentation-context ids 1, 3, 5, … in order (accepted ones keep theirs)
        for cid, (abstract, ts) in zip(range(1, 256, 2), self._contexts):
            neg = self._client.accepted_ts(abstract)
            if (neg is not None) == accepted:
                out.append(PresentationContext(
                    cid, abstract, [neg] if neg is not None else ts,
                    0x00 if neg is not None else 0x03))
        return out

    @_empty_on_net_error()
    def send_c_echo(self, msg_id=1, context_id=None):
        self._require()
        return _status(self._client.echo(repeat=1)["status"])

    @_empty_on_net_error()
    def send_c_store(self, dataset, msg_id=1, originator_aet=None,
                     originator_id=None, context_id=None):
        self._require()
        # The persistent association negotiated ONE transfer syntax per SOP class; the
        # native store ships verbatim (no transcode), so the data set must be in exactly
        # that syntax. Re-encode to the negotiated TS when the instance differs (the peer
        # does the same for native datasets) — never silently drop the instance.
        sc, si, ts, data = _encode(dataset)
        neg = self._client.accepted_ts(sc)
        if neg is None:
            raise ValueError(
                f"no accepted presentation context for SOP Class {sc}; "
                f"add_requested_context(<sop_class>) before associate()")
        if neg != ts:
            data, ts = _reencode_to(dataset, ts, neg), neg
        r = self._client.store(instances=[(sc, si, ts, data)])
        if r["succeeded"] == 0 and r["sent"] > 0:        # transmitted-and-refused, never success
            return _status(r["last_status"] or 0xA700)   # 0xA700: Refused — Out of Resources
        return _status(r["last_status"])

    @_empty_gen_on_net_error
    def send_c_find(self, dataset, query_model, msg_id=1, priority=2):
        # query_model IS the FIND SOP class UID (a Q/R root, Modality Worklist, UPS Pull, …).
        self._require()
        ident = _identifier_bytes(dataset)
        for data in self._client.find(sop_class=str(query_model), identifier=ident):
            yield _status(STATUS_PENDING), _to_dataset(data)
        yield _status(STATUS_SUCCESS), None

    @_empty_gen_on_net_error
    def send_c_move(self, dataset, move_aet, query_model, msg_id=1, priority=2):
        self._require()
        ident = _identifier_bytes(dataset)
        r = self._client.move(sop_class=str(query_model),
                              destination=str(move_aet), identifier=ident)
        yield _status_with_counts(r), None

    @_empty_gen_on_net_error
    def send_c_get(self, dataset, query_model, msg_id=1, priority=2):
        """C-GET. Matched instances are delivered to the EVT_C_STORE handler registered on
        this AE. When the AE has add_supported_context'd the
        storage classes (negotiated with SCP role at associate()), it runs on the PERSISTENT
        association; otherwise it falls back to a one-shot association that negotiates a CT/MR
        default set to receive the inbound C-STORE-RQs."""
        self._require()
        ident = _identifier_bytes(dataset)
        store_handler = self._ae._get_store_handler()

        def on_store(sop_class, sop_instance, ts, data):
            if store_handler is None:
                return STATUS_SUCCESS
            evt_obj = _StoreEvent(_to_dataset(data, ts, sop_class, sop_instance),
                                  sop_class, sop_instance, ts, self)
            rv = store_handler(evt_obj)
            return int(getattr(rv, "Status", rv) if rv is not None else STATUS_SUCCESS)

        if self._get_storage:        # storage classes negotiated scp_role at associate()
            r = self._client.get(str(query_model), ident, on_store)
        else:                        # one-shot fallback (no add_supported_context)
            accept = ["1.2.840.10008.5.1.4.1.1.2", "1.2.840.10008.5.1.4.1.1.4"]  # CT/MR
            r = _dimse.get(model=_model_of(query_model), identifier=ident,
                           accept_storage_classes=accept, on_store=on_store, **self._common)
        yield _status_with_counts(r), None

    # --------------------------------------------------------------------- #
    #  DIMSE-N (Normalized) — MPPS, Storage Commitment, Print, UPS, …
    #  Each reuses the persistent association (negotiate the N SOP class via
    #  add_requested_context first). Datasets travel as bare IVR-LE attribute
    #  lists; replies parse back to a Dataset (None when the peer sent none).
    # --------------------------------------------------------------------- #
    @_empty_on_net_error(reply=True)
    def send_n_action(self, dataset, action_type, class_uid, instance_uid, msg_id=1, meta_uid=None):
        self._require()
        info = _identifier_bytes(dataset) if dataset is not None else b""
        r = self._client.n_action(str(class_uid), str(instance_uid), int(action_type), info)
        return _status(r["status"]), _reply_dataset(r["reply"])

    @_empty_on_net_error(reply=True)
    def send_n_create(self, dataset, class_uid, instance_uid=None, msg_id=1, meta_uid=None):
        self._require()
        attrs = _identifier_bytes(dataset) if dataset is not None else b""
        r = self._client.n_create(str(class_uid), str(instance_uid) if instance_uid else None, attrs)
        st = _status(r["status"])
        if r.get("affected_sop_instance_uid"):
            st.AffectedSOPInstanceUID = r["affected_sop_instance_uid"]
        return st, _reply_dataset(r["attribute_list"])

    @_empty_on_net_error(reply=True)
    def send_n_set(self, dataset, class_uid, instance_uid, msg_id=1, meta_uid=None):
        self._require()
        r = self._client.n_set(str(class_uid), str(instance_uid), _identifier_bytes(dataset))
        return _status(r["status"]), _reply_dataset(r["attribute_list"])

    @_empty_on_net_error(reply=True)
    def send_n_get(self, identifier_list, class_uid, instance_uid, msg_id=1, meta_uid=None):
        self._require()
        tags = [_tag_pair(t) for t in (identifier_list or [])]
        r = self._client.n_get(str(class_uid), str(instance_uid), tags)
        return _status(r["status"]), _reply_dataset(r["attribute_list"])

    @_empty_on_net_error()
    def send_n_delete(self, class_uid, instance_uid, msg_id=1, meta_uid=None):
        self._require()
        r = self._client.n_delete(str(class_uid), str(instance_uid))
        return _status(r["status"])

    @_empty_on_net_error(reply=True)
    def send_n_event_report(self, dataset, event_type, class_uid, instance_uid, msg_id=1, meta_uid=None):
        self._require()
        info = _identifier_bytes(dataset) if dataset is not None else b""
        r = self._client.n_event_report(str(class_uid), str(instance_uid), int(event_type), info)
        return _status(r["status"]), _reply_dataset(r["reply"])

    # ---- UPS (PS3.4 §CC) SCU helpers — thin sugar over send_n_* + the UPS ----
    # vocabulary; the action-info datasets are the only Python-resident bit, as
    # the native shapes layer leaves data-set encoding to each consumer.
    def ups_push(self, workitem, instance_uid=None):
        """UPS Push: create a workitem (N-CREATE). ``instance_uid`` proposes the
        new SOP Instance UID, else the SCP mints one."""
        return self.send_n_create(workitem, UPS.PUSH, instance_uid=instance_uid)

    def ups_get(self, workitem_uid, attributes=None):
        """UPS Pull: read a workitem (N-GET); ``attributes`` empty => whole workitem."""
        return self.send_n_get(attributes or [], UPS.PULL, workitem_uid)

    def ups_set(self, workitem_uid, modifications, transaction_uid):
        """UPS Pull: merge a modification list into a claimed workitem (N-SET).
        Stamps the claiming Transaction UID."""
        modifications = modifications or Dataset()
        modifications.TransactionUID = transaction_uid
        return self.send_n_set(modifications, UPS.PULL, workitem_uid)

    def ups_change_state(self, workitem_uid, state, transaction_uid):
        """UPS Pull: change Procedure Step State (N-ACTION type 1)."""
        ds = Dataset()
        ds.ProcedureStepState = str(state)
        ds.TransactionUID = transaction_uid
        return self.send_n_action(ds, UPS.CHANGE_STATE, UPS.PULL, workitem_uid)

    def ups_claim(self, workitem_uid, transaction_uid=None):
        """Claim a SCHEDULED workitem (-> IN PROGRESS). Mints a Transaction UID if
        not given; returns ``(status, transaction_uid)``."""
        transaction_uid = transaction_uid or generate_uid()
        status, _ = self.ups_change_state(workitem_uid, UPS.IN_PROGRESS, transaction_uid)
        return status, transaction_uid

    def ups_complete(self, workitem_uid, transaction_uid):
        """Transition a claimed workitem to COMPLETED."""
        return self.ups_change_state(workitem_uid, UPS.COMPLETED, transaction_uid)

    def ups_cancel(self, workitem_uid, transaction_uid):
        """Transition a claimed workitem to CANCELED (owner-side; use
        :meth:`ups_request_cancel` to ask another performer to cancel)."""
        return self.ups_change_state(workitem_uid, UPS.CANCELED, transaction_uid)

    def ups_request_cancel(self, workitem_uid, reason=None):
        """UPS Pull: request cancellation of a workitem you don't own (N-ACTION
        type 2); optional (0074,1238) Reason For Cancellation."""
        ds = None
        if reason is not None:
            ds = Dataset()
            ds.ReasonForCancellation = reason
        return self.send_n_action(ds, UPS.REQUEST_CANCEL, UPS.PULL, workitem_uid)

    def ups_subscribe(self, target=None, receiving_ae=None):
        """UPS Watch: subscribe to event reports (N-ACTION type 3). ``target``
        defaults to the global subscription instance (watch all workitems)."""
        ds = None
        if receiving_ae is not None:
            ds = Dataset()
            ds.ReceivingAE = receiving_ae
        return self.send_n_action(ds, UPS.SUBSCRIBE, UPS.WATCH,
                                  target or UPS.GLOBAL_SUBSCRIPTION)

    def ups_unsubscribe(self, target=None, receiving_ae=None):
        """UPS Watch: cancel a subscription (N-ACTION type 4)."""
        ds = None
        if receiving_ae is not None:
            ds = Dataset()
            ds.ReceivingAE = receiving_ae
        return self.send_n_action(ds, UPS.UNSUBSCRIBE, UPS.WATCH,
                                  target or UPS.GLOBAL_SUBSCRIPTION)

    # ---- MPPS (PS3.4 §F) SCU helpers — thin sugar over send_n_create/_set. --
    # The attribute / modification-list datasets are the caller's (PS3.3 §C.4.10
    # defines the schema); these just bind the SOP class + mint/echo the UID.
    def mpps_create(self, attributes, instance_uid=None):
        """MPPS N-CREATE — start a Performed Procedure Step (the dataset should
        carry PerformedProcedureStepStatus == ``MPPS.IN_PROGRESS`` plus the
        Scheduled/Performed step attributes). Mints a SOP Instance UID when not
        given; returns ``(status, sop_instance_uid)``."""
        instance_uid = instance_uid or generate_uid()
        status, _ = self.send_n_create(attributes, MPPS.SOP_CLASS, instance_uid=instance_uid)
        echoed = getattr(status, "AffectedSOPInstanceUID", None)
        return status, (str(echoed) if echoed else instance_uid)

    def mpps_set(self, instance_uid, modifications):
        """MPPS N-SET — update a step (typically transition
        PerformedProcedureStepStatus to COMPLETED / DISCONTINUED plus the
        Performed Series Sequence, end date/time)."""
        return self.send_n_set(modifications, MPPS.SOP_CLASS, instance_uid)

    def mpps_complete(self, instance_uid, modifications=None):
        """MPPS N-SET transitioning the step to COMPLETED (stamps the status;
        merge any final Performed* attributes via ``modifications``)."""
        modifications = modifications or Dataset()
        modifications.PerformedProcedureStepStatus = MPPS.COMPLETED
        return self.mpps_set(instance_uid, modifications)

    def mpps_discontinue(self, instance_uid, modifications=None):
        """MPPS N-SET transitioning the step to DISCONTINUED (operator cancelled
        the exam mid-procedure)."""
        modifications = modifications or Dataset()
        modifications.PerformedProcedureStepStatus = MPPS.DISCONTINUED
        return self.mpps_set(instance_uid, modifications)

    # ---- Storage Commitment Push Model (PS3.4 §J) SCU helper. ---------------
    # N-ACTION-RQ (action 1) asks the SCP to commit the listed instances; the
    # SCP acks immediately, then sends the real determination LATER as an
    # N-EVENT-REPORT-RQ — handle that via EVT_N_EVENT_REPORT on this
    # association, or a listening AE (start_server) on a fresh one (PS3.4 §J.3.3).
    def request_storage_commitment(self, referenced_sop_instances, *, transaction_uid=None):
        """Storage Commitment N-ACTION (Request Storage). ``referenced_sop_instances``
        is an iterable of ``(sop_class_uid, sop_instance_uid)`` pairs to commit.
        ``transaction_uid`` (0008,1195) is PS3.4 Table J.3-1 Type 1 — minted here
        if not given. Returns ``(status, transaction_uid)``: keep the Transaction
        UID to correlate the commitment RESULT, which arrives asynchronously as an
        N-EVENT-REPORT (``EVENT_SUCCESS`` / ``EVENT_FAILURES``)."""
        transaction_uid = transaction_uid or generate_uid()
        ds = Dataset()
        ds.TransactionUID = transaction_uid
        items = []
        for cls_uid, inst_uid in referenced_sop_instances:
            item = Dataset()
            item.ReferencedSOPClassUID = str(cls_uid)
            item.ReferencedSOPInstanceUID = str(inst_uid)
            items.append(item)
        ds.ReferencedSOPSequence = items
        status, _ = self.send_n_action(ds, StorageCommitment.ACTION_REQUEST_STORAGE,
                                       StorageCommitment.SOP_CLASS,
                                       StorageCommitment.PUSH_MODEL_INSTANCE)
        return status, transaction_uid

    # ---- Instance Availability Notification (PS3.4 §R) SCU helper. ----------
    # A one-shot N-CREATE on the IAN SOP class announcing that instances are
    # available (PS3.4 §R N-CREATE-RQ).
    def notify_instances_available(self, study_uid, instances, *, retrieve_ae=None,
                                   availability=None, instance_uid=None):
        """IAN N-CREATE (PS3.4 §R): announce that ``instances`` of ``study_uid`` are
        available. ``instances`` is an iterable of ``(series_uid, sop_class_uid,
        sop_instance_uid)`` tuples (grouped by series here). ``availability`` is an
        ``IAN.*`` value (default ``IAN.ONLINE``); ``retrieve_ae`` is the (0008,0054)
        Retrieve AE Title applied to every instance. Mints the IAN SOP Instance UID
        if not given. Returns ``(status, sop_instance_uid)``. For per-instance
        availability/retrieve, build the Attribute List ``Dataset`` yourself and call
        :meth:`send_n_create` on ``IAN.SOP_CLASS``."""
        availability = availability or IAN.ONLINE
        instance_uid = instance_uid or generate_uid()
        by_series, order = {}, []
        for series_uid, cls_uid, inst_uid in instances:
            series_uid = str(series_uid)
            if series_uid not in by_series:
                by_series[series_uid] = []
                order.append(series_uid)
            by_series[series_uid].append((str(cls_uid), str(inst_uid)))
        series_items = []
        for series_uid in order:
            sop_items = []
            for cls_uid, inst_uid in by_series[series_uid]:
                it = Dataset()
                it.ReferencedSOPClassUID = cls_uid
                it.ReferencedSOPInstanceUID = inst_uid
                it.InstanceAvailability = availability
                if retrieve_ae:
                    it.RetrieveAETitle = retrieve_ae
                sop_items.append(it)
            si = Dataset()
            si.SeriesInstanceUID = series_uid
            si.ReferencedSOPSequence = sop_items
            series_items.append(si)
        ds = Dataset()
        ds.ReferencedPerformedProcedureStepSequence = []   # (0008,1111) Type 2, may be empty
        ds.ReferencedSeriesSequence = series_items
        ds.StudyInstanceUID = str(study_uid)
        status, _ = self.send_n_create(ds, IAN.SOP_CLASS, instance_uid=instance_uid)
        echoed = getattr(status, "AffectedSOPInstanceUID", None)
        return status, (str(echoed) if echoed else instance_uid)

    # ---- Modality Worklist (PS3.4 §K) — no special method needed. -----------
    # MWL is a plain C-FIND on the Modality Worklist Information Model; use
    #   assoc.send_c_find(ds, pydcm.sop_class.ModalityWorklistInformationModelFind)
    # with an identifier carrying the (0040,0100) Scheduled Procedure Step
    # Sequence. send_c_find already accepts any FIND SOP class UID.

    # ---- §H Meta SOP-class PC alias -----------------------------------------
    def alias_presentation_context(self, sop_class_uid, negotiated_as):
        """Route N-services for ``sop_class_uid`` over the presentation context
        negotiated for ``negotiated_as`` (PS3.4 §H Meta SOP-class alias).

        Call after :meth:`associate` when you negotiated the §H Grayscale or
        Color Print Management Meta SOP class, then alias each member class
        (Film Session, Film Box, Image Box) onto the Meta so the engine selects
        the right presentation context for every N-* command::

            assoc.alias_presentation_context(Print.FILM_SESSION, Print.GRAYSCALE_META)
            assoc.alias_presentation_context(Print.FILM_BOX,     Print.GRAYSCALE_META)
            assoc.alias_presentation_context(Print.GRAYSCALE_IMAGE_BOX, Print.GRAYSCALE_META)

        No-op if the association is not established.
        """
        self._require()
        self._client.alias_presentation_context(str(sop_class_uid), str(negotiated_as))

    # ---- Print Management (PS3.4 §H) SCU helpers ----------------------------
    def print_film_session(self, image_display_format="STANDARD\\1,1"):
        """§H Film Session + Film Box N-CREATE workflow (steps 1–2).

        Opens a Film Session, creates a Film Box with the given
        ``image_display_format``, and returns
        ``(status, film_session_uid, film_box_uid, image_box_uid)`` — the
        Image Box UID is what you pass to :meth:`print_set_image_box`.
        Returns ``(status, None, None, None)`` on first N-CREATE failure.
        """
        self._require()
        r = self._client.n_create(str(Print.FILM_SESSION), None, b"")
        st = r["status"]
        if st != 0x0000:
            return _status(st), None, None, None
        session_uid = r.get("affected_sop_instance_uid") or ""

        attrs = _dimse.build_film_box_attrs(image_display_format, session_uid)
        r2 = self._client.n_create(str(Print.FILM_BOX), None, attrs)
        st2 = r2["status"]
        if st2 != 0x0000:
            return _status(st2), session_uid, None, None
        box_uid = r2.get("affected_sop_instance_uid") or ""
        image_box_uid = _dimse.parse_image_box_uid(r2["attribute_list"])
        return _status(st2), session_uid, box_uid, image_box_uid

    def print_set_image_box(self, image_box_uid, pixel_bytes,
                            rows, cols, samples_per_pixel=1, position=1):
        """§H Image Box N-SET (step 2b): load rendered pixel data into the
        Image Box the SCP assigned.

        ``pixel_bytes`` must be raw 8-bit display pixels in row-major order:
        ``rows × cols × samples_per_pixel`` bytes, where ``samples_per_pixel``
        is 1 for MONOCHROME2 (grayscale) or 3 for RGB (color, interleaved).
        Uses :func:`build_image_box_mods` to build the validated wire
        bytes so the caller doesn't need to know the DICOM attribute layout.
        """
        self._require()
        mods = _dimse.build_image_box_mods(
            position, rows, cols, pixel_bytes, samples_per_pixel)
        sop_class_uid = str(Print.COLOR_IMAGE_BOX if samples_per_pixel == 3
                            else Print.GRAYSCALE_IMAGE_BOX)
        r = self._client.n_set(sop_class_uid, str(image_box_uid), mods)
        return _status(r["status"])

    def print_action(self, film_box_uid):
        """§H N-ACTION Print (step 3): ask the SCP to print the Film Box."""
        self._require()
        r = self._client.n_action(str(Print.FILM_BOX), str(film_box_uid),
                                  Print.ACTION_PRINT, b"")
        return _status(r["status"])

    def print_cleanup(self, film_box_uid=None, film_session_uid=None):
        """§H N-DELETE cleanup (step 4): best-effort; ignores failures."""
        if not self.is_established:
            return
        if film_box_uid:
            try:
                self._client.n_delete(str(Print.FILM_BOX), str(film_box_uid))
            except Exception:
                pass
        if film_session_uid:
            try:
                self._client.n_delete(str(Print.FILM_SESSION), str(film_session_uid))
            except Exception:
                pass

    def release(self):
        if not self.is_released:
            if self._client is not None:
                self._client.release()
            self.is_released = True
            self.is_established = False
            self._fire("released")

    def abort(self):
        if not self.is_aborted:
            if self._client is not None:
                self._client.release()              # native graceful teardown (no A-ABORT PDU)
            self.is_aborted = True
            self.is_established = False
            self._fire("aborted")

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.release()
        return False


def _status_with_counts(r) -> Dataset:
    ds = _status(r["status"])
    ds.NumberOfCompletedSuboperations = r.get("completed", 0)
    ds.NumberOfFailedSuboperations    = r.get("failed", 0)
    ds.NumberOfWarningSuboperations   = r.get("warning", 0)
    ds.NumberOfRemainingSuboperations = r.get("remaining", 0)
    return ds


# --------------------------------------------------------------------------- #
#  Events (evt)
# --------------------------------------------------------------------------- #
class _Event:
    def __init__(self, name):
        self.name = name
    def __repr__(self):
        return f"<evt {self.name}>"


class _StoreEvent:
    """The EVT_C_STORE event object passed to a handler."""
    def __init__(self, dataset, sop_class, sop_instance, ts, assoc):
        self._dataset = dataset
        self.sop_class_uid = sop_class
        self.sop_instance_uid = sop_instance
        self.transfer_syntax = ts
        self.assoc = assoc

    @property
    def dataset(self):
        return self._dataset

    def file_meta(self):
        return self._dataset.file_meta


class _FindEvent:
    """The EVT_C_FIND event passed to a query handler: ``.identifier``
    is the query keys dataset."""
    def __init__(self, identifier):
        self.identifier = identifier
        self.is_cancelled = False


class _MoveEvent:
    """The EVT_C_MOVE event: ``.identifier`` is the query, and
    ``.move_destination`` is the destination AE title the SCU named."""
    def __init__(self, identifier, move_destination):
        self.identifier = identifier
        self.move_destination = move_destination
        self.is_cancelled = False


class _NEvent:
    """A DIMSE-N SCP event. The service-specific data set is exposed
    under every common alias (action_information / attribute_list / modification_list
    / event_information) so handlers written for any N service work unchanged."""
    def __init__(self, sop_class, sop_instance, *, dataset=None, type_id=None, identifiers=None):
        self.sop_class_uid = sop_class
        self.sop_instance_uid = sop_instance
        self._dataset = dataset
        self.action_type = type_id            # N-ACTION
        self.event_type = type_id             # N-EVENT-REPORT
        self.attribute_identifiers = identifiers or []   # N-GET (list of (group, element))
        self.is_cancelled = False

    # exposes the inbound data set under a per-service name; alias them all.
    action_information = property(lambda self: self._dataset)
    attribute_list     = property(lambda self: self._dataset)
    modification_list  = property(lambda self: self._dataset)
    event_information  = property(lambda self: self._dataset)


class _LifecycleEvent:
    """A notification event for an association lifecycle transition:
    ``.assoc`` is the Association it fired for."""
    def __init__(self, assoc):
        self.assoc = assoc


class evt:                                          # noqa: N801 (lowercase event namespace)
    EVT_C_STORE = _Event("C_STORE")
    EVT_C_ECHO  = _Event("C_ECHO")
    EVT_C_FIND  = _Event("C_FIND")
    EVT_C_MOVE  = _Event("C_MOVE")
    EVT_C_GET   = _Event("C_GET")
    EVT_N_ACTION       = _Event("N_ACTION")
    EVT_N_CREATE       = _Event("N_CREATE")
    EVT_N_SET          = _Event("N_SET")
    EVT_N_GET          = _Event("N_GET")
    EVT_N_DELETE       = _Event("N_DELETE")
    EVT_N_EVENT_REPORT = _Event("N_EVENT_REPORT")
    # association lifecycle notifications (requestor side; fired on the real transition)
    EVT_ESTABLISHED = _Event("ESTABLISHED")
    EVT_RELEASED    = _Event("RELEASED")
    EVT_ABORTED     = _Event("ABORTED")
    # Accepted for source compatibility. pydcm's native engine fires the
    # intercept/lifecycle events above; these low-level protocol notifications (ACSE /
    # DIMSE / PDU / data / connection / FSM) are not surfaced by the native stack, so
    # handlers registered for them simply never trigger.
    EVT_ACCEPTED       = _Event("ACCEPTED")
    EVT_REJECTED       = _Event("REJECTED")
    EVT_REQUESTED      = _Event("REQUESTED")
    EVT_ACSE_RECV      = _Event("ACSE_RECV")
    EVT_ACSE_SENT      = _Event("ACSE_SENT")
    EVT_DIMSE_RECV     = _Event("DIMSE_RECV")
    EVT_DIMSE_SENT     = _Event("DIMSE_SENT")
    EVT_PDU_RECV       = _Event("PDU_RECV")
    EVT_PDU_SENT       = _Event("PDU_SENT")
    EVT_DATA_RECV      = _Event("DATA_RECV")
    EVT_DATA_SENT      = _Event("DATA_SENT")
    EVT_CONN_OPEN      = _Event("CONN_OPEN")
    EVT_CONN_CLOSE     = _Event("CONN_CLOSE")
    EVT_FSM_TRANSITION = _Event("FSM_TRANSITION")
    EVT_ASYNC_OPS      = _Event("ASYNC_OPS")
    EVT_SOP_COMMON     = _Event("SOP_COMMON")
    EVT_SOP_EXTENDED   = _Event("SOP_EXTENDED")
    EVT_USER_ID        = _Event("USER_ID")


# --------------------------------------------------------------------------- #
#  AE (application entity)
# --------------------------------------------------------------------------- #
class AE:
    """An Application Entity over the native DIMSE engine."""

    def __init__(self, ae_title="PYDCM"):
        self.ae_title = str(ae_title)
        self._requested: list[str] = []
        self._supported: list[str] = []
        self.tls = None                              # dict: ca_file/cert_file/key_file/verify_peer/server_name/ciphers/check_hostname
        self._store_handler = None
        # timeout knobs (seconds; None = no limit). connection_timeout +
        # acse_timeout bound association ESTABLISHMENT (TCP connect + ACSE handshake), which
        # the native engine enforces. network_timeout maps to the SCP's idle timeout;
        # dimse_timeout bounds EACH DIMSE operation's response wait — the native engine
        # enforces it per-op and A-ABORTs the association on timeout (a half-sent op
        # leaves it unusable). 0/None = unbounded.
        self.network_timeout = 60
        self.acse_timeout = 30
        self.dimse_timeout = 30
        self.connection_timeout = None
        # AE config knobs (accepted; honoured by the native engine where
        # applicable, stored for source compatibility otherwise).
        self.maximum_pdu_size = 16382
        self.maximum_associations = 10
        self.require_called_aet = False
        self.require_calling_aet: list = []
        self._servers: list = []        # servers started by make_server / start_server
        self._active: list = []         # associations opened by associate()

    @property
    def implementation_class_uid(self):
        return PYDCM_DIMSE_IMPLEMENTATION_UID

    @implementation_class_uid.setter
    def implementation_class_uid(self, value):          # accepted
        pass

    @property
    def implementation_version_name(self):
        return PYDCM_DIMSE_IMPLEMENTATION_VERSION

    @implementation_version_name.setter
    def implementation_version_name(self, value):
        pass

    @property
    def active_associations(self):
        """The associations opened by this AE that are still established."""
        return [a for a in self._active if getattr(a, "is_established", False)]

    def remove_requested_context(self, abstract_syntax, transfer_syntax=None):
        """Remove a requested presentation context."""
        a = str(abstract_syntax)
        ts = ([transfer_syntax] if isinstance(transfer_syntax, str)
              else None if transfer_syntax is None else [str(t) for t in transfer_syntax])
        self._requested = [(ab, t) for (ab, t) in self._requested
                           if not (ab == a and (ts is None or list(t) == ts))]

    def remove_supported_context(self, abstract_syntax, transfer_syntax=None):
        """Remove a supported presentation context."""
        a = str(abstract_syntax)
        self._supported = [s for s in self._supported if s != a]

    def make_server(self, address, ae_title=None, contexts=None, evt_handlers=None,
                    ssl_context=None, **kw):
        """Return a non-blocking SCP server handle; call
        ``.serve_forever()`` / ``.shutdown()`` on it. ``== start_server(block=False)``."""
        return self.start_server(address, block=False, evt_handlers=evt_handlers,
                                 ssl_context=ssl_context, **kw)

    def shutdown(self):
        """Stop all servers started by this AE."""
        for s in self._servers:
            try:
                s.shutdown()
            except Exception:
                pass
        self._servers = []

    def _establish_timeout_ms(self) -> int:
        """The association-establishment timeout in ms (0 = none) from the timeout knobs."""
        secs = [t for t in (self.connection_timeout, self.acse_timeout) if t]
        return int(min(secs) * 1000) if secs else 0

    # SCU contexts: each is (abstract_syntax, [transfer_syntaxes]) — the persistent
    # association negotiates exactly these (pass the syntaxes you intend to send, incl.
    # compressed). transfer_syntax may be a str, a list, or None (⇒ Implicit+Explicit VR LE).
    def add_requested_context(self, abstract_syntax, transfer_syntax=None):
        if transfer_syntax is None:
            ts = list(_DEFAULT_TS)
        elif isinstance(transfer_syntax, str):
            ts = [transfer_syntax]
        else:
            ts = [str(t) for t in transfer_syntax]
        self._requested.append((str(abstract_syntax), ts))

    def add_supported_context(self, abstract_syntax, transfer_syntax=None):
        self._supported.append(str(abstract_syntax))

    # property idiom: ``ae.requested_contexts = StoragePresentationContexts``
    # (a list of PresentationContext). Mirrors add_requested_context, in bulk.
    @property
    def requested_contexts(self):
        return [PresentationContext(abstract_syntax=a, transfer_syntax=ts)
                for a, ts in self._requested]

    @requested_contexts.setter
    def requested_contexts(self, contexts):
        self._requested = [(str(c.abstract_syntax), [str(t) for t in c.transfer_syntax])
                           for c in contexts]

    @property
    def supported_contexts(self):
        return [PresentationContext(abstract_syntax=a, transfer_syntax=list(_DEFAULT_TS))
                for a in self._supported]

    @supported_contexts.setter
    def supported_contexts(self, contexts):
        self._supported = [str(c.abstract_syntax) for c in contexts]

    def _get_store_handler(self):
        return self._store_handler

    def associate(self, addr, port, ae_title="ANY-SCP", contexts=None, tls_args=None,
                  evt_handlers=None, **_kw):
        if tls_args is not None:
            self.tls = _tls_from_args(tls_args)
        n_event_handler, lifecycle = None, {}
        for entry in (evt_handlers or []):
            ev = entry[0]
            if ev is evt.EVT_C_STORE:
                self._store_handler = entry[1]
            elif ev is evt.EVT_N_EVENT_REPORT:
                n_event_handler = entry[1]    # inbound N-EVENT-REPORT on the request association
            elif ev is evt.EVT_ESTABLISHED:
                lifecycle["established"] = entry[1]
            elif ev is evt.EVT_RELEASED:
                lifecycle["released"] = entry[1]
            elif ev is evt.EVT_ABORTED:
                lifecycle["aborted"] = entry[1]
        assoc = Association(self, addr, port, str(ae_title),
                            n_event_handler=n_event_handler, lifecycle=lifecycle)
        self._active.append(assoc)
        return assoc

    def start_server(self, address, block=True, evt_handlers=None, ssl_context=None,
                     **_kw):
        host, port = address
        on_store = on_echo = on_find = on_get = on_move = None
        on_n_action = on_n_create = on_n_set = on_n_get = on_n_delete = on_n_event = None
        for entry in (evt_handlers or []):
            ev, handler = entry[0], entry[1]
            if ev is evt.EVT_C_STORE:
                def on_store(sc, si, ts, data, _h=handler):
                    rv = _h(_StoreEvent(_to_dataset(data, ts, sc, si), sc, si, ts, None))
                    return int(getattr(rv, "Status", rv) if rv is not None else STATUS_SUCCESS)
            elif ev is evt.EVT_C_ECHO:
                def on_echo(_h=handler):
                    rv = _h(_Event("C_ECHO"))
                    return int(getattr(rv, "Status", rv) if rv is not None else STATUS_SUCCESS)
            elif ev is evt.EVT_C_FIND:
                def on_find(query_bytes, _h=handler):
                    # C-FIND handler is a generator yielding (status, identifier);
                    # collect the pending matches' encoded bytes for the native facade.
                    out = []
                    for status, identifier in _h(_FindEvent(_to_dataset(query_bytes))):
                        code = int(getattr(status, "Status", status))
                        if identifier is not None and code in (STATUS_PENDING, 0xFF01):
                            out.append(_identifier_bytes(identifier))
                    return out
            elif ev is evt.EVT_C_GET:
                def on_get(query_bytes, _h=handler):
                    # C-GET handler yields the instances to send (the first
                    # yield is the match count, then (status, dataset) per instance).
                    out = []
                    for item in _h(_FindEvent(_to_dataset(query_bytes))):
                        if isinstance(item, int):
                            continue                         # the leading sub-operation count
                        status, ds = item
                        if ds is not None:
                            sc, si, ts, data = _encode(ds)
                            out.append((sc, si, ts, data))
                    return out
            elif ev is evt.EVT_C_MOVE:
                def on_move(query_bytes, dest_ae, _h=handler):
                    # C-MOVE handler: first yield (addr, port[, contexts]) of the
                    # destination, then (status, dataset) per instance to push there.
                    gen = iter(_h(_MoveEvent(_to_dataset(query_bytes), dest_ae)))
                    dest = next(gen, (None, None))
                    if not dest or dest[0] is None:
                        return ("", 0, [])                   # unknown destination → 0xA801
                    out = []
                    for item in gen:
                        if isinstance(item, int):
                            continue
                        status, ds = item
                        if ds is not None:
                            out.append(_encode(ds))
                    return (str(dest[0]), int(dest[1]), out)
            # --- DIMSE-N SCP handlers (Normalized services) ---
            elif ev is evt.EVT_N_ACTION:
                def on_n_action(sc, si, at, info, _h=handler):
                    return _norm_n_return(_h(_NEvent(sc, si, dataset=_reply_dataset(info), type_id=at)))
            elif ev is evt.EVT_N_CREATE:
                def on_n_create(sc, si, attrs, _h=handler):
                    return _norm_n_return(_h(_NEvent(sc, si, dataset=_reply_dataset(attrs))))
            elif ev is evt.EVT_N_SET:
                def on_n_set(sc, si, mods, _h=handler):
                    return _norm_n_return(_h(_NEvent(sc, si, dataset=_reply_dataset(mods))))
            elif ev is evt.EVT_N_GET:
                def on_n_get(sc, si, tags, _h=handler):
                    return _norm_n_return(_h(_NEvent(sc, si, identifiers=[tuple(t) for t in tags])))
            elif ev is evt.EVT_N_DELETE:
                def on_n_delete(sc, si, _h=handler):
                    return _norm_status(_h(_NEvent(sc, si)))
            elif ev is evt.EVT_N_EVENT_REPORT:
                def on_n_event(sc, si, et, info, _h=handler):
                    return _norm_n_return(_h(_NEvent(sc, si, dataset=_reply_dataset(info), type_id=et)))
        tls = self.tls if ssl_context is None else self.tls
        native = _dimse.server_start(port=int(port), ae_title=self.ae_title,
                                     on_store=on_store, on_echo=on_echo, on_find=on_find,
                                     on_get=on_get, on_move=on_move,
                                     on_n_action=on_n_action, on_n_create=on_n_create,
                                     on_n_set=on_n_set, on_n_get=on_n_get,
                                     on_n_delete=on_n_delete, on_n_event_report=on_n_event,
                                     tls=tls)
        server = ServerHandle(native)
        self._servers.append(server)
        if block:
            try:
                import threading
                threading.Event().wait()             # block until interrupted
            except KeyboardInterrupt:
                server.shutdown()
            return None
        return server


class ServerHandle:
    """Handle for a non-blocking SCP (returned from start_server(block=False))."""
    def __init__(self, native):
        self._native = native

    def shutdown(self):
        if self._native is not None:
            self._native.stop()
            self._native = None

    def serve_forever(self):
        """Block serving until interrupted."""
        import threading
        try:
            threading.Event().wait()
        except KeyboardInterrupt:
            self.shutdown()

    @property
    def bound_port(self):
        return self._native.bound_port() if self._native else 0

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.shutdown()
        return False


def _tls_from_args(tls_args):
    """Best-effort map tls_args=(ssl.SSLContext, server_hostname) to pydcm's
    dict form. A Python ssl.SSLContext can't be reused as an OpenSSL context, so callers
    needing TLS should set ``ae.tls = {ca_file, cert_file, key_file, ...}`` explicitly."""
    if isinstance(tls_args, dict):
        return tls_args
    if isinstance(tls_args, (tuple, list)) and tls_args:
        host = tls_args[1] if len(tls_args) > 1 else ""
        return {"verify_peer": True, "server_name": str(host or "")}
    return None


# --------------------------------------------------------------------------- #
#  module-level constants / helpers / presentation-context bundles
# --------------------------------------------------------------------------- #
from .uid import (UID, UID_dictionary, ImplicitVRLittleEndian, ExplicitVRLittleEndian,
                  DeflatedExplicitVRLittleEndian, ExplicitVRBigEndian, AllTransferSyntaxes,
                  UID_REGISTRY, generate_uid)

DEFAULT_TRANSFER_SYNTAXES = [ImplicitVRLittleEndian, ExplicitVRLittleEndian,
                             DeflatedExplicitVRLittleEndian, ExplicitVRBigEndian]
ALL_TRANSFER_SYNTAXES = list(AllTransferSyntaxes)

# pydcm.dimse's OWN implementation identity — pydcm's registered UID arc, stamped
# into the Implementation Class UID / Version Name of every association it opens.
# A SCU must announce who it really is, so these carry pydcm's identity, not a
# borrowed one.
PYDCM_DIMSE_UID_PREFIX = "1.2.826.0.1.3680043.9.7155."
PYDCM_DIMSE_IMPLEMENTATION_VERSION = "PYDCM_DIMSE_1"
PYDCM_DIMSE_IMPLEMENTATION_UID = UID(PYDCM_DIMSE_UID_PREFIX + "1")


def build_context(abstract_syntax, transfer_syntax=None):
    """Return a :class:`PresentationContext`."""
    if transfer_syntax is None:
        transfer_syntax = DEFAULT_TRANSFER_SYNTAXES
    elif isinstance(transfer_syntax, str):
        transfer_syntax = [transfer_syntax]
    return PresentationContext(abstract_syntax=str(abstract_syntax),
                               transfer_syntax=list(transfer_syntax))


class SCP_SCU_RoleSelection:                          # noqa: N801 (DICOM role-selection name)
    """An SCP/SCU role-selection item."""
    def __init__(self, uid, scu_role=False, scp_role=False):
        self.sop_class_uid = UID(str(uid))
        self.scu_role = bool(scu_role)
        self.scp_role = bool(scp_role)


def build_role(uid, scu_role=False, scp_role=False):
    """Return an SCP/SCU role-selection item."""
    return SCP_SCU_RoleSelection(uid, scu_role, scp_role)


def debug_logger():
    """Enable pydcm.dimse debug logging to stderr."""
    import logging
    log = logging.getLogger("pydcm.dimse")
    log.setLevel(logging.DEBUG)
    if not any(isinstance(h, logging.StreamHandler) for h in log.handlers):
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(asctime)s %(levelname).1s: %(message)s"))
        log.addHandler(h)


_REGISTERED_UIDS: dict = {}


def register_uid(uid, keyword, service_class=None, **_kw):
    """Register a private SOP Class UID for source-compatibility (recorded for lookup)."""
    _REGISTERED_UIDS[str(uid)] = keyword


# Presentation-context bundles, projected from the UID registry (so they track the native
# dict, not a hardcoded list) — validated set-equal to the curated bundles. DICOM
# caps an association at 128 presentation contexts, so Storage is capped too.
def _bundle(pred, cap=None):
    uids = sorted(u for u, (n, t, i, r, k) in UID_REGISTRY.items() if pred(t, k, bool(r)))
    return [build_context(u) for u in (uids[:cap] if cap else uids)]


def _ctx(*keywords):
    by_kw = {k: u for u, (n, t, i, r, k) in UID_REGISTRY.items() if k}
    return [build_context(by_kw[k]) for k in keywords if k in by_kw]


_NPO = {"CTDefinedProcedureProtocolStorage", "ProtocolApprovalStorage",
        "XADefinedProcedureProtocolStorage", "InventoryStorage", "HangingProtocolStorage",
        "ColorPaletteStorage", "GenericImplantTemplateStorage",
        "ImplantAssemblyTemplateStorage", "ImplantTemplateGroupStorage"}

AllStoragePresentationContexts = _bundle(
    lambda t, k, r: t == "SOP Class" and "Storage" in k and not r and k not in _NPO
    and not k.startswith(("DICOS", "DICONDE", "StorageCommitment", "EddyCurrent", "Thermography"))
    and k != "MediaStorageDirectoryStorage")
StoragePresentationContexts = AllStoragePresentationContexts[:128]
VerificationPresentationContexts = _ctx("Verification")
QueryRetrievePresentationContexts = _bundle(
    lambda t, k, r: (k.endswith(("QueryRetrieveInformationModelFind", "QueryRetrieveInformationModelMove",
                                  "QueryRetrieveInformationModelGet"))
                     and k.startswith(("PatientRoot", "StudyRoot", "PatientStudyOnly")))
    or k.startswith("CompositeInstance") or k == "RepositoryQuery")
BasicWorklistManagementPresentationContexts = _ctx("ModalityWorklistInformationModelFind")
RelevantPatientInformationPresentationContexts = _bundle(
    lambda t, k, r: k.endswith("RelevantPatientInformationQuery"))
StorageCommitmentPresentationContexts = _ctx("StorageCommitmentPushModel")
ColorPalettePresentationContexts = _bundle(
    lambda t, k, r: k.startswith("ColorPaletteQueryRetrieveInformationModel"))
HangingProtocolPresentationContexts = _bundle(
    lambda t, k, r: k.startswith("HangingProtocolInformationModel"))
DefinedProcedureProtocolPresentationContexts = _bundle(
    lambda t, k, r: k.startswith("DefinedProcedureProtocolInformationModel"))
ProtocolApprovalPresentationContexts = _bundle(
    lambda t, k, r: k.startswith("ProtocolApprovalInformationModel"))
ImplantTemplatePresentationContexts = _bundle(
    lambda t, k, r: "InformationModel" in k and
    ("ImplantTemplateGroup" in k or "ImplantAssemblyTemplate" in k or "GenericImplantTemplate" in k))
ProcedureStepPresentationContexts = _bundle(
    lambda t, k, r: k.startswith("ModalityPerformedProcedureStep"))
ModalityPerformedPresentationContexts = ProcedureStepPresentationContexts
UnifiedProcedurePresentationContexts = _bundle(
    lambda t, k, r: k.startswith("UnifiedProcedureStep") and not r)
NonPatientObjectPresentationContexts = _ctx(*sorted(_NPO))
DisplaySystemPresentationContexts = _ctx("DisplaySystem")
InstanceAvailabilityPresentationContexts = _ctx("InstanceAvailabilityNotification")
MediaCreationManagementPresentationContexts = _ctx("MediaCreationManagement")
MediaStoragePresentationContexts = _ctx("MediaStorageDirectoryStorage")
RTMachineVerificationPresentationContexts = _ctx(
    "RTConventionalMachineVerification", "RTIonMachineVerification")
SubstanceAdministrationPresentationContexts = _ctx(
    "ProductCharacteristicsQuery", "SubstanceApprovalQuery")
ApplicationEventLoggingPresentationContexts = _ctx(
    "ProceduralEventLogging", "SubstanceAdministrationLogging")
PrintManagementPresentationContexts = _ctx(
    "BasicFilmSession", "PrintJob", "BasicAnnotationBox", "Printer",
    "PrinterConfigurationRetrieval", "BasicColorPrintManagementMeta", "BasicFilmBox",
    "PresentationLUT", "BasicGrayscalePrintManagementMeta", "BasicGrayscaleImageBox",
    "BasicColorImageBox")


def print_image(host, port, pixel_bytes, rows, cols,
                samples_per_pixel=1,
                image_display_format="STANDARD\\1,1",
                called_ae="PRINTER", calling_ae="PYDCM",
                tls=None, connect_timeout=30):
    """Drive a complete §H Print Management SCU workflow.

    Connects to the DICOM printer at ``host:port``, negotiates the
    Grayscale or Color Print Management Meta SOP class (chosen by
    ``samples_per_pixel``), and executes the full Film Session → Film Box
    → Image Box N-SET → N-ACTION Print → N-DELETE pipeline.

    Parameters
    ----------
    host, port
        Printer address.
    pixel_bytes
        Raw 8-bit display pixels (pre-rendered; row-major, interleaved for
        color).  ``rows × cols × samples_per_pixel`` bytes.
    rows, cols
        Image dimensions in pixels.
    samples_per_pixel
        1 for MONOCHROME2 (grayscale), 3 for RGB (color).
    image_display_format
        DICOM ImageDisplayFormat string (default ``"STANDARD\\\\1,1"``).
    called_ae, calling_ae
        DICOM AE titles.
    tls
        Optional TLS config dict (``ca_file``, ``cert_file``, ``key_file``, …).
    connect_timeout
        Association establishment timeout in seconds (default 30).

    Returns
    -------
    tuple[Dataset, str]
        ``(status_dataset, film_box_uid)`` where ``status_dataset.Status``
        is the DIMSE status from the N-ACTION Print step.  ``film_box_uid``
        is the SCP-assigned Film Box UID (for logging / audit).

    Raises
    ------
    ConnectionError
        If the association cannot be established.
    RuntimeError
        If the SCP does not accept the Print Management Meta SOP class, or if a
        mandatory setup step (Film Session, Film Box, or Image Box N-SET) fails
        with a non-Success DIMSE status. The N-ACTION Print result is *returned*
        (see ``status_dataset.Status``), not raised — so a printer-side failure
        (e.g. out of film) is reported to the caller rather than thrown.
    """
    color = samples_per_pixel == 3
    meta_class = str(Print.COLOR_META if color else Print.GRAYSCALE_META)
    image_box_class = str(Print.COLOR_IMAGE_BOX if color else Print.GRAYSCALE_IMAGE_BOX)

    ae = AE(ae_title=calling_ae)
    ae.acse_timeout = connect_timeout
    if tls:
        ae.tls = tls

    # Propose ONLY the §H Meta SOP class, Implicit VR LE only: the shared
    # attribute-list builders emit IVR-LE bytes, so the Meta presentation
    # context must negotiate it. The member classes (Film Session/Box/Image
    # Box) then ride this PC via alias_presentation_context() below. Uses
    # pydcm's standard requested_contexts idiom — the same association path
    # every other pydcm SCU takes (don't hand-roll the proposal).
    ae.requested_contexts = [build_context(meta_class, _IMPLICIT_VR_LE)]
    assoc = ae.associate(host, port, ae_title=called_ae)
    if not assoc.is_established:
        raise ConnectionError(f"Could not associate with {host}:{port} (called_ae={called_ae})")

    try:
        # The §H member-class N-services ride the negotiated Meta PC; bail with
        # a clear error if the SCP rejected the Meta (e.g. a grayscale-only
        # printer asked to print colour) instead of failing opaquely on the
        # first N-CREATE — matches the CLI's accepted_pc() guard.
        if not any(str(c.abstract_syntax) == meta_class for c in assoc.accepted_contexts):
            raise RuntimeError(
                f"SCP did not accept the Print Management Meta SOP class ({meta_class})")
        # Set up §H Meta PC aliases so member-class N-services ride the Meta PC.
        for member in (str(Print.FILM_SESSION), str(Print.FILM_BOX), image_box_class):
            assoc.alias_presentation_context(member, meta_class)

        # Steps 1–2: Film Session + Film Box → Image Box UID.
        status, session_uid, film_box_uid, image_box_uid = \
            assoc.print_film_session(image_display_format)
        if int(status.Status) != 0x0000 or not image_box_uid:
            raise RuntimeError(
                f"Film Session/Box creation failed (status=0x{int(status.Status):04x})")

        # Step 2b: N-SET the rendered pixels into the Image Box.
        st2 = assoc.print_set_image_box(
            image_box_uid, pixel_bytes, rows, cols, samples_per_pixel)
        if int(st2.Status) != 0x0000:
            raise RuntimeError(f"Image Box N-SET failed (status=0x{int(st2.Status):04x})")

        # Step 3: N-ACTION Print.
        st3 = assoc.print_action(film_box_uid)
        # Step 4: cleanup (best-effort).
        assoc.print_cleanup(film_box_uid, session_uid)
        return st3, film_box_uid
    finally:
        assoc.release()

__all__ = ["AE", "Association", "evt", "ServerHandle", "sop_class", "UPS", "Print", "print_image",
           "PresentationContext", "STATUS_SUCCESS", "STATUS_PENDING",
           "UID", "UID_dictionary", "build_context", "build_role", "debug_logger",
           "register_uid", "SCP_SCU_RoleSelection",
           "DEFAULT_TRANSFER_SYNTAXES", "ALL_TRANSFER_SYNTAXES",
           "PYDCM_DIMSE_IMPLEMENTATION_UID", "PYDCM_DIMSE_IMPLEMENTATION_VERSION",
           "PYDCM_DIMSE_UID_PREFIX",
           "AllStoragePresentationContexts", "StoragePresentationContexts",
           "VerificationPresentationContexts", "QueryRetrievePresentationContexts",
           "BasicWorklistManagementPresentationContexts",
           "RelevantPatientInformationPresentationContexts",
           "StorageCommitmentPresentationContexts", "ColorPalettePresentationContexts",
           "HangingProtocolPresentationContexts", "DefinedProcedureProtocolPresentationContexts",
           "ProtocolApprovalPresentationContexts", "ImplantTemplatePresentationContexts",
           "ProcedureStepPresentationContexts", "ModalityPerformedPresentationContexts",
           "UnifiedProcedurePresentationContexts", "NonPatientObjectPresentationContexts",
           "DisplaySystemPresentationContexts", "InstanceAvailabilityPresentationContexts",
           "MediaCreationManagementPresentationContexts", "MediaStoragePresentationContexts",
           "RTMachineVerificationPresentationContexts", "SubstanceAdministrationPresentationContexts",
           "ApplicationEventLoggingPresentationContexts", "PrintManagementPresentationContexts"]
