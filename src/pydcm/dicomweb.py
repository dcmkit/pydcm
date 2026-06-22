# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Fuli Wu
# NOT a medical device — not for clinical or diagnostic use.

"""DICOMweb client (QIDO-RS) — native request builders + native HTTP transport.

Query a DICOMweb server for studies/series/instances and get DICOM-JSON back::

    studies = pydcm.dicomweb.search_studies("http://pacs:8042", base_path="/dicom-web",
                                            matches={"00100020": "PAT001"}, limit=10)

Self-contained (no Python HTTP dependency): the request is built by the native
zero-alloc DICOMweb builders — and executed
through the native async HTTP transport driven synchronously, so this is the
conformance-tested path.

Covers the three core transactions: QIDO-RS search (``search_studies``/``_series``/
``_instances``), WADO-RS retrieve (``retrieve_study``/``_series``/``_instance`` → Part-10
``bytes``), and STOW-RS store (``store_instances``). Requires the optional ``_dicomweb``
extension.
"""

from __future__ import annotations

import json
import queue
import threading

#: Per-call sliding work buffer (MiB) for the streaming iter_* path. A single instance body must
#: fit; bump it for very large multi-frame instances.
STREAM_WORK_MIB = 16

try:
    from . import _dicomweb
except ImportError as _e:                            # pragma: no cover
    raise ImportError(
        "pydcm.dicomweb requires the optional native _dicomweb extension, "
        "which is not present in this build."
    ) from _e


def _search(server, base_path, scope, study_uid, series_uid, matches, includefields,
            limit, offset, auth) -> list[dict]:
    body = _dicomweb.search(str(server), str(base_path or ""), scope,
                            str(study_uid or ""), str(series_uid or ""),
                            {str(k): str(v) for k, v in (matches or {}).items()},
                            [str(f) for f in (includefields or [])],
                            int(limit), int(offset), str(auth or ""))
    return json.loads(body) if body.strip() else []     # 204 No Content → []


def search_studies(server, *, base_path="", matches=None, includefields=None,
                   limit=0, offset=0, auth="") -> list[dict]:
    """QIDO-RS study search → list of DICOM-JSON study records.

    ``matches`` is ``{tag_or_keyword: value}`` (e.g. ``{"00100020": "PAT001"}``);
    ``includefields`` is a list of tags/keywords (or ``["all"]``); ``auth`` is an
    ``Authorization`` header value (e.g. ``"Bearer …"``/``"Basic …"``). Returns ``[]`` on 204.
    """
    return _search(server, base_path, "studies", "", "", matches, includefields, limit, offset, auth)


def search_series(server, study_uid="", *, base_path="", matches=None, includefields=None,
                  limit=0, offset=0, auth="") -> list[dict]:
    """QIDO-RS series search (all series, or within ``study_uid``)."""
    return _search(server, base_path, "series", study_uid, "", matches, includefields, limit, offset, auth)


def search_instances(server, study_uid="", series_uid="", *, base_path="", matches=None,
                     includefields=None, limit=0, offset=0, auth="") -> list[dict]:
    """QIDO-RS instance search (optionally scoped to a study/series)."""
    return _search(server, base_path, "instances", study_uid, series_uid, matches,
                   includefields, limit, offset, auth)


def retrieve_study(server, study_uid, *, base_path="", transfer_syntax="", auth="") -> list[bytes]:
    """WADO-RS: retrieve every Part-10 instance of a study → list of ``bytes`` blobs.

    ``transfer_syntax`` optionally negotiates the wire encoding (a TS UID, e.g.
    ``"1.2.840.10008.1.2.4.50"`` for JPEG baseline, or ``"*"`` for any) — the server falls back
    to the default if it cannot honour it.
    """
    return _dicomweb.retrieve(str(server), str(base_path or ""), "study",
                              str(study_uid), "", "", str(transfer_syntax or ""), str(auth or ""))


def retrieve_series(server, study_uid, series_uid, *, base_path="", transfer_syntax="",
                    auth="") -> list[bytes]:
    """WADO-RS: retrieve every Part-10 instance of a series → list of ``bytes`` blobs."""
    return _dicomweb.retrieve(str(server), str(base_path or ""), "series", str(study_uid),
                              str(series_uid), "", str(transfer_syntax or ""), str(auth or ""))


def retrieve_instance(server, study_uid, series_uid, instance_uid, *, base_path="",
                      transfer_syntax="", auth="") -> bytes:
    """WADO-RS: retrieve one Part-10 instance → ``bytes`` (raises if the server returns none)."""
    parts = _dicomweb.retrieve(str(server), str(base_path or ""), "instance", str(study_uid),
                               str(series_uid), str(instance_uid), str(transfer_syntax or ""),
                               str(auth or ""))
    if not parts:
        raise LookupError("WADO-RS retrieve returned no instance")
    return parts[0]


def retrieve_instance_wado_uri(server, study_uid, series_uid, instance_uid, *, base_path="/wado",
                               content_type="application/dicom", transfer_syntax="", frame_number=0,
                               rows=0, columns=0, image_quality=0, anonymize=False, auth="") -> bytes:
    """WADO-URI: retrieve one object via the classic query-parameter GET (PS3.18 §9) → ``bytes``.

    WADO-URI predates WADO-RS and is the only retrieval protocol on many older PACS; for modern
    servers prefer :func:`retrieve_instance` (WADO-RS), which this mirrors. Returns the object
    bytes — a Part-10 instance when ``content_type="application/dicom"`` (the default), or the
    server-rendered image bytes for an image MIME (e.g. ``"image/jpeg"``), in which case
    ``rows``/``columns``/``image_quality`` apply.

    ``base_path`` is the WADO-URI endpoint (default ``"/wado"`` — a standalone endpoint, *not*
    under the DICOMweb ``/dicom-web`` root); ``frame_number`` is 1-based (0 = unset);
    ``transfer_syntax`` optionally negotiates the wire encoding; ``anonymize=True`` asks the
    server for a de-identified object.
    """
    return _dicomweb.retrieve_wado_uri(str(server), str(base_path or ""), str(study_uid),
                                       str(series_uid), str(instance_uid), str(content_type or ""),
                                       str(transfer_syntax or ""), int(frame_number), int(rows),
                                       int(columns), int(image_quality), bool(anonymize),
                                       str(auth or ""))


def retrieve_study_metadata(server, study_uid, *, base_path="", auth="") -> list[dict]:
    """WADO-RS: study metadata → list of per-instance DICOM-JSON records (no pixel data)."""
    return json.loads(_dicomweb.retrieve_metadata(str(server), str(base_path or ""), "study",
                                                  str(study_uid), "", "", str(auth or "")))


def retrieve_series_metadata(server, study_uid, series_uid, *, base_path="", auth="") -> list[dict]:
    """WADO-RS: series metadata → list of per-instance DICOM-JSON records."""
    return json.loads(_dicomweb.retrieve_metadata(str(server), str(base_path or ""), "series",
                                                  str(study_uid), str(series_uid), "", str(auth or "")))


def retrieve_instance_metadata(server, study_uid, series_uid, instance_uid, *,
                               base_path="", auth="") -> dict:
    """WADO-RS: one instance's metadata → a single DICOM-JSON record."""
    arr = json.loads(_dicomweb.retrieve_metadata(str(server), str(base_path or ""), "instance",
                                                 str(study_uid), str(series_uid), str(instance_uid),
                                                 str(auth or "")))
    return arr[0] if arr else {}


def _frames_spec(frames) -> str:
    """Normalize a frames argument (int / iterable of ints / spec string) to a WADO spec."""
    if isinstance(frames, str):
        return frames
    if isinstance(frames, (list, tuple, set)):
        return ",".join(str(int(f)) for f in frames)
    return str(int(frames))


def retrieve_frames(server, study_uid, series_uid, instance_uid, frames, *,
                    base_path="", transfer_syntax="", auth="") -> list[bytes]:
    """WADO-RS: retrieve specific frames of an instance → list of raw frame ``bytes``.

    ``frames`` is a 1-based frame number, an iterable of them, or a spec string
    (``"1"``, ``"1,3-5,7"``). ``transfer_syntax`` optionally negotiates the frame pixel encoding.
    """
    return _dicomweb.retrieve_frames(str(server), str(base_path or ""), str(study_uid),
                                     str(series_uid), str(instance_uid), _frames_spec(frames),
                                     str(transfer_syntax or ""), str(auth or ""))


def _iter_retrieve(level, server, study_uid, series_uid, instance_uid, base_path,
                   transfer_syntax, auth):
    """Drive _dicomweb.retrieve_streaming on a worker thread and yield each instance as it
    arrives. Memory peak is one instance (plus the bounded queue), not the whole study — the
    native side pulls parts through a stream_decoder and pushes them across a small backpressured
    queue. The worker holds the GIL during network I/O; a full ``put`` blocks it (releasing the
    GIL) so the consumer keeps pace."""
    q: queue.Queue = queue.Queue(maxsize=4)
    sentinel = object()
    box: dict = {}

    def worker():
        try:
            _dicomweb.retrieve_streaming(str(server), str(base_path or ""), level, str(study_uid),
                                         str(series_uid or ""), str(instance_uid or ""),
                                         str(transfer_syntax or ""), str(auth or ""),
                                         int(STREAM_WORK_MIB), q.put)
        except BaseException as e:                        # surfaced to the consumer after drain
            box["err"] = e
        finally:
            q.put(sentinel)

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    try:
        while True:
            item = q.get()
            if item is sentinel:
                break
            yield item
        if box.get("err") is not None:
            raise box["err"]
    finally:
        if t.is_alive():                                  # consumer stopped early: let worker drain
            for _ in iter(q.get, sentinel):
                pass
            t.join(timeout=5)


def iter_study(server, study_uid, *, base_path="", transfer_syntax="", auth=""):
    """WADO-RS: yield each Part-10 instance of a study as ``bytes`` (memory-efficient stream)."""
    yield from _iter_retrieve("study", server, study_uid, "", "", base_path, transfer_syntax, auth)


def iter_series(server, study_uid, series_uid, *, base_path="", transfer_syntax="", auth=""):
    """WADO-RS: yield each Part-10 instance of a series as ``bytes`` (memory-efficient stream)."""
    yield from _iter_retrieve("series", server, study_uid, series_uid, "", base_path,
                              transfer_syntax, auth)


def retrieve_bulkdata(uri, *, auth="") -> list[bytes]:
    """WADO-RS: follow a server-issued ``BulkDataURI`` → list of ``bytes``.

    ``uri`` is the absolute URL the server handed out (e.g. a metadata record's ``BulkDataURI``
    for PixelData); it is fetched verbatim, since its path layout is server-specific.
    """
    return _dicomweb.retrieve_bulkdata(str(uri), str(auth or ""))


def retrieve_rendered(server, study_uid, series_uid="", instance_uid="", *, base_path="",
                      level=None, quality=0, window=None, viewport=None, auth="") -> bytes:
    """WADO-RS: a server-rendered image (default image/jpeg) → ``bytes``.

    ``level`` defaults to the deepest UID supplied (instance > series > study). ``window`` is an
    optional ``(center, width)`` tuple; ``quality`` an optional JPEG quality (1–100); ``viewport``
    an optional ``(width, height)`` output size.
    """
    lvl = level or ("instance" if instance_uid else "series" if series_uid else "study")
    has_w = window is not None
    wc, ww = (float(window[0]), float(window[1])) if has_w else (0.0, 0.0)
    vw, vh = (int(viewport[0]), int(viewport[1])) if viewport else (0, 0)
    return _dicomweb.retrieve_rendered(str(server), str(base_path or ""), lvl, str(study_uid),
                                       str(series_uid or ""), str(instance_uid or ""),
                                       int(quality), bool(has_w), wc, ww, vw, vh, str(auth or ""))


def delete_study(server, study_uid, *, base_path="", auth="") -> int:
    """DICOMweb DELETE a whole study → HTTP status (200/204). Server must support the (non-core)
    delete transaction."""
    return _dicomweb.delete(str(server), str(base_path or ""), "study",
                            str(study_uid), "", "", str(auth or ""))


def delete_series(server, study_uid, series_uid, *, base_path="", auth="") -> int:
    """DICOMweb DELETE a series → HTTP status."""
    return _dicomweb.delete(str(server), str(base_path or ""), "series",
                            str(study_uid), str(series_uid), "", str(auth or ""))


def delete_instance(server, study_uid, series_uid, instance_uid, *, base_path="", auth="") -> int:
    """DICOMweb DELETE a single instance → HTTP status."""
    return _dicomweb.delete(str(server), str(base_path or ""), "instance",
                            str(study_uid), str(series_uid), str(instance_uid), str(auth or ""))


def store_instances(server, instances, *, study_uid="", base_path="", auth="") -> dict:
    """STOW-RS: store Part-10 instances on the server.

    ``instances`` is an iterable of ``bytes`` (path-like / file objects are NOT accepted — pass
    raw DICOM bytes, e.g. ``open(p, "rb").read()`` or ``ds.to_bytes()``). If ``study_uid`` is
    given, the POST is bound to that study (server rejects mismatched StudyInstanceUID).

    Returns ``{"status", "stored", "failed"}`` — status 200 (all stored) / 202 (partial) / 409
    (none); ``stored`` and ``failed`` are lists of ``{sop_class_uid, sop_instance_uid, ...}``
    parsed natively from the ReferencedSOPSequence (00081199) / FailedSOPSequence (00081198) by
    the native store-response parser, so a 202 partial-success is directly
    inspectable (``failed[i]["failure_reason"]`` holds the PS3.4 code).
    """
    blobs = [bytes(x) for x in instances]
    return _dicomweb.store(str(server), str(base_path or ""), str(study_uid or ""),
                           blobs, str(auth or ""))


# --------------------------------------------------------------------------- #
#  UPS-RS — Unified Procedure Step (PS3.4 §CC / PS3.18 §11)
# --------------------------------------------------------------------------- #
# Workitem worklist over HTTP — the RS twin of pydcm.dimse's UPS N-services.
# Thin wrappers over the native _dicomweb.ups_* verbs. Action verbs return
# the native ``{status, location, body}`` dict; search/retrieve parse the body.

def _json_body(payload) -> str:
    """A Dataset / dict / JSON-string → a DICOM-JSON body string ("" for None)."""
    if payload is None:
        return ""
    if isinstance(payload, str):
        return payload
    if hasattr(payload, "to_json"):
        return payload.to_json()
    return json.dumps(payload)


def create_workitem(server, workitem, *, workitem_uid="", base_path="", auth=""):
    """UPS-RS create (POST /workitems). ``workitem`` is a Dataset/dict of attributes;
    pass ``workitem_uid`` to propose the SOP Instance UID, else the server assigns one
    (returned in the result's ``location``). Returns ``{status, location, body}``."""
    return _dicomweb.ups_create(str(server), str(base_path or ""), str(workitem_uid or ""),
                                _json_body(workitem), str(auth or ""))


def search_workitems(server, *, base_path="", matches=None, includefields=None,
                     limit=0, offset=0, auth=""):
    """UPS-RS search (GET /workitems) → list of DICOM-JSON workitem records. ``[]`` on 204."""
    r = _dicomweb.ups_search(str(server), str(base_path or ""),
                             {str(k): str(v) for k, v in (matches or {}).items()},
                             [str(f) for f in (includefields or [])],
                             int(limit), int(offset), str(auth or ""))
    return json.loads(r["body"]) if r["body"].strip() else []


def retrieve_workitem(server, workitem_uid, *, base_path="", auth=""):
    """UPS-RS retrieve (GET /workitems/{w}) → the workitem as DICOM JSON (dict)."""
    r = _dicomweb.ups_retrieve(str(server), str(base_path or ""), str(workitem_uid), str(auth or ""))
    return json.loads(r["body"]) if r["body"].strip() else {}


def update_workitem(server, workitem_uid, changes, transaction_uid, *, base_path="", auth=""):
    """UPS-RS update (POST /workitems/{w}?transaction=…). ``changes`` = Dataset/dict of
    attributes to merge; ``transaction_uid`` is the one obtained when the workitem was claimed."""
    return _dicomweb.ups_update(str(server), str(base_path or ""), str(workitem_uid),
                                str(transaction_uid), _json_body(changes), str(auth or ""))


def change_workitem_state(server, workitem_uid, state, transaction_uid, *, base_path="", auth=""):
    """UPS-RS change state (PUT /workitems/{w}/state). ``state`` is the CS value — e.g.
    ``"IN PROGRESS"`` / ``"COMPLETED"`` / ``"CANCELED"`` (use ``pydcm.dimse.UPS.IN_PROGRESS`` etc.);
    the body is built natively from (state, transaction_uid)."""
    return _dicomweb.ups_change_state(str(server), str(base_path or ""), str(workitem_uid),
                                      str(state), str(transaction_uid), str(auth or ""))


def request_cancel_workitem(server, workitem_uid, *, reason=None, base_path="", auth=""):
    """UPS-RS cancel request (POST /workitems/{w}/cancelrequest). ``reason`` = Dataset/dict/None."""
    return _dicomweb.ups_request_cancel(str(server), str(base_path or ""), str(workitem_uid),
                                        _json_body(reason), str(auth or ""))


def subscribe_workitem(server, workitem_uid, ae_title, *, base_path="", auth=""):
    """UPS-RS subscribe (POST /workitems/{w}/subscribers/{ae}). Pass the global subscription
    instance UID as ``workitem_uid`` to watch all workitems."""
    return _dicomweb.ups_subscribe(str(server), str(base_path or ""), str(workitem_uid),
                                   str(ae_title), str(auth or ""))


def unsubscribe_workitem(server, workitem_uid, ae_title, *, base_path="", auth=""):
    """UPS-RS unsubscribe (DELETE /workitems/{w}/subscribers/{ae})."""
    return _dicomweb.ups_unsubscribe(str(server), str(base_path or ""), str(workitem_uid),
                                     str(ae_title), str(auth or ""))

__all__ = ['search_studies', 'search_series', 'search_instances', 'search_workitems',
           'retrieve_study', 'retrieve_series', 'retrieve_instance', 'retrieve_frames',
           'retrieve_bulkdata', 'retrieve_rendered', 'retrieve_study_metadata',
           'retrieve_series_metadata', 'retrieve_instance_metadata',
           'retrieve_instance_wado_uri', 'iter_study', 'iter_series', 'store_instances',
           'delete_study', 'delete_series', 'delete_instance', 'create_workitem',
           'retrieve_workitem', 'update_workitem', 'change_workitem_state',
           'request_cancel_workitem', 'subscribe_workitem', 'unsubscribe_workitem']
