# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Fuli Wu
# NOT a medical device — not for clinical or diagnostic use.

"""HL7 v2.5 — parse messages + build ORU^R01 results, over the native HL7 engine.

The HL7 side of the imaging↔EHR seam: read an inbound order/result and emit a radiology
result back to the HIS::

    segs = pydcm.hl7.parse(open("oru.hl7").read())   # -> [{"id": "MSH", "fields": [...]}, ...]
    msg  = pydcm.hl7.build_oru(config, context, observations)   # -> ER7 string

Parsing returns each segment as ``{id, fields}`` (fields split on the field separator — the
universal ER7 shape). ``build_oru`` takes plain dicts. MLLP networking (send/listen) is not
bound here yet — this is the message layer. Requires the optional ``_hl7`` extension.
"""

from __future__ import annotations

try:
    from . import _hl7
except ImportError as _e:                            # pragma: no cover
    raise ImportError(
        "pydcm.hl7 requires the optional native _hl7 extension, "
        "which is not present in this build."
    ) from _e


def parse(text: str) -> list[dict]:
    """Parse an HL7 v2 message into a list of segments.

    Each segment is ``{"id": "MSH"|"PID"|…, "fields": [str, …]}`` where ``fields`` is the
    segment split on its field separator (ER7). Raises on a malformed message.
    """
    return _hl7.parse(str(text))


def build_oru(config: dict, context: dict, observations: list[dict]) -> str:
    """Build an HL7 v2.5 ``ORU^R01`` result message (ER7 string).

    ``config`` → MSH (``our_app``/``our_facility``/``their_app``/``their_facility``/
    ``control_id``/``timestamp``/``version``/``specific_character_set``); ``context`` → PID +
    ORC/OBR order identifiers (patient_*, *_order_number, accession_number, procedure_*,
    modality, ordering_provider, observation_datetime); ``observations`` → OBX rows
    (``observation_id``, ``value``, ``value_type`` def "TX", ``status`` def "F"). Absent keys
    keep the native defaults.
    """
    return _hl7.build_oru(dict(config), dict(context), list(observations))

__all__ = ['parse', 'build_oru']
