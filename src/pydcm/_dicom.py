# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Fuli Wu
# NOT a medical device — not for clinical or diagnostic use.

"""An object model over the native engine.

``dcmread(path)`` returns a :class:`Dataset`:

    ds = pydcm.dcmread("scan.dcm")
    ds.PatientName            # -> PersonName('CompressedSamples^MR1')
    ds.Rows, ds.Columns       # -> 64, 64
    ds[0x0010, 0x0010].value  # element access by tag / keyword / (group,elem)
    for elem in ds: ...       # iterate elements in tag order
    ds.pixel_array            # -> NumPy array (lazy native decode)

Nothing here re-parses DICOM: the bytes are decoded once in C++ (every transfer
syntax, charset → UTF-8) into the DICOM JSON Model via ``_core.read_json``; this
module only maps that model onto Python types. Keyword ⇄ tag ⇄
VR all resolve through the native superset dictionary (``_core``'s
17,699-entry table), so attribute names follow the standard keyword set.
"""

from __future__ import annotations

import base64
import json
from typing import Any

from . import _native

# VR groupings that drive JSON-value → Python-type conversion (PS3.5 / PS3.18 §F).
_VR_INT = frozenset({"US", "UL", "SS", "SL", "SV", "UV", "IS"})
_VR_FLOAT = frozenset({"FL", "FD", "DS"})
_VR_BINARY = frozenset({"OB", "OW", "OF", "OD", "OL", "OV", "UN",
                        "OB or OW", "US or OW", "US or SS or OW"})
# everything else with a textual Value (AE AS CS DA DT LO LT SH ST TM UC UI UR UT)
# falls through to str.


class Tag(int):
    """A DICOM tag — an ``int`` (``group << 16 | element``) that prints ``(gggg,eeee)``.

    construction: ``Tag(0x00100010)``, ``Tag(0x0010, 0x0010)``,
    ``Tag((0x0010, 0x0010))``, ``Tag('PatientName')`` and ``Tag('0010,0010')``."""

    __slots__ = ()

    def __new__(cls, arg, arg2=None):
        if arg2 is not None:                              # Tag(group, element)
            return super().__new__(cls, ((int(arg) & 0xFFFF) << 16) | (int(arg2) & 0xFFFF))
        if isinstance(arg, tuple):                        # Tag((group, element))
            return super().__new__(cls, ((int(arg[0]) & 0xFFFF) << 16) | (int(arg[1]) & 0xFFFF))
        if isinstance(arg, str):                          # Tag('PatientName') / Tag('0010,0010')
            t = _native.tag_for_keyword(arg)
            if t is not None:
                return super().__new__(cls, t)
            return super().__new__(
                cls, int(arg.replace("(", "").replace(")", "").replace(",", "").strip(), 16))
        return super().__new__(cls, int(arg))             # Tag(int) / Tag(Tag)

    @property
    def group(self) -> int:
        return self >> 16

    @property
    def element(self) -> int:
        return self & 0xFFFF

    @property
    def is_private(self) -> bool:
        return bool(self.group & 1)

    @property
    def is_private_creator(self) -> bool:
        return self.is_private and 0x0010 <= self.element <= 0x00FF

    def __str__(self) -> str:
        return f"({self.group:04X},{self.element:04X})"

    __repr__ = __str__


# BaseTag is the int subclass exposed for API compatibility; Tag() is the factory.
BaseTag = Tag


class PersonName(str):
    """A PN value. ``str(pn)`` is the full wire form 'Alphabetic=Ideographic=Phonetic'
    (trailing empty groups trimmed); ``.alphabetic`` /
    ``.ideographic`` / ``.phonetic`` are the component groups (PS3.5 §6.2.1) and the
    name-part properties (family_name, …) read the alphabetic group.
    """

    __slots__ = ("alphabetic", "ideographic", "phonetic")

    def __new__(cls, alphabetic="", ideographic="", phonetic=""):
        # compat: a single string with '=' group separators is the wire form
        # 'Alphabetic=Ideographic=Phonetic' (PS3.5 §6.2.1.1) — split it.
        if isinstance(alphabetic, str) and "=" in alphabetic and not ideographic and not phonetic:
            g = alphabetic.split("=")
            alphabetic = g[0]
            ideographic = g[1] if len(g) > 1 else ""
            phonetic = g[2] if len(g) > 2 else ""
        groups = [alphabetic, ideographic, phonetic]
        last = max((i for i, g in enumerate(groups) if g), default=0)
        self = super().__new__(cls, "=".join(groups[: last + 1]))
        self.alphabetic = alphabetic
        self.ideographic = ideographic
        self.phonetic = phonetic
        return self

    def _component(self, i: int) -> str:
        parts = self.alphabetic.split("^")
        return parts[i] if i < len(parts) else ""

    family_name = property(lambda self: self._component(0))
    given_name = property(lambda self: self._component(1))
    middle_name = property(lambda self: self._component(2))
    name_prefix = property(lambda self: self._component(3))
    name_suffix = property(lambda self: self._component(4))

    @property
    def components(self):
        g = [self.alphabetic, self.ideographic, self.phonetic]
        last = max((i for i, x in enumerate(g) if x), default=0)
        return tuple(g[: last + 1])

    def family_comma_given(self) -> str:
        return f"{self.family_name}, {self.given_name}"      # puts a space after the comma

    @property
    def encodings(self):
        return getattr(self, "_encodings", None)

    def formatted(self, format_str: str) -> str:
        """Format the name with a ``%(component)s`` template."""
        return format_str % {
            "family_name": self.family_name, "given_name": self.given_name,
            "middle_name": self.middle_name, "name_prefix": self.name_prefix,
            "name_suffix": self.name_suffix,
            "ideographic": self.ideographic, "phonetic": self.phonetic,
        }

    def decode(self, encodings=None) -> "PersonName":
        """Already-decoded (pydcm decodes at read time); returns self."""
        return self

    def encode(self, encodings=None) -> bytes:
        """The on-the-wire bytes for this name. ``encodings`` is a SpecificCharacterSet
        list; pydcm encodes the unicode wire form to the requested charset,
        defaulting to UTF-8 when none is given."""
        from .charset import _python_encodings
        encs = _python_encodings(encodings) if encodings else ["utf-8"]
        try:
            return str(self).encode(encs[-1])
        except (LookupError, UnicodeEncodeError):
            return str(self).encode("utf-8")

    @classmethod
    def from_named_components(cls, family_name="", given_name="", middle_name="",
                              name_prefix="", name_suffix="",
                              family_name_ideographic="", given_name_ideographic="",
                              middle_name_ideographic="", name_prefix_ideographic="",
                              name_suffix_ideographic="",
                              family_name_phonetic="", given_name_phonetic="",
                              middle_name_phonetic="", name_prefix_phonetic="",
                              name_suffix_phonetic="", encodings=None) -> "PersonName":
        """Build a PersonName from individual name components."""
        def join(*parts):
            s = "^".join(parts)
            return s.rstrip("^")
        return cls(join(family_name, given_name, middle_name, name_prefix, name_suffix),
                   join(family_name_ideographic, given_name_ideographic,
                        middle_name_ideographic, name_prefix_ideographic, name_suffix_ideographic),
                   join(family_name_phonetic, given_name_phonetic,
                        middle_name_phonetic, name_prefix_phonetic, name_suffix_phonetic))

    @classmethod
    def from_named_components_veterinary(cls, responsible_party_name="",
                                         patient_name="", encodings=None) -> "PersonName":
        """Build a veterinary PersonName (ResponsibleParty^PatientName)."""
        s = "^".join([responsible_party_name, patient_name]).rstrip("^")
        return cls(s)


class DSfloat(float):
    """A DS value — a ``float`` that remembers its original string."""

    __slots__ = ("original_string",)

    def __new__(cls, val):
        self = super().__new__(cls, val)
        self.original_string = val if isinstance(val, str) else _fmt_ds(val)
        return self

    def __str__(self):
        return self.original_string

    def __repr__(self):
        return f"'{self.original_string}'"


class IS(int):
    """An IS value — an ``int`` that remembers its original string."""

    # NB: int is variable-length, so a nonempty __slots__ is not permitted — IS
    # instances carry a __dict__ for ``original_string``.

    def __new__(cls, val):
        self = super().__new__(cls, val if not isinstance(val, str) or val.strip() else 0)
        self.original_string = val if isinstance(val, str) else str(val)
        return self

    def __str__(self):
        return self.original_string

    def __repr__(self):
        return f"'{self.original_string}'"


class ISfloat(float):
    """An IS value whose text is numeric but not an integer (e.g. '2.5') — a ``float``
    subclass preserving the original string."""

    __slots__ = ("original_string",)

    def __new__(cls, val):
        self = super().__new__(cls, val)
        self.original_string = val if isinstance(val, str) else repr(float(val))
        return self

    def __str__(self):
        return self.original_string


def _make_is(v):
    """IS value: an :class:`IS` (int) for integer text, else an :class:`ISfloat`
    (float subclass) for numeric-but-non-integer text."""
    try:
        return IS(v)
    except (ValueError, TypeError):
        try:
            return ISfloat(v)
        except (ValueError, TypeError):
            return v


def _fmt_ds(v) -> str:
    """DS string formatting (≤16 chars)."""
    s = repr(float(v))
    return s if len(s) <= 16 else f"{float(v):.10g}"


class MultiValue(list):
    """A multi-valued element value (VM > 1) — a ``list``."""

    __slots__ = ()

    def __repr__(self) -> str:
        # renders numeric (DS/IS) elements bare but quotes strings:
        # [0.66, 0.66] / [1, 2] / ['ORIGINAL', 'PRIMARY'].
        def render(v):
            return str(v) if isinstance(v, (DSfloat, IS, ISfloat)) else repr(v)
        return "[" + ", ".join(render(v) for v in self) + "]"


class Sequence(list):
    """An SQ value — a ``list`` of :class:`Dataset` items."""

    __slots__ = ()

    def __repr__(self) -> str:
        return f"<Sequence, length {len(self)}>"


# --------------------------------------------------------------------------- #
#  JSON DICOM Model  →  Python values
# --------------------------------------------------------------------------- #
def _pn(component: Any) -> PersonName:
    if isinstance(component, dict):
        return PersonName(component.get("Alphabetic", ""),
                          component.get("Ideographic", ""),
                          component.get("Phonetic", ""))
    return PersonName(str(component))


def _pn_to_json(v: Any) -> dict:
    """PersonName → DICOM-JSON PN component object (omit empty groups)."""
    node = {"Alphabetic": getattr(v, "alphabetic", str(v))}
    ideo = getattr(v, "ideographic", "")
    phon = getattr(v, "phonetic", "")
    if ideo:
        node["Ideographic"] = ideo
    if phon:
        node["Phonetic"] = phon
    return node


def _scalar(vr: str, v: Any) -> Any:
    if vr == "PN":
        return _pn(v)
    try:
        if vr == "DS":
            return DSfloat(v)            # float subclass, preserves the original string
        if vr == "IS":
            return _make_is(v)           # IS(int) or ISfloat(float)
        if vr in _VR_INT:
            return int(v)
        if vr in _VR_FLOAT:
            return float(v)
        if vr == "AT":
            return Tag(int(v, 16)) if isinstance(v, str) else Tag(int(v))
        if vr == "UI" and isinstance(v, str) and v:
            from .uid import UID
            return UID(v)               # so .name/.is_little_endian/… work
    except (TypeError, ValueError):
        return v   # malformed numeric → keep the raw value
    return v  # string VRs (and anything already a str)


def _convert(vr: str, node: dict, path) -> Any:
    """Map a DICOM-JSON element node to a Python value."""
    if vr == "SQ":
        return Sequence(_build(item, path) for item in node.get("Value", []))
    if "InlineBinary" in node:                       # OB/OW/… inline base64
        return base64.b64decode(node["InlineBinary"])
    if "BulkDataURI" in node:
        return node["BulkDataURI"]
    values = node.get("Value")
    if not values:                                   # absent / empty element
        if vr in _VR_BINARY:
            return None                              # empty binary → None
        if vr in _VR_INT or vr in _VR_FLOAT or vr == "AT":
            return None                              # empty numeric → None
        if vr == "PN":
            return PersonName("")
        return ""                                    # empty text VR → ''
    out = [_scalar(vr, v) for v in values]
    return out[0] if len(out) == 1 else MultiValue(out)


def _build(jmap: dict, path=None) -> "Dataset":
    """Build a Dataset from a tag-keyed DICOM-JSON map (recursing into SQ items)."""
    elements: dict[int, DataElement] = {}
    for key in sorted(jmap):
        node = jmap[key]
        if not isinstance(node, dict) or "vr" not in node:
            continue
        tag = Tag(int(key, 16))
        vr = node["vr"]
        # Bulk pixel data (PixelData / FloatPixelData / DoubleFloatPixelData) arrives as a
        # value-less {"vr":...} stub: read_json carries the real VR out of the single decode,
        # but the bulk bytes stay lazy (loaded from the file on first access).
        # Keeps the file's true VR (OB/OW/OF/OD) — no guess, no reread.
        if int(tag) in _PIXEL_DATA_TAGS and path is not None and "Value" not in node \
                and "InlineBinary" not in node and "BulkDataURI" not in node:
            elements[tag] = DataElement(tag, vr, _LAZY, keyword=_PIXEL_DATA_TAGS[int(tag)],
                                        lazy=(lambda pp=path: _native.read_pixel_data(pp)))
            continue
        # Lazy per-element conversion: most workflows touch a handful of the
        # ~100+ elements in a dataset, and the eager _convert pass dominated
        # dcmread (fast on small files). The JSON node is kept and
        # converted on first .value access via DataElement's lazy mechanism
        # (the same one bulk PixelData uses).
        elements[tag] = DataElement(tag, vr, _LAZY,
                                    lazy=(lambda _v=vr, _n=node, _p=path: _convert(_v, _n, _p)))
    return Dataset._from_elements(elements, path=path)


# --------------------------------------------------------------------------- #
#  Python values  →  JSON DICOM Model  (for save_as / to_json)
# --------------------------------------------------------------------------- #
def _encode(elem: "DataElement") -> dict:
    vr, value = elem.VR, elem.value
    node: dict[str, Any] = {"vr": vr}
    if vr == "SQ":
        node["Value"] = [{f"{e.tag.group:04X}{e.tag.element:04X}": _encode(e)
                          for e in ds} for ds in (value or [])]
        return node
    if isinstance(value, (bytes, bytearray)):
        if value:
            node["InlineBinary"] = base64.b64encode(bytes(value)).decode("ascii")
        return node
    if value is None:
        return node
    seq = value if isinstance(value, list) else [value]
    if vr == "PN":
        node["Value"] = [_pn_to_json(v) for v in seq]
    elif vr == "AT":
        node["Value"] = [f"{int(v):08X}" for v in seq]
    else:
        node["Value"] = list(seq)
    return node


def _edit_one(v) -> "str | None":
    """One scalar → DICOM string for an edit op; None if not string-encodable."""
    if isinstance(v, (bytes, bytearray)):
        return None
    if isinstance(v, PersonName):
        return str(v)                       # full 'Alphabetic=Ideographic=Phonetic'
    if isinstance(v, Tag):
        return f"{v.group:04X}{v.element:04X}"
    return "" if v is None else str(v)


def _edit_value_str(value) -> "str | None":
    """Value → DICOM string form (backslash-joined for VM>1); None if binary."""
    if isinstance(value, (bytes, bytearray)):
        return None
    seq = value if isinstance(value, (list, MultiValue)) else [value]
    parts = [_edit_one(x) for x in seq]
    return None if any(p is None for p in parts) else "\\".join(parts)


# --------------------------------------------------------------------------- #
#  DataElement
# --------------------------------------------------------------------------- #
_LAZY = object()        # sentinel: a DataElement whose value loads on first access

# The three bulk pixel-data tags → keyword. A dataset carries at most one of them.
_PIXEL_DATA_TAGS = {
    0x7FE00010: "PixelData",
    0x7FE00008: "FloatPixelData",
    0x7FE00009: "DoubleFloatPixelData",
}


def _pixel_data_vr(path, ds):
    """VR for a lazily-reconstructed PixelData element: the file's AUTHORITATIVE on-disk
    VR for Explicit-VR transfer syntaxes (preserving the encoder's OB-vs-OW choice for
    <=8-bit data, PS3.5), else inferred from Bits Allocated (Implicit VR stores no VR)."""
    vr = _native.pixel_data_vr(path)
    if vr:
        return vr
    ba = ds.get("BitsAllocated", 16) or 16
    return "OB" if ba <= 8 else "OW"

# VRs summarised as "Array of N elements" once the value exceeds the display
# limit (byte-like + long-text VRs; the ambiguous forms included in case
# one survives unresolved).
_LONG_VALUE_VR = frozenset({
    "LT", "OB", "OW", "OD", "OF", "OL", "OV", "UC", "UN", "UT",
    "OB or OW", "US or OW", "US or SS", "US or SS or OW",
})


class DataElement:
    """One (tag, VR, value) triple — surface."""

    __slots__ = ("tag", "VR", "_value", "_kw", "_name", "_owner", "_lazy")

    # repr-formatting knobs (class-level, overridable).
    descripWidth = 35
    maxBytesToDisplay = 16
    showVR = True

    def __init__(self, tag, vr, value, *, keyword=None, name=None, _from_json=False,
                 lazy=None):
        self.tag = tag if isinstance(tag, Tag) else Tag(tag)
        self.VR = vr
        self._owner = None          # (Dataset, tag) — set when the element lives in a Dataset
        self._value = value
        self._kw = keyword
        self._name = name
        self._lazy = lazy           # callable → value, loaded once on first .value access

    @property
    def value(self):
        if self._value is _LAZY and self._lazy is not None:
            self._value = self._lazy()
            self._lazy = None
        return self._value

    @value.setter
    def value(self, v):
        # In-place mutation (ds[tag].value = x  /  for e in ds: e.value = x) MUST record
        # an edit, or save_as patches the original bytes and silently drops it (the
        # anonymise idiom). Record it on the owning Dataset, exactly like ds.kw = x does.
        self._value = v
        if self._owner is not None:
            ds, tag = self._owner
            ds._edits[tag] = ("insert", v, self.VR)

    def _info(self):
        if self._kw is None or self._name is None:
            info = _native.describe_tag(int(self.tag))
            self._kw = "" if info is None else info["keyword"]
            self._name = (self._kw or "") if info is None else info["name"]
        return self

    @property
    def keyword(self) -> str:
        return self._info()._kw

    @property
    def name(self) -> str:
        # DataElement.name semantics, including private-dictionary resolution
        # against the native 12,608-pattern private dict (a superset).
        if self.tag.is_private:
            creator = self.private_creator
            if creator:
                info = _native.describe_private(
                    creator, self.tag.group, self.tag.element & 0xFF)
                if info is not None:                 # known private element → "[name]"
                    return f"[{info['name']}]"
            elif (self.tag.element >> 8) == 0:
                return "Private Creator"
            return "Private tag data"                # unknown private element
        nm = self._info()._name
        if nm:
            return nm
        if self.tag.element == 0:
            return "Group Length"
        return ""

    def description(self) -> str:
        """The element's name (its data-dictionary description)."""
        return self.name

    @property
    def is_private(self) -> bool:
        return self.tag.is_private

    @property
    def is_private_creator(self) -> bool:
        return self.tag.is_private_creator

    @property
    def is_retired(self) -> bool:
        info = _native.describe_tag(int(self.tag))
        return bool(info and info.get("retired"))

    @property
    def private_creator(self):
        """The private-creator string owning this private element, or None."""
        if not self.tag.is_private or self._owner is None:
            return None
        block = (self.tag.element >> 8) & 0xFF
        if block == 0:
            return None
        ds = self._owner[0]
        creator = ds._dict.get(Tag((self.tag.group << 16) | block))
        return creator.value if creator is not None else None

    @property
    def is_little_endian(self) -> bool:
        return self._owner[0].is_little_endian if self._owner is not None else True

    @property
    def VM(self) -> int:
        v = self.value
        if v is None or isinstance(v, (bytes, bytearray, str)):
            return 0 if v is None or v == "" else 1
        return len(v) if isinstance(v, list) else 1

    @property
    def is_empty(self) -> bool:
        return self.value in (None, "", b"")

    @property
    def is_raw(self) -> bool:
        return False                     # pydcm decodes eagerly — never a RawDataElement

    @property
    def is_buffered(self) -> bool:
        return isinstance(self._value, (bytes, bytearray, memoryview))

    @property
    def empty_value(self):
        """The value used to represent an empty element of this VR."""
        if self.VR in ("PN", "SH", "LO", "ST", "LT", "UT", "UC", "AE", "AS", "CS",
                       "DA", "DT", "TM", "UI", "UR", "DS", "IS"):
            return ""
        if self.VR == "SQ":
            return None
        return None

    def clear(self) -> None:
        """Clear this element's value."""
        self.value = self.empty_value

    @property
    def repval(self) -> str:
        """A string representation of the value for display."""
        return self._value_repr()

    def validate(self, value=None) -> None:
        """Accepted for source compatibility; pydcm validates leniently (no-op)."""
        return None

    def to_json_dict(self, bulk_data_threshold=1024, bulk_data_element_handler=None):
        """This element as a one-key DICOM JSON Model dict."""
        return {f"{self.tag.group:04X}{self.tag.element:04X}": _encode(self)}

    def to_json(self, bulk_data_threshold=1024, bulk_data_element_handler=None,
                dump_handler=None):
        """This element as a DICOM JSON Model string."""
        node = _encode(self)
        return json.dumps(node)

    @classmethod
    def from_json(cls, tag, vr, value, value_key="Value", bulk_data_uri_handler=None):
        """Build a :class:`DataElement` from DICOM JSON Model pieces."""
        node = {"vr": vr, value_key: value} if not isinstance(value, dict) else value
        return _build({f"{int(Tag(tag)):08X}": node})[Tag(tag)]

    def _value_repr(self) -> str:
        """The element value as ``repval`` would render it."""
        v = self.value
        if isinstance(v, Sequence):                  # only via direct repr(); _lines bypasses SQ
            return repr(v)
        # byte-like / long-text VRs: summarise once longer than maxBytesToDisplay
        if self.VR in _LONG_VALUE_VR:
            try:
                length = len(v)
            except TypeError:
                pass
            else:
                if length > self.maxBytesToDisplay:
                    return f"Array of {length} elements"
        if self.VM > self.maxBytesToDisplay:         # multi-valued beyond the display cap
            return f"Array of {self.VM} elements"
        if self.VR == "UI" and isinstance(v, str) and v:
            # shows a UID as its NAME, unquoted (registered name when known,
            # otherwise the raw UID string) — e.g. "RT Plan Storage", "1.3.6.1.4...".
            from .uid import UID
            return UID(v).name or v
        return repr(v)

    def __repr__(self) -> str:
        # DataElement.__str__: (gggg,eeee) <name truncated+padded to descripWidth> VR: value
        name = f"{self.name[:self.descripWidth]:<{self.descripWidth}}"
        if self.showVR:
            return f"{self.tag} {name} {self.VR}: {self._value_repr()}"
        return f"{self.tag} {name} {self._value_repr()}"


class PrivateBlock:
    """A reserved private-tag block: elements ``(group, (block<<8)|offset)``
    owned by a private creator. ``block.add_new(offset, VR, value)`` / ``block[offset]``."""

    __slots__ = ("dataset", "group", "block", "private_creator")

    def __init__(self, dataset, group, block, private_creator):
        self.dataset = dataset
        self.group = group
        self.block = block                       # the reservation element (0x10..0xFF)
        self.private_creator = private_creator

    def get_tag(self, element_offset: int) -> Tag:
        return Tag((self.group << 16) | (self.block << 8) | (element_offset & 0xFF))

    def __contains__(self, element_offset: int) -> bool:
        return self.get_tag(element_offset) in self.dataset._dict

    def __getitem__(self, element_offset: int) -> DataElement:
        return self.dataset[self.get_tag(element_offset)]

    def __delitem__(self, element_offset: int) -> None:
        del self.dataset[self.get_tag(element_offset)]

    def add_new(self, element_offset: int, VR: str, value) -> None:
        self.dataset.add_new(self.get_tag(element_offset), VR, value)


# --------------------------------------------------------------------------- #
#  Dataset
# --------------------------------------------------------------------------- #
class Dataset:
    """An ordered collection of :class:`DataElement`, keyed by tag.

    Supports keyword attributes (``ds.PatientName``), item access by
    tag / keyword / ``(group, element)``, ``in`` / iteration / ``len`` / ``get``,
    a lazy ``pixel_array`` and ``save_as``. Construct via :func:`dcmread`.
    """

    __slots__ = ("_dict", "_path", "_pixels", "_file_meta", "_edits", "_source_bytes")

    # str()-formatting knobs (class-level, overridable).
    default_element_format = "{tag} {name:<35} {VR}: {repval}"
    default_sequence_element_format = "{tag} {name:<35} {VR}: {repval}"
    indent_chars = "   "

    def __init__(self, *args, **kwargs):
        if len(args) > 1:
            raise TypeError(
                f"Dataset expected at most 1 positional argument, got {len(args)}")
        object.__setattr__(self, "_dict", {})
        object.__setattr__(self, "_path", None)
        object.__setattr__(self, "_pixels", None)
        object.__setattr__(self, "_file_meta", None)  # group-0002 (lazy: see file_meta)
        # Edits applied after construction (tag -> ('insert', value, vr) | ('erase',…)).
        # Lets save_as patch the ORIGINAL file byte-verbatim (keeping TS + pixels)
        # instead of re-serialising the lossy metadata JSON.
        object.__setattr__(self, "_edits", {})
        # In-memory Part-10 source for the byte-verbatim save path when there is no backing
        # file (`_path` is None) — set for received DIMSE data sets so save_as preserves their
        # exact bytes / Transfer Syntax (incl. compressed) without a persistent temp file.
        object.__setattr__(self, "_source_bytes", None)
        if args:
            self._load_initial(args[0])
        for key, value in kwargs.items():
            if _native.tag_for_keyword(key) is not None:
                setattr(self, key, value)

    @classmethod
    def _from_elements(cls, elements=None, *, path=None):
        ds = cls()
        object.__setattr__(ds, "_path", path)
        ds._load_initial(elements)
        object.__setattr__(ds, "_edits", {})
        return ds

    def _load_initial(self, elements) -> None:
        if not elements:
            return
        if isinstance(elements, Dataset):
            iterable = elements._dict.items()
        elif hasattr(elements, "items"):
            iterable = elements.items()
        else:
            iterable = dict(elements).items()
        for key, value in iterable:
            if isinstance(value, DataElement):
                tag = self._as_tag(getattr(value, "tag", key))
                elem = value
            elif hasattr(value, "tag") and hasattr(value, "VR"):
                tag = self._as_tag(getattr(value, "tag"))
                elem = DataElement(tag, str(value.VR), getattr(value, "value", None))
            else:
                tag = self._as_tag(key)
                info = _native.describe_tag(int(tag))
                elem = DataElement(tag, info["vr"] if info else "UN", value,
                                   keyword=info and info.get("keyword"),
                                   name=info and info.get("name"))
            self._dict[tag] = elem
            elem._owner = (self, tag)

    # -- file meta (lazy) ------------------------------------------------------ #
    @property
    def file_meta(self):
        """Group-0002 File Meta Information, built lazily from the backing file
        on first access (dcmread no longer pays the second native read when the
        caller never looks at it). ``None`` for path-less, hand-built Datasets."""
        fm = self._file_meta
        if fm is None and self._path is not None:
            fm = _build_file_meta(self._path)
            object.__setattr__(self, "_file_meta", fm)
        return fm

    @file_meta.setter
    def file_meta(self, value) -> None:
        object.__setattr__(self, "_file_meta", value)

    # -- key normalisation --------------------------------------------------- #
    @staticmethod
    def _as_tag(key) -> Tag:
        if isinstance(key, Tag):
            return key
        if isinstance(key, tuple):
            return Tag((int(key[0]) << 16) | int(key[1]))
        if isinstance(key, int):
            return Tag(key)
        if isinstance(key, str):
            t = _native.tag_for_keyword(key)
            if t is not None:
                return Tag(t)
            cleaned = key.replace("(", "").replace(")", "").replace(",", "").strip()
            return Tag(int(cleaned, 16))
        raise TypeError(f"cannot interpret {key!r} as a DICOM tag")

    # -- mapping protocol ---------------------------------------------------- #
    def __getitem__(self, key):
        if isinstance(key, slice):
            return self._slice(key)
        tag = self._as_tag(key)
        elem = self._dict[tag]
        elem._owner = (self, tag)            # so an in-place elem.value = x records an edit
        return elem

    def _slice(self, sl: slice) -> "Dataset":
        """Tag-range slicing ``ds[start:stop:step]`` → sub-:class:`Dataset`."""
        lo = 0 if sl.start is None else int(self._as_tag(sl.start))
        hi = (1 << 32) if sl.stop is None else int(self._as_tag(sl.stop))
        step = sl.step or 1
        out = Dataset()
        for t in sorted(self._dict):
            if lo <= int(t) < hi and (int(t) - lo) % step == 0:
                out._dict[t] = self._dict[t]
        return out

    def __setitem__(self, key, elem: DataElement) -> None:
        tag = self._as_tag(key)
        self._dict[tag] = elem
        elem._owner = (self, tag)
        self._edits[tag] = ("insert", elem.value, elem.VR)

    def __delitem__(self, key) -> None:
        tag = self._as_tag(key)
        del self._dict[tag]
        self._edits[tag] = ("erase", None, None)

    def __contains__(self, key) -> bool:
        try:
            return self._as_tag(key) in self._dict
        except (ValueError, TypeError):
            return False

    def __iter__(self):
        for tag in sorted(self._dict):
            elem = self._dict[tag]
            elem._owner = (self, tag)
            yield elem

    def __len__(self) -> int:
        return len(self._dict)

    elements = __iter__

    def __eq__(self, other) -> bool:
        if not isinstance(other, Dataset):
            return NotImplemented
        return set(self._dict) == set(other._dict) and all(
            self._dict[t].value == other._dict[t].value for t in self._dict)

    def __ne__(self, other):
        r = self.__eq__(other)
        return r if r is NotImplemented else not r

    __hash__ = None        # Datasets are unhashable

    def get(self, key, default=None):
        """``Dataset.get``: a keyword string returns the VALUE; a tag/int/tuple
        returns the :class:`DataElement` (so ``ds.get(tag).value`` works)."""
        if isinstance(key, str):
            try:
                return self._dict[self._as_tag(key)].value
            except (KeyError, ValueError, TypeError):
                return default
        try:
            return self[key]
        except (KeyError, ValueError, TypeError):
            return default

    # -- full Mapping surface ---------------------------------------- #
    def get_item(self, key) -> DataElement:
        """The raw :class:`DataElement` for ``key`` (no value conversion)."""
        return self[key]

    def data_element(self, keyword):
        """The :class:`DataElement` for ``keyword``, or ``None``."""
        try:
            return self[keyword]
        except (KeyError, ValueError, TypeError):
            return None

    def keys(self):
        return sorted(self._dict)

    def values(self):
        return [self[t] for t in sorted(self._dict)]

    def items(self):
        return [(t, self[t]) for t in sorted(self._dict)]

    def pop(self, key, *default):
        try:
            tag = self._as_tag(key)
        except (ValueError, TypeError):
            tag = None
        if tag is not None and tag in self._dict:
            elem = self._dict.pop(tag)
            self._edits[tag] = ("erase", None, None)
            return elem
        if default:
            return default[0]
        raise KeyError(key)

    def setdefault(self, key, default=None):
        tag = self._as_tag(key)
        if tag in self._dict:
            return self._dict[tag]
        if isinstance(default, DataElement):
            elem = default
        else:
            info = _native.describe_tag(int(tag))
            elem = DataElement(tag, info["vr"] if info else "UN", default)
        self[tag] = elem
        return elem

    def update(self, other) -> None:
        if isinstance(other, Dataset):
            for e in other:
                self[e.tag] = DataElement(e.tag, e.VR, e.value)
        else:
            for k, v in dict(other).items():
                if isinstance(v, DataElement):
                    self[k] = v
                elif isinstance(k, str) and _native.tag_for_keyword(k):
                    setattr(self, k, v)

    def clear(self) -> None:
        for t in list(self._dict):
            self._edits[t] = ("erase", None, None)
        self._dict.clear()

    def __copy__(self) -> "Dataset":
        # shallow copy: a NEW dict container but the SAME DataElement objects,
        # so mutating a shared element's value is visible in both, while add/remove is not.
        new = type(self)()
        object.__setattr__(new, "_dict", dict(self._dict))
        object.__setattr__(new, "_path", self._path)
        object.__setattr__(new, "_edits", dict(self._edits))
        object.__setattr__(new, "_source_bytes", self._source_bytes)
        object.__setattr__(new, "file_meta", self.file_meta)
        return new

    copy = __copy__

    def popitem(self):
        """Remove and return the highest-tag ``(tag, DataElement)`` pair."""
        if not self._dict:
            raise KeyError("popitem(): dataset is empty")
        tag = sorted(self._dict)[-1]
        elem = self._dict.pop(tag)
        self._edits[tag] = ("erase", None, None)
        return tag, elem

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False

    @property
    def is_little_endian(self) -> bool:
        ts = self.file_meta.get("TransferSyntaxUID") if self.file_meta is not None else None
        return ts != "1.2.840.10008.1.2.2"          # only EVR Big Endian is big

    @property
    def is_implicit_VR(self) -> bool:
        ts = self.file_meta.get("TransferSyntaxUID") if self.file_meta is not None else None
        return ts == "1.2.840.10008.1.2"

    def __deepcopy__(self, memo) -> "Dataset":
        import copy as _copy
        new = type(self)()
        object.__setattr__(new, "_dict",
                           {t: DataElement(e.tag, e.VR, _copy.deepcopy(e.value, memo),
                                           keyword=e._kw, name=e._name)
                            for t, e in self._dict.items()})
        object.__setattr__(new, "_path", self._path)
        object.__setattr__(new, "_edits", dict(self._edits))
        object.__setattr__(new, "_source_bytes", self._source_bytes)   # bytes are immutable — share
        object.__setattr__(new, "file_meta",
                           _copy.deepcopy(self.file_meta, memo) if self.file_meta is not None else None)
        return new

    def walk(self, callback, recursive=True) -> None:
        """Call ``callback(ds, elem)`` for each element, recursing into SQ."""
        for elem in self:
            callback(self, elem)
            if recursive and elem.VR == "SQ":
                for item in (elem.value or []):
                    item.walk(callback, recursive)

    def iterall(self):
        """Iterate every element, descending into sequences."""
        for elem in self:
            yield elem
            if elem.VR == "SQ":
                for item in (elem.value or []):
                    yield from item.iterall()

    def group_dataset(self, group) -> "Dataset":
        """A sub-:class:`Dataset` of the elements in ``group``."""
        out = Dataset()
        for e in self:
            if e.tag.group == group:
                out._dict[e.tag] = DataElement(e.tag, e.VR, e.value)
        return out

    def remove_private_tags(self) -> None:
        """Delete every private (odd-group) element."""
        for t in [t for t in list(self._dict) if Tag(t).is_private]:
            del self._dict[t]
            self._edits[t] = ("erase", None, None)

    # -- private-tag blocks ---------------------------------------- #
    def private_creators(self, group: int) -> list:
        """The private-creator strings reserved in ``group``."""
        out = []
        for t in sorted(self._dict):
            tg = Tag(t)
            if tg.group == group and 0x10 <= tg.element <= 0xFF:
                v = self._dict[t].value
                if v:
                    out.append(str(v))
        return out

    def private_block(self, group: int, private_creator: str, create: bool = False):
        """Return the :class:`PrivateBlock` for ``private_creator`` in ``group``.

        Locates the (group,00xx) reservation element whose value is ``private_creator``;
        with ``create=True`` reserves the next free block if absent.
        """
        for elem in range(0x10, 0x100):
            t = Tag((group << 16) | elem)
            e = self._dict.get(t)
            if e is not None and str(e.value) == private_creator:
                return PrivateBlock(self, group, elem, private_creator)
        if not create:
            raise KeyError(
                f"private creator {private_creator!r} not found in group {group:04X}")
        for elem in range(0x10, 0x100):
            t = Tag((group << 16) | elem)
            if t not in self._dict:
                self.add_new(t, "LO", private_creator)
                return PrivateBlock(self, group, elem, private_creator)
        raise ValueError(f"no free private block in group {group:04X}")

    def get_private_item(self, group: int, element_offset: int,
                         private_creator: str) -> DataElement:
        """The private :class:`DataElement` at ``element_offset`` for ``private_creator``."""
        return self.private_block(group, private_creator)[element_offset]

    def add_new_private(self, private_creator, group, element_offset, value, vr=None):
        """Add a private element under ``private_creator`` reserving its block."""
        block = self.private_block(group, private_creator, create=True)
        if vr is None:
            info = _native.describe_private(private_creator, group, element_offset & 0xFF)
            vr = info["vr"] if info else "UN"
        block.add_new(element_offset, vr, value)

    # -- encoding info (reflects how the source file was encoded) ---- #
    @property
    def original_encoding(self):
        """``(is_implicit_VR, is_little_endian)`` of the source file, or ``(None, None)``."""
        if self.file_meta is None:
            return (None, None)
        return (self.is_implicit_VR, self.is_little_endian)

    read_implicit_vr = property(lambda self: self.original_encoding[0])
    read_little_endian = property(lambda self: self.original_encoding[1])
    read_encoding = property(lambda self: self.get("SpecificCharacterSet"))

    @property
    def is_original_encoding(self) -> bool:
        return True

    @property
    def original_character_set(self):
        return self.get("SpecificCharacterSet")

    def set_original_encoding(self, is_implicit_vr, is_little_endian,
                              character_encoding=None) -> None:
        """Accepted for source compatibility (pydcm derives encoding from the TS)."""
        return None

    def ensure_file_meta(self) -> None:
        """Create an empty :class:`FileMetaDataset` if this dataset has none."""
        if self.file_meta is None:
            object.__setattr__(self, "file_meta", FileMetaDataset())

    def convert_pixel_data(self, handler_name="") -> None:
        """Force pixel decoding (pydcm decodes lazily; this just primes the cache)."""
        _ = self.pixel_array

    def decode(self) -> None:
        """No-op: pydcm decodes every text VR to unicode at read time."""
        return None

    def update_raw_element(self, tag, *, vr=None, value=None):
        """Update a (raw) element's VR/value in place."""
        e = self[tag]
        if vr is not None:
            e.VR = vr
        if value is not None:
            e.value = value

    def top(self) -> str:
        """A string of only the top-level elements (sequences shown as their
        ``N item(s)`` header, items omitted)."""
        return self._pretty_str(top_level_only=True)

    def formatted_lines(self, element_format="{elem}",
                        sequence_element_format="{elem}", indent_format=None):
        """Yield a formatted line per element."""
        for e in self:
            yield element_format.format(elem=repr(e), name=e.name, tag=e.tag, vr=e.VR)

    def trait_names(self):
        """IPython completion hook — same as :meth:`dir`."""
        return self.dir()

    def dir(self, *filters):
        """Sorted keywords of present elements, optionally substring-filtered."""
        kws = sorted({e.keyword for e in self if e.keyword})
        if filters:
            fl = [f.lower() for f in filters]
            kws = [k for k in kws if any(f in k.lower() for f in fl)]
        return kws

    def __dir__(self):
        base = set(object.__dir__(self))
        base.update(e.keyword for e in self if e.keyword)
        return sorted(base)

    # -- attribute (keyword) access ------------------------------------------ #
    def __getattr__(self, name: str):
        # only reached when normal lookup fails (so never for _dict/_path/_pixels)
        tag = _native.tag_for_keyword(name)
        if tag is None:
            raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")
        try:
            return self._dict[Tag(tag)].value
        except KeyError:
            # Cold fallback: a bulk pixel element (PixelData / Float / DoubleFloat) accessed
            # on a dataset that has no stub (e.g. built without read_json). Fetch the raw
            # bytes lazily from the backing file, reusing the native extractor. Cache as a
            # real element — but NOT as an _edit: it already exists in the file byte-verbatim.
            if tag in _PIXEL_DATA_TAGS and (self._path is not None
                                            or self._source_bytes is not None):
                if self._path is not None:
                    raw = _native.read_pixel_data(self._path)
                    vr = _pixel_data_vr(self._path, self)
                else:
                    raw = _native.read_pixel_data_bytes(self._source_bytes)
                    vr = _native.pixel_data_vr_bytes(self._source_bytes) \
                        or ("OB" if (self.get("BitsAllocated", 16) or 16) <= 8 else "OW")
                if raw is not None:
                    el = DataElement(Tag(tag), vr, raw,
                                     keyword=_PIXEL_DATA_TAGS[tag])
                    self._dict[Tag(tag)] = el
                    return raw
            raise AttributeError(
                f"Dataset has no element {name!r} (tag {Tag(tag)})") from None

    def __setattr__(self, name: str, value) -> None:
        if name in Dataset.__slots__ or name == "file_meta":
            object.__setattr__(self, name, value)
            return
        tag = _native.tag_for_keyword(name)
        if tag is None:
            object.__setattr__(self, name, value)
            return
        info = _native.describe_tag(tag)
        vr = info["vr"] if info else "UN"
        self._dict[Tag(tag)] = DataElement(Tag(tag), vr, value,
                                           keyword=name, name=info and info["name"])
        self._edits[Tag(tag)] = ("insert", value, vr)

    def __delattr__(self, name: str) -> None:
        tag = _native.tag_for_keyword(name)
        if tag is not None and Tag(tag) in self._dict:
            del self._dict[Tag(tag)]
            self._edits[Tag(tag)] = ("erase", None, None)
        else:
            object.__delattr__(self, name)

    # -- pixels -------------------------------------------------------------- #
    @property
    def pixel_array(self):
        """Stored pixel values as a NumPy array (lazy native decode, no rescale).

        Single-frame → ``[rows, cols(, samples)]``; multi-frame keeps the frame
        axis. PALETTE COLOR returns the stored index values; apply
        :func:`pydcm.pixels.apply_color_lut` explicitly to obtain RGB. For
        Hounsfield units use :func:`pydcm.decode` with ``rescale=True`` or
        ``apply_modality_lut``.
        """
        if self._pixels is None:
            from .pixels import _decode_all
            arr = _decode_all(self)
            object.__setattr__(self, "_pixels",
                               arr[0] if arr.shape[0] == 1 else arr)
        return self._pixels

    def waveform_array(self, index: int = 0):
        """The (samples × channels) array of waveform multiplex group ``index``."""
        from .waveforms import multiplex_array
        return multiplex_array(self, index, as_raw=False)

    def pixel_array_options(self, **kwargs) -> None:
        """Accepted for source compatibility; pydcm decodes natively (no plugins)."""
        return None

    def overlay_array(self, group: int):
        """The (rows × cols) {0,1} overlay plane of group ``group`` (60xx), PS3.3 C.9."""
        import numpy as np
        rows = int(self[(group, 0x0010)].value)
        cols = int(self[(group, 0x0011)].value)
        data = self[(group, 0x3000)].value          # OverlayData (OW/OB, 1 bit/pixel)
        bits = np.unpackbits(np.frombuffer(bytes(data), dtype="uint8"), bitorder="little")
        return bits[: rows * cols].reshape(rows, cols).astype("uint8")

    def set_pixel_data(self, arr, photometric_interpretation, bits_stored,
                       *, generate_instance_uid=True) -> None:
        """Set PixelData and the related Image Pixel module elements from ``arr``."""
        import numpy as np
        a = np.asarray(arr)
        frames = a.shape[0] if a.ndim >= 3 and a.shape[-1] not in (3, 4) else 1
        samples = a.shape[-1] if (a.ndim == 3 and a.shape[-1] in (3, 4)) or a.ndim == 4 else 1
        rows, cols = (a.shape[-3], a.shape[-2]) if samples > 1 else a.shape[-2:]
        ba = a.dtype.itemsize * 8
        self.Rows, self.Columns = int(rows), int(cols)
        self.SamplesPerPixel = int(samples)
        self.PhotometricInterpretation = photometric_interpretation
        self.BitsAllocated = int(ba)
        self.BitsStored = int(bits_stored)
        self.HighBit = int(bits_stored) - 1
        self.PixelRepresentation = 1 if a.dtype.kind == "i" else 0
        if frames > 1:
            self.NumberOfFrames = int(frames)
        if samples > 1:
            self.PlanarConfiguration = 0
        self._dict[Tag(0x7FE00010)] = DataElement(
            Tag(0x7FE00010), "OW" if ba > 8 else "OB", a.tobytes(), keyword="PixelData")
        self._edits[Tag(0x7FE00010)] = ("insert", a.tobytes(), "OW" if ba > 8 else "OB")
        object.__setattr__(self, "_pixels", None)
        if generate_instance_uid:
            from .uid import generate_uid
            self.SOPInstanceUID = generate_uid()

    def compress(self, transfer_syntax_uid, arr=None, *, encoding_plugin="",
                 generate_instance_uid=True, quality=0, **_kw) -> None:
        """Compress PixelData to ``transfer_syntax_uid`` in place.

        Re-encodes via the native transcoder. Lossless targets — RLE Lossless
        (1.2.840.10008.1.2.5), JPEG 2000 Lossless (.4.90), JPEG-LS Lossless (.4.80),
        HTJ2K Lossless (.4.201), JPEG-XL Lossless (.4.110) — never take ``quality``.
        The lossy-capable targets (JPEG Baseline .4.50 / Extended .4.51, JPEG-LS
        near-lossless .4.81, J2K .4.91, HTJ2K .4.203, JPEG-XL .4.112) accept
        ``quality`` 1..100 (0 = codec default); LOSSY compression must be this
        explicit — it is never a fallback from a lossless request. When the encode
        actually loses data the result carries LossyImageCompression "01", the ISO
        method in LossyImageCompressionMethod, and a NEW SOP Instance UID, per
        PS3.3 C.7.6.1.1.5. ``arr`` (optional) replaces the pixels first. Raises
        :class:`ValueError` for an unsupported target.
        """
        import os, tempfile, atexit
        if arr is not None:
            self.set_pixel_data(arr, self.get("PhotometricInterpretation", "MONOCHROME2"),
                                int(self.get("BitsStored", self.get("BitsAllocated", 16) or 16)),
                                generate_instance_uid=False)
        ts = str(transfer_syntax_uid)
        try:
            out = _native.transcode(self._encode_part10(), ts, quality)
        except Exception as exc:
            raise ValueError(f"cannot compress to {ts} [{exc}]") from exc
        fd, tmp = tempfile.mkstemp(suffix=".dcm", prefix="pydcm_cmp_")
        os.write(fd, out); os.close(fd)
        atexit.register(lambda p=tmp: os.path.exists(p) and os.unlink(p))
        new = dcmread(tmp)                       # adopt the compressed dataset's state
        object.__setattr__(self, "_dict", new._dict)
        object.__setattr__(self, "_path", new._path)
        object.__setattr__(self, "_pixels", None)
        object.__setattr__(self, "_edits", {})
        # Drop any in-memory verbatim source (a received data set's original bytes): the state
        # is now backed by the freshly transcoded file (_path), so the old bytes are stale.
        object.__setattr__(self, "_source_bytes", new._source_bytes)
        object.__setattr__(self, "file_meta", new.file_meta)
        if generate_instance_uid:
            from .uid import generate_uid
            self.SOPInstanceUID = generate_uid()
            # File meta must name the same instance the dataset does — PS3.10
            # requires (0002,0003) == (0008,0018).
            if self.file_meta is not None:
                self.file_meta.MediaStorageSOPInstanceUID = self.SOPInstanceUID

    def decompress(self, handler_name="", *, as_rgb=True,
                   generate_instance_uid=True, decoding_plugin="", **_kw) -> None:
        """Decode compressed PixelData to Explicit VR Little Endian in place.

        The native transcoder owns frame reconstruction and native packing, so
        PALETTE COLOR retains its LUT plus stored indices and multi-frame 1-bit
        data is packed as one continuous PS3.5 bitstream. ``as_rgb=False`` is
        unsupported for YBR sources because the native transcoder normalizes
        decoded YBR samples to RGB.
        """
        if handler_name:
            raise NotImplementedError(
                "handler_name is not supported by the native decompressor")
        if decoding_plugin:
            raise NotImplementedError(
                "decoding_plugin is not supported by the native decompressor")
        if _kw:
            raise NotImplementedError(
                "decoder options are not supported by the native decompressor: "
                + ", ".join(sorted(_kw)))
        if "PixelData" not in self:
            raise AttributeError(
                "Unable to decompress as the dataset has no (7FE0,0010) 'Pixel Data' element")
        fm = self.file_meta
        ts = str(fm.get("TransferSyntaxUID") or "") if fm is not None else ""
        if not ts:
            raise AttributeError(
                "Unable to determine the initial compression state as there's no "
                "(0002,0010) 'Transfer Syntax UID' element in file_meta")
        from .uid import UID
        if not UID(ts).is_compressed:
            raise ValueError("The dataset is already uncompressed")
        pi = str(self.get("PhotometricInterpretation", "") or "").upper()
        if not as_rgb and pi.startswith("YBR"):
            raise NotImplementedError(
                "as_rgb=False is not supported for compressed YBR Pixel Data")

        try:
            out = _native.transcode(self._encode_part10(), "1.2.840.10008.1.2.1")
        except Exception as exc:
            raise RuntimeError(f"Unable to decompress Pixel Data [{exc}]") from exc

        import os, tempfile, atexit
        fd, tmp = tempfile.mkstemp(suffix=".dcm", prefix="pydcm_dec_")
        try:
            os.write(fd, out)
        finally:
            os.close(fd)
        atexit.register(lambda p=tmp: os.path.exists(p) and os.unlink(p))
        new = dcmread(tmp)
        object.__setattr__(self, "_dict", new._dict)
        object.__setattr__(self, "_path", new._path)
        object.__setattr__(self, "_pixels", None)
        object.__setattr__(self, "_edits", {})
        object.__setattr__(self, "_source_bytes", new._source_bytes)
        object.__setattr__(self, "file_meta", new.file_meta)
        if generate_instance_uid:
            from .uid import generate_uid
            instance_uid = generate_uid()
            self.SOPInstanceUID = instance_uid
            self.file_meta.MediaStorageSOPInstanceUID = instance_uid

    # -- authoring ----------------------------------------------------------- #
    def add_new(self, tag, VR, value) -> None:
        """Add a :class:`DataElement` ``(tag, VR, value)``."""
        t = self._as_tag(tag)
        info = _native.describe_tag(int(t))
        self._dict[t] = DataElement(t, VR, value,
                                    keyword=info and info.get("keyword"),
                                    name=info and info.get("name"))
        self._edits[t] = ("insert", value, VR)

    def add(self, elem: "DataElement") -> None:
        """Add an existing :class:`DataElement`."""
        self[elem.tag] = elem

    # -- serialisation ------------------------------------------------------- #
    def to_json_dict(self) -> dict:
        """The DICOM JSON Model (PS3.18 §F) of this dataset."""
        return {f"{e.tag.group:04X}{e.tag.element:04X}": _encode(e) for e in self}

    def to_json(self, indent=None) -> str:
        return json.dumps(self.to_json_dict(), indent=indent, ensure_ascii=False)

    @classmethod
    def from_json(cls, json_dataset, bulk_data_uri_handler=None) -> "Dataset":
        """Build a :class:`Dataset` from the DICOM JSON Model.

        ``json_dataset`` may be a JSON string or an already-parsed mapping. Reuses the
        same element builder as :func:`dcmread` — no separate JSON parser.
        """
        jmap = json.loads(json_dataset) if isinstance(json_dataset, (str, bytes, bytearray)) else json_dataset
        return _build(jmap)

    def _edit_ops(self):
        """Tracked edits → (tag, kind, value, vr_code) ops for edit_part10. A binary
        (OB/OW/UN/…) value is passed as raw ``bytes`` (the native editor writes it
        verbatim); a text/numeric value as its backslash-joined string."""
        ops = []
        for tag, (kind, value, vr) in self._edits.items():
            if kind == "erase":
                ops.append((int(tag), "erase", b"", 0))
                continue
            vr_code = (ord(vr[0]) | (ord(vr[1]) << 8)) if vr and len(vr) == 2 else 0
            if isinstance(value, (bytes, bytearray)):
                ops.append((int(tag), "insert", bytes(value), vr_code))   # raw binary
                continue
            vs = _edit_value_str(value)
            if vs is None:           # unencodable value — can't patch via op
                continue
            ops.append((int(tag), "insert", vs, vr_code))
        return ops

    def _has_nested_edits(self) -> bool:
        """True if any element inside a sequence item was modified/added/erased — the
        byte-verbatim top-level editor can't express these, so save_as routes through the
        full-model JSON path instead."""
        for e in self._dict.values():
            if e.VR == "SQ" and isinstance(e._value, list):
                for item in e._value:
                    if isinstance(item, Dataset) and (item._edits or item._has_nested_edits()):
                        return True
        return False

    def _encode_part10(self, target_ts: str = "") -> bytes:
        """The Part-10 bytes for this dataset.

        - Top-level edits on a Little-Endian source → byte-verbatim editor (keeps the
          Transfer Syntax, compressed/encapsulated PixelData, sequences, private tags).
        - Nested sequence-item edits → full-model JSON path (the in-memory model already
          reflects them; PixelData rides along as InlineBinary).
        - Big-Endian / Deflated source, or a nested edit on an encapsulated source → the
          JSON path can't preserve the pixels, so for a file WITH image pixels this raises
          rather than silently drop them. Metadata-only datasets re-serialise safely.

        ``target_ts`` (Explicit VR LE) recodes the from-scratch JSON build, which is
        otherwise Implicit VR LE; it does not affect the byte-verbatim editor path
        (that keeps the source Transfer Syntax).
        """
        ts = str(self.file_meta.get("TransferSyntaxUID") or "") if self.file_meta is not None else ""
        be_or_deflated = ts in ("1.2.840.10008.1.2.2", "1.2.840.10008.1.2.1.99")
        encapsulated = ts.startswith("1.2.840.10008.1.2.4") or ts == "1.2.840.10008.1.2.5"
        # Has image pixels? has_pixel_data can't parse BE, so also accept the Rows/Columns
        # signal — the conservative (no-silent-loss) choice when we can't be sure.
        has_pixels = (Tag(0x7FE00010) in self._dict
                      or (self.get("Rows") and self.get("Columns")))
        nested = self._has_nested_edits()

        # A requested uncompressed-VR-LE Transfer-Syntax CHANGE on an uncompressed
        # source re-encodes through the MODEL path (to_json → EVR/IVR), so every VR our
        # dictionary resolved on read — INCLUDING private tags — survives (the native
        # re-encodes on save too). Compare against the ON-DISK TS, not self.file_meta
        # (the caller may have just changed it to request the switch). Same-TS or a
        # compressed source falls through to the byte-verbatim editor below, which keeps
        # the source Transfer Syntax and compressed PixelData exactly.
        recode = False
        if target_ts in ("1.2.840.10008.1.2", "1.2.840.10008.1.2.1"):
            disk_ts = ""
            if self._path is not None:
                try:
                    disk_ts = str(_build_file_meta(self._path).get("TransferSyntaxUID") or "")
                except Exception:
                    disk_ts = ""
            elif self._source_bytes is not None:
                disk_ts = ts          # received data set: its file_meta TS is the source TS
            recode = disk_ts in ("1.2.840.10008.1.2", "1.2.840.10008.1.2.1") and target_ts != disk_ts

        # Byte-verbatim editor: patch edits onto the original Part-10 bytes — the backing file,
        # or the in-memory source kept for a received DIMSE data set (no temp file). Preserves the
        # exact Transfer Syntax and PixelData (incl. compressed), which the model path cannot.
        original = None
        if self._path is not None:
            try:
                with open(self._path, "rb") as f:
                    original = f.read()
            except Exception:
                original = None       # unexpected (malformed) — fall through to the model path
        elif self._source_bytes is not None:
            original = self._source_bytes
        if original is not None and not be_or_deflated and not nested and not recode:
            try:
                return _native.edit_part10(original, self._edit_ops())
            except Exception:
                pass                  # unexpected — try the model path below

        if has_pixels and (be_or_deflated or (nested and encapsulated)):
            why = ("a Big-Endian / Deflated file" if be_or_deflated
                   else "a nested sequence-item edit on a compressed file")
            raise NotImplementedError(
                f"save_as cannot rewrite {why} without losing/recompressing its PixelData "
                f"[{ts or 'unknown TS'}]; transcode to Explicit VR Little Endian (or "
                "decompress()) first. The byte-verbatim editor handles every LE syntax.")
        return _native.write_part10(self.to_json(), target_ts)

    def _requested_ts(self, implicit_vr, little_endian) -> str:
        """The from-scratch output Transfer Syntax: the explicit ``implicit_vr`` /
        ``little_endian`` save_as args win, else the dataset's file_meta
        TransferSyntaxUID, which is passed through UNCHANGED.

        This used to whitelist the two uncompressed VR-LE UIDs and silently return
        ``""`` (= write Implicit VR LE) for everything else, so asking for J2K or
        MPEG-2 produced a file whose (0002,0010) contradicted its own bytes. The
        native writer now takes the UID itself and rejects what it cannot produce,
        so the honest answer is to hand the caller's choice down and let it speak.
        """
        if implicit_vr is not None:
            # PS3.5: Explicit VR Big Endian is the only big-endian syntax, and the
            # writer does not emit it. Name it so the error says what was asked for
            # rather than falling back to something else.
            if little_endian is False:
                return "1.2.840.10008.1.2.2"
            return "1.2.840.10008.1.2" if implicit_vr else "1.2.840.10008.1.2.1"
        ts = str(self.file_meta.get("TransferSyntaxUID") or "") if self.file_meta is not None else ""
        # No file_meta at all ⇒ unspecified; the DICOM default is Implicit VR LE.
        return ts or "1.2.840.10008.1.2"

    def save_as(self, filename, /, __write_like_original=None, *,
                implicit_vr=None, little_endian=None, enforce_file_format=False,
                overwrite=True, **kwargs) -> None:
        """Write this dataset as a Part-10 file.

        ``filename`` may be a path or a writable binary file-like object (e.g. an open
        file or ``io.BytesIO``). ``overwrite=False`` raises :class:`FileExistsError` if
        the target exists. When the dataset was read from a file, edits are patched onto
        the ORIGINAL bytes byte-verbatim (Transfer Syntax, PixelData, sequences, private
        tags all survive) via the native editor; a from-scratch dataset is serialised
        from the metadata model — Explicit VR LE when the file_meta TransferSyntaxUID (or
        ``implicit_vr=False``) asks for it, otherwise Implicit VR LE. ``implicit_vr`` /
        ``little_endian`` select the from-scratch VR form (Explicit VR Big Endian is not
        emitted); ``write_like_original`` / ``enforce_file_format`` are accepted for
        signature compatibility.
        """
        if __write_like_original is None and "write_like_original" in kwargs:
            __write_like_original = kwargs.pop("write_like_original")
        data = self._encode_part10(self._requested_ts(implicit_vr, little_endian))
        if hasattr(filename, "write"):                # file-like (BytesIO / open file)
            filename.write(data)
            return
        import os
        target = os.fspath(filename)
        if not overwrite and os.path.exists(target):
            raise FileExistsError(f"[Errno 17] File exists: {target!r}")
        with open(target, "wb") as f:
            f.write(data)

    # -- display (pretty-printed element tree) -------------------------------- #
    def _lines(self, indent=0, top_level_only=False):
        """Yield one formatted line per element,:
        3-space indent per level, sequences shown with a ``N item(s) ----`` header
        and each item closed by a ``---------`` separator."""
        indent_str = self.indent_chars * indent
        nextindent_str = self.indent_chars * (indent + 1)
        for elem in self:
            if elem.VR == "SQ":
                yield (f"{indent_str}{elem.tag}  {elem.name}  "
                       f"{len(elem.value or [])} item(s) ---- ")
                if not top_level_only:
                    for item in (elem.value or []):
                        yield from item._lines(indent + 1)
                        yield nextindent_str + "---------"
            else:
                yield indent_str + repr(elem)

    def _pretty_str(self, indent=0, top_level_only=False) -> str:
        """File-meta section (when present + config.show_file_meta)
        followed by the indented element lines."""
        from . import config
        strings = []
        fm = getattr(self, "file_meta", None)
        if fm is not None and len(fm) and getattr(config, "show_file_meta", True):
            strings.append(f"{'Dataset.file_meta ':-<49}")
            strings.extend(self.indent_chars * indent + repr(e) for e in fm)
            strings.append(f"{'':-<49}")
        strings.extend(self._lines(indent, top_level_only))
        return "\n".join(strings)

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        return self._pretty_str() or "<empty Dataset>"


class FileMetaDataset(Dataset):
    """The group-0002 File Meta Information (ds.file_meta), a thin Dataset."""

    __slots__ = ()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.validate(self)

    @staticmethod
    def validate(init_value) -> None:
        return _validate_file_meta_dataset(init_value)


def _validate_file_meta_dataset(init_value) -> None:
    bad = []
    if isinstance(init_value, Dataset):
        keys = init_value._dict.keys()
    elif hasattr(init_value, "keys"):
        keys = init_value.keys()
    else:
        keys = dict(init_value).keys()
    for key in keys:
        try:
            tag = Dataset._as_tag(key)
        except (TypeError, ValueError):
            continue
        if tag.group != 0x0002:
            bad.append(str(tag))
    if bad:
        raise ValueError(
            "File meta datasets may only contain group 2 elements but the "
            f"following elements are present: {', '.join(bad)}")
    return None


class FileDataset(Dataset):
    """A :class:`Dataset` bound to a file, with ``file_meta`` + preamble — the
    the authoring class. ``FileDataset(filename, ds, file_meta=fm)`` then
    ``.save_as(path)`` writes a Part-10 file (reusing the same write path as
    :meth:`Dataset.save_as`; the dataset must carry SOPClassUID + SOPInstanceUID)."""

    __slots__ = ("filename", "preamble", "_is_implicit_VR", "_is_little_endian")

    def __init__(self, filename_or_obj=None, dataset=None, preamble=None,
                 file_meta=None, is_implicit_VR=True, is_little_endian=True):
        elements = dataset._dict if isinstance(dataset, Dataset) else dict(dataset or {})
        super().__init__(elements)
        object.__setattr__(self, "filename",
                           filename_or_obj if isinstance(filename_or_obj, str) else None)
        object.__setattr__(self, "preamble", preamble if preamble is not None else b"\x00" * 128)
        object.__setattr__(self, "_is_implicit_VR", is_implicit_VR)
        object.__setattr__(self, "_is_little_endian", is_little_endian)
        object.__setattr__(self, "file_meta",
                           file_meta if file_meta is not None else FileMetaDataset())

    # Lazy: dcmread leaves these unset (None) so the TransferSyntaxUID lookup —
    # which materialises file_meta — happens only if the caller actually asks.
    @property
    def is_implicit_VR(self) -> bool:
        v = self._is_implicit_VR
        if v is None:
            fm = self.file_meta
            ts = fm.get("TransferSyntaxUID") if fm is not None else None
            v = (str(ts) == "1.2.840.10008.1.2")
            object.__setattr__(self, "_is_implicit_VR", v)
        return v

    @is_implicit_VR.setter
    def is_implicit_VR(self, value) -> None:
        object.__setattr__(self, "_is_implicit_VR", value)

    @property
    def is_little_endian(self) -> bool:
        v = self._is_little_endian
        if v is None:
            fm = self.file_meta
            ts = fm.get("TransferSyntaxUID") if fm is not None else None
            v = (str(ts) != "1.2.840.10008.1.2.2")
            object.__setattr__(self, "_is_little_endian", v)
        return v

    @is_little_endian.setter
    def is_little_endian(self, value) -> None:
        object.__setattr__(self, "_is_little_endian", value)

    @property
    def timestamp(self):
        """The backing file's mtime (epoch seconds), or ``None``."""
        import os
        if self.filename is None or not os.path.exists(self.filename):
            return None
        return os.stat(self.filename).st_mtime


# Single-valued US-or-SS tags whose VR follows (0028,0103) PixelRepresentation:
# Smallest/Largest Image (Plane / in-Series) Pixel Value + Pixel Padding Value/Range Limit.
_US_OR_SS = (0x00280106, 0x00280107, 0x00280108, 0x00280109,
             0x00280110, 0x00280111, 0x00280120, 0x00280121)


def _build_file_meta(path: str) -> FileMetaDataset:
    """Full group-0002 File Meta Information, built with the SAME element builder the
    dataset uses (``_build`` over ``read_meta_json``) — so ds.file_meta exposes every
    (0002,xxxx), not just the 3 mandatory UIDs. Naked files (no group-2) fall back to
    the sniffed UIDs from ``read_file_meta``."""
    fm = FileMetaDataset()
    mj = _native.read_meta_json(path)
    if mj:
        object.__setattr__(fm, "_dict", dict(_build(json.loads(mj))._dict))
        return fm
    meta = _native.read_file_meta(path)              # naked / no real group-2
    for tag, key in ((0x00020002, "sop_class"),
                     (0x00020003, "sop_instance"),
                     (0x00020010, "transfer_syntax")):
        val = (meta or {}).get(key)
        if val:
            fm._dict[Tag(tag)] = DataElement(Tag(tag), "UI", _scalar("UI", val))
    return fm


def _spool(data: bytes) -> str:
    """Write in-memory DICOM bytes to a temp file (the native reader is path-based)."""
    import tempfile
    fd, name = tempfile.mkstemp(suffix=".dcm", prefix="pydcm_")
    import os
    try:
        os.write(fd, data)
    finally:
        os.close(fd)
    return name


def dcmread(fp, defer_size=None, stop_before_pixels=False, force=False,
            specific_tags=None, *, charset_override: str = "", **kwargs) -> Dataset:
    """Read a DICOM Part-10 file into a :class:`Dataset`.

    ``fp`` may be a path, an ``os.PathLike``, raw ``bytes``, or a readable binary
    file-like object (e.g. ``io.BytesIO``). ``charset_override``
    forces a SpecificCharacterSet when a file omits/misdeclares one. ``stop_before_pixels``
    / ``defer_size`` / ``specific_tags`` are accepted for signature compatibility (pixels
    are always lazy here). A non-DICOM input raises :class:`~pydcm.errors.InvalidDicomError`
    unless ``force=True``.
    """
    import os
    from .errors import InvalidDicomError

    tmp = None
    src_bytes = None                              # in-memory source, kept on the Dataset
    if hasattr(fp, "read"):                       # file-like (BytesIO, open file, …)
        src_bytes = fp.read()
        tmp = _spool(src_bytes)
        p = tmp
    elif isinstance(fp, (bytes, bytearray)):
        src_bytes = bytes(fp)
        tmp = _spool(src_bytes)
        p = tmp
    else:
        p = os.fspath(fp)

    try:
        ds = _build(json.loads(_native.read_json(p, charset_override)), path=p)
        if tmp is not None:
            # The spooled temp file is unlinked on return — materialise
            # file_meta now; real paths defer to the lazy property.
            ds.file_meta = _build_file_meta(p)
    except InvalidDicomError:
        raise
    except Exception as exc:
        if force:
            ds = _build({}, path=p)
            ds.file_meta = FileMetaDataset()
        else:
            raise InvalidDicomError(
                f"{getattr(fp, 'name', p)!r} is not a valid DICOM file; "
                f"use force=True to read it anyway ({exc})") from exc
    finally:
        if tmp is not None:
            try:
                os.unlink(tmp)
            except OSError:
                pass

    # US-or-SS tags: their VR is resolved purely by (0028,0103) PixelRepresentation,
    # regardless of how the bytes were typed on the wire —
    # 1 → signed (SS), 0 → unsigned (US). Reinterpret both directions on the 16-bit value.
    pr = ds.get("PixelRepresentation")
    if pr in (0, 1):
        for tag in _US_OR_SS:
            e = ds._dict.get(Tag(tag))
            if e is not None and isinstance(e.value, int):
                v = e.value & 0xFFFF
                if pr == 1:
                    e._value = v - 0x10000 if v >= 0x8000 else v
                    e.VR = "SS"
                else:
                    e._value = v
                    e.VR = "US"

    # Lazy PixelData (7FE0,0010) is built directly from read_json's value-less VR stub
    # (present in the mapping protocol — 'PixelData' in ds, len, iter, keys, ds[tag] —
    # without loading the bytes). For in-memory (tmp) input the spooled file is already
    # deleted, so keep the SOURCE BYTES on the dataset instead and point the stub's lazy
    # loader at them — the same _source_bytes a received DIMSE data set uses, which also
    # gives BytesIO-read datasets the byte-verbatim save_as path. These stubs used to be
    # dropped here, so a dataset read from BytesIO simply had no PixelData at all.
    if tmp is not None:
        # The spool path is unlinked — a dataset still pointing at it would fail
        # every _path-guarded fallback later. The bytes are the source now.
        object.__setattr__(ds, "_path", None)
        object.__setattr__(ds, "_source_bytes", src_bytes)
        for _pt in _PIXEL_DATA_TAGS:
            e = ds._dict.get(Tag(_pt))
            if e is not None:
                e._lazy = (lambda b=src_bytes: _native.read_pixel_data_bytes(b))

    # Return a FileDataset — gives .filename/.preamble/.timestamp
    # and is_implicit_VR/is_little_endian, while preserving _path/_edits/_pixels so the
    # byte-verbatim save_as path keeps working. Use __new__ to avoid re-copying _dict.
    fds = FileDataset.__new__(FileDataset)
    object.__setattr__(fds, "_dict", ds._dict)
    object.__setattr__(fds, "_path", None if tmp is not None else ds._path)
    object.__setattr__(fds, "_pixels", ds._pixels)
    # Raw backing slot, NOT the property: keeps file_meta lazy for real paths
    # (the spooled-tmp case stored it eagerly above).
    object.__setattr__(fds, "_file_meta", ds._file_meta)
    object.__setattr__(fds, "_edits", ds._edits)
    object.__setattr__(fds, "_source_bytes", ds._source_bytes)
    object.__setattr__(fds, "filename", None if tmp is not None else os.fspath(p))
    object.__setattr__(fds, "preamble", b"\x00" * 128)
    # None = derive from file_meta on first access (lazy properties above).
    object.__setattr__(fds, "_is_implicit_VR", None)
    object.__setattr__(fds, "_is_little_endian", None)
    return fds


def dcmwrite(filename, dataset: Dataset, /, __write_like_original=None, *,
             implicit_vr=None, little_endian=None, enforce_file_format=False,
             force_encoding=False, overwrite=True, **kwargs) -> None:
    """Write ``dataset`` to ``path`` as Part-10 (name)."""
    dataset.save_as(filename, __write_like_original, implicit_vr=implicit_vr,
                    little_endian=little_endian,
                    enforce_file_format=enforce_file_format,
                    overwrite=overwrite, **kwargs)


# ── VOI / Modality LUT (PS3.3 C.11.1-2) ────────────────────────────────────────────────────────────
# pydicom-style spellings of the two transfer stages, so code that reaches for
# `pydicom.pixel_data_handlers.util.apply_{modality,voi}_lut` has a native equivalent here. Neither
# reimplements anything: the per-pixel math is the native engine's header-only VOI transfer, bound as
# _native.voi_apply, i.e. the SAME inline the native render path uses — one implementation, so a
# Python caller and a rendered frame can never disagree about what a window or a VOI LUT means.

def apply_modality_lut(arr, ds):
    """Stored pixel values → MODALITY values (PS3.3 C.11.1).

    Modality LUT Sequence when present, else Rescale Slope/Intercept, else the array unchanged.
    Mirrors ``pydicom.pixel_data_handlers.util.apply_modality_lut``.
    """
    import numpy as np
    seq = getattr(ds, "ModalityLUTSequence", None)
    if seq:
        item = seq[0]
        n, first, depth = (int(x) for x in item.LUTDescriptor[:3])
        data = item.LUTData
        lut = (np.frombuffer(data, dtype=np.uint8 if depth <= 8 else np.uint16)
               if isinstance(data, (bytes, bytearray)) else np.asarray(data))
        idx = np.clip(np.asarray(arr, dtype=np.int64) - first, 0, max(len(lut) - 1, 0))
        return lut[idx]
    slope = getattr(ds, "RescaleSlope", None)
    inter = getattr(ds, "RescaleIntercept", None)
    if slope is None and inter is None:
        return arr
    return np.asarray(arr, dtype=np.float64) * float(slope if slope is not None else 1.0) \
        + float(inter if inter is not None else 0.0)


def apply_voi_lut(arr, ds, index: int = 0):
    """Modality values → PRESENTATION values (PS3.3 C.11.2).

    A VOI LUT Sequence supersedes Window Center/Width when present (C.11.2), matching both the
    standard and ``pydicom.pixel_data_handlers.util.apply_voi_lut``. The return is on the LUT's own
    output range (0 … 2**bits−1) for the LUT branch and on the window's [0,1] range scaled to the
    input's own span for the window branch — the same convention pydicom uses, so a caller that
    scales by ``2**(16 - bits)`` (what dcmtk's ``dcmj2pnm --use-voi-lut`` writes) keeps working.

    `index` picks the item of a multi-item sequence (GE mammography ships NORMAL/HARDER/SOFTER).
    """
    import numpy as np
    a = np.ascontiguousarray(arr, dtype=np.float32)

    seq = getattr(ds, "VOILUTSequence", None)
    if seq:
        item = seq[min(index, len(seq) - 1)]
        n, first, depth = (int(x) for x in item.LUTDescriptor[:3])
        data = item.LUTData
        if not isinstance(data, (bytes, bytearray)):
            data = np.asarray(data, dtype=np.uint8 if depth <= 8 else np.uint16).tobytes()
        out = _native.voi_apply(a, 0.0, 0.0, 0, bytes(data), first, depth)
        # voi_apply normalises by the descriptor's output range (C.11.2.1.1); pydicom returns the
        # raw table entry, so scale back. Exact for depth ≤ 24.
        maxv = float((1 << depth) - 1)
        return np.rint(np.asarray(out, dtype=np.float64) * maxv)

    wc, ww = getattr(ds, "WindowCenter", None), getattr(ds, "WindowWidth", None)
    if wc is None or ww is None:
        return arr
    if isinstance(wc, (list, tuple)): wc = wc[min(index, len(wc) - 1)]
    if isinstance(ww, (list, tuple)): ww = ww[min(index, len(ww) - 1)]
    fn = {"LINEAR": 0, "LINEAR_EXACT": 1, "SIGMOID": 2}.get(
        str(getattr(ds, "VOILUTFunction", "LINEAR") or "LINEAR").upper(), 0)
    u = _native.voi_apply(a, float(wc), float(ww), fn, None, 0, 16)
    lo, hi = float(np.min(a)), float(np.max(a))
    return np.asarray(u, dtype=np.float64) * (hi - lo) + lo
