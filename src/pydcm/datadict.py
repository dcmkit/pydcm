# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Fuli Wu
# NOT a medical device — not for clinical or diagnostic use.
"""data-dictionary access (`pydcm.datadict`) over the native dict."""
from . import _native
from ._dicom import Tag


def _info(tag):
    return _native.describe_tag(int(Tag(tag)))


def dictionary_VR(tag):
    i = _info(tag); return i["vr"] if i else "UN"


def dictionary_VM(tag):
    i = _info(tag); return i["vm"] if i else "1"


def dictionary_description(tag):
    i = _info(tag); return i["name"] if i else ""


def dictionary_keyword(tag):
    i = _info(tag); return i["keyword"] if i else ""


def keyword_for_tag(tag):
    return dictionary_keyword(tag)


def tag_for_keyword(keyword):
    return _native.tag_for_keyword(keyword)


def dictionary_has_tag(tag):
    return _info(tag) is not None


def dictionary_is_retired(tag):
    i = _info(tag); return bool(i and i.get("retired"))


def get_entry(tag):
    """``(VR, VM, description, is_retired, keyword)`` tuple for ``tag``."""
    i = _info(tag)
    if not i:
        raise KeyError(f"Tag {Tag(tag)} not in DICOM dictionary")
    return (i["vr"], i["vm"], i["name"], "Retired" if i.get("retired") else "", i["keyword"])


def repeater_has_tag(tag):
    """True if ``tag`` matches a repeater group (60xx overlays, 50xx curves, 7Fxx)."""
    g = Tag(tag).group
    return (g & 0xFF00) in (0x5000, 0x6000) or g == 0x7FE0


def _private_info(creator, tag):
    t = Tag(tag)
    return _native.describe_private(creator, t.group, t.element & 0xFF)


def private_dictionary_VR(tag, private_creator):
    i = _private_info(private_creator, tag); return i["vr"] if i else "UN"


def private_dictionary_VM(tag, private_creator):
    i = _private_info(private_creator, tag); return i["vm"] if i else "1"


def private_dictionary_description(tag, private_creator):
    i = _private_info(private_creator, tag); return i["name"] if i else ""


# pydcm's dictionary is the native superset dict (read-only); these overlays accept
# runtime additions for source compatibility.
DicomDictionary: dict = {}
RepeatersDictionary: dict = {}


def add_dict_entries(new_entries: dict) -> None:
    """Add/override (tag → (VR, VM, name, is_retired, keyword)) entries."""
    DicomDictionary.update(new_entries)


def add_dict_entry(tag, VR, description, keyword, VM="1", is_retired="") -> None:
    DicomDictionary[int(Tag(tag))] = (VR, VM, description, is_retired, keyword)


# Runtime private-dictionary overlay. The authoritative private
# dict is the native union (12,608 patterns) queried by private_dictionary_*;
# this holds user-registered additions and is what ported code mutates/reads.
private_dictionaries: dict = {}


def add_private_dict_entries(private_creator, new_entries: dict) -> None:
    """Add ``{tag_pattern: (VR, VM, name, keyword)}`` private entries."""
    private_dictionaries.setdefault(private_creator, {}).update(new_entries)


def add_private_dict_entry(private_creator, tag, VR, description, VM="1") -> None:
    """Add a single private-dictionary entry."""
    key = f"{int(Tag(tag)):08X}"
    private_dictionaries.setdefault(private_creator, {})[key] = (VR, VM, description, "")


def get_private_entry(tag, private_creator):
    """Return ``(VR, VM, name, keyword)`` for a private ``tag`` under ``private_creator``.

    Checks the runtime overlay first, then the native private dictionary; raises
    ``KeyError`` if neither knows the tag/creator."""
    t = Tag(tag)
    block = private_dictionaries.get(private_creator)
    if block:
        gs, es = f"{t.group:04X}", f"{t.element:04X}"
        for k in (f"{gs}{es}", f"{gs}xx{es[-2:]}", f"{gs[:2]}xxxx{es[-2:]}"):
            if k in block:
                return block[k]
    i = _private_info(private_creator, tag)
    if i:
        return (i["vr"], i["vm"], i["name"], i.get("keyword", ""))
    raise KeyError(
        f"Tag '{t}' not in private dictionary for private creator '{private_creator}'")


# The DICOM repeaters-group element keywords (overlay 60xx / curve 50xx / variable-pixel).
REPEATER_KEYWORDS = {
    "AudioComments", "AudioSampleData", "AudioSampleFormat", "AudioType", "AxisLabels",
    "AxisUnits", "BitsForCodeWord", "CodeLabel", "CodeTableLocation", "CoefficientCoding",
    "CoefficientCodingPointers", "ColumnsForNthOrderCoefficients", "CoordinateStartValue",
    "CoordinateStepValue", "CurveActivationLayer", "CurveData", "CurveDataDescriptor",
    "CurveDescription", "CurveDimensions", "CurveLabel", "CurveRange",
    "CurveReferencedOverlayGroup", "CurveReferencedOverlaySequence", "DataValueRepresentation",
    "EscapeTriplet", "HuffmanTableSize", "HuffmanTableTriplet", "ImageDataLocation",
    "ImageFrameOrigin", "MaximumCoordinateValue", "MinimumCoordinateValue", "NumberOfChannels",
    "NumberOfFramesInOverlay", "NumberOfPoints", "NumberOfSamples", "NumberOfTables",
    "OverlayActivationLayer", "OverlayBitPosition", "OverlayBitsAllocated", "OverlayBitsForCodeWord",
    "OverlayBitsGrouped", "OverlayCodeLabel", "OverlayCodeTableLocation", "OverlayColumns",
    "OverlayComments", "OverlayCompressionCode", "OverlayCompressionDescription",
    "OverlayCompressionLabel", "OverlayCompressionOriginator", "OverlayCompressionStepPointers",
    "OverlayData", "OverlayDescription", "OverlayDescriptorBlue", "OverlayDescriptorGray",
    "OverlayDescriptorGreen", "OverlayDescriptorRed", "OverlayFormat", "OverlayLabel",
    "OverlayLocation", "OverlayNumberOfTables", "OverlayOrigin", "OverlayPlaneOrigin",
    "OverlayPlanes", "OverlayRepeatInterval", "OverlayRows", "OverlaySubtype", "OverlayType",
    "OverlaysBlue", "OverlaysGray", "OverlaysGreen", "OverlaysRed", "ROIArea", "ROIMean",
    "ROIStandardDeviation", "RowsForNthOrderCoefficients", "RunLengthTriplet", "SampleRate",
    "ShiftTableSize", "ShiftTableTriplet", "SourceImageIDs", "TotalTime", "TypeOfData",
    "VariableCoefficientsSDDN", "VariableCoefficientsSDHN", "VariableCoefficientsSDVN",
    "VariableNextDataGroup", "VariablePixelData", "ZonalMap",
}


def repeater_has_keyword(keyword) -> bool:
    """Return ``True`` if ``keyword`` is in the DICOM repeaters data dictionary."""
    return keyword in REPEATER_KEYWORDS


__all__ = ["dictionary_VR", "dictionary_VM", "dictionary_description",
           "dictionary_keyword", "keyword_for_tag", "tag_for_keyword",
           "dictionary_has_tag", "dictionary_is_retired", "get_entry",
           "repeater_has_tag", "repeater_has_keyword", "REPEATER_KEYWORDS",
           "private_dictionary_VR", "private_dictionary_VM", "get_private_entry",
           "add_private_dict_entry", "private_dictionaries",
           "private_dictionary_description", "DicomDictionary", "RepeatersDictionary",
           "add_dict_entries", "add_dict_entry", "add_private_dict_entries"]
