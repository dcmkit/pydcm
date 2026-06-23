# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Fuli Wu
# NOT a medical device — not for clinical or diagnostic use.

"""Secondary Capture images (`pydcm.sc`) — write an ndarray as an SC DICOM object.

Functional API ``write_sc`` plus an ``SCImage`` constructor
(returns a :class:`pydcm.Dataset` built over the native ``set_pixel_data``)."""

from __future__ import annotations

from . import dataset as _ds_mod
from .dataset import Dataset, FileMetaDataset
from .uid import generate_uid, ExplicitVRLittleEndian
from .pixels import set_pixel_data

_SC_STORAGE = "1.2.840.10008.5.1.4.1.1.7"     # Secondary Capture Image Storage


def write_sc(pixel_array, photometric_interpretation="MONOCHROME2", *, bits_stored=None,
             study_instance_uid=None, series_instance_uid=None, sop_instance_uid=None,
             series_number=1, instance_number=1, manufacturer="pydcm",
             patient_id="", patient_name="", patient_birth_date="", patient_sex="",
             accession_number="", study_id="", study_date="", study_time="",
             referring_physician_name="", conversion_type="WSD",
             pixel_spacing=None, output=None):
    """Author a Secondary Capture image from `pixel_array`.

    `pixel_array` is uint8/uint16 shaped (rows, cols), (frames, rows, cols),
    (rows, cols, 3) or (frames, rows, cols, 3). Returns a :class:`Dataset` (or writes
    to `output` and returns ``None``).
    """
    ds = Dataset()
    ds.file_meta = FileMetaDataset()
    ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds.file_meta.MediaStorageSOPClassUID = _SC_STORAGE
    ds.SOPClassUID = _SC_STORAGE
    # Patient / Study / Series / SC modules
    ds.PatientName = patient_name
    ds.PatientID = patient_id
    ds.PatientBirthDate = patient_birth_date
    ds.PatientSex = patient_sex
    ds.StudyInstanceUID = study_instance_uid or generate_uid()
    ds.SeriesInstanceUID = series_instance_uid or generate_uid()
    ds.StudyID = study_id
    ds.AccessionNumber = accession_number
    ds.StudyDate = study_date
    ds.StudyTime = study_time
    ds.ReferringPhysicianName = referring_physician_name
    ds.SeriesNumber = int(series_number)
    ds.InstanceNumber = int(instance_number)
    ds.Modality = "OT"                          # Other (SC default)
    ds.ConversionType = conversion_type
    ds.Manufacturer = manufacturer
    if pixel_spacing is not None:
        ds.PixelSpacing = [float(pixel_spacing[0]), float(pixel_spacing[1])]
    # Pixel data + Image Pixel module via the native-backed helper.
    if bits_stored is None:
        bits_stored = pixel_array.dtype.itemsize * 8
    sop_instance_uid = sop_instance_uid or generate_uid()
    set_pixel_data(ds, pixel_array, photometric_interpretation, bits_stored,
                   generate_instance_uid=False)
    ds.SOPInstanceUID = sop_instance_uid
    ds.file_meta.MediaStorageSOPInstanceUID = sop_instance_uid
    if output is not None:
        ds.save_as(output, enforce_file_format=True)
        return None
    return ds


def SCImage(pixel_array, photometric_interpretation, bits_allocated, coordinate_system,
            study_instance_uid, series_instance_uid, series_number, sop_instance_uid,
            instance_number, manufacturer, *, patient_id=None, patient_name=None,
            patient_birth_date=None, patient_sex=None, accession_number=None, study_id=None,
            study_date=None, study_time=None, referring_physician_name=None,
            pixel_spacing=None, **_kwargs) -> Dataset:
    """``SCImage`` constructor — returns a :class:`Dataset`.

    pydcm builds the object over the native ``set_pixel_data`` (a thin Dataset, not a
    bespoke class); ``coordinate_system`` and other extra kwargs are accepted
    for source compatibility.
    """
    return write_sc(
        pixel_array, str(photometric_interpretation), bits_stored=int(bits_allocated),
        study_instance_uid=study_instance_uid, series_instance_uid=series_instance_uid,
        sop_instance_uid=sop_instance_uid, series_number=series_number,
        instance_number=instance_number, manufacturer=manufacturer,
        patient_id=patient_id or "", patient_name=patient_name or "",
        patient_birth_date=patient_birth_date or "", patient_sex=str(patient_sex or ""),
        accession_number=accession_number or "", study_id=study_id or "",
        study_date=str(study_date or ""), study_time=str(study_time or ""),
        referring_physician_name=referring_physician_name or "", pixel_spacing=pixel_spacing)


__all__ = ["write_sc", "SCImage"]
