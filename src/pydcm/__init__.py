# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Fuli Wu
# NOT a medical device — not for clinical or diagnostic use.

"""pydcm — decode DICOM pixels for NumPy / PyTorch.

A compiled native extension decodes EVERY transfer syntax (JPEG / JPEG-2000 /
HTJ2K / JPEG-LS / JPEG-XL / RLE) to native integer pixels — or Hounsfield units — with no
separate codec plugins to install, and assembles a directory of slices into a 3D volume.

    import pydcm
    arr = pydcm.decode("scan.dcm")             # ndarray [frames, rows, cols(, samples)]
    hu  = pydcm.decode("ct.dcm", rescale=True)  # float32 Hounsfield units
    vol = pydcm.load_series("ct_dir/")          # spatially-ordered 3D HU volume

    from torch.utils.data import DataLoader
    ds = pydcm.DICOMDataset("study_dir/", to_torch=True)
    for batch in DataLoader(ds, batch_size=8, num_workers=4):
        ...

NOT a medical device — not for clinical or diagnostic use; research/engineering only.
"""

from __future__ import annotations

from . import _native
from .torchdata import DICOMDataset, scan
from .volume import Volume, Volume4D, Axis, load_series, load_4d, from_nifti, bids_sidecar
from .diffusion import diffusion_table, load_dwi, save_dwi
from ._dicom import (
    Dataset, DataElement, Tag, BaseTag, PersonName, Sequence, MultiValue,
    DSfloat, IS, FileMetaDataset, FileDataset, dcmread, dcmwrite,
)
# submodules — make `from pydcm.<mod> import X` (i.e. ported
# `from pydicom.<mod> import X`) resolve to the same objects.
from . import (uid, valuerep, config, errors, dataset, dataelem, tag, sequence,
               multival, filereader, filewriter, datadict, pixels, encaps,
               fileset, pixel_data_handlers, charset, values, jsonrep, data,
               examples, waveforms, util, misc, hooks, fileutil, filebase,
               overlays, encoders, dicomio, env_info, sr, paramap, ko, pr,
               legacy_converted, ann, opv, dce)
from .uid import generate_uid, UID
from .errors import InvalidDicomError
from .fileset import FileSet, FileInstance
from .pixels import pixel_array, iter_pixels      # pydicom 3.x top-level re-exports
from .radiomics import radiomics                      # IBSI features (+ )
from .content import content                           # unified structured-object reader
from .deident import deidentify, deidentify_series, clean_pixel_data  # PS3.15 Annex E de-id + burned-in pixel blackout (native engine)
from .seg import (write_seg, write_seg_fractional, read_seg,  # SEG authoring + reading
                  write_seg_from_prediction)                  # model prediction -> original grid -> SEG
from .sr import (write_sr, write_report, read_report, sr_to_html,   # SR authoring (generic + TID 1500) + HTML render
                 write_measurement_report, read_measurement_report,  # typed TID 1500 templates
                 sr_code_meaning, sr_validate_code,             # + code lookup
                 sr_cid_has, sr_validate)                       # + content-tree validation
from .paramap import write_paramap, read_paramap                # Parametric Map author + read
from .ko import write_ko, read_ko                               # Key Object Selection author + read
from .pr import write_pr, read_pr                               # Presentation State (GSPS) author + read
from .legacy_converted import write_legacy_converted            # Legacy Converted Enhanced CT/MR/PET author
from .ann import read_ann                                       # Microscopy Bulk Simple Annotations reader
from .opv import (OPVDicom, OPVDicomSet, read_visual_field)      # Ophthalmic Visual Field (Supplement 146 static perimetry)
from ._core import iod_validate                                 # IOD/module Type-1/2 conformance
from .rt import (read_rtdose, DoseGrid,                         # RT line (native engine): dose grid,
                 write_rtdose, dvhcalc, ComputedDVH)            # dose authoring + DVH compute
from .encapdoc import (write_encapsulated, read_encapsulated,   # Encapsulated Documents (PDF/CDA/
                       EncapsulatedDocument)                    # STL/OBJ/MTL)
from . import transforms                                        # deterministic transforms (the native "ITK")

__all__ = ["decode", "dcmread", "dcmwrite", "Dataset", "DataElement", "Tag",
           "BaseTag", "PersonName", "Sequence", "MultiValue", "DSfloat", "IS",
           "FileMetaDataset", "FileDataset", "FileSet", "FileInstance",
           "InvalidDicomError",
           "DICOMDataset", "scan", "load_series", "load_4d", "from_nifti", "bids_sidecar", "diffusion_table", "load_dwi", "save_dwi", "Volume", "Volume4D", "Axis",
           "uid", "valuerep", "config", "errors", "dataset", "dataelem", "tag",
           "sequence", "multival", "filereader", "filewriter", "datadict",
           "pixels", "encaps", "pixel_data_handlers", "charset", "values",
           "jsonrep", "data", "examples", "waveforms",
           "overlays", "encoders", "dicomio", "env_info", "sr", "paramap", "ko", "pr", "dce",
           "legacy_converted", "ann", "rt", "read_rtdose", "DoseGrid", "write_rtdose",
           "dvhcalc", "ComputedDVH", "transforms",
           "encapdoc", "write_encapsulated", "read_encapsulated", "EncapsulatedDocument",
           "pixel_array", "iter_pixels", "generate_uid", "UID",
           "content", "deidentify", "deidentify_series", "clean_pixel_data",
           "radiomics", "write_seg", "write_seg_fractional", "read_seg",
           "write_seg_from_prediction", "write_sr",
           "write_report", "read_report", "write_measurement_report", "read_measurement_report",
           "write_paramap", "read_paramap", "write_ko", "read_ko", "write_pr", "read_pr",
           "write_legacy_converted", "read_ann",
           "opv", "OPVDicom", "OPVDicomSet", "read_visual_field", "iod_validate",
           "sr_code_meaning", "sr_validate_code", "sr_cid_has", "sr_validate", "sr_to_html",
           "__version__", "__disclaimer__"]
__version__ = "0.3.0"
__disclaimer__ = ("pydcm is not a medical device and is not for clinical or "
                  "diagnostic use; outputs are for research/engineering only.")


def decode(path, frame: int = 0, *, rescale: bool = False,
           to_torch: bool = False, with_meta: bool = False):
    """Decode a DICOM file's pixels to an array.

    Parameters
    ----------
    path : str | os.PathLike
        A Part-10 DICOM file (any transfer syntax).
    frame : int
        1-based frame to extract; ``0`` (default) returns all frames.
    rescale : bool
        ``True`` → real-world values (**HU** for CT) as float32 (per-frame rescale
        applied); ``False`` (default) → native **stored** integers (lossless).
    to_torch : bool
        Return a ``torch.Tensor`` instead of a NumPy array.
    with_meta : bool
        Also return the geometry sidecar (rescale_slope/intercept, pixel_spacing,
        image_position/orientation_patient, slice_thickness, window_center/width,
        modality, *_instance_uid, …).

    Returns
    -------
    ndarray (or Tensor), shape ``[frames, rows, cols(, samples)]`` — or
    ``(array, meta)`` when ``with_meta=True``.
    """
    arr, meta = _native.decode(path, frame, rescale)
    if to_torch:
        import torch
        arr = torch.from_numpy(arr.copy())
    return (arr, meta) if with_meta else arr
