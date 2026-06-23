# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Fuli Wu
# NOT a medical device — not for clinical or diagnostic use.

"""DICOMDIR / File-set reading (``pydcm.FileSet``).

A File-set is a DICOMDIR plus the instance files it indexes. ``dcmread`` already
parses the DICOMDIR's ``DirectoryRecordSequence`` (PATIENT→STUDY→SERIES→IMAGE), so
this is a thin navigation layer over those records — it reuses ``dcmread`` for both
the DICOMDIR and each referenced instance; no separate DICOM parsing.

    fs = pydcm.FileSet("/media/DICOMDIR")
    for inst in fs:                     # FileInstance per leaf record
        ds = inst.load()               # -> a Dataset (reuses dcmread)
    for inst in fs.find(PatientID="1"):
        ...
"""

from __future__ import annotations

import os
import shutil

from . import _native
from ._dicom import Dataset, MultiValue, dcmread

_LEVELS = ("PATIENT", "STUDY", "SERIES")   # record-hierarchy order (reader view)


class FileInstance:
    """One referenced instance (a leaf DICOMDIR record).

    Attribute access falls back through the record hierarchy: the instance/IMAGE
    record first, then its SERIES, STUDY and PATIENT records — so ``inst.PatientID``,
    ``inst.StudyInstanceUID`` etc. work without loading the file."""

    __slots__ = ("_fs", "_record", "_ctx", "path")

    def __init__(self, fs: "FileSet", record: Dataset, ctx: dict, path: str):
        self._fs = fs
        self._record = record
        self._ctx = ctx                       # {"PATIENT": rec, "STUDY": rec, "SERIES": rec}
        self.path = path

    def load(self) -> Dataset:
        """Read and return the referenced :class:`Dataset` (reuses ``dcmread``)."""
        return dcmread(self.path)

    @property
    def SOPInstanceUID(self):
        return (self._record.get("ReferencedSOPInstanceUIDInFile")
                or self._record.get("SOPInstanceUID"))

    @property
    def SOPClassUID(self):
        return (self._record.get("ReferencedSOPClassUIDInFile")
                or self._record.get("SOPClassUID"))

    @property
    def FileID(self):
        fid = self._record.get("ReferencedFileID")
        return fid if isinstance(fid, (list, MultiValue)) else ([fid] if fid else [])

    def __getattr__(self, name):
        # record first, then SERIES/STUDY/PATIENT (the inherited record view)
        v = self._record.get(name)
        if v is not None:
            return v
        for lvl in reversed(_LEVELS):
            rec = self._ctx.get(lvl)
            if rec is not None:
                v = rec.get(name)
                if v is not None:
                    return v
        raise AttributeError(
            f"{type(self).__name__!r} (and its DICOMDIR records) has no {name!r}")

    def __repr__(self):
        return f"<FileInstance {self.SOPInstanceUID} @ {os.path.basename(self.path)}>"


class FileSet:
    """A DICOMDIR File-set — iterate :class:`FileInstance` leaves, ``find`` + ``load``.

    Construct from a DICOMDIR path or an already-read DICOMDIR :class:`Dataset`."""

    def __init__(self, ds_or_path=None):
        if ds_or_path is None:
            self._ds = Dataset()
            self._root = os.getcwd()
        elif isinstance(ds_or_path, Dataset):
            self._ds = ds_or_path
            self._root = os.path.dirname(getattr(ds_or_path, "_path", None) or "") or os.getcwd()
        else:
            p = os.fspath(ds_or_path)
            if os.path.isdir(p):
                p = os.path.join(p, "DICOMDIR")
            self._root = os.path.dirname(p) or os.getcwd()
            self._ds = dcmread(p)
        self._added = []          # (dataset, source_path|None) staged for write()
        self._instances = self._walk()

    # -- record-tree navigation (records are written depth-first per PS3.10) -- #
    def _walk(self):
        records = self._ds.get("DirectoryRecordSequence") or []
        out, ctx = [], {}
        for rec in records:
            rt = rec.get("DirectoryRecordType")
            if rt == "PATIENT":
                ctx = {"PATIENT": rec}
            elif rt == "STUDY":
                ctx = {k: ctx[k] for k in ("PATIENT",) if k in ctx}
                ctx["STUDY"] = rec
            elif rt == "SERIES":
                ctx = {k: ctx[k] for k in ("PATIENT", "STUDY") if k in ctx}
                ctx["SERIES"] = rec
            else:                                  # IMAGE / RT / SR … leaf with a file ref
                fid = rec.get("ReferencedFileID")
                if fid:
                    out.append(FileInstance(self, rec, dict(ctx), self._resolve(fid)))
        return out

    def _resolve(self, fid) -> str:
        parts = [str(x) for x in (fid if isinstance(fid, (list, MultiValue)) else [fid])]
        cand = os.path.join(self._root, *parts)
        if os.path.exists(cand):
            return cand
        # ISO-9660 media frequently uppercases file IDs; fall back case-insensitively.
        cur = self._root
        for part in parts:
            if not os.path.isdir(cur):
                return cand
            match = next((e for e in os.listdir(cur) if e.lower() == part.lower()), part)
            cur = os.path.join(cur, match)
        return cur

    # -- surface ------------------------------------------ #
    def __iter__(self):
        return iter(self._instances)

    def __len__(self):
        return len(self._instances)

    def find(self, load: bool = False, **filters):
        """Return instances matching ``filters`` (record attributes; with ``load=True``
        the referenced file is read and its values are matched too)."""
        hits = []
        for inst in self._instances:
            src = inst.load() if load else inst
            ok = True
            for key, want in filters.items():
                got = (src.get(key) if isinstance(src, Dataset) else getattr(src, key, None))
                if str(got) != str(want):
                    ok = False
                    break
            if ok:
                hits.append(inst)
        return hits

    def find_values(self, element, instances=None):
        """Distinct values of ``element`` across the (given or all) instances."""
        seen = []
        for inst in (instances if instances is not None else self._instances):
            v = getattr(inst, element, None)
            if v is not None and v not in seen:
                seen.append(v)
        return seen

    @property
    def path(self):
        return self._root

    # -- writing (synthesise a DICOMDIR) ------------------------------------- #
    def add(self, ds_or_path) -> None:
        """Stage an instance (a :class:`Dataset` or a file path) for :meth:`write`."""
        if isinstance(ds_or_path, Dataset):
            self._added.append((ds_or_path, getattr(ds_or_path, "_path", None)))
        else:
            p = os.fspath(ds_or_path)
            self._added.append((dcmread(p), p))

    def write(self, path, file_set_id="PYDCM_FILESET") -> str:
        """Write the staged instances + a conformant DICOMDIR under ``path``; returns
        the DICOMDIR path. Instances staged from a file are copied byte-verbatim.

        The DICOMDIR itself is built by the native engine: it reads each instance's
        key attributes,
        groups them PATIENT→STUDY→SERIES→leaf, and emits a conformant Explicit-VR-LE
        directory with correct inter-record byte offsets. This wrapper only does the
        filesystem side — assign each instance a media File ID, place the file, and
        write the returned DICOMDIR."""
        os.makedirs(path, exist_ok=True)
        inputs = []                                       # (bytes, media File ID)
        for i, (ds, src) in enumerate(self._added):
            fid = "IMG%05d" % (i + 1)
            dest = os.path.join(path, fid)
            if src and os.path.exists(src):
                shutil.copyfile(src, dest)
            else:
                ds.save_as(dest)
            with open(dest, "rb") as f:
                inputs.append((f.read(), fid))
        data = _native.build_dicomdir(inputs, file_set_id)
        out = os.path.join(path, "DICOMDIR")
        with open(out, "wb") as f:
            f.write(data)
        self._root = path                                 # re-read our own output
        self._ds = dcmread(out)
        self._instances = self._walk()
        return out

    def __repr__(self):
        return f"<FileSet: {len(self._instances)} instances under {self._root}>"

__all__ = ['FileSet', 'FileInstance']
