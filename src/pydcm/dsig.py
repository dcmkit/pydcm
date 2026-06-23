# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Fuli Wu
# NOT a medical device — not for clinical or diagnostic use.

"""DICOM digital signatures (PS3.15) — sign and verify, over the native engine.

The integrity / non-repudiation seam (PS3.15 digital signatures)::

    signed = pydcm.dsig.sign("ct.dcm", "key.pem", "cert.pem")   # -> bytes
    sigs   = pydcm.dsig.verify(signed)                          # -> [{uid, mac, ...}]

``sign`` computes the PS3.15 canonical MAC over the dataset, signs it with the
PEM private key, and embeds the signer's X.509 certificate; it self-checks the
result and raises on a key/cert mismatch. ``verify`` checks every signature
cryptographically against its embedded certificate and reports validity per
signature; it does NOT validate the certificate's trust chain (a separate PKI
concern). RIPEMD160 (the legacy default MAC), SHA1/256/384/512 and MD5 are
supported; the output is byte-exact and standards-conformant so signatures cross-verify with other implementations.

The logic lives in C++ (a crypto-free canonical engine plus the OpenSSL crypto) —
this module is the thin marshaller.
Requires the optional ``_dsig`` extension.
"""
from __future__ import annotations

import os

try:
    from . import _dsig
except ImportError as _e:                            # pragma: no cover
    raise ImportError(
        "pydcm.dsig requires the optional native _dsig extension, "
        "which is not present in this build."
    ) from _e

__all__ = ["sign", "verify"]


def _read_bytes(data) -> bytes:
    """bytes → bytes; path-like → file bytes."""
    if isinstance(data, (bytes, bytearray)):
        return bytes(data)
    if isinstance(data, (str, os.PathLike)):
        with open(data, "rb") as f:
            return f.read()
    raise TypeError("data must be bytes or a path to a Part-10 file")


def _read_pem(pem) -> str:
    """PEM text passes through; anything else is treated as a path to read.

    A PEM always contains ``-----BEGIN``; otherwise the value is taken as a file
    path, so callers can hand either the certificate/key text or its file. Files
    are read latin-1 (lossless byte→str): the meaningful base64 + ``-----BEGIN/
    END`` markers are ASCII, but a ``.pem`` may carry non-ASCII text in the
    human-readable preamble that a strict-ASCII read would wrongly reject."""
    if isinstance(pem, (bytes, bytearray)):
        return bytes(pem).decode("latin-1")
    if isinstance(pem, (str, os.PathLike)):
        s = os.fspath(pem)
        if "-----BEGIN" in s:
            return s
        with open(s, "r", encoding="latin-1") as f:
            return f.read()
    raise TypeError("expected a PEM string or a path to a PEM file")


def sign(data, key, cert, *, mac: str = "SHA256") -> bytes:
    """Sign a DICOM file and return the signed Part-10 bytes.

    Parameters
    ----------
    data : bytes | str | os.PathLike
        The DICOM file to sign (Part-10, uncompressed VR-LE).
    key : str | os.PathLike
        The PEM private key — its text, or a path to the ``.pem`` file.
    cert : str | os.PathLike
        The signer's PEM X.509 certificate (text or path); embedded in the
        signature as Certificate of Signer (0400,0115).
    mac : str
        MAC algorithm (0400,0015): ``SHA256`` (default), ``SHA1``, ``SHA384``,
        ``SHA512``, ``RIPEMD160`` or ``MD5``.

    Returns
    -------
    bytes
        The signed Part-10 file. Signing an already-signed file adds a second
        signature (counter-signing), merged into the existing sequences.

    Raises
    ------
    RuntimeError
        On a signing error, or if the new signature does not verify against
        ``cert`` (a key/cert mismatch).
    """
    return _dsig.sign(_read_bytes(data), _read_pem(key), _read_pem(cert), mac)


def verify(data) -> list[dict]:
    """Verify every digital signature in a DICOM file.

    Parameters
    ----------
    data : bytes | str | os.PathLike
        The DICOM file to verify.

    Returns
    -------
    list[dict]
        One entry per signature: ``{"uid": str, "mac": str,
        "signed_elements": int, "valid": bool}``. Empty when the file carries
        no signatures. ``valid`` is the cryptographic verdict against the
        embedded signer certificate.
    """
    return _dsig.verify(_read_bytes(data))
