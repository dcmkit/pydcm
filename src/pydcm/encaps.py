# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Fuli Wu
# NOT a medical device — not for clinical or diagnostic use.
"""encapsulated-pixel framing (`pydcm.encaps`).

The PS3.5 §A.4 wire framing — Basic Offset Table, per-fragment items, even-length
padding, the offset arithmetic — belongs to the native engine
(the native Part-10 encapsulated-value writer) and is NOT reimplemented here. This
module owns only the policy the standard leaves to the caller: how a frame is split
into fragments, and which of the read-side views to present.

Item walking on the read side is deliberately local: it works on a bare value that
has been handed to Python, which the native parsers reach only through a whole file.
"""
import struct as _struct

from . import _native

_ITEM = b"\xFE\xFF\x00\xE0"
_SEQ_DELIM = b"\xFE\xFF\xDD\xE0\x00\x00\x00\x00"


def _core():
    return _native.require()


# ── Write side ──────────────────────────────────────────────────────────────

def fragment_frame(frame, nr_fragments=1):
    """Yield ``nr_fragments`` even-length fragments of a single ``frame``.

    The first N-1 fragments are ``len(frame) // nr_fragments`` rounded up to even;
    the last takes whatever remains and is padded if odd. Splitting is the caller's
    policy, not the engine's, so it lives here.
    """
    f = bytes(frame)
    n = len(f)
    if nr_fragments > (n + 1) / 2.0:
        raise ValueError(
            "Too many fragments requested (the minimum fragment size is 2 bytes)")
    length = int(n / nr_fragments)
    if length % 2:
        length += 1
    for offset in range(0, length * (nr_fragments - 1), length):
        yield f[offset:offset + length]
    offset = length * (nr_fragments - 1)
    last = f[offset:]
    if (n - offset) % 2:
        last += b"\x00"
    yield last


def itemize_frame(frame, nr_fragments=1):
    """Yield each fragment of ``frame`` as a complete (FFFE,E000) item."""
    for fragment in fragment_frame(frame, nr_fragments):
        yield itemize_fragment(fragment)


def itemize_fragment(fragment):
    """One fragment -> one (FFFE,E000) item, padded to even length."""
    f = bytes(fragment)
    if len(f) % 2:
        f += b"\x00"
    return _ITEM + _struct.pack("<I", len(f)) + f


def _fragment_frames(frames, fragments_per_frame):
    """(all fragments in order, index of each frame's first fragment)."""
    frags, starts = [], []
    for frame in frames:
        starts.append(len(frags))
        frags.extend(fragment_frame(frame, fragments_per_frame))
    return frags, starts


def encapsulate(frames, fragments_per_frame=1, has_bot=True):
    """Encapsulate a list of frame byte-strings into a PixelData OB value.

    ``has_bot=False`` still emits the Basic Offset Table ITEM, empty — PS3.5 §A.4
    allows an unpopulated table but not a missing one, and a value without it is
    unreadable to conformant parsers.
    """
    frags, starts = _fragment_frames(frames, fragments_per_frame)
    return _core().encapsulate_pixel_value(frags, starts if has_bot else [])


def encapsulate_buffer(frames, fragments_per_frame=1, has_bot=True):
    return encapsulate(frames, fragments_per_frame, has_bot)


def encapsulate_extended(frames):
    """``(pixel_data, offsets_bytes, lengths_bytes)`` — the Extended Offset Table form.

    The Basic Offset Table is emitted EMPTY: (7FE0,0001)/(7FE0,0002) supersede it,
    and duplicating the offsets in both invites them to disagree. Offsets are to each
    frame's item tag measured from the first byte after the Basic Offset Table item;
    lengths are the item VALUE lengths, header excluded.
    """
    frames = [bytes(f) for f in frames]
    pixel_data = _core().encapsulate_pixel_value(frames, [])
    offsets, lengths, pos = [], [], 0
    for f in frames:
        padded = len(f) + (len(f) % 2)
        offsets.append(pos)
        lengths.append(padded)
        pos += 8 + padded
    return (pixel_data,
            b"".join(_struct.pack("<Q", o) for o in offsets),
            b"".join(_struct.pack("<Q", n) for n in lengths))


# ── Read side ───────────────────────────────────────────────────────────────
#
# `buffer` throughout is the raw (7FE0,0010) VALUE: the Basic Offset Table item
# followed by the fragment items. The BOT is item 0 — it is a fragment as far as
# the item walk is concerned, and only the frame-oriented views treat it specially.

def _items(buffer):
    """Yield ``(item_offset, value)`` for each FFFE,E000 item, in order."""
    d = bytes(buffer)
    p = 0
    while p + 8 <= len(d):
        if d[p:p + 4] != _ITEM:
            break
        ln = _struct.unpack_from("<I", d, p + 4)[0]
        if p + 8 + ln > len(d):
            break                        # truncated item — stop rather than over-read
        yield p, d[p + 8:p + 8 + ln]
        p += 8 + ln


def generate_fragments(buffer, *, endianness="<"):
    """Yield every item's value, the Basic Offset Table included as the first."""
    for _off, val in _items(buffer):
        yield val


def parse_fragments(buffer, *, endianness="<"):
    """``(item count, [item offsets])`` — offsets from the START of ``buffer``."""
    offsets = [off for off, _v in _items(buffer)]
    return len(offsets), offsets


def parse_basic_offsets(buffer, *, endianness="<"):
    """The Basic Offset Table as a list of frame offsets ( ``[]`` when empty)."""
    for _off, bot in _items(buffer):     # item 0 is the Basic Offset Table
        n = len(bot) // 4
        return list(_struct.unpack(f"{endianness}{n}I", bot)) if n else []
    return []


def _fragments_after_bot(buffer):
    """``(offset relative to the end of the BOT item, value)`` per fragment.

    That reference frame — the first byte after the Basic Offset Table item — is the
    one the Basic Offset Table itself uses, so these offsets are directly comparable
    with its entries.
    """
    base = None
    for i, (off, val) in enumerate(_items(buffer)):
        if i == 0:
            base = off + 8 + len(val)
            continue
        yield off - base, val


def decode_data_sequence(data):
    """The fragment values, Basic Offset Table excluded."""
    return [val for _off, val in _fragments_after_bot(data)]


def defragment_data(data):
    return b"".join(decode_data_sequence(data))


def generate_pixel_data_frame(data, number_of_frames=None):
    yield from generate_frames(data, number_of_frames=number_of_frames)


def generate_frames(buffer, *, number_of_frames=None, extended_offsets=None,
                    endianness="<"):
    """Yield each frame's complete (de-fragmented) bytes.

    Fragments are grouped into frames by, in priority: the Extended Offset Table,
    a populated Basic Offset Table, then ``number_of_frames`` when it matches the
    fragment count one-to-one. Failing all of those the fragments are one frame.
    """
    frags = list(_fragments_after_bot(buffer))

    def _group(starts):
        for i, lo in enumerate(starts):
            hi = starts[i + 1] if i + 1 < len(starts) else None
            yield b"".join(v for off, v in frags
                           if off >= lo and (hi is None or off < hi))

    if extended_offsets:
        offs = extended_offsets[0]
        if isinstance(offs, (bytes, bytearray)):
            offs = list(_struct.unpack(f"{endianness}{len(offs) // 8}Q", bytes(offs)))
        yield from _group(list(offs))
        return

    bot = parse_basic_offsets(buffer, endianness=endianness)
    if len(bot) > 1:
        yield from _group(bot)
        return

    vals = [v for _o, v in frags]
    if number_of_frames is not None and number_of_frames > 1 \
            and len(vals) == number_of_frames:
        yield from vals                  # one fragment per frame
        return
    yield b"".join(vals)                 # single frame, or unresolvable grouping


def get_frame(buffer, index, *, extended_offsets=None, number_of_frames=None,
              endianness="<"):
    """The complete bytes of frame ``index``."""
    for i, frame in enumerate(generate_frames(
            buffer, number_of_frames=number_of_frames,
            extended_offsets=extended_offsets, endianness=endianness)):
        if i == index:
            return frame
    raise IndexError(f"There is no frame at index {index}")


__all__ = ["encapsulate", "encapsulate_extended", "encapsulate_buffer",
           "decode_data_sequence", "defragment_data", "generate_pixel_data_frame",
           "itemize_frame", "itemize_fragment", "fragment_frame",
           "parse_basic_offsets", "generate_fragments", "parse_fragments",
           "generate_frames", "get_frame"]
