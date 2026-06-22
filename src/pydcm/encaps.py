# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Fuli Wu
# NOT a medical device — not for clinical or diagnostic use.
"""encapsulated-pixel framing (`pydcm.encaps`). The native decoder
handles encapsulation on read; these helpers cover the common encode/split needs."""
import struct as _struct

_ITEM = b"\xFE\xFF\x00\xE0"
_SEQ_DELIM = b"\xFE\xFF\xDD\xE0\x00\x00\x00\x00"


def itemize_frame(frame, *_a, **_k):
    f = bytes(frame)
    if len(f) % 2:
        f += b"\x00"
    return _ITEM + _struct.pack("<I", len(f)) + f


def encapsulate(frames, fragments_per_frame=1, has_bot=True):
    """Encapsulate a list of frame byte-strings into a PixelData OB value."""
    items = [itemize_frame(f) for f in frames]
    if has_bot:
        bot = _ITEM + _struct.pack("<I", 0)        # empty Basic Offset Table
    else:
        bot = b""
    return bot + b"".join(items)


def decode_data_sequence(data):
    """Split an encapsulated PixelData value into its fragment byte-strings."""
    out, p, d = [], 0, bytes(data)
    while p + 8 <= len(d):
        if d[p:p + 4] != _ITEM:
            break
        ln = _struct.unpack_from("<I", d, p + 4)[0]
        out.append(d[p + 8:p + 8 + ln])
        p += 8 + ln
    return out[1:] if out else out                 # drop the Basic Offset Table item


def defragment_data(data):
    return b"".join(decode_data_sequence(data))


def generate_pixel_data_frame(data, number_of_frames=None):
    for frag in decode_data_sequence(data):
        yield frag


def fragment_frame(frame, nr_fragments=1):
    """Yield ``nr_fragments`` even-length fragments of a single ``frame``."""
    f = bytes(frame)
    if nr_fragments <= 1:
        yield f if len(f) % 2 == 0 else f + b"\x00"
        return
    n = (len(f) + nr_fragments - 1) // nr_fragments
    if n % 2:
        n += 1
    for i in range(0, len(f), n):
        frag = f[i:i + n]
        yield frag if len(frag) % 2 == 0 else frag + b"\x00"


def itemize_fragment(fragment):
    return itemize_frame(fragment)


def encapsulate_extended(frames):
    """Encapsulate ``frames`` and also return the Extended Offset Table + Lengths
   : ``(pixel_data, offsets_bytes, lengths_bytes)``."""
    items, offsets, lengths, pos = [], [], [], 0
    for f in frames:
        item = itemize_frame(f)
        offsets.append(pos)
        lengths.append(len(item) - 8)              # value length (excl. the 8-byte item tag)
        pos += len(item)
        items.append(item)
    pixel_data = b"".join(items)
    off_bytes = b"".join(_struct.pack("<Q", o) for o in offsets)
    len_bytes = b"".join(_struct.pack("<Q", n) for n in lengths)
    return pixel_data, off_bytes, len_bytes


def encapsulate_buffer(frames, fragments_per_frame=1, has_bot=True):
    return encapsulate(frames, fragments_per_frame, has_bot)


# ── Read-side helpers (pydicom 3.0 names) ───────────────────────────────────
def _items(buffer):
    """Yield ``(item_offset, value)`` for each FFFE,E000 item in ``buffer``, in order."""
    d = bytes(buffer)
    p = 0
    while p + 8 <= len(d):
        if d[p:p + 4] != _ITEM:
            break
        ln = _struct.unpack_from("<I", d, p + 4)[0]
        yield p, d[p + 8:p + 8 + ln]
        p += 8 + ln


def _fragments_with_offsets(buffer):
    """Yield ``(offset_in_fragment_area, value)`` for each fragment after the BOT.

    The offset is to the fragment's Item Tag measured from the first byte following
    the Basic Offset Table item — the reference frame the Basic Offset Table uses.
    """
    base = None
    for i, (off, val) in enumerate(_items(buffer)):
        if i == 0:                       # the Basic Offset Table item
            base = off + 8 + len(val)
            continue
        yield off - base, val


def parse_basic_offsets(buffer, *, endianness="<"):
    """Return the Basic Offset Table as a list of frame offsets."""
    for _off, bot in _items(buffer):     # first item is the BOT
        n = len(bot) // 4
        return list(_struct.unpack(f"{endianness}{n}I", bot)) if n else []
    return []


def generate_fragments(buffer, *, endianness="<"):
    """Yield each fragment's bytes (after the Basic Offset Table)."""
    for _off, val in _fragments_with_offsets(buffer):
        yield val


def parse_fragments(buffer, *, endianness="<"):
    """Return ``(number_of_fragments, [item offsets])`` after the BOT."""
    offsets = [off for off, _v in _fragments_with_offsets(buffer)]
    return len(offsets), offsets


def generate_frames(buffer, *, number_of_frames=None, extended_offsets=None, endianness="<"):
    """Yield each frame's complete (de-fragmented) bytes from encapsulated *Pixel Data*.

    Groups fragments into frames using, in priority: the Extended Offset Table, the
    Basic Offset Table, then ``number_of_frames`` (1 frame, or 1 fragment per frame)."""
    frags = list(_fragments_with_offsets(buffer))

    def _group(starts):
        for i, lo in enumerate(starts):
            hi = starts[i + 1] if i + 1 < len(starts) else None
            yield b"".join(v for off, v in frags if off >= lo and (hi is None or off < hi))

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
    if number_of_frames is None or number_of_frames == 1 or len(vals) != number_of_frames:
        yield b"".join(vals)            # single frame (or unresolved -> all fragments = 1)
        return
    yield from vals                     # one fragment per frame


def get_frame(buffer, index, *, extended_offsets=None, number_of_frames=None, endianness="<"):
    """Return the complete bytes of frame ``index`` from encapsulated *Pixel Data*."""
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
