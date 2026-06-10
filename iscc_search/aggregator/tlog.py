"""
C2SP tlog-tiles wire-format helpers (pure, Sans-IO).

Ported faithfully from iscc-hub (``iscc_hub/merkle.py`` and
``iscc_hub/checkpoint_note.py``; iscc-hub is a Django app, not an importable
library). Only the trust-mode subset needed to follow a hub's transparency
log: checkpoint tree-size parsing, entry-bundle decoding, and entry-bundle
path encoding. No Merkle or signature verification (the aggregator trusts the
authoritative hub list plus TLS).
"""

import struct

# Tile height is fixed at 8 by the tlog-tiles profile: a full entry bundle holds 256 records.
TILE_WIDTH = 256


def parse_checkpoint(text):
    # type: (str) -> int
    """
    Parse a C2SP checkpoint and return its tree size (no signature check).

    The checkpoint body is three newline-terminated lines: origin, decimal
    tree size, base64 root hash — followed by signature lines we ignore.

    :param text: Full checkpoint text (or just the body)
    :return: The committed tree size from line 2
    :raises ValueError: If the body is malformed
    """
    lines = text.split("\n")
    if len(lines) < 3:
        raise ValueError("checkpoint body has too few lines")
    try:
        tree_size = int(lines[1])
    except ValueError:
        raise ValueError("checkpoint tree size is not an integer") from None
    if tree_size < 0 or (lines[1] != "0" and lines[1].startswith("0")):
        raise ValueError("checkpoint tree size has leading zeros or is negative")
    return tree_size


def parse_entry_bundle(data):
    # type: (bytes) -> list[bytes]
    """
    Decode tlog-tiles entry-bundle bytes into record byte strings.

    Inverse of the hub's ``entry_bundle`` framing: each record is a big-endian
    uint16 length prefix followed by the record bytes; frames are concatenated
    with no trailing length.

    :param data: Entry-bundle bytes (up to 256 framed records)
    :return: Record byte strings in leaf order
    :raises ValueError: If a frame is truncated
    """
    records = []
    offset = 0
    total = len(data)
    while offset < total:
        if offset + 2 > total:
            raise ValueError("entry bundle truncated in length prefix")
        (length,) = struct.unpack_from(">H", data, offset)
        offset += 2
        if offset + length > total:
            raise ValueError("entry bundle truncated in record body")
        records.append(data[offset : offset + length])
        offset += length
    return records


def format_index(n):
    # type: (int) -> str
    """
    Encode a tile/bundle index in the tlog-tiles thousands-grouped path form.

    Digits are grouped in threes from the least-significant end, each group
    zero-padded to three digits; every group except the least-significant is
    ``x``-prefixed and groups are slash-separated (e.g. ``1234067`` ->
    ``x001/x234/067``).

    :param n: Tile/bundle index
    :return: Thousands-grouped path segment
    """
    s = f"{n % 1000:03d}"
    n //= 1000
    while n > 0:
        s = f"x{n % 1000:03d}/{s}"
        n //= 1000
    return s


def entries_path(index, width=0):
    # type: (int, int) -> str
    """
    Return the tlog-tiles entry-bundle path ``tile/entries/<N>[.p/<W>]``.

    :param index: Entry-bundle index
    :param width: 0 for a full bundle, 1-255 for the in-progress partial bundle
    :return: Path relative to the hub's /log/ mount
    """
    suffix = f".p/{width}" if width else ""
    return f"tile/entries/{format_index(index)}{suffix}"
