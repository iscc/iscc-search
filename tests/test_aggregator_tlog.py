"""Tests for tlog-tiles wire-format helpers (KAT-backed, byte-exact)."""

import json
import struct
from pathlib import Path
import pytest
from iscc_search.aggregator import tlog
from iscc_search.aggregator.poller import plan_bundles


KAT = json.loads((Path(__file__).parent / "data" / "tlog_kat.json").read_text(encoding="utf-8"))


def test_parse_checkpoint_full_signed_note():
    """Tree size is read from line 2 of a full signed checkpoint."""
    text = "sb0.iscc.id\n12345\nq83vASNFZ4mrze8BI0VniavN7wEjRWeJq83vASNFZ4k=\n\n— sb0.iscc.id c2lnbmF0dXJl\n"
    assert tlog.parse_checkpoint(text) == 12345


def test_parse_checkpoint_body_only():
    """A bare three-line body parses without signature lines."""
    assert tlog.parse_checkpoint("origin\n0\nroot\n") == 0


def test_parse_checkpoint_too_few_lines():
    """Fewer than three lines is rejected."""
    with pytest.raises(ValueError, match="too few lines"):
        tlog.parse_checkpoint("origin\n5")


def test_parse_checkpoint_non_integer():
    """A non-integer tree size is rejected."""
    with pytest.raises(ValueError, match="not an integer"):
        tlog.parse_checkpoint("origin\nfive\nroot\n")


def test_parse_checkpoint_leading_zeros():
    """Leading zeros in the tree size are rejected."""
    with pytest.raises(ValueError, match="leading zeros"):
        tlog.parse_checkpoint("origin\n01\nroot\n")


def test_parse_checkpoint_negative():
    """A negative tree size is rejected."""
    with pytest.raises(ValueError, match="negative"):
        tlog.parse_checkpoint("origin\n-1\nroot\n")


def test_parse_entry_bundle_roundtrip():
    """Framed records decode back to the original byte strings."""
    records = [b"a", b"bb" * 100, b"", b"record"]
    framed = b"".join(struct.pack(">H", len(r)) + r for r in records)
    assert tlog.parse_entry_bundle(framed) == records


def test_parse_entry_bundle_empty():
    """An empty bundle decodes to no records."""
    assert tlog.parse_entry_bundle(b"") == []


def test_parse_entry_bundle_truncated_prefix():
    """A frame cut inside the length prefix is rejected."""
    with pytest.raises(ValueError, match="length prefix"):
        tlog.parse_entry_bundle(b"\x00")


def test_parse_entry_bundle_truncated_body():
    """A frame cut inside the record body is rejected."""
    with pytest.raises(ValueError, match="record body"):
        tlog.parse_entry_bundle(b"\x00\x05abc")


def test_format_index():
    """Indexes encode in the thousands-grouped path form."""
    assert tlog.format_index(0) == "000"
    assert tlog.format_index(67) == "067"
    assert tlog.format_index(999) == "999"
    assert tlog.format_index(1000) == "x001/000"
    assert tlog.format_index(1234067) == "x001/x234/067"


def test_entries_path():
    """Entry-bundle paths carry the .p/<width> suffix only for partials."""
    assert tlog.entries_path(0) == "tile/entries/000"
    assert tlog.entries_path(1, width=1) == "tile/entries/001.p/1"
    assert tlog.entries_path(273, width=112) == "tile/entries/273.p/112"


def test_kat_entry_bundles():
    """plan_bundles + entries_path + parse_entry_bundle match the Go-generated KAT vectors byte-exactly."""
    for size_str, bundles in KAT["entry_bundles"].items():
        tree_size = int(size_str)
        plan = plan_bundles(0, tree_size)
        planned_paths = {tlog.entries_path(i, w): (i, w) for i, w in plan}
        if tree_size <= 2 * tlog.TILE_WIDTH:
            # Small trees: the KAT lists every bundle, the plan must match exactly
            assert sorted(planned_paths) == sorted(bundles)
        else:
            # Large trees: the KAT lists a subset (first full + last partial)
            assert set(bundles) <= set(planned_paths)
        for path, data_hex in bundles.items():
            bundle_index, width = planned_paths[path]
            records = tlog.parse_entry_bundle(bytes.fromhex(data_hex))
            start = bundle_index * tlog.TILE_WIDTH
            expected_count = width or min(tlog.TILE_WIDTH, tree_size - start)
            assert len(records) == expected_count
            assert records == [f"iscc-log-entry-{start + i}".encode() for i in range(expected_count)]
