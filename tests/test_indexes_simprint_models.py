"""Tests for simprint msgspec models."""

from iscc_search.indexes.simprint.models import MatchedChunkRaw, SimprintMatchRaw


def test_matched_chunk_raw_creation():
    # type: () -> None
    """Test MatchedChunkRaw struct instantiation."""
    chunk = MatchedChunkRaw(
        query=b"\x01\x02\x03\x04\x05\x06\x07\x08",
        match=b"\x01\x02\x03\x04\x05\x06\x07\x08",
        score=0.95,
        offset=1024,
        size=512,
        freq=5,
    )

    assert chunk.query == b"\x01\x02\x03\x04\x05\x06\x07\x08"
    assert chunk.match == b"\x01\x02\x03\x04\x05\x06\x07\x08"
    assert chunk.score == 0.95
    assert chunk.offset == 1024
    assert chunk.size == 512
    assert chunk.freq == 5


def test_simprint_match_raw_creation_with_chunks():
    # type: () -> None
    """Test SimprintMatchRaw struct with chunk details."""
    chunks = [
        MatchedChunkRaw(
            query=b"\x01\x02\x03\x04\x05\x06\x07\x08",
            match=b"\x01\x02\x03\x04\x05\x06\x07\x08",
            score=1.0,
            offset=0,
            size=256,
            freq=2,
        ),
        MatchedChunkRaw(
            query=b"\x11\x12\x13\x14\x15\x16\x17\x18",
            match=b"\x11\x12\x13\x14\x15\x16\x17\x18",
            score=1.0,
            offset=256,
            size=256,
            freq=3,
        ),
    ]

    match = SimprintMatchRaw(
        iscc_id_body=b"\x12\x34\x56\x78\x9a\xbc\xde\xf0",
        score=0.87,
        queried=10,
        matches=2,
        chunks=chunks,
    )

    assert match.iscc_id_body == b"\x12\x34\x56\x78\x9a\xbc\xde\xf0"
    assert match.score == 0.87
    assert match.queried == 10
    assert match.matches == 2
    assert match.chunks is not None
    assert len(match.chunks) == 2
    assert match.chunks[0].offset == 0
    assert match.chunks[1].offset == 256


def test_simprint_match_raw_creation_without_chunks():
    # type: () -> None
    """Test SimprintMatchRaw struct without chunk details (detailed=False)."""
    match = SimprintMatchRaw(
        iscc_id_body=b"\xaa\xbb\xcc\xdd\xee\xff\x00\x11",
        score=0.75,
        queried=5,
        matches=3,
        chunks=None,
    )

    assert match.iscc_id_body == b"\xaa\xbb\xcc\xdd\xee\xff\x00\x11"
    assert match.score == 0.75
    assert match.queried == 5
    assert match.matches == 3
    assert match.chunks is None
