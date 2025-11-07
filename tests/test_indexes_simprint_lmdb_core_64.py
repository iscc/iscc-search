"""Tests for LMDB-based 64-bit simprint index."""

import struct
import pytest
from iscc_search.indexes.simprint.lmdb_core_64 import LmdbSimprintIndex64


class MockSimprintRaw:
    """Mock implementation of SimprintRaw protocol."""

    def __init__(self, simprint, offset, size):
        # type: (bytes, int, int) -> None
        self.simprint = simprint
        self.offset = offset
        self.size = size


class MockSimprintEntryRaw:
    """Mock implementation of SimprintEntryRaw protocol."""

    def __init__(self, iscc_id_body, simprints):
        # type: (bytes, list[MockSimprintRaw]) -> None
        self.iscc_id_body = iscc_id_body
        self.simprints = simprints


@pytest.fixture
def temp_index_path(tmp_path):
    # type: (Path) -> Path
    """Create a temporary directory for index storage."""
    return tmp_path / "simprint_index"


@pytest.fixture
def index(temp_index_path):
    # type: (Path) -> LmdbSimprintIndex64
    """Create and return a fresh index instance."""
    idx = LmdbSimprintIndex64(str(temp_index_path))
    yield idx
    idx.close()


def test_init_creates_directory(temp_index_path):
    # type: (Path) -> None
    """Test that __init__ creates the index directory."""
    idx = LmdbSimprintIndex64(str(temp_index_path))
    assert temp_index_path.exists()
    assert temp_index_path.is_dir()
    idx.close()


def test_init_with_file_uri(tmp_path):
    # type: (Path) -> None
    """Test initialization with file:// URI."""
    path = tmp_path / "file_uri_index"
    uri = f"file://{path}"
    idx = LmdbSimprintIndex64(uri)
    assert path.exists()
    idx.close()


def test_init_with_lmdb_uri(tmp_path):
    # type: (Path) -> None
    """Test initialization with lmdb:// URI."""
    path = tmp_path / "lmdb_uri_index"
    uri = f"lmdb://{path}"
    idx = LmdbSimprintIndex64(uri)
    assert path.exists()
    idx.close()


def test_pack_chunk_pointer_valid(index):
    # type: (LmdbSimprintIndex64) -> None
    """Test packing chunk pointer with valid inputs."""
    iscc_id_body = b"\x01\x02\x03\x04\x05\x06\x07\x08"
    offset = 1024
    size = 512

    packed = index._pack_chunk_pointer(iscc_id_body, offset, size)

    assert len(packed) == 16
    assert packed[:8] == iscc_id_body
    assert struct.unpack("!II", packed[8:16]) == (offset, size)


def test_pack_chunk_pointer_invalid_iscc_id_length(index):
    # type: (LmdbSimprintIndex64) -> None
    """Test packing with incorrect ISCC-ID body length."""
    with pytest.raises(ValueError, match="ISCC-ID body must be 8 bytes"):
        index._pack_chunk_pointer(b"\x01\x02", 0, 0)


def test_pack_chunk_pointer_offset_exceeds_max(index):
    # type: (LmdbSimprintIndex64) -> None
    """Test packing with offset exceeding maximum."""
    iscc_id_body = b"\x01\x02\x03\x04\x05\x06\x07\x08"
    with pytest.raises(ValueError, match="Offset .* exceeds max"):
        index._pack_chunk_pointer(iscc_id_body, 2**32, 0)


def test_pack_chunk_pointer_size_exceeds_max(index):
    # type: (LmdbSimprintIndex64) -> None
    """Test packing with size exceeding maximum."""
    iscc_id_body = b"\x01\x02\x03\x04\x05\x06\x07\x08"
    with pytest.raises(ValueError, match="Size .* exceeds max"):
        index._pack_chunk_pointer(iscc_id_body, 0, 2**32)


def test_unpack_chunk_pointer_valid(index):
    # type: (LmdbSimprintIndex64) -> None
    """Test unpacking chunk pointer."""
    iscc_id_body = b"\xaa\xbb\xcc\xdd\xee\xff\x00\x11"
    offset = 2048
    size = 1024

    packed = index._pack_chunk_pointer(iscc_id_body, offset, size)
    unpacked_id, unpacked_offset, unpacked_size = index._unpack_chunk_pointer(packed)

    assert unpacked_id == iscc_id_body
    assert unpacked_offset == offset
    assert unpacked_size == size


def test_unpack_chunk_pointer_invalid_length(index):
    # type: (LmdbSimprintIndex64) -> None
    """Test unpacking with incorrect data length."""
    with pytest.raises(ValueError, match="Expected 16 bytes"):
        index._unpack_chunk_pointer(b"\x01\x02\x03")


def test_add_raw_empty_list(index):
    # type: (LmdbSimprintIndex64) -> None
    """Test adding empty entry list (no-op)."""
    index.add_raw([])
    assert len(index) == 0


def test_add_raw_single_entry(index):
    # type: (LmdbSimprintIndex64) -> None
    """Test adding a single entry with simprints."""
    iscc_id = b"\x01\x02\x03\x04\x05\x06\x07\x08"
    simprints = [
        MockSimprintRaw(b"\x11\x12\x13\x14\x15\x16\x17\x18", 0, 256),
        MockSimprintRaw(b"\x21\x22\x23\x24\x25\x26\x27\x28", 256, 256),
    ]
    entry = MockSimprintEntryRaw(iscc_id, simprints)

    index.add_raw([entry])

    assert len(index) == 1
    assert iscc_id in index


def test_add_raw_duplicate_ignored(index):
    # type: (LmdbSimprintIndex64) -> None
    """Test that duplicate ISCC-ID bodies are silently ignored."""
    iscc_id = b"\x01\x02\x03\x04\x05\x06\x07\x08"
    simprints = [MockSimprintRaw(b"\x11\x12\x13\x14\x15\x16\x17\x18", 0, 256)]
    entry = MockSimprintEntryRaw(iscc_id, simprints)

    index.add_raw([entry])
    assert len(index) == 1

    # Add duplicate - should be ignored
    index.add_raw([entry])
    assert len(index) == 1


def test_add_raw_multiple_entries(index):
    # type: (LmdbSimprintIndex64) -> None
    """Test adding multiple unique entries in one batch."""
    entries = [
        MockSimprintEntryRaw(
            b"\x01\x02\x03\x04\x05\x06\x07\x08",
            [MockSimprintRaw(b"\x11\x12\x13\x14\x15\x16\x17\x18", 0, 256)],
        ),
        MockSimprintEntryRaw(
            b"\x11\x12\x13\x14\x15\x16\x17\x18",
            [MockSimprintRaw(b"\x21\x22\x23\x24\x25\x26\x27\x28", 0, 512)],
        ),
    ]

    index.add_raw(entries)
    assert len(index) == 2


def test_add_raw_with_duplicates_in_batch(index):
    # type: (LmdbSimprintIndex64) -> None
    """Test batch with duplicate ISCC-IDs within the batch."""
    iscc_id = b"\x01\x02\x03\x04\x05\x06\x07\x08"
    entries = [
        MockSimprintEntryRaw(iscc_id, [MockSimprintRaw(b"\x11" * 8, 0, 256)]),
        MockSimprintEntryRaw(iscc_id, [MockSimprintRaw(b"\x22" * 8, 256, 256)]),
    ]

    index.add_raw(entries)
    assert len(index) == 1  # Only first occurrence counted


def test_search_raw_empty_query(index):
    # type: (LmdbSimprintIndex64) -> None
    """Test search with empty simprint list."""
    results = index.search_raw([])
    assert results == []


def test_search_raw_no_matches(index):
    # type: (LmdbSimprintIndex64) -> None
    """Test search when no simprints match."""
    # Add entry
    iscc_id = b"\x01\x02\x03\x04\x05\x06\x07\x08"
    simprints = [MockSimprintRaw(b"\x11\x12\x13\x14\x15\x16\x17\x18", 0, 256)]
    entry = MockSimprintEntryRaw(iscc_id, simprints)
    index.add_raw([entry])

    # Search for different simprint
    results = index.search_raw([b"\x99" * 8])
    assert results == []


def test_search_raw_exact_match(index):
    # type: (LmdbSimprintIndex64) -> None
    """Test search with exact simprint match."""
    iscc_id = b"\x01\x02\x03\x04\x05\x06\x07\x08"
    simprint_bytes = b"\x11\x12\x13\x14\x15\x16\x17\x18"
    simprints = [MockSimprintRaw(simprint_bytes, 0, 256)]
    entry = MockSimprintEntryRaw(iscc_id, simprints)
    index.add_raw([entry])

    results = index.search_raw([simprint_bytes])

    assert len(results) == 1
    assert results[0].iscc_id_body == iscc_id
    assert results[0].score > 0.0
    assert results[0].queried == 1
    assert results[0].matches == 1


def test_search_raw_with_detailed_chunks(index):
    # type: (LmdbSimprintIndex64) -> None
    """Test search returns detailed chunk information when detailed=True."""
    iscc_id = b"\x01\x02\x03\x04\x05\x06\x07\x08"
    simprint_bytes = b"\x11" * 8
    simprints = [MockSimprintRaw(simprint_bytes, 1024, 512)]
    entry = MockSimprintEntryRaw(iscc_id, simprints)
    index.add_raw([entry])

    results = index.search_raw([simprint_bytes], detailed=True)

    assert len(results) == 1
    assert results[0].chunks is not None
    assert len(results[0].chunks) == 1
    assert results[0].chunks[0].offset == 1024
    assert results[0].chunks[0].size == 512
    assert results[0].chunks[0].score == 1.0


def test_search_raw_without_detailed_chunks(index):
    # type: (LmdbSimprintIndex64) -> None
    """Test search excludes chunk details when detailed=False."""
    iscc_id = b"\x01\x02\x03\x04\x05\x06\x07\x08"
    simprint_bytes = b"\x11" * 8
    simprints = [MockSimprintRaw(simprint_bytes, 0, 256)]
    entry = MockSimprintEntryRaw(iscc_id, simprints)
    index.add_raw([entry])

    results = index.search_raw([simprint_bytes], detailed=False)

    assert len(results) == 1
    assert results[0].chunks is None


def test_search_raw_respects_limit(index):
    # type: (LmdbSimprintIndex64) -> None
    """Test that search respects the limit parameter."""
    simprint_bytes = b"\x11" * 8

    # Add 5 entries with same simprint (each in separate batch to avoid duplicate detection)
    for i in range(1, 6):
        iscc_id = bytes([i] * 8)
        simprints = [MockSimprintRaw(simprint_bytes, 0, 256)]
        entry = MockSimprintEntryRaw(iscc_id, simprints)
        index.add_raw([entry])

    results = index.search_raw([simprint_bytes], limit=3, threshold=0.0)
    assert len(results) == 3


def test_search_raw_respects_threshold(index):
    # type: (LmdbSimprintIndex64) -> None
    """Test that search respects the threshold parameter."""
    # Add one asset with single matching simprint out of query with many
    iscc_id = b"\x01" * 8
    simprint_bytes = b"\x11" * 8
    simprints = [MockSimprintRaw(simprint_bytes, 0, 256)]
    entry = MockSimprintEntryRaw(iscc_id, simprints)
    index.add_raw([entry])

    # Search with 10 simprints but only 1 matches
    # This gives score = 1/10 = 0.1 (below typical threshold)
    query_simprints = [simprint_bytes] + [bytes([i] * 8) for i in range(2, 11)]
    results = index.search_raw(query_simprints, threshold=0.5)

    # Should have no results because score is too low
    assert len(results) == 0


def test_search_raw_multiple_matches(index):
    # type: (LmdbSimprintIndex64) -> None
    """Test search with multiple matching assets."""
    simprint1 = b"\x11" * 8
    simprint2 = b"\x22" * 8

    # Add two assets with different simprints
    entry1 = MockSimprintEntryRaw(b"\x01" * 8, [MockSimprintRaw(simprint1, 0, 256)])
    entry2 = MockSimprintEntryRaw(b"\x02" * 8, [MockSimprintRaw(simprint2, 0, 512)])
    index.add_raw([entry1, entry2])

    # Search for both simprints
    results = index.search_raw([simprint1, simprint2], threshold=0.0)

    assert len(results) == 2


def test_search_raw_idf_scoring(index):
    # type: (LmdbSimprintIndex64) -> None
    """Test IDF-weighted scoring reduces impact of common simprints."""
    common_simprint = b"\xaa" * 8
    rare_simprint = b"\xbb" * 8

    # Add 10 assets with common simprint
    for i in range(10):
        iscc_id = bytes([i] * 8)
        entry = MockSimprintEntryRaw(iscc_id, [MockSimprintRaw(common_simprint, 0, 256)])
        index.add_raw([entry])

    # Add 1 asset with rare simprint
    rare_id = b"\xff" * 8
    entry = MockSimprintEntryRaw(rare_id, [MockSimprintRaw(rare_simprint, 0, 256)])
    index.add_raw([entry])

    # Search for rare simprint - should have high score
    results_rare = index.search_raw([rare_simprint], threshold=0.0)
    rare_score = results_rare[0].score

    # Search for common simprint - should have lower score due to IDF
    results_common = index.search_raw([common_simprint], threshold=0.0, limit=1)
    common_score = results_common[0].score

    # Rare simprint should score higher than common one
    assert rare_score > common_score


def test_contains_existing_asset(index):
    # type: (LmdbSimprintIndex64) -> None
    """Test __contains__ for existing asset."""
    iscc_id = b"\x01\x02\x03\x04\x05\x06\x07\x08"
    simprints = [MockSimprintRaw(b"\x11" * 8, 0, 256)]
    entry = MockSimprintEntryRaw(iscc_id, simprints)
    index.add_raw([entry])

    assert iscc_id in index


def test_contains_nonexistent_asset(index):
    # type: (LmdbSimprintIndex64) -> None
    """Test __contains__ for non-existent asset."""
    iscc_id = b"\x99" * 8
    assert iscc_id not in index


def test_len_empty_index(index):
    # type: (LmdbSimprintIndex64) -> None
    """Test __len__ on empty index."""
    assert len(index) == 0


def test_len_after_additions(index):
    # type: (LmdbSimprintIndex64) -> None
    """Test __len__ after adding entries."""
    entries = [MockSimprintEntryRaw(bytes([i] * 8), [MockSimprintRaw(b"\x11" * 8, 0, 256)]) for i in range(5)]
    index.add_raw(entries)
    assert len(index) == 5


def test_context_manager(temp_index_path):
    # type: (Path) -> None
    """Test context manager protocol."""
    with LmdbSimprintIndex64(str(temp_index_path)) as idx:
        assert idx is not None
        idx.add_raw([])

    # Index should be closed after exiting context
    assert temp_index_path.exists()


def test_close_and_reopen(temp_index_path):
    # type: (Path) -> None
    """Test closing and reopening an index preserves data."""
    iscc_id = b"\x01" * 8
    simprints = [MockSimprintRaw(b"\x11" * 8, 0, 256)]
    entry = MockSimprintEntryRaw(iscc_id, simprints)

    # Create and populate index
    idx1 = LmdbSimprintIndex64(str(temp_index_path))
    idx1.add_raw([entry])
    idx1.close()

    # Reopen and verify data persisted
    idx2 = LmdbSimprintIndex64(str(temp_index_path))
    assert len(idx2) == 1
    assert iscc_id in idx2
    idx2.close()


def test_simprint_truncation_to_64bit(index):
    # type: (LmdbSimprintIndex64) -> None
    """Test that simprints longer than 64-bit are truncated."""
    iscc_id = b"\x01" * 8
    # Provide 128-bit simprint
    long_simprint = b"\x11" * 16
    simprints = [MockSimprintRaw(long_simprint, 0, 256)]
    entry = MockSimprintEntryRaw(iscc_id, simprints)
    index.add_raw([entry])

    # Search with first 8 bytes should match
    results = index.search_raw([long_simprint[:8]])
    assert len(results) == 1

    # Search with full 16 bytes should also match (gets truncated)
    results = index.search_raw([long_simprint])
    assert len(results) == 1


def test_calculate_idf_score_empty_matches(index):
    # type: (LmdbSimprintIndex64) -> None
    """Test IDF score calculation with no matches."""
    score = index._calculate_idf_score([], {}, 10, 5)
    assert score == 0.0


def test_calculate_idf_score_basic(index):
    # type: (LmdbSimprintIndex64) -> None
    """Test basic IDF score calculation."""
    matches = [(b"\x11" * 8, b"\x11" * 8, 0, 256)]
    doc_frequencies = {b"\x11" * 8: 1}
    total_assets = 10
    num_queried = 5

    score = index._calculate_idf_score(matches, doc_frequencies, total_assets, num_queried)

    # Score should be between 0 and 1
    assert 0.0 <= score <= 1.0
    # With 1 match out of 5 queried, score should be relatively low
    assert score < 1.0


def test_format_match_result_with_freq(index):
    # type: (LmdbSimprintIndex64) -> None
    """Test match result formatting includes frequency information."""
    iscc_id = b"\x01" * 8
    simprint_bytes = b"\x11" * 8
    matches = [(simprint_bytes, simprint_bytes, 0, 256)]
    doc_frequencies = {simprint_bytes: 5}

    result = index._format_match_result(iscc_id, matches, 0.8, doc_frequencies, 1, True)

    assert result.chunks is not None
    assert len(result.chunks) == 1
    assert result.chunks[0].freq == 5


def test_add_raw_all_existing_after_check(index):
    # type: (LmdbSimprintIndex64) -> None
    """Test that adding already-existing entries returns early without executing insert."""
    # First add some entries
    iscc_id1 = b"\x01" * 8
    iscc_id2 = b"\x02" * 8
    entries = [
        MockSimprintEntryRaw(iscc_id1, [MockSimprintRaw(b"\x11" * 8, 0, 256)]),
        MockSimprintEntryRaw(iscc_id2, [MockSimprintRaw(b"\x22" * 8, 0, 256)]),
    ]
    index.add_raw(entries)
    assert len(index) == 2

    # Try to add them again - should return early
    index.add_raw(entries)
    assert len(index) == 2
