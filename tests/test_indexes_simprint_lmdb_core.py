"""Tests for LMDB-based variable-length simprint index."""

import struct
import pytest
from pathlib import Path
from iscc_search.indexes.simprint.lmdb_core import LmdbSimprintIndex


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
    """Create a temporary file path for index storage (flat file with subdir=False)."""
    return tmp_path / "simprint_index.lmdb"


@pytest.fixture
def index(temp_index_path):
    # type: (Path) -> LmdbSimprintIndex
    """Create and return a fresh index instance."""
    idx = LmdbSimprintIndex(str(temp_index_path))
    yield idx
    idx.close()


def test_init_creates_file(temp_index_path):
    # type: (Path) -> None
    """Test that __init__ creates the index file with subdir=False."""
    idx = LmdbSimprintIndex(str(temp_index_path))
    assert temp_index_path.exists()
    assert temp_index_path.is_file()
    # Lock file should also be created
    lock_file = Path(str(temp_index_path) + "-lock")
    assert lock_file.exists()
    idx.close()


def test_init_with_file_uri(tmp_path):
    # type: (Path) -> None
    """Test initialization with file:// URI."""
    path = tmp_path / "file_uri_index.lmdb"
    uri = f"file://{path}"
    idx = LmdbSimprintIndex(uri)
    assert path.exists()
    assert path.is_file()
    idx.close()


def test_init_with_lmdb_uri(tmp_path):
    # type: (Path) -> None
    """Test initialization with lmdb:// URI."""
    path = tmp_path / "lmdb_uri_index.lmdb"
    uri = f"lmdb://{path}"
    idx = LmdbSimprintIndex(uri)
    assert path.exists()
    assert path.is_file()
    idx.close()


def test_init_with_explicit_ndim(tmp_path):
    # type: (Path) -> None
    """Test initialization with explicit ndim parameter."""
    path = tmp_path / "ndim_index.lmdb"
    idx = LmdbSimprintIndex(str(path), ndim=128)
    assert idx.ndim == 128
    assert idx.simprint_bytes == 16
    idx.close()


def test_pack_chunk_pointer_valid(index):
    # type: (LmdbSimprintIndex) -> None
    """Test packing chunk pointer with valid inputs."""
    iscc_id_body = b"\x01\x02\x03\x04\x05\x06\x07\x08"
    offset = 1024
    size = 512

    packed = index._pack_chunk_pointer(iscc_id_body, offset, size)

    assert len(packed) == 16
    assert packed[:8] == iscc_id_body
    assert struct.unpack("!II", packed[8:16]) == (offset, size)


def test_pack_chunk_pointer_invalid_iscc_id_length(index):
    # type: (LmdbSimprintIndex) -> None
    """Test packing with incorrect ISCC-ID body length."""
    with pytest.raises(ValueError, match="ISCC-ID body must be 8 bytes"):
        index._pack_chunk_pointer(b"\x01\x02", 0, 0)


def test_pack_chunk_pointer_offset_exceeds_max(index):
    # type: (LmdbSimprintIndex) -> None
    """Test packing with offset exceeding maximum."""
    iscc_id_body = b"\x01\x02\x03\x04\x05\x06\x07\x08"
    with pytest.raises(ValueError, match="Offset .* exceeds max"):
        index._pack_chunk_pointer(iscc_id_body, 2**32, 0)


def test_pack_chunk_pointer_size_exceeds_max(index):
    # type: (LmdbSimprintIndex) -> None
    """Test packing with size exceeding maximum."""
    iscc_id_body = b"\x01\x02\x03\x04\x05\x06\x07\x08"
    with pytest.raises(ValueError, match="Size .* exceeds max"):
        index._pack_chunk_pointer(iscc_id_body, 0, 2**32)


def test_unpack_chunk_pointer_valid(index):
    # type: (LmdbSimprintIndex) -> None
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
    # type: (LmdbSimprintIndex) -> None
    """Test unpacking with incorrect data length."""
    with pytest.raises(ValueError, match="Expected 16 bytes"):
        index._unpack_chunk_pointer(b"\x01\x02\x03")


def test_add_raw_empty_list(index):
    # type: (LmdbSimprintIndex) -> None
    """Test adding empty entry list (no-op)."""
    index.add_raw([])
    assert len(index) == 0


def test_add_raw_single_entry_auto_detect(index):
    # type: (LmdbSimprintIndex) -> None
    """Test adding a single entry with simprints auto-detects ndim."""
    iscc_id = b"\x01\x02\x03\x04\x05\x06\x07\x08"
    simprints = [
        MockSimprintRaw(b"\x11\x12\x13\x14\x15\x16\x17\x18", 0, 256),
        MockSimprintRaw(b"\x21\x22\x23\x24\x25\x26\x27\x28", 256, 256),
    ]
    entry = MockSimprintEntryRaw(iscc_id, simprints)

    index.add_raw([entry])

    assert len(index) == 1
    assert iscc_id in index
    assert index.ndim == 64  # Auto-detected from 8-byte simprint
    assert index.simprint_bytes == 8


def test_add_raw_duplicate_ignored(index):
    # type: (LmdbSimprintIndex) -> None
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
    # type: (LmdbSimprintIndex) -> None
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
    # type: (LmdbSimprintIndex) -> None
    """Test batch with duplicate ISCC-IDs within the batch."""
    iscc_id = b"\x01\x02\x03\x04\x05\x06\x07\x08"
    entries = [
        MockSimprintEntryRaw(iscc_id, [MockSimprintRaw(b"\x11" * 8, 0, 256)]),
        MockSimprintEntryRaw(iscc_id, [MockSimprintRaw(b"\x22" * 8, 256, 256)]),
    ]

    index.add_raw(entries)
    assert len(index) == 1  # Only first occurrence counted


def test_add_raw_validates_simprint_length(tmp_path):
    # type: (Path) -> None
    """Test that adding simprints with wrong length raises ValueError."""
    path = tmp_path / "validate_index.lmdb"
    idx = LmdbSimprintIndex(str(path), ndim=64)

    iscc_id = b"\x01" * 8
    # Try to add 128-bit simprint to 64-bit index
    simprints = [MockSimprintRaw(b"\x11" * 16, 0, 256)]
    entry = MockSimprintEntryRaw(iscc_id, simprints)

    with pytest.raises(ValueError, match="Simprint length mismatch"):
        idx.add_raw([entry])

    idx.close()


def test_search_raw_empty_query(index):
    # type: (LmdbSimprintIndex) -> None
    """Test search with empty simprint list."""
    results = index.search_raw([])
    assert results == []


def test_search_raw_no_matches(index):
    # type: (LmdbSimprintIndex) -> None
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
    # type: (LmdbSimprintIndex) -> None
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
    # type: (LmdbSimprintIndex) -> None
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
    # type: (LmdbSimprintIndex) -> None
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
    # type: (LmdbSimprintIndex) -> None
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
    # type: (LmdbSimprintIndex) -> None
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
    # type: (LmdbSimprintIndex) -> None
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
    # type: (LmdbSimprintIndex) -> None
    """Test frequency-based scoring penalizes common simprints."""
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

    # Add 1 asset with both rare and common simprints
    mixed_id = b"\xfe" * 8
    entry = MockSimprintEntryRaw(
        mixed_id, [MockSimprintRaw(rare_simprint, 0, 256), MockSimprintRaw(common_simprint, 256, 256)]
    )
    index.add_raw([entry])

    # Search with both simprints - mixed asset should match both
    results = index.search_raw([rare_simprint, common_simprint], threshold=0.0, limit=20)

    # Find results
    mixed_result = next((r for r in results if r.iscc_id_body == mixed_id), None)
    rare_result = next((r for r in results if r.iscc_id_body == rare_id), None)
    common_results = [r for r in results if r.iscc_id_body in [bytes([i] * 8) for i in range(10)]]

    assert mixed_result is not None, f"Mixed asset {mixed_id.hex()} not found"
    assert rare_result is not None, f"Rare asset {rare_id.hex()} not found"
    assert len(common_results) == 10, "Should find all 10 common assets"

    # Verify scoring behavior:
    # - Mixed asset matches both: coverage=1.0, quality penalized by freq difference
    # - Rare-only asset matches one: coverage=0.5, quality=1.0 (single match)
    # - Common-only assets match one: coverage=0.5, quality=1.0 (single match)
    # All should have scores in (0, 1) range and none should exceed 1.0
    assert 0.0 < mixed_result.score <= 1.0, f"Mixed score {mixed_result.score} out of range"
    assert 0.0 < rare_result.score <= 1.0, f"Rare score {rare_result.score} out of range"
    for r in common_results:
        assert 0.0 < r.score <= 1.0, f"Common score {r.score} out of range"

    # Key property: scores must not exceed 1.0 (the original bug we're fixing)
    all_scores = [r.score for r in results]
    assert all(s <= 1.0 for s in all_scores), f"Found scores > 1.0: {[s for s in all_scores if s > 1.0]}"


def test_contains_existing_asset(index):
    # type: (LmdbSimprintIndex) -> None
    """Test __contains__ for existing asset."""
    iscc_id = b"\x01\x02\x03\x04\x05\x06\x07\x08"
    simprints = [MockSimprintRaw(b"\x11" * 8, 0, 256)]
    entry = MockSimprintEntryRaw(iscc_id, simprints)
    index.add_raw([entry])

    assert iscc_id in index


def test_contains_nonexistent_asset(index):
    # type: (LmdbSimprintIndex) -> None
    """Test __contains__ for non-existent asset."""
    iscc_id = b"\x99" * 8
    assert iscc_id not in index


def test_len_empty_index(index):
    # type: (LmdbSimprintIndex) -> None
    """Test __len__ on empty index."""
    assert len(index) == 0


def test_len_after_additions(index):
    # type: (LmdbSimprintIndex) -> None
    """Test __len__ after adding entries."""
    entries = [MockSimprintEntryRaw(bytes([i] * 8), [MockSimprintRaw(b"\x11" * 8, 0, 256)]) for i in range(5)]
    index.add_raw(entries)
    assert len(index) == 5


def test_context_manager(temp_index_path):
    # type: (Path) -> None
    """Test context manager protocol."""
    with LmdbSimprintIndex(str(temp_index_path)) as idx:
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
    idx1 = LmdbSimprintIndex(str(temp_index_path))
    idx1.add_raw([entry])
    idx1.close()

    # Reopen and verify data persisted
    idx2 = LmdbSimprintIndex(str(temp_index_path))
    assert len(idx2) == 1
    assert iscc_id in idx2
    assert idx2.ndim == 64  # Should be loaded from metadata
    idx2.close()


def test_calculate_idf_score_empty_matches(index):
    # type: (LmdbSimprintIndex) -> None
    """Test IDF score calculation with no matches."""
    score = index._calculate_idf_score([], {}, 5)
    assert score == 0.0


def test_calculate_idf_score_basic(index):
    # type: (LmdbSimprintIndex) -> None
    """Test basic IDF score calculation."""
    matches = [(b"\x11" * 8, b"\x11" * 8, 0, 256)]
    doc_frequencies = {b"\x11" * 8: 1}
    num_queried = 5

    score = index._calculate_idf_score(matches, doc_frequencies, num_queried)

    # Score should be between 0 and 1
    assert 0.0 <= score <= 1.0
    # With 1 match out of 5 queried, coverage = 1/5 = 0.2, quality = 1.0 (single match)
    # Expected score = 0.2 * 1.0 = 0.2
    assert score == 0.2


def test_format_match_result_with_freq(index):
    # type: (LmdbSimprintIndex) -> None
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
    # type: (LmdbSimprintIndex) -> None
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


def test_metadata_persistence(tmp_path):
    # type: (Path) -> None
    """Test that ndim and realm_id are persisted in LMDB metadata database."""
    import json
    import lmdb

    path = tmp_path / "metadata_index.lmdb"
    realm_id = b"\xaa\xbb"
    idx = LmdbSimprintIndex(str(path), ndim=128, realm_id=realm_id)
    idx.close()

    # Open LMDB and check metadata database
    env = lmdb.open(str(path), max_dbs=3, readonly=True, subdir=False)
    metadata_db = env.open_db(b"index_metadata")
    with env.begin() as txn:
        raw_data = txn.get(b"metadata", db=metadata_db)
        assert raw_data is not None
        metadata = json.loads(raw_data.decode("utf-8"))
        assert metadata["ndim"] == 128
        assert metadata["realm_id"] == "aabb"
    env.close()


def test_ndim_mismatch_raises_error(tmp_path):
    # type: (Path) -> None
    """Test that reopening with different ndim raises ValueError."""
    path = tmp_path / "mismatch_index.lmdb"

    # Create with ndim=64
    idx1 = LmdbSimprintIndex(str(path), ndim=64)
    idx1.close()

    # Try to reopen with ndim=128
    with pytest.raises(ValueError, match="Index has ndim=64 but constructor specified ndim=128"):
        LmdbSimprintIndex(str(path), ndim=128)


def test_128bit_simprints(tmp_path):
    # type: (Path) -> None
    """Test index with 128-bit simprints."""
    path = tmp_path / "index_128.lmdb"
    idx = LmdbSimprintIndex(str(path), ndim=128)

    # Add entry with 128-bit simprints (16 bytes)
    iscc_id = b"\x01" * 8
    simprint_128 = b"\x11" * 16
    simprints = [MockSimprintRaw(simprint_128, 0, 512)]
    entry = MockSimprintEntryRaw(iscc_id, simprints)

    idx.add_raw([entry])
    assert len(idx) == 1
    assert idx.ndim == 128
    assert idx.simprint_bytes == 16

    # Search with 128-bit simprint
    results = idx.search_raw([simprint_128])
    assert len(results) == 1
    assert results[0].iscc_id_body == iscc_id

    idx.close()


def test_256bit_simprints(tmp_path):
    # type: (Path) -> None
    """Test index with 256-bit simprints."""
    path = tmp_path / "index_256.lmdb"
    idx = LmdbSimprintIndex(str(path), ndim=256)

    # Add entry with 256-bit simprints (32 bytes)
    iscc_id = b"\x02" * 8
    simprint_256 = b"\x22" * 32
    simprints = [MockSimprintRaw(simprint_256, 0, 1024)]
    entry = MockSimprintEntryRaw(iscc_id, simprints)

    idx.add_raw([entry])
    assert len(idx) == 1
    assert idx.ndim == 256
    assert idx.simprint_bytes == 32

    # Search with 256-bit simprint
    results = idx.search_raw([simprint_256])
    assert len(results) == 1
    assert results[0].iscc_id_body == iscc_id

    idx.close()


def test_auto_detect_128bit(tmp_path):
    # type: (Path) -> None
    """Test auto-detection of 128-bit simprints."""
    path = tmp_path / "auto_128.lmdb"
    idx = LmdbSimprintIndex(str(path))

    # Add entry with 128-bit simprints without specifying ndim
    iscc_id = b"\x03" * 8
    simprint_128 = b"\x33" * 16
    simprints = [MockSimprintRaw(simprint_128, 0, 512)]
    entry = MockSimprintEntryRaw(iscc_id, simprints)

    idx.add_raw([entry])

    # Should auto-detect ndim=128
    assert idx.ndim == 128
    assert idx.simprint_bytes == 16

    # Verify search works
    results = idx.search_raw([simprint_128])
    assert len(results) == 1

    idx.close()


def test_different_lengths_different_indexes(tmp_path):
    # type: (Path) -> None
    """Test that different indexes can have different simprint lengths."""
    path_64 = tmp_path / "index_64.lmdb"
    path_128 = tmp_path / "index_128.lmdb"

    # Create 64-bit index
    idx_64 = LmdbSimprintIndex(str(path_64), ndim=64)
    iscc_id1 = b"\x01" * 8
    simprint_64 = b"\x11" * 8
    idx_64.add_raw([MockSimprintEntryRaw(iscc_id1, [MockSimprintRaw(simprint_64, 0, 256)])])

    # Create 128-bit index
    idx_128 = LmdbSimprintIndex(str(path_128), ndim=128)
    iscc_id2 = b"\x02" * 8
    simprint_128 = b"\x22" * 16
    idx_128.add_raw([MockSimprintEntryRaw(iscc_id2, [MockSimprintRaw(simprint_128, 0, 512)])])

    # Verify both work correctly
    assert idx_64.ndim == 64
    assert idx_128.ndim == 128

    results_64 = idx_64.search_raw([simprint_64])
    assert len(results_64) == 1

    results_128 = idx_128.search_raw([simprint_128])
    assert len(results_128) == 1

    idx_64.close()
    idx_128.close()


def test_persistence_128bit(tmp_path):
    # type: (Path) -> None
    """Test that 128-bit ndim persists across reopening."""
    path = tmp_path / "persist_128.lmdb"

    # Create with 128-bit
    idx1 = LmdbSimprintIndex(str(path), ndim=128)
    iscc_id = b"\x04" * 8
    simprint_128 = b"\x44" * 16
    idx1.add_raw([MockSimprintEntryRaw(iscc_id, [MockSimprintRaw(simprint_128, 0, 512)])])
    idx1.close()

    # Reopen without specifying ndim
    idx2 = LmdbSimprintIndex(str(path))
    assert idx2.ndim == 128
    assert idx2.simprint_bytes == 16

    # Verify data is still accessible
    results = idx2.search_raw([simprint_128])
    assert len(results) == 1
    assert results[0].iscc_id_body == iscc_id

    idx2.close()


def test_search_with_wrong_length_simprints(tmp_path):
    # type: (Path) -> None
    """Test that searching auto-truncates larger simprints and rejects smaller ones."""
    path = tmp_path / "search_validation.lmdb"
    idx = LmdbSimprintIndex(str(path), ndim=64)

    # Add 64-bit entry
    iscc_id = b"\x01" * 8
    idx.add_raw([MockSimprintEntryRaw(iscc_id, [MockSimprintRaw(b"\x11" * 8, 0, 256)])])

    # Search with 128-bit simprint - should auto-truncate to 64-bit
    results = idx.search_raw([b"\x22" * 16])
    # Search succeeds (truncates), but won't find matches since simprints don't match
    assert isinstance(results, list)

    # Search with 32-bit simprint - should raise error (too small)
    with pytest.raises(ValueError, match="Simprint too small"):
        idx.search_raw([b"\x33" * 4])

    idx.close()


def test_search_empty_index_no_ndim(tmp_path):
    # type: (Path) -> None
    """Test searching on empty index with no ndim configured."""
    path = tmp_path / "empty_search.lmdb"
    idx = LmdbSimprintIndex(str(path))

    # Search should work (ndim is None, so no validation)
    results = idx.search_raw([b"\x11" * 8])
    assert results == []

    idx.close()


def test_auto_detect_with_empty_simprints_list(tmp_path):
    # type: (Path) -> None
    """Test auto-detect fails when entry has no simprints."""
    path = tmp_path / "empty_simprints.lmdb"
    idx = LmdbSimprintIndex(str(path))

    # Try to add entry with empty simprints list
    iscc_id = b"\x01" * 8
    entry = MockSimprintEntryRaw(iscc_id, [])

    with pytest.raises(ValueError, match="Cannot auto-detect ndim"):
        idx.add_raw([entry])

    idx.close()


def test_load_metadata_without_ndim_key(tmp_path):
    # type: (Path) -> None
    """Test loading LMDB metadata without ndim key."""
    import json
    import lmdb

    path = tmp_path / "no_ndim_key.lmdb"

    # Pre-create LMDB with empty metadata (flat file with subdir=False)
    env = lmdb.open(str(path), max_dbs=3, subdir=False)
    metadata_db = env.open_db(b"index_metadata")
    with env.begin(write=True) as txn:
        # Store metadata without ndim key (using new "metadata" key)
        txn.put(b"metadata", json.dumps({}).encode("utf-8"), db=metadata_db)
    env.close()

    # Open index - should work, ndim remains None
    idx = LmdbSimprintIndex(str(path))
    assert idx.ndim is None
    assert idx.realm_id is None
    idx.close()


def test_realm_id_parameter(tmp_path):
    # type: (Path) -> None
    """Test initialization with realm_id parameter."""
    path = tmp_path / "realm_index.lmdb"
    realm_id = b"\xaa\xbb"
    idx = LmdbSimprintIndex(str(path), realm_id=realm_id)
    assert idx.realm_id == realm_id
    idx.close()


def test_realm_id_invalid_length(tmp_path):
    # type: (Path) -> None
    """Test that realm_id with wrong length raises ValueError."""
    path = tmp_path / "bad_realm.lmdb"
    with pytest.raises(ValueError, match="realm_id must be 2 bytes"):
        LmdbSimprintIndex(str(path), realm_id=b"\xaa")


def test_realm_id_mismatch_raises_error(tmp_path):
    # type: (Path) -> None
    """Test that reopening with different realm_id raises ValueError."""
    path = tmp_path / "realm_mismatch.lmdb"

    # Create with realm_id
    idx1 = LmdbSimprintIndex(str(path), realm_id=b"\xaa\xbb")
    idx1.close()

    # Try to reopen with different realm_id
    with pytest.raises(ValueError, match="Index has realm_id=aabb but constructor specified realm_id=ccdd"):
        LmdbSimprintIndex(str(path), realm_id=b"\xcc\xdd")


def test_realm_id_persistence(tmp_path):
    # type: (Path) -> None
    """Test that realm_id persists across reopening."""
    path = tmp_path / "realm_persist.lmdb"
    realm_id = b"\xaa\xbb"

    # Create with realm_id
    idx1 = LmdbSimprintIndex(str(path), ndim=64, realm_id=realm_id)
    idx1.close()

    # Reopen without specifying realm_id
    idx2 = LmdbSimprintIndex(str(path))
    assert idx2.realm_id == realm_id
    idx2.close()


def test_calculate_idf_score_duplicate_query_simprints(index):
    # type: (LmdbSimprintIndex) -> None
    """Test IDF score when same query simprint matches multiple times (line 472)."""
    # Scenario: query_simprint appears twice with different frequencies
    # This triggers the min() path on line 472
    query_sp = b"\x11" * 8
    match_sp1 = b"\xaa" * 8
    match_sp2 = b"\xbb" * 8

    # Same query simprint matches two different simprints with different frequencies
    matches = [
        (query_sp, match_sp1, 0, 256),  # First match
        (query_sp, match_sp2, 256, 256),  # Second match with same query_sp
    ]
    doc_frequencies = {
        match_sp1: 10,  # Common
        match_sp2: 2,  # Rare
    }
    num_queried = 1

    score = index._calculate_idf_score(matches, doc_frequencies, num_queried)

    # Should use minimum frequency (2) for this query simprint
    # Coverage = 1/1 = 1.0, Quality = 1.0 (single unique query)
    assert score == 1.0


def test_calculate_idf_score_same_frequencies(index):
    # type: (LmdbSimprintIndex) -> None
    """Test IDF score when multiple query simprints have identical frequencies (line 490)."""
    # Scenario: multiple different query simprints, but all match simprints with same frequency
    # This triggers min_freq == max_freq check on line 488-490
    query_sp1 = b"\x11" * 8
    query_sp2 = b"\x22" * 8
    match_sp1 = b"\xaa" * 8
    match_sp2 = b"\xbb" * 8

    matches = [
        (query_sp1, match_sp1, 0, 256),
        (query_sp2, match_sp2, 256, 256),
    ]
    # Both match simprints have identical frequency
    doc_frequencies = {
        match_sp1: 5,
        match_sp2: 5,
    }
    num_queried = 2

    score = index._calculate_idf_score(matches, doc_frequencies, num_queried)

    # Coverage = 2/2 = 1.0, Quality = 1.0 (all same frequency)
    assert score == 1.0
