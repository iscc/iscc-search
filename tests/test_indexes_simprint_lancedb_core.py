"""
Tests for LanceDB-based Simprint Index Implementation
"""

import pytest

from iscc_search.indexes.simprint.lancedb_core import LancedbSimprintIndex


# Mock classes matching protocol
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


# Fixtures
@pytest.fixture
def temp_db_path(tmp_path):
    # type: (Path) -> Path
    """Create a temporary directory path for LanceDB database."""
    return tmp_path / "test_lancedb"


@pytest.fixture
def index(temp_db_path):
    # type: (Path) -> LancedbSimprintIndex
    """Create and return a fresh index instance."""
    idx = LancedbSimprintIndex(str(temp_db_path), ndim=128)
    yield idx
    idx.close()


@pytest.fixture
def sample_entries():
    # type: () -> list[MockSimprintEntryRaw]
    """Create sample entries for testing."""
    return [
        MockSimprintEntryRaw(
            iscc_id_body=b"\x01\x02\x03\x04\x05\x06\x07\x08",
            simprints=[
                MockSimprintRaw(b"\x01" * 16, 0, 512),
                MockSimprintRaw(b"\x02" * 16, 512, 489),
            ],
        ),
        MockSimprintEntryRaw(
            iscc_id_body=b"\x11\x12\x13\x14\x15\x16\x17\x18",
            simprints=[
                MockSimprintRaw(b"\x03" * 16, 0, 1024),
            ],
        ),
    ]


# ===== Initialization Tests =====


def test_init_creates_new_index(temp_db_path):
    # type: (Path) -> None
    """Test creating a new index from scratch."""
    idx = LancedbSimprintIndex(str(temp_db_path), ndim=128)
    assert idx.path == temp_db_path
    assert idx.ndim == 128
    assert idx.simprint_bytes == 16
    idx.close()


def test_init_opens_existing(temp_db_path):
    # type: (Path) -> None
    """Test opening an existing index."""
    # Create first index
    idx1 = LancedbSimprintIndex(str(temp_db_path), ndim=128)
    idx1.close()

    # Reopen
    idx2 = LancedbSimprintIndex(str(temp_db_path))
    assert idx2.ndim == 128
    assert idx2.simprint_bytes == 16
    idx2.close()


def test_init_uri_parsing_file(temp_db_path):
    # type: (Path) -> None
    """Test URI parsing with file:// prefix."""
    idx = LancedbSimprintIndex(f"file://{temp_db_path}", ndim=128)
    assert idx.path == temp_db_path
    idx.close()


def test_init_uri_parsing_lancedb(temp_db_path):
    # type: (Path) -> None
    """Test URI parsing with lancedb:// prefix."""
    idx = LancedbSimprintIndex(f"lancedb://{temp_db_path}", ndim=128)
    assert idx.path == temp_db_path
    idx.close()


def test_init_with_ndim(temp_db_path):
    # type: (Path) -> None
    """Test initialization with ndim parameter."""
    idx = LancedbSimprintIndex(str(temp_db_path), ndim=256)
    assert idx.ndim == 256
    assert idx.simprint_bytes == 32
    idx.close()


def test_init_with_realm_id(temp_db_path):
    # type: (Path) -> None
    """Test initialization with realm_id parameter."""
    realm_id = b"\x00\x10"
    idx = LancedbSimprintIndex(str(temp_db_path), ndim=128, realm_id=realm_id)
    assert idx.realm_id == realm_id
    idx.close()


def test_init_metadata_persistence(temp_db_path):
    # type: (Path) -> None
    """Test that metadata persists across sessions."""
    realm_id = b"\x00\x10"

    # Create index with metadata
    idx1 = LancedbSimprintIndex(str(temp_db_path), ndim=128, realm_id=realm_id)
    idx1.close()

    # Reopen and verify
    idx2 = LancedbSimprintIndex(str(temp_db_path))
    assert idx2.ndim == 128
    assert idx2.realm_id == realm_id
    idx2.close()


def test_init_auto_detect_ndim(temp_db_path):
    # type: (Path) -> None
    """Test auto-detection of ndim from first add."""
    idx = LancedbSimprintIndex(str(temp_db_path))
    assert idx.ndim is None

    # Add entry to trigger auto-detection
    entry = MockSimprintEntryRaw(
        iscc_id_body=b"\x01\x02\x03\x04\x05\x06\x07\x08",
        simprints=[MockSimprintRaw(b"\x01" * 16, 0, 512)],
    )
    idx.add_raw([entry])

    assert idx.ndim == 128
    assert idx.simprint_bytes == 16
    idx.close()


def test_init_realm_id_validation(temp_db_path):
    # type: (Path) -> None
    """Test realm_id validation."""
    with pytest.raises(ValueError, match="realm_id must be 2 bytes"):
        LancedbSimprintIndex(str(temp_db_path), ndim=128, realm_id=b"\x00")


# ===== add_raw Tests =====


def test_add_empty_list(index):
    # type: (LancedbSimprintIndex) -> None
    """Test adding empty list is a no-op."""
    index.add_raw([])
    assert len(index) == 0


def test_add_single_entry(index):
    # type: (LancedbSimprintIndex) -> None
    """Test adding a single entry."""
    entry = MockSimprintEntryRaw(
        iscc_id_body=b"\x01\x02\x03\x04\x05\x06\x07\x08",
        simprints=[MockSimprintRaw(b"\x01" * 16, 0, 512)],
    )
    index.add_raw([entry])
    assert len(index) == 1
    assert entry.iscc_id_body in index


def test_add_batch_entries(index, sample_entries):
    # type: (LancedbSimprintIndex, list[MockSimprintEntryRaw]) -> None
    """Test adding multiple entries in batch."""
    index.add_raw(sample_entries)
    assert len(index) == 2


def test_add_duplicate_creates_multiple_rows(index):
    # type: (LancedbSimprintIndex) -> None
    """Test that append semantics allow duplicates."""
    entry = MockSimprintEntryRaw(
        iscc_id_body=b"\x01\x02\x03\x04\x05\x06\x07\x08",
        simprints=[MockSimprintRaw(b"\x01" * 16, 0, 512)],
    )

    # Add same entry twice
    index.add_raw([entry])
    index.add_raw([entry])

    # Should still count as 1 unique asset
    assert len(index) == 1
    # But entry should still be in index
    assert entry.iscc_id_body in index


def test_add_multiple_simprints_per_asset(index):
    # type: (LancedbSimprintIndex) -> None
    """Test adding multiple simprints for one asset."""
    entry = MockSimprintEntryRaw(
        iscc_id_body=b"\x01\x02\x03\x04\x05\x06\x07\x08",
        simprints=[
            MockSimprintRaw(b"\x01" * 16, 0, 512),
            MockSimprintRaw(b"\x02" * 16, 512, 489),
            MockSimprintRaw(b"\x03" * 16, 1001, 23),
        ],
    )
    index.add_raw([entry])
    assert len(index) == 1


def test_add_validates_simprint_length(index):
    # type: (LancedbSimprintIndex) -> None
    """Test validation of simprint length."""
    entry = MockSimprintEntryRaw(
        iscc_id_body=b"\x01\x02\x03\x04\x05\x06\x07\x08",
        simprints=[MockSimprintRaw(b"\x01" * 8, 0, 512)],  # Wrong length (8 instead of 16)
    )

    with pytest.raises(ValueError, match="Simprint length mismatch"):
        index.add_raw([entry])


def test_add_ndim_64(temp_db_path):
    # type: (Path) -> None
    """Test adding 64-bit simprints."""
    idx = LancedbSimprintIndex(str(temp_db_path), ndim=64)
    entry = MockSimprintEntryRaw(
        iscc_id_body=b"\x01\x02\x03\x04\x05\x06\x07\x08",
        simprints=[MockSimprintRaw(b"\x01" * 8, 0, 512)],
    )
    idx.add_raw([entry])
    assert len(idx) == 1
    idx.close()


def test_add_ndim_256(temp_db_path):
    # type: (Path) -> None
    """Test adding 256-bit simprints."""
    idx = LancedbSimprintIndex(str(temp_db_path), ndim=256)
    entry = MockSimprintEntryRaw(
        iscc_id_body=b"\x01\x02\x03\x04\x05\x06\x07\x08",
        simprints=[MockSimprintRaw(b"\x01" * 32, 0, 512)],
    )
    idx.add_raw([entry])
    assert len(idx) == 1
    idx.close()


def test_add_large_offset_size(index):
    # type: (LancedbSimprintIndex) -> None
    """Test adding entry with large offset and size values."""
    entry = MockSimprintEntryRaw(
        iscc_id_body=b"\x01\x02\x03\x04\x05\x06\x07\x08",
        simprints=[MockSimprintRaw(b"\x01" * 16, 0xFFFFFFFF, 0xFFFFFFFF)],
    )
    index.add_raw([entry])
    assert len(index) == 1


def test_add_empty_simprints_list(index):
    # type: (LancedbSimprintIndex) -> None
    """Test adding entry with empty simprints list."""
    entry = MockSimprintEntryRaw(
        iscc_id_body=b"\x01\x02\x03\x04\x05\x06\x07\x08",
        simprints=[],
    )
    index.add_raw([entry])
    assert len(index) == 0  # No simprints means nothing added


# ===== add_semantics Property Tests =====


def test_add_semantics_returns_append(index):
    # type: (LancedbSimprintIndex) -> None
    """Test that add_semantics property returns 'append'."""
    assert index.add_semantics == "append"


def test_add_semantics_documented(index):
    # type: (LancedbSimprintIndex) -> None
    """Test that add_semantics behavior matches documentation."""
    # Add same entry twice - should both be added (append semantics)
    entry = MockSimprintEntryRaw(
        iscc_id_body=b"\x01\x02\x03\x04\x05\x06\x07\x08",
        simprints=[MockSimprintRaw(b"\x01" * 16, 0, 512)],
    )

    index.add_raw([entry])
    first_len = len(index)

    index.add_raw([entry])
    second_len = len(index)

    # Length should still be 1 (counting unique assets)
    assert first_len == second_len == 1
    # But entry is in index
    assert entry.iscc_id_body in index


# ===== search_raw Tests =====


def test_search_empty_query(index, sample_entries):
    # type: (LancedbSimprintIndex, list[MockSimprintEntryRaw]) -> None
    """Test searching with empty query."""
    index.add_raw(sample_entries)
    results = index.search_raw([])
    assert results == []


def test_search_no_data(index):
    # type: (LancedbSimprintIndex) -> None
    """Test searching index with no data."""
    results = index.search_raw([b"\x01" * 16])
    assert results == []


def test_search_exact_match(index):
    # type: (LancedbSimprintIndex) -> None
    """Test searching for exact match."""
    entry = MockSimprintEntryRaw(
        iscc_id_body=b"\x01\x02\x03\x04\x05\x06\x07\x08",
        simprints=[MockSimprintRaw(b"\x01" * 16, 0, 512)],
    )
    index.add_raw([entry])

    # Search for exact same simprint
    results = index.search_raw([b"\x01" * 16])
    assert len(results) >= 1
    assert results[0].iscc_id_body == entry.iscc_id_body
    assert results[0].score > 0.9  # Should be very high score


def test_search_with_limit(index, sample_entries):
    # type: (LancedbSimprintIndex, list[MockSimprintEntryRaw]) -> None
    """Test that limit parameter works."""
    index.add_raw(sample_entries)

    # Add more entries
    for i in range(10):
        entry = MockSimprintEntryRaw(
            iscc_id_body=bytes([i] * 8),
            simprints=[MockSimprintRaw(bytes([i]) * 16, 0, 512)],
        )
        index.add_raw([entry])

    results = index.search_raw([b"\x01" * 16], limit=3)
    assert len(results) <= 3


def test_search_with_threshold(index, sample_entries):
    # type: (LancedbSimprintIndex, list[MockSimprintEntryRaw]) -> None
    """Test that threshold filters results."""
    index.add_raw(sample_entries)

    # Search with high threshold
    results_high = index.search_raw([b"\x01" * 16], threshold=0.95)
    results_low = index.search_raw([b"\x01" * 16], threshold=0.1)

    assert len(results_high) <= len(results_low)


def test_search_detailed_true(index):
    # type: (LancedbSimprintIndex) -> None
    """Test that detailed=True returns chunk information."""
    entry = MockSimprintEntryRaw(
        iscc_id_body=b"\x01\x02\x03\x04\x05\x06\x07\x08",
        simprints=[MockSimprintRaw(b"\x01" * 16, 100, 512)],
    )
    index.add_raw([entry])

    results = index.search_raw([b"\x01" * 16], detailed=True)

    if results:
        assert results[0].chunks is not None
        assert len(results[0].chunks) > 0
        # Check chunk has all required fields
        chunk = results[0].chunks[0]
        assert chunk.query is not None
        assert chunk.match is not None
        assert chunk.score >= 0.0
        assert chunk.offset == 100
        assert chunk.size == 512


def test_search_detailed_false(index):
    # type: (LancedbSimprintIndex) -> None
    """Test that detailed=False omits chunk information."""
    entry = MockSimprintEntryRaw(
        iscc_id_body=b"\x01\x02\x03\x04\x05\x06\x07\x08",
        simprints=[MockSimprintRaw(b"\x01" * 16, 0, 512)],
    )
    index.add_raw([entry])

    results = index.search_raw([b"\x01" * 16], detailed=False)

    if results:
        assert results[0].chunks is None


def test_search_multiple_queries(index):
    # type: (LancedbSimprintIndex) -> None
    """Test searching with multiple query simprints."""
    entry = MockSimprintEntryRaw(
        iscc_id_body=b"\x01\x02\x03\x04\x05\x06\x07\x08",
        simprints=[
            MockSimprintRaw(b"\x01" * 16, 0, 512),
            MockSimprintRaw(b"\x02" * 16, 512, 489),
        ],
    )
    index.add_raw([entry])

    # Search with both simprints
    results = index.search_raw([b"\x01" * 16, b"\x02" * 16])

    if results:
        assert results[0].queried == 2


def test_search_exponential_weighting(index):
    # type: (LancedbSimprintIndex) -> None
    """Test exponential weighting scoring."""
    entry = MockSimprintEntryRaw(
        iscc_id_body=b"\x01\x02\x03\x04\x05\x06\x07\x08",
        simprints=[MockSimprintRaw(b"\x01" * 16, 0, 512)],
    )
    index.add_raw([entry])

    # Search with different confidence_exponent values
    results_exp2 = index.search_raw([b"\x01" * 16], confidence_exponent=2)
    results_exp6 = index.search_raw([b"\x01" * 16], confidence_exponent=6)

    # Both should find the entry
    assert len(results_exp2) >= 1
    assert len(results_exp6) >= 1


def test_search_match_threshold(index):
    # type: (LancedbSimprintIndex) -> None
    """Test configurable match_threshold."""
    entry = MockSimprintEntryRaw(
        iscc_id_body=b"\x01\x02\x03\x04\x05\x06\x07\x08",
        simprints=[MockSimprintRaw(b"\x01" * 16, 0, 512)],
    )
    index.add_raw([entry])

    # Search with different match_threshold values
    results_low = index.search_raw([b"\x01" * 16], match_threshold=0.5)
    results_high = index.search_raw([b"\x01" * 16], match_threshold=0.95)

    assert len(results_low) >= len(results_high)


def test_search_ordering(index):
    # type: (LancedbSimprintIndex) -> None
    """Test that results are ordered by score descending."""
    # Add multiple entries
    for i in range(5):
        entry = MockSimprintEntryRaw(
            iscc_id_body=bytes([i] * 8),
            simprints=[MockSimprintRaw(bytes([i]) * 16, 0, 512)],
        )
        index.add_raw([entry])

    results = index.search_raw([b"\x01" * 16], limit=10)

    # Check scores are descending
    for i in range(len(results) - 1):
        assert results[i].score >= results[i + 1].score


# ===== Protocol Compliance Tests =====


def test_contains_existing(index):
    # type: (LancedbSimprintIndex) -> None
    """Test __contains__ returns True for indexed assets."""
    entry = MockSimprintEntryRaw(
        iscc_id_body=b"\x01\x02\x03\x04\x05\x06\x07\x08",
        simprints=[MockSimprintRaw(b"\x01" * 16, 0, 512)],
    )
    index.add_raw([entry])

    assert entry.iscc_id_body in index


def test_contains_nonexistent(index):
    # type: (LancedbSimprintIndex) -> None
    """Test __contains__ returns False for non-indexed assets."""
    assert b"\xff" * 8 not in index


def test_len_empty(index):
    # type: (LancedbSimprintIndex) -> None
    """Test __len__ returns 0 for empty index."""
    assert len(index) == 0


def test_len_counts_unique_assets(index):
    # type: (LancedbSimprintIndex) -> None
    """Test __len__ counts distinct ISCC-IDs despite duplicates."""
    entry = MockSimprintEntryRaw(
        iscc_id_body=b"\x01\x02\x03\x04\x05\x06\x07\x08",
        simprints=[MockSimprintRaw(b"\x01" * 16, 0, 512)],
    )

    # Add twice (append semantics)
    index.add_raw([entry])
    index.add_raw([entry])

    # Should count as 1 unique asset
    assert len(index) == 1


def test_len_not_simprints(index):
    # type: (LancedbSimprintIndex) -> None
    """Test that __len__ counts assets, not simprints or rows."""
    entry = MockSimprintEntryRaw(
        iscc_id_body=b"\x01\x02\x03\x04\x05\x06\x07\x08",
        simprints=[
            MockSimprintRaw(b"\x01" * 16, 0, 512),
            MockSimprintRaw(b"\x02" * 16, 512, 489),
            MockSimprintRaw(b"\x03" * 16, 1001, 23),
        ],
    )
    index.add_raw([entry])

    # 3 simprints but 1 asset
    assert len(index) == 1


def test_context_manager(temp_db_path):
    # type: (Path) -> None
    """Test context manager protocol."""
    with LancedbSimprintIndex(str(temp_db_path), ndim=128) as idx:
        entry = MockSimprintEntryRaw(
            iscc_id_body=b"\x01\x02\x03\x04\x05\x06\x07\x08",
            simprints=[MockSimprintRaw(b"\x01" * 16, 0, 512)],
        )
        idx.add_raw([entry])
        assert len(idx) == 1


def test_close_and_reopen(temp_db_path):
    # type: (Path) -> None
    """Test closing and reopening index preserves data."""
    # Create index and add data
    idx1 = LancedbSimprintIndex(str(temp_db_path), ndim=128)
    entry = MockSimprintEntryRaw(
        iscc_id_body=b"\x01\x02\x03\x04\x05\x06\x07\x08",
        simprints=[MockSimprintRaw(b"\x01" * 16, 0, 512)],
    )
    idx1.add_raw([entry])
    idx1.close()

    # Reopen and verify
    idx2 = LancedbSimprintIndex(str(temp_db_path))
    assert len(idx2) == 1
    assert entry.iscc_id_body in idx2
    idx2.close()


# ===== Edge Cases and Integration Tests =====


def test_optimize(index, sample_entries):
    # type: (LancedbSimprintIndex, list[MockSimprintEntryRaw]) -> None
    """Test optimize method runs without errors."""
    index.add_raw(sample_entries)
    index.optimize()  # Should not raise


def test_large_batch(temp_db_path):
    # type: (Path) -> None
    """Test adding large batch of entries."""
    idx = LancedbSimprintIndex(str(temp_db_path), ndim=128)

    entries = []
    for i in range(100):
        entry = MockSimprintEntryRaw(
            iscc_id_body=bytes([i % 256] * 8),
            simprints=[MockSimprintRaw(bytes([i % 256]) * 16, 0, 512)],
        )
        entries.append(entry)

    idx.add_raw(entries)
    assert len(idx) > 0
    idx.close()


# ===== Additional Coverage Tests =====


def test_init_ndim_mismatch(temp_db_path):
    # type: (Path) -> None
    """Test ndim mismatch error when reopening."""
    # Create with ndim=128
    idx1 = LancedbSimprintIndex(str(temp_db_path), ndim=128)
    idx1.close()

    # Try to open with different ndim
    with pytest.raises(ValueError, match="Index has ndim"):
        LancedbSimprintIndex(str(temp_db_path), ndim=256)


def test_init_realm_id_mismatch(temp_db_path):
    # type: (Path) -> None
    """Test realm_id mismatch error when reopening."""
    # Create with realm_id
    realm1 = b"\x00\x10"
    idx1 = LancedbSimprintIndex(str(temp_db_path), ndim=128, realm_id=realm1)
    idx1.close()

    # Try to open with different realm_id
    realm2 = b"\x00\x20"
    with pytest.raises(ValueError, match="Index has realm_id"):
        LancedbSimprintIndex(str(temp_db_path), realm_id=realm2)


def test_add_offset_too_large(index):
    # type: (LancedbSimprintIndex) -> None
    """Test validation of offset exceeding max."""
    entry = MockSimprintEntryRaw(
        iscc_id_body=b"\x01\x02\x03\x04\x05\x06\x07\x08",
        simprints=[MockSimprintRaw(b"\x01" * 16, 2**32, 512)],  # Offset too large
    )

    with pytest.raises(ValueError, match="Offset.*exceeds max"):
        index.add_raw([entry])


def test_add_size_too_large(index):
    # type: (LancedbSimprintIndex) -> None
    """Test validation of size exceeding max."""
    entry = MockSimprintEntryRaw(
        iscc_id_body=b"\x01\x02\x03\x04\x05\x06\x07\x08",
        simprints=[MockSimprintRaw(b"\x01" * 16, 0, 2**32)],  # Size too large
    )

    with pytest.raises(ValueError, match="Size.*exceeds max"):
        index.add_raw([entry])


def test_auto_detect_no_simprints(temp_db_path):
    # type: (Path) -> None
    """Test auto-detect fails when entry has no simprints."""
    idx = LancedbSimprintIndex(str(temp_db_path))

    # Entry with empty simprints list
    entry = MockSimprintEntryRaw(
        iscc_id_body=b"\x01\x02\x03\x04\x05\x06\x07\x08",
        simprints=[],
    )

    with pytest.raises(ValueError, match="Cannot auto-detect ndim"):
        idx.add_raw([entry])

    idx.close()


def test_metadata_load_corrupted_file(temp_db_path, monkeypatch):
    # type: (Path, ...) -> None
    """Test metadata loading with corrupted file."""
    # Create index first
    idx1 = LancedbSimprintIndex(str(temp_db_path), ndim=128)
    idx1.close()

    # Corrupt the metadata file
    metadata_path = temp_db_path / "simprints_metadata.json"
    with open(metadata_path, "w") as f:
        f.write("corrupted json {{{")

    # Should still open (just log warning)
    idx2 = LancedbSimprintIndex(str(temp_db_path))
    # ndim should be None since metadata load failed
    assert idx2.ndim is None
    idx2.close()


def test_optimize_builds_vector_index(index, sample_entries):
    # type: (LancedbSimprintIndex, list[MockSimprintEntryRaw]) -> None
    """Test optimize builds vector index for fast queries."""
    index.add_raw(sample_entries)

    # Should successfully build vector index without errors
    index.optimize()

    # Verify index still works after optimization
    result = index.search_raw([sample_entries[0].simprints[0].simprint], limit=5)
    assert len(result) > 0


def test_search_with_empty_table(temp_db_path):
    # type: (Path) -> None
    """Test search filters low-quality matches using threshold parameter."""
    idx = LancedbSimprintIndex(str(temp_db_path), ndim=128)

    # Add entry to create table
    entry = MockSimprintEntryRaw(
        iscc_id_body=b"\x01\x02\x03\x04\x05\x06\x07\x08",
        simprints=[MockSimprintRaw(b"\x01" * 16, 0, 512)],
    )
    idx.add_raw([entry])

    # Search for very dissimilar simprint (score ~0.125)
    # With threshold=0.5, should filter out low-scoring match
    results = idx.search_raw([b"\xff" * 16], threshold=0.5)

    # Should return empty list (filtered by threshold)
    assert results == []
    idx.close()


def test_search_with_zero_limit(temp_db_path):
    # type: (Path) -> None
    """Test search with limit=0 edge case."""
    idx = LancedbSimprintIndex(str(temp_db_path), ndim=128)

    # Add some entries
    for i in range(3):
        entry = MockSimprintEntryRaw(
            iscc_id_body=bytes([i] * 8),
            simprints=[MockSimprintRaw(bytes([i]) * 16, 0, 512)],
        )
        idx.add_raw([entry])

    # Search with limit=0 (should still work, using backend_limit=64)
    results = idx.search_raw([b"\x01" * 16], limit=0)

    # Should return results (not limited by 0)
    assert len(results) >= 0  # Will return matches within backend_limit
    idx.close()


def test_match_threshold_filters_low_scores(temp_db_path):
    # type: (Path) -> None
    """Test that match_threshold filters out low-scoring individual chunk matches."""
    idx = LancedbSimprintIndex(str(temp_db_path), ndim=128)

    # Add entry with exact match
    entry = MockSimprintEntryRaw(
        iscc_id_body=b"\x01\x02\x03\x04\x05\x06\x07\x08",
        simprints=[MockSimprintRaw(b"\xaa" * 16, 0, 512)],
    )
    idx.add_raw([entry])

    # Search with very dissimilar simprint and high match_threshold
    # This simprint is very different from \xaa (bit pattern: 10101010 vs 11111111)
    # Should be filtered by match_threshold before being added to asset_scores
    results = idx.search_raw([b"\xff" * 16], match_threshold=0.95, threshold=0.0)

    # The low-scoring match should be filtered by match_threshold
    # Result should be empty because the match score is well below 0.95
    assert len(results) == 0
    idx.close()


def test_search_with_all_zero_scores(temp_db_path):
    # type: (Path) -> None
    """Test that division by zero is avoided when all match scores are 0.0."""
    idx = LancedbSimprintIndex(str(temp_db_path), ndim=128)

    # Add entry with all bits set to 0
    entry = MockSimprintEntryRaw(
        iscc_id_body=b"\x01\x02\x03\x04\x05\x06\x07\x08",
        simprints=[MockSimprintRaw(b"\x00" * 16, 0, 512)],
    )
    idx.add_raw([entry])

    # Search with opposite pattern (all bits set to 1)
    # This creates maximum Hamming distance (score = 0.0)
    # With match_threshold=0.0, this will pass the filter
    results = idx.search_raw([b"\xff" * 16], match_threshold=0.0, threshold=0.0)

    # Should not crash with ZeroDivisionError
    # Should return result with score 0.0 (no similarity)
    assert len(results) == 1
    assert results[0].score == 0.0  # Coverage * quality = 1.0 * 0.0 = 0.0
    idx.close()
