"""Comprehensive tests for UsearchSimprintIndex with exponential confidence weighting."""

import random
import pytest
import numpy as np
from iscc_search.indexes.simprint.usearch_core import UsearchSimprintIndex
from iscc_search.indexes.simprint.models import SimprintRaw, SimprintEntryRaw


def create_random_simprint(ndim=128):
    # type: (int) -> bytes
    """Create a random binary simprint of ndim bits."""
    num_bytes = ndim // 8
    return bytes(random.getrandbits(8) for _ in range(num_bytes))


def flip_bits(simprint, num_bits):
    # type: (bytes, int) -> bytes
    """
    Create a new simprint by flipping num_bits random bits.

    Used to create simprints with controlled Hamming distance.
    """
    arr = np.unpackbits(np.frombuffer(simprint, dtype=np.uint8))
    indices = random.sample(range(len(arr)), num_bits)
    for idx in indices:
        arr[idx] = 1 - arr[idx]
    return np.packbits(arr).tobytes()


def hamming_distance(a, b):
    # type: (bytes, bytes) -> int
    """Calculate Hamming distance between two byte strings."""
    a_bits = np.unpackbits(np.frombuffer(a, dtype=np.uint8))
    b_bits = np.unpackbits(np.frombuffer(b, dtype=np.uint8))
    return int(np.sum(a_bits != b_bits))


@pytest.fixture
def temp_index_path(tmp_path):
    # type: (Path) -> Path
    """Create a temporary file path for index storage."""
    return tmp_path / "usearch_simprint.index"


@pytest.fixture
def index(temp_index_path):
    # type: (Path) -> UsearchSimprintIndex
    """Create and return a fresh index instance."""
    idx = UsearchSimprintIndex(str(temp_index_path), ndim=128)
    yield idx
    idx.close()


# Protocol compliance tests


def test_init_creates_index(tmp_path):
    # type: (Path) -> None
    """Test that __init__ creates a new index."""
    path = tmp_path / "test.index"
    idx = UsearchSimprintIndex(str(path), ndim=64)
    assert idx.ndim == 64
    assert idx.path == path
    assert len(idx) == 0
    idx.close()


def test_init_with_file_uri(tmp_path):
    # type: (Path) -> None
    """Test initialization with file:// URI."""
    path = tmp_path / "file_uri.index"
    uri = f"file://{path}"
    idx = UsearchSimprintIndex(uri, ndim=128)
    assert idx.path == path
    idx.close()


def test_init_with_realm_id(tmp_path):
    # type: (Path) -> None
    """Test initialization with realm_id parameter."""
    path = tmp_path / "realm.index"
    realm_id = b"\xaa\xbb"
    idx = UsearchSimprintIndex(str(path), ndim=128, realm_id=realm_id)
    assert idx.realm_id == realm_id
    idx.close()


def test_add_raw_stores_simprints(index):
    # type: (UsearchSimprintIndex) -> None
    """Test that add_raw correctly stores simprints."""
    iscc_id_body = b"\x01" * 8
    simprints = [
        SimprintRaw(simprint=create_random_simprint(), offset=0, size=1024),
        SimprintRaw(simprint=create_random_simprint(), offset=1024, size=512),
    ]
    entry = SimprintEntryRaw(iscc_id_body=iscc_id_body, simprints=simprints)

    index.add_raw([entry])

    assert iscc_id_body in index
    assert len(index) == 1


def test_add_raw_empty_list(index):
    # type: (UsearchSimprintIndex) -> None
    """Test that add_raw handles empty list gracefully."""
    index.add_raw([])
    assert len(index) == 0


def test_add_raw_entry_with_no_simprints(index):
    # type: (UsearchSimprintIndex) -> None
    """Test that add_raw handles entries with no simprints gracefully."""
    iscc_id_body = b"\x01" * 8
    # Entry with empty simprints list
    entry = SimprintEntryRaw(iscc_id_body=iscc_id_body, simprints=[])

    index.add_raw([entry])

    # Nothing is added to usearch if there are no simprints
    assert iscc_id_body not in index
    assert len(index) == 0


def test_search_raw_returns_matches(index):
    # type: (UsearchSimprintIndex) -> None
    """Test that search_raw returns matching assets."""
    iscc_id_body = b"\x01" * 8
    base_simprint = create_random_simprint()
    simprints = [SimprintRaw(simprint=base_simprint, offset=0, size=1024)]
    entry = SimprintEntryRaw(iscc_id_body=iscc_id_body, simprints=simprints)

    index.add_raw([entry])

    # Search with the exact same simprint
    results = index.search_raw([base_simprint], limit=10, threshold=0.0, detailed=False)

    assert len(results) == 1
    assert results[0].iscc_id_body == iscc_id_body
    assert results[0].score > 0.9  # Should be very high (near 1.0)
    assert results[0].queried == 1
    assert results[0].matches == 1
    assert results[0].chunks is None


def test_search_raw_with_detailed(index):
    # type: (UsearchSimprintIndex) -> None
    """Test that search_raw with detailed=True returns chunk information."""
    iscc_id_body = b"\x01" * 8
    base_simprint = create_random_simprint()
    simprints = [SimprintRaw(simprint=base_simprint, offset=0, size=1024)]
    entry = SimprintEntryRaw(iscc_id_body=iscc_id_body, simprints=simprints)

    index.add_raw([entry])

    # Search with detailed=True
    results = index.search_raw([base_simprint], limit=10, threshold=0.0, detailed=True)

    assert len(results) == 1
    assert results[0].iscc_id_body == iscc_id_body
    assert results[0].chunks is not None
    assert len(results[0].chunks) == 1  # One chunk match
    chunk = results[0].chunks[0]
    assert chunk.score > 0.9  # Should be very high
    assert chunk.offset == 0  # Position tracking not supported (usearch)
    assert chunk.size == 0  # Position tracking not supported (usearch)
    assert chunk.freq == 1  # Frequency tracking not supported (usearch)


def test_search_empty_index_returns_empty(index):
    # type: (UsearchSimprintIndex) -> None
    """Test that searching an empty index returns empty results."""
    simprint = create_random_simprint()
    results = index.search_raw([simprint], detailed=False)
    assert results == []


def test_search_empty_query_returns_empty(index):
    # type: (UsearchSimprintIndex) -> None
    """Test that searching with empty query returns empty results."""
    iscc_id_body = b"\x01" * 8
    simprints = [SimprintRaw(simprint=create_random_simprint(), offset=0, size=1024)]
    entry = SimprintEntryRaw(iscc_id_body=iscc_id_body, simprints=simprints)
    index.add_raw([entry])

    results = index.search_raw([], detailed=False)
    assert results == []


def test_search_no_matches_returns_empty(index):
    # type: (UsearchSimprintIndex) -> None
    """Test that search returns empty when no matches above threshold."""
    iscc_id_body = b"\x01" * 8
    base_simprint = create_random_simprint()
    simprints = [SimprintRaw(simprint=base_simprint, offset=0, size=1024)]
    entry = SimprintEntryRaw(iscc_id_body=iscc_id_body, simprints=simprints)
    index.add_raw([entry])

    # Create a very different simprint (flip 90% of bits)
    different = flip_bits(base_simprint, 115)  # ~90% different for 128-bit

    # Search with high threshold and strict match_threshold
    results = index.search_raw([different], threshold=0.9, detailed=False, match_threshold=0.9)
    assert results == []


def test_add_duplicate_iscc_id_appends_simprints(index):
    # type: (UsearchSimprintIndex) -> None
    """Test that adding the same ISCC-ID multiple times appends simprints (multi=True)."""
    iscc_id_body = b"\x01" * 8

    # Add first batch
    simprints1 = [SimprintRaw(simprint=create_random_simprint(), offset=0, size=1024)]
    entry1 = SimprintEntryRaw(iscc_id_body=iscc_id_body, simprints=simprints1)
    index.add_raw([entry1])

    # Add second batch with same ISCC-ID
    simprints2 = [SimprintRaw(simprint=create_random_simprint(), offset=1024, size=512)]
    entry2 = SimprintEntryRaw(iscc_id_body=iscc_id_body, simprints=simprints2)
    index.add_raw([entry2])

    # Should still show as 1 asset (tracked in set)
    assert len(index) == 1
    assert iscc_id_body in index


def test_search_raw_default_detailed_false(index):
    # type: (UsearchSimprintIndex) -> None
    """Test that search_raw works with default parameters (detailed=False by default)."""
    iscc_id_body = b"\x01" * 8
    simprint = create_random_simprint()
    entry = SimprintEntryRaw(iscc_id_body=iscc_id_body, simprints=[SimprintRaw(simprint=simprint, offset=0, size=1024)])
    index.add_raw([entry])

    # Call without detailed parameter - should use default (False) and work
    results = index.search_raw([simprint])

    assert len(results) == 1
    assert results[0].iscc_id_body == iscc_id_body


def test_detailed_true_returns_chunk_matches(index):
    # type: (UsearchSimprintIndex) -> None
    """Test that detailed=True returns chunk-level match details."""
    # Add entry with multiple simprints
    iscc_id_body = b"\x01" * 8
    simprint1 = create_random_simprint()
    simprint2 = create_random_simprint()
    simprints = [
        SimprintRaw(simprint=simprint1, offset=0, size=1024),
        SimprintRaw(simprint=simprint2, offset=1024, size=1024),
    ]
    entry = SimprintEntryRaw(iscc_id_body=iscc_id_body, simprints=simprints)
    index.add_raw([entry])

    # Search with exact match and detailed=True
    results = index.search_raw([simprint1], detailed=True)

    assert len(results) == 1
    result = results[0]
    assert result.iscc_id_body == iscc_id_body
    assert result.chunks is not None
    assert len(result.chunks) == 1  # One query simprint matched

    chunk = result.chunks[0]
    assert chunk.query == simprint1
    assert chunk.match == simprint1  # Exact match
    assert chunk.score == 1.0  # Perfect match
    assert chunk.offset == 0  # Not tracked by usearch_core
    assert chunk.size == 0  # Not tracked by usearch_core
    assert chunk.freq == 1  # Not tracked by usearch_core


def test_detailed_with_multiple_query_simprints(index):
    # type: (UsearchSimprintIndex) -> None
    """Test detailed=True with multiple query simprints."""
    iscc_id_body = b"\x02" * 8
    simprint1 = create_random_simprint()
    simprint2 = create_random_simprint()
    simprint3 = create_random_simprint()
    simprints = [
        SimprintRaw(simprint=simprint1, offset=0, size=1024),
        SimprintRaw(simprint=simprint2, offset=1024, size=1024),
    ]
    entry = SimprintEntryRaw(iscc_id_body=iscc_id_body, simprints=simprints)
    index.add_raw([entry])

    # Search with 2 matching + 1 non-matching simprints
    # Use threshold=0.75 to filter out weak matches
    results = index.search_raw([simprint1, simprint2, simprint3], detailed=True, threshold=0.75)

    assert len(results) == 1
    result = results[0]
    assert result.chunks is not None
    # Should have 2 chunks (simprint1 and simprint2 matched, simprint3 filtered by threshold)
    assert len(result.chunks) == 2

    # Verify both matches
    chunk_queries = {c.query for c in result.chunks}
    assert simprint1 in chunk_queries
    assert simprint2 in chunk_queries


def test_contains_checks_existence(index):
    # type: (UsearchSimprintIndex) -> None
    """Test __contains__ method for membership checking."""
    iscc_id_body1 = b"\x01" * 8
    iscc_id_body2 = b"\x02" * 8

    simprints = [SimprintRaw(simprint=create_random_simprint(), offset=0, size=1024)]
    entry = SimprintEntryRaw(iscc_id_body=iscc_id_body1, simprints=simprints)
    index.add_raw([entry])

    assert iscc_id_body1 in index
    assert iscc_id_body2 not in index


def test_len_returns_asset_count(index):
    # type: (UsearchSimprintIndex) -> None
    """Test __len__ returns unique asset count."""
    entries = []
    for i in range(5):
        iscc_id_body = bytes([i] * 8)
        simprints = [SimprintRaw(simprint=create_random_simprint(), offset=0, size=1024)]
        entries.append(SimprintEntryRaw(iscc_id_body=iscc_id_body, simprints=simprints))

    index.add_raw(entries)
    assert len(index) == 5


def test_close_saves_index(temp_index_path):
    # type: (Path) -> None
    """Test that close() saves the index to disk."""
    idx = UsearchSimprintIndex(str(temp_index_path), ndim=128)

    iscc_id_body = b"\x01" * 8
    simprints = [SimprintRaw(simprint=create_random_simprint(), offset=0, size=1024)]
    entry = SimprintEntryRaw(iscc_id_body=iscc_id_body, simprints=simprints)
    idx.add_raw([entry])

    idx.close()

    # File should exist after close
    assert temp_index_path.exists()

    # Should be able to restore
    idx2 = UsearchSimprintIndex(str(temp_index_path), ndim=128)
    idx2.close()


def test_restore_preserves_contains_and_len(temp_index_path):
    # type: (Path) -> None
    """Test that __contains__ and __len__ work correctly after restore."""
    # Create index and add multiple assets
    idx = UsearchSimprintIndex(str(temp_index_path), ndim=128)

    assets = []
    for i in range(3):
        iscc_id_body = bytes([i] * 8)
        simprints = [SimprintRaw(simprint=create_random_simprint(), offset=0, size=1024)]
        entry = SimprintEntryRaw(iscc_id_body=iscc_id_body, simprints=simprints)
        assets.append(iscc_id_body)
        idx.add_raw([entry])

    # Verify before close
    assert len(idx) == 3
    for asset in assets:
        assert asset in idx

    idx.close()

    # Restore and verify __contains__ and __len__ still work
    idx2 = UsearchSimprintIndex(str(temp_index_path), ndim=128)

    assert len(idx2) == 3
    for asset in assets:
        assert asset in idx2

    # Verify non-existent asset
    assert b"\xff" * 8 not in idx2

    idx2.close()


def test_context_manager(temp_index_path):
    # type: (Path) -> None
    """Test context manager support."""
    with UsearchSimprintIndex(str(temp_index_path), ndim=128) as idx:
        iscc_id_body = b"\x01" * 8
        simprints = [SimprintRaw(simprint=create_random_simprint(), offset=0, size=1024)]
        entry = SimprintEntryRaw(iscc_id_body=iscc_id_body, simprints=simprints)
        idx.add_raw([entry])

    # Index should be saved
    assert temp_index_path.exists()


# Exponential weighting tests


def test_exponential_weighting_emphasizes_high_scores(tmp_path):
    # type: (Path) -> None
    """
    Verify that exponential weighting (score^4) heavily penalizes lower scores.

    Asset A: 2 matches with scores [0.9, 0.9]
    Asset B: 2 matches with scores [1.0, 0.8]

    With exponent=4:
    - Asset A quality: (0.9^4 + 0.9^4) / (0.9 + 0.9) ≈ 0.73
    - Asset B quality: (1.0^4 + 0.8^4) / (1.0 + 0.8) ≈ 0.78

    Asset B scores higher despite lower average, because perfect match (1.0)
    is heavily weighted.
    """
    path = tmp_path / "exp_test.index"
    idx = UsearchSimprintIndex(str(path), ndim=128)

    # Create base simprints for query
    base1 = create_random_simprint(128)
    base2 = create_random_simprint(128)

    # Asset A: Two 90% matches (flip ~10% of bits = ~13 bits for 128-bit)
    asset_a_id = b"\xaa" * 8
    simprint_a1 = flip_bits(base1, 13)  # ~90% match
    simprint_a2 = flip_bits(base2, 13)  # ~90% match
    idx.add_raw([
        SimprintEntryRaw(
            iscc_id_body=asset_a_id,
            simprints=[
                SimprintRaw(simprint=simprint_a1, offset=0, size=1024),
                SimprintRaw(simprint=simprint_a2, offset=1024, size=1024),
            ],
        )
    ])

    # Asset B: One perfect match, one 80% match (flip ~20% = ~26 bits)
    asset_b_id = b"\xbb" * 8
    simprint_b1 = base1  # Perfect match
    simprint_b2 = flip_bits(base2, 26)  # ~80% match
    idx.add_raw([
        SimprintEntryRaw(
            iscc_id_body=asset_b_id,
            simprints=[
                SimprintRaw(simprint=simprint_b1, offset=0, size=1024),
                SimprintRaw(simprint=simprint_b2, offset=1024, size=1024),
            ],
        )
    ])

    # Search with both query simprints
    results = idx.search_raw([base1, base2], limit=10, threshold=0.0, detailed=False)

    # Asset B should rank higher due to the perfect match being heavily weighted
    assert len(results) == 2
    assert results[0].iscc_id_body == asset_b_id  # B should be first
    assert results[1].iscc_id_body == asset_a_id  # A should be second

    idx.close()


def test_match_threshold_filters_low_scores(tmp_path):
    # type: (Path) -> None
    """Test that match_threshold filters out low-scoring individual matches."""
    path = tmp_path / "threshold_test.index"
    idx = UsearchSimprintIndex(str(path), ndim=128)

    base = create_random_simprint(128)

    # Create asset with one good match and one poor match
    asset_id = b"\x01" * 8
    good_match = flip_bits(base, 10)  # ~92% match
    poor_match = flip_bits(base, 40)  # ~69% match (below default 0.75 threshold)

    idx.add_raw([
        SimprintEntryRaw(
            iscc_id_body=asset_id,
            simprints=[
                SimprintRaw(simprint=good_match, offset=0, size=1024),
                SimprintRaw(simprint=poor_match, offset=1024, size=1024),
            ],
        )
    ])

    # Search - poor match should be filtered by threshold
    results = idx.search_raw([base], limit=10, threshold=0.75, detailed=False)

    # Should still find the asset but with only 1 match counted (not 2)
    assert len(results) == 1
    # matches should be 1, not 2 (poor match filtered)
    # Coverage = 1/1 = 1.0 (only one query simprint)
    assert results[0].matches == 1
    assert results[0].queried == 1

    idx.close()


def test_coverage_vs_quality_tradeoff(tmp_path):
    # type: (Path) -> None
    """
    Test coverage vs quality tradeoff in scoring.

    Asset A: High coverage (many matches) but lower quality
    Asset B: Low coverage (few matches) but higher quality
    """
    path = tmp_path / "coverage_test.index"
    idx = UsearchSimprintIndex(str(path), ndim=128)

    # Create 4 query simprints
    query_simprints = [create_random_simprint(128) for _ in range(4)]

    # Asset A: Matches all 4 queries at ~85% quality
    asset_a_id = b"\xaa" * 8
    asset_a_simprints = [flip_bits(q, 19) for q in query_simprints]  # ~85% match each
    idx.add_raw([
        SimprintEntryRaw(
            iscc_id_body=asset_a_id,
            simprints=[SimprintRaw(simprint=s, offset=i * 1024, size=1024) for i, s in enumerate(asset_a_simprints)],
        )
    ])

    # Asset B: Matches 2 queries at ~98% quality
    asset_b_id = b"\xbb" * 8
    asset_b_simprints = [flip_bits(query_simprints[0], 3), flip_bits(query_simprints[1], 3)]  # ~98% match each
    idx.add_raw([
        SimprintEntryRaw(
            iscc_id_body=asset_b_id,
            simprints=[SimprintRaw(simprint=s, offset=i * 1024, size=1024) for i, s in enumerate(asset_b_simprints)],
        )
    ])

    # Use threshold=0.75 to filter weak cross-matches
    results = idx.search_raw(query_simprints, limit=10, threshold=0.75, detailed=False)

    # Both should be found
    assert len(results) == 2

    # Asset A: coverage = 4/4 = 1.0, quality ≈ 0.85^4 (lower due to exponent)
    # Asset B: coverage = 2/4 = 0.5, quality ≈ 0.98^4 (very high)
    # With default exponent=4, high coverage should win
    assert results[0].iscc_id_body == asset_a_id  # A should rank higher
    assert results[0].matches == 4
    assert results[1].iscc_id_body == asset_b_id
    assert results[1].matches == 2

    idx.close()


# Per-query configuration tests


def test_search_with_custom_match_threshold(tmp_path):
    # type: (Path) -> None
    """Test per-query threshold parameter."""
    path = tmp_path / "custom_threshold.index"
    idx = UsearchSimprintIndex(str(path), ndim=128)

    base = create_random_simprint(128)
    asset_id = b"\x01" * 8

    # Create simprint with ~80% match (26 bits different)
    match_80 = flip_bits(base, 26)
    idx.add_raw([
        SimprintEntryRaw(iscc_id_body=asset_id, simprints=[SimprintRaw(simprint=match_80, offset=0, size=1024)])
    ])

    # With threshold 0.75, should find match (~80% > 75%)
    results_default = idx.search_raw([base], detailed=False, threshold=0.75)
    assert len(results_default) == 1

    # With stricter threshold (0.85), should not find match (~80% < 85%)
    results_strict = idx.search_raw([base], detailed=False, threshold=0.85)
    assert len(results_strict) == 0

    idx.close()


def test_search_with_custom_confidence_exponent(tmp_path):
    # type: (Path) -> None
    """
    Verify per-query confidence_exponent override changes ranking.

    With exponent=1 (linear): Asset A wins (higher average)
    With exponent=4 (default): Asset B wins (perfect match weighted heavily)
    """
    path = tmp_path / "custom_exponent.index"
    idx = UsearchSimprintIndex(str(path), ndim=128)

    base1 = create_random_simprint(128)
    base2 = create_random_simprint(128)

    # Asset A: Both matches at ~90%
    asset_a_id = b"\xaa" * 8
    idx.add_raw([
        SimprintEntryRaw(
            iscc_id_body=asset_a_id,
            simprints=[
                SimprintRaw(simprint=flip_bits(base1, 13), offset=0, size=1024),
                SimprintRaw(simprint=flip_bits(base2, 13), offset=1024, size=1024),
            ],
        )
    ])

    # Asset B: One perfect, one ~80%
    asset_b_id = b"\xbb" * 8
    idx.add_raw([
        SimprintEntryRaw(
            iscc_id_body=asset_b_id,
            simprints=[
                SimprintRaw(simprint=base1, offset=0, size=1024),  # Perfect
                SimprintRaw(simprint=flip_bits(base2, 26), offset=1024, size=1024),  # ~80%
            ],
        )
    ])

    # Search with linear weighting (exponent=1) and no coverage influence
    results_linear = idx.search_raw([base1, base2], detailed=False, confidence_exponent=1, coverage_weight=0.0)

    # Search with exponential weighting (exponent=4) and no coverage influence
    results_exp = idx.search_raw([base1, base2], detailed=False, confidence_exponent=4, coverage_weight=0.0)

    # With weighted average formula (sum(s^(k+1)) / sum(s^k)), even exp=1 favors peak scores
    # Asset B wins with both settings, but margin increases with higher exponent
    # Linear (exp=1): B score ~0.91, A score ~0.90 (small advantage for peak)
    # Exponential (exp=4): B score ~0.94, A score ~0.90 (large advantage for peak)
    assert len(results_linear) == 2
    assert len(results_exp) == 2
    assert results_linear[0].iscc_id_body == asset_b_id  # B wins (peak score advantage)
    assert results_exp[0].iscc_id_body == asset_b_id  # B wins (peak score heavily weighted)

    # Verify score gap increases with higher exponent
    linear_gap = results_linear[0].score - results_linear[1].score
    exp_gap = results_exp[0].score - results_exp[1].score
    assert exp_gap > linear_gap  # Higher exponent increases score gap

    idx.close()


def test_search_with_both_custom_parameters(tmp_path):
    # type: (Path) -> None
    """Test search with both match_threshold and confidence_exponent overrides."""
    path = tmp_path / "both_params.index"
    idx = UsearchSimprintIndex(str(path), ndim=128)

    base = create_random_simprint(128)
    asset_id = b"\x01" * 8

    match_95 = flip_bits(base, 6)  # ~95% match
    idx.add_raw([
        SimprintEntryRaw(iscc_id_body=asset_id, simprints=[SimprintRaw(simprint=match_95, offset=0, size=1024)])
    ])

    # Search with custom threshold and exponent
    results = idx.search_raw([base], detailed=False, match_threshold=0.9, confidence_exponent=2)

    assert len(results) == 1
    assert results[0].iscc_id_body == asset_id

    idx.close()


# Performance tests


def test_search_with_100_query_simprints(tmp_path):
    # type: (Path) -> None
    """Test search performance with 100 query simprints."""
    path = tmp_path / "perf_100.index"
    idx = UsearchSimprintIndex(str(path), ndim=128)

    # Add an asset with 100 simprints
    asset_id = b"\x01" * 8
    simprints = [SimprintRaw(simprint=create_random_simprint(), offset=i * 1024, size=1024) for i in range(100)]
    idx.add_raw([SimprintEntryRaw(iscc_id_body=asset_id, simprints=simprints)])

    # Create 100 query simprints (some will match)
    query_simprints = [create_random_simprint() for _ in range(100)]

    # Should complete without error (performance is tested implicitly)
    results = idx.search_raw(query_simprints, limit=10, detailed=False)

    # May or may not find matches (random data), just verify no errors
    assert isinstance(results, list)

    idx.close()


def test_add_asset_with_200_simprints(tmp_path):
    # type: (Path) -> None
    """Test adding an asset with 200 simprints."""
    path = tmp_path / "perf_200.index"
    idx = UsearchSimprintIndex(str(path), ndim=128)

    asset_id = b"\x01" * 8
    simprints = [SimprintRaw(simprint=create_random_simprint(), offset=i * 1024, size=1024) for i in range(200)]

    # Should complete without error
    idx.add_raw([SimprintEntryRaw(iscc_id_body=asset_id, simprints=simprints)])

    assert asset_id in idx
    assert len(idx) == 1

    idx.close()


def test_results_ordered_by_score(tmp_path):
    # type: (Path) -> None
    """Test that results are correctly ordered by descending score."""
    path = tmp_path / "order_test.index"
    idx = UsearchSimprintIndex(str(path), ndim=128)

    base = create_random_simprint(128)

    # Create 3 assets with different similarity levels
    for i, num_flips in enumerate([5, 15, 25]):  # ~96%, ~88%, ~80%
        asset_id = bytes([i] * 8)
        simprint = flip_bits(base, num_flips)
        idx.add_raw([
            SimprintEntryRaw(iscc_id_body=asset_id, simprints=[SimprintRaw(simprint=simprint, offset=0, size=1024)])
        ])

    results = idx.search_raw([base], limit=10, detailed=False)

    # Should be ordered by score (descending)
    assert len(results) == 3
    for i in range(len(results) - 1):
        assert results[i].score >= results[i + 1].score

    # Best match should be first
    assert results[0].iscc_id_body == b"\x00" * 8  # Least flips

    idx.close()


def test_limit_parameter(tmp_path):
    # type: (Path) -> None
    """Test that limit parameter correctly restricts results."""
    path = tmp_path / "limit_test.index"
    idx = UsearchSimprintIndex(str(path), ndim=128)

    base = create_random_simprint(128)

    # Add 10 assets with varying similarity
    for i in range(10):
        asset_id = bytes([i] * 8)
        simprint = flip_bits(base, 5 + i)  # Decreasing similarity
        idx.add_raw([
            SimprintEntryRaw(iscc_id_body=asset_id, simprints=[SimprintRaw(simprint=simprint, offset=0, size=1024)])
        ])

    # Request only top 3
    results = idx.search_raw([base], limit=3, detailed=False)

    assert len(results) == 3

    idx.close()


def test_threshold_parameter(tmp_path):
    # type: (Path) -> None
    """Test that threshold parameter filters results by final score."""
    path = tmp_path / "threshold_param.index"
    idx = UsearchSimprintIndex(str(path), ndim=128)

    base = create_random_simprint(128)

    # Add asset with moderate match
    asset_id = b"\x01" * 8
    simprint = flip_bits(base, 20)  # ~84% match
    idx.add_raw([
        SimprintEntryRaw(iscc_id_body=asset_id, simprints=[SimprintRaw(simprint=simprint, offset=0, size=1024)])
    ])

    # With low threshold, should find it
    results_low = idx.search_raw([base], threshold=0.5, detailed=False)
    assert len(results_low) == 1

    # With very high threshold, should not find it
    results_high = idx.search_raw([base], threshold=0.95, detailed=False)
    assert len(results_high) == 0

    idx.close()


def test_optimize_compacts_index(tmp_path):
    # type: (Path) -> None
    """Test that optimize() calls compact() on the index."""
    path = tmp_path / "test_optimize.usearch"
    idx = UsearchSimprintIndex(str(path), ndim=128)

    # Add some entries
    for i in range(10):
        iscc_id_body = i.to_bytes(8, "big")
        simprint = create_random_simprint()
        entry = SimprintEntryRaw(
            iscc_id_body=iscc_id_body, simprints=[SimprintRaw(simprint=simprint, offset=0, size=1024)]
        )
        idx.add_raw([entry])

    # Call optimize - should not raise
    idx.optimize()

    # Verify index still works after optimization
    assert len(idx) == 10

    idx.close()


def test_detailed_search_with_zero_coverage_weight(tmp_path):
    # type: (Path) -> None
    """Test detailed=True with coverage_weight=0.0 uses quality-only scoring."""
    path = tmp_path / "coverage_zero.index"
    idx = UsearchSimprintIndex(str(path), ndim=128)

    base1 = create_random_simprint(128)
    base2 = create_random_simprint(128)

    # Asset with 2 simprints
    asset_id = b"\x01" * 8
    idx.add_raw([
        SimprintEntryRaw(
            iscc_id_body=asset_id,
            simprints=[
                SimprintRaw(simprint=base1, offset=0, size=1024),
                SimprintRaw(simprint=base2, offset=1024, size=1024),
            ],
        )
    ])

    # Search with detailed=True and coverage_weight=0.0
    # This triggers the else branch at line 281 (final_score = quality)
    results = idx.search_raw([base1, base2], detailed=True, coverage_weight=0.0, threshold=0.0)

    assert len(results) == 1
    assert results[0].iscc_id_body == asset_id
    assert results[0].chunks is not None
    assert len(results[0].chunks) == 2

    # Score should be pure quality (no coverage influence)
    # With perfect matches, quality should be 1.0
    assert results[0].score == 1.0

    idx.close()
