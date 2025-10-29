"""
Tests for UsearchIndex class - low-level index implementation.

Tests UsearchIndex directly to cover edge cases not exercised through UsearchIndexManager.
"""

import pytest
import iscc_core as ic
from iscc_search.indexes.usearch.index import UsearchIndex
from iscc_search.schema import IsccAsset


@pytest.fixture
def usearch_index(tmp_path):
    """Create UsearchIndex instance for testing."""
    index_path = tmp_path / "test_index"
    idx = UsearchIndex(index_path, realm_id=0, max_dim=256)
    yield idx
    idx.close()


def test_usearch_index_with_custom_lmdb_options(tmp_path):
    """Test UsearchIndex accepts custom LMDB options."""
    index_path = tmp_path / "custom_index"
    custom_options = {
        "map_size": 10 * 1024 * 1024,  # 10MB
        "max_readers": 64,
    }

    idx = UsearchIndex(index_path, realm_id=0, max_dim=256, lmdb_options=custom_options)

    # Verify index was created
    assert len(idx) == 0
    assert idx.map_size == 10 * 1024 * 1024

    idx.close()


def test_usearch_index_empty_length(usearch_index):
    """Test __len__ returns 0 for empty index."""
    assert len(usearch_index) == 0


def test_usearch_index_get_asset_readonly_error(tmp_path):
    """Test get_asset on completely empty index handles ReadonlyError."""
    # Create index with just the directory
    index_path = tmp_path / "empty_index"
    idx = UsearchIndex(index_path, realm_id=0, max_dim=256)

    # Try to get non-existent asset before any database operations
    iscc_id = ic.gen_iscc_id(timestamp=5000000, hub_id=1, realm_id=0)["iscc"]

    with pytest.raises(FileNotFoundError, match="not found"):
        idx.get_asset(iscc_id)

    idx.close()


def test_usearch_index_bidirectional_instance_matching_256bit(usearch_index, sample_iscc_ids):
    """Test bidirectional prefix matching with 256-bit INSTANCE codes."""
    # Create 256-bit INSTANCE code (32 bytes)
    instance_256 = ic.Code.rnd(ic.MT.INSTANCE, bits=256)
    instance_str = f"ISCC:{instance_256}"

    # Also create shorter versions for testing
    instance_128_obj = ic.Code.rnd(ic.MT.INSTANCE, bits=128)
    instance_128 = f"ISCC:{instance_128_obj}"

    instance_64_obj = ic.Code.rnd(ic.MT.INSTANCE, bits=64)
    instance_64 = f"ISCC:{instance_64_obj}"

    # Add asset with 256-bit INSTANCE
    content_unit = ic.gen_text_code_v0("Test content")["iscc"]
    asset = IsccAsset(
        iscc_id=sample_iscc_ids[0],
        units=[instance_str, content_unit],
    )
    usearch_index.add_assets([asset])

    # Search with 256-bit query - should match itself
    query = IsccAsset(units=[instance_str, content_unit])
    result = usearch_index.search_assets(query, limit=10)

    assert len(result.matches) == 1
    assert result.matches[0].iscc_id == sample_iscc_ids[0]

    # Search with 128-bit - tests the reverse matching code path (query_len == 32 branch)
    query_128 = IsccAsset(units=[instance_128, content_unit])
    _ = usearch_index.search_assets(query_128, limit=10)
    # May or may not match depending on prefix - just testing code path executes

    # Search with 64-bit - tests the reverse matching code path (query_len >= 16 branch)
    query_64 = IsccAsset(units=[instance_64, content_unit])
    _ = usearch_index.search_assets(query_64, limit=10)
    # May or may not match depending on prefix - just testing code path executes


def test_usearch_index_bidirectional_instance_matching_128bit(usearch_index, sample_iscc_ids):
    """Test bidirectional prefix matching with 128-bit INSTANCE codes."""
    # Create 128-bit INSTANCE code (16 bytes)
    instance_128 = ic.Code.rnd(ic.MT.INSTANCE, bits=128)
    instance_str = f"ISCC:{instance_128}"

    # Add asset with 128-bit INSTANCE
    content_unit = ic.gen_text_code_v0("Test content 128")["iscc"]
    asset = IsccAsset(
        iscc_id=sample_iscc_ids[1],
        units=[instance_str, content_unit],
    )
    usearch_index.add_assets([asset])

    # Search with 128-bit query - should match itself
    query = IsccAsset(units=[instance_str, content_unit])
    result = usearch_index.search_assets(query, limit=10)

    assert len(result.matches) == 1
    assert result.matches[0].iscc_id == sample_iscc_ids[1]

    # Create a different 64-bit code to test the reverse matching path (query_len >= 16 branch)
    instance_64_obj = ic.Code.rnd(ic.MT.INSTANCE, bits=64)
    instance_64 = f"ISCC:{instance_64_obj}"
    query_64 = IsccAsset(units=[instance_64, content_unit])
    _ = usearch_index.search_assets(query_64, limit=10)
    # Just testing the code path executes


def test_usearch_index_instance_proportional_scoring(usearch_index, sample_iscc_ids):
    """Test that INSTANCE matches are scored proportionally to match length."""
    # Create INSTANCE codes of different lengths with shared prefixes
    # We'll use predictable prefixes to ensure matches
    base_bytes = bytes([1, 2, 3, 4, 5, 6, 7, 8])  # 64 bits
    prefix_128 = base_bytes + bytes([9, 10, 11, 12, 13, 14, 15, 16])  # 128 bits
    prefix_256 = prefix_128 + bytes(range(17, 33))  # 256 bits

    # Create ISCC codes from these bytes
    ic_64 = "ISCC:" + ic.encode_component(ic.MT.INSTANCE, ic.ST.NONE, ic.VS.V0, 64, base_bytes)
    ic_128 = "ISCC:" + ic.encode_component(ic.MT.INSTANCE, ic.ST.NONE, ic.VS.V0, 128, prefix_128)
    ic_256 = "ISCC:" + ic.encode_component(ic.MT.INSTANCE, ic.ST.NONE, ic.VS.V0, 256, prefix_256)

    # Add assets with different INSTANCE lengths
    content_unit = ic.gen_text_code_v0("Shared content for scoring test")["iscc"]

    asset_64 = IsccAsset(iscc_id=sample_iscc_ids[0], units=[ic_64, content_unit])
    asset_128 = IsccAsset(iscc_id=sample_iscc_ids[1], units=[ic_128, content_unit])
    asset_256 = IsccAsset(iscc_id=sample_iscc_ids[2], units=[ic_256, content_unit])

    usearch_index.add_assets([asset_64, asset_128, asset_256])

    # Test 1: 64-bit exact match should score 0.25
    query_64 = IsccAsset(units=[ic_64, content_unit])
    result_64 = usearch_index.search_assets(query_64, limit=10)

    # Find the exact match
    match_64 = next(m for m in result_64.matches if m.iscc_id == sample_iscc_ids[0])
    assert match_64.matches["INSTANCE_NONE_V0"] == 0.25, "64-bit exact match should score 0.25"

    # Test 2: 128-bit exact match should score 0.5
    query_128 = IsccAsset(units=[ic_128, content_unit])
    result_128 = usearch_index.search_assets(query_128, limit=10)

    match_128 = next(m for m in result_128.matches if m.iscc_id == sample_iscc_ids[1])
    assert match_128.matches["INSTANCE_NONE_V0"] == 0.5, "128-bit exact match should score 0.5"

    # Test 3: 256-bit exact match should score 1.0
    query_256 = IsccAsset(units=[ic_256, content_unit])
    result_256 = usearch_index.search_assets(query_256, limit=10)

    match_256 = next(m for m in result_256.matches if m.iscc_id == sample_iscc_ids[2])
    assert match_256.matches["INSTANCE_NONE_V0"] == 1.0, "256-bit exact match should score 1.0"

    # Test 4: Forward prefix match - 64-bit query matches 256-bit stored (64-bit overlap)
    # Query with 64-bit should match all three (they all share the 64-bit prefix)
    all_matches_64 = {m.iscc_id for m in result_64.matches}
    assert sample_iscc_ids[0] in all_matches_64  # Exact 64-bit match
    assert sample_iscc_ids[1] in all_matches_64  # 128-bit starts with 64-bit prefix
    assert sample_iscc_ids[2] in all_matches_64  # 256-bit starts with 64-bit prefix

    # All should score 0.25 (64-bit overlap)
    for match in result_64.matches:
        if match.iscc_id in [sample_iscc_ids[0], sample_iscc_ids[1], sample_iscc_ids[2]]:
            assert match.matches["INSTANCE_NONE_V0"] == 0.25

    # Test 5: Reverse prefix match - 256-bit query matches 64-bit stored
    # The 256-bit query contains the 64-bit and 128-bit as prefixes, so should match via reverse search
    all_matches_256 = {m.iscc_id for m in result_256.matches}
    assert sample_iscc_ids[2] in all_matches_256  # Exact 256-bit match (score 1.0)
    assert sample_iscc_ids[1] in all_matches_256  # 128-bit is prefix (score 0.5)
    assert sample_iscc_ids[0] in all_matches_256  # 64-bit is prefix (score 0.25)

    # Verify individual scores
    match_256_exact = next(m for m in result_256.matches if m.iscc_id == sample_iscc_ids[2])
    match_256_to_128 = next(m for m in result_256.matches if m.iscc_id == sample_iscc_ids[1])
    match_256_to_64 = next(m for m in result_256.matches if m.iscc_id == sample_iscc_ids[0])

    assert match_256_exact.matches["INSTANCE_NONE_V0"] == 1.0
    assert match_256_to_128.matches["INSTANCE_NONE_V0"] == 0.5
    assert match_256_to_64.matches["INSTANCE_NONE_V0"] == 0.25


def test_usearch_index_map_size_property(usearch_index):
    """Test map_size property returns LMDB map size."""
    map_size = usearch_index.map_size

    # Default map_size should be positive
    assert map_size > 0
    assert isinstance(map_size, int)


def test_usearch_index_add_empty_list(usearch_index):
    """Test add_assets with empty list returns empty results."""
    results = usearch_index.add_assets([])
    assert results == []


def test_usearch_index_get_asset_not_found(tmp_path):
    """Test get_asset raises FileNotFoundError for missing asset."""
    index_path = tmp_path / "notfound_index"
    idx = UsearchIndex(index_path, realm_id=0, max_dim=256)

    # Try to get non-existent asset
    iscc_id = ic.gen_iscc_id(timestamp=7777777, hub_id=77, realm_id=0)["iscc"]

    with pytest.raises(FileNotFoundError, match="not found"):
        idx.get_asset(iscc_id)

    idx.close()


def test_usearch_index_search_with_no_similarity_units(tmp_path):
    """Test search_assets when no similarity units are indexed."""
    # Create fresh index
    index_path = tmp_path / "no_sim_index"
    idx = UsearchIndex(index_path, realm_id=0, max_dim=256)

    # Add asset
    iscc_id = ic.gen_iscc_id(timestamp=8888888, hub_id=88, realm_id=0)["iscc"]
    instance_unit = f"ISCC:{ic.Code.rnd(ic.MT.INSTANCE, bits=128)}"
    content_unit = ic.gen_text_code_v0("Test content no sim unique")["iscc"]

    asset = IsccAsset(
        iscc_id=iscc_id,
        units=[instance_unit, content_unit],
    )
    idx.add_assets([asset])

    # Search with CONTENT unit that doesn't exist in index (unit type not in _nphd_indexes)
    # This tests the branch: if unit_type in self._nphd_indexes
    different_content = ic.gen_text_code_v0("Different content completely unique")["iscc"]
    query = IsccAsset(units=[different_content, instance_unit])
    result = idx.search_assets(query, limit=10)

    # Should still find via INSTANCE match even though CONTENT not in index
    assert len(result.matches) >= 1
    idx.close()
