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
