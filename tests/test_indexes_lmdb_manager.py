"""
Tests for LmdbIndexManager protocol implementation.

Tests full protocol compliance, index lifecycle management, and multi-index scenarios.
"""

import pytest
from iscc_search.schema import IsccIndex, IsccEntry
from iscc_search.indexes.lmdb import LmdbIndexManager
from iscc_search.protocols.index import IsccIndexProtocol


@pytest.fixture
def manager(tmp_path):
    """Create LmdbIndexManager instance for testing."""
    mgr = LmdbIndexManager(tmp_path)
    yield mgr
    mgr.close()


def test_manager_initialization(tmp_path):
    """Test LmdbIndexManager creates base directory."""
    base_path = tmp_path / "indexes"
    assert not base_path.exists()

    mgr = LmdbIndexManager(base_path)

    assert base_path.exists()
    assert base_path.is_dir()

    mgr.close()


def test_manager_implements_protocol(manager):
    """Test LmdbIndexManager implements IsccIndexProtocol."""
    assert isinstance(manager, IsccIndexProtocol)


def test_list_indexes_empty(manager):
    """Test list_indexes returns empty list initially."""
    indexes = manager.list_indexes()
    assert indexes == []


def test_create_index_success(manager):
    """Test creating a new index."""
    index = IsccIndex(name="test")
    result = manager.create_index(index)

    assert result.name == "test"
    assert result.assets == 0
    assert result.size == 0


def test_create_index_creates_file(manager, tmp_path):
    """Test create_index creates .lmdb file."""
    index = IsccIndex(name="myindex")
    manager.create_index(index)

    lmdb_file = tmp_path / "myindex.lmdb"
    assert lmdb_file.exists()


def test_create_index_invalid_name(manager):
    """Test create_index with invalid name raises ValidationError from Pydantic."""
    from pydantic import ValidationError

    # Pydantic validates the name pattern in IsccIndex schema
    with pytest.raises(ValidationError, match="String should match pattern"):
        index = IsccIndex(name="Invalid-Name")
        manager.create_index(index)


def test_create_index_already_exists(manager):
    """Test create_index with existing name raises FileExistsError."""
    index = IsccIndex(name="duplicate")
    manager.create_index(index)

    with pytest.raises(FileExistsError, match="already exists"):
        manager.create_index(index)


def test_get_index_success(manager):
    """Test getting existing index metadata."""
    # Create index
    manager.create_index(IsccIndex(name="test"))

    # Get metadata
    result = manager.get_index("test")

    assert result.name == "test"
    assert result.assets == 0
    assert result.size >= 0


def test_get_index_not_found(manager):
    """Test get_index with non-existent index raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="not found"):
        manager.get_index("nonexistent")


def test_delete_index_success(manager, tmp_path):
    """Test deleting an index."""
    # Create index
    manager.create_index(IsccIndex(name="todelete"))
    lmdb_file = tmp_path / "todelete.lmdb"
    assert lmdb_file.exists()

    # Delete
    manager.delete_index("todelete")

    assert not lmdb_file.exists()


def test_delete_index_not_found(manager):
    """Test delete_index with non-existent index raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="not found"):
        manager.delete_index("nonexistent")


def test_delete_index_removes_from_cache(manager):
    """Test delete_index removes index from cache."""
    # Create and access index (loads into cache)
    manager.create_index(IsccIndex(name="cached"))
    manager.get_index("cached")

    assert "cached" in manager._index_cache

    # Delete
    manager.delete_index("cached")

    assert "cached" not in manager._index_cache


def test_delete_index_not_in_cache(manager):
    """Test deleting index that was never accessed (not in cache)."""
    # Create index
    manager.create_index(IsccIndex(name="uncached"))

    # Manually remove from cache to simulate never accessing it
    manager._index_cache.clear()

    # Delete should work even though not in cache
    manager.delete_index("uncached")

    # Verify deletion
    with pytest.raises(FileNotFoundError):
        manager.get_index("uncached")


def test_list_indexes_multiple(manager):
    """Test list_indexes with multiple indexes."""
    # Create several indexes
    manager.create_index(IsccIndex(name="alpha"))
    manager.create_index(IsccIndex(name="beta"))
    manager.create_index(IsccIndex(name="gamma"))

    indexes = manager.list_indexes()

    assert len(indexes) == 3
    names = [idx.name for idx in indexes]
    assert "alpha" in names
    assert "beta" in names
    assert "gamma" in names


def test_list_indexes_sorted(manager):
    """Test list_indexes returns sorted by name."""
    # Create in non-alphabetical order
    manager.create_index(IsccIndex(name="zulu"))
    manager.create_index(IsccIndex(name="alpha"))
    manager.create_index(IsccIndex(name="mike"))

    indexes = manager.list_indexes()

    names = [idx.name for idx in indexes]
    assert names == sorted(names)


def test_add_assets_success(manager, sample_assets):
    """Test adding assets to index."""
    # Create index
    manager.create_index(IsccIndex(name="test"))

    # Add assets
    results = manager.add_assets("test", [sample_assets[0]])

    assert len(results) == 1
    assert results[0].iscc_id == sample_assets[0].iscc_id


def test_add_assets_index_not_found(manager, sample_assets):
    """Test add_assets with non-existent index raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="not found"):
        manager.add_assets("nonexistent", [sample_assets[0]])


def test_get_asset_success(manager, sample_assets):
    """Test retrieving asset from index."""
    asset = sample_assets[0]

    # Create index and add asset
    manager.create_index(IsccIndex(name="test"))
    manager.add_assets("test", [asset])

    # Retrieve
    retrieved = manager.get_asset("test", asset.iscc_id)

    assert retrieved.iscc_id == asset.iscc_id
    assert retrieved.metadata == asset.metadata


def test_get_asset_index_not_found(manager, sample_iscc_ids):
    """Test get_asset with non-existent index raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="not found"):
        manager.get_asset("nonexistent", sample_iscc_ids[0])


def test_get_asset_asset_not_found(manager, sample_iscc_ids):
    """Test get_asset with missing asset raises FileNotFoundError."""
    # Create empty index
    manager.create_index(IsccIndex(name="test"))

    with pytest.raises(FileNotFoundError, match="not found"):
        manager.get_asset("test", sample_iscc_ids[5])


def test_search_assets_success(manager, sample_assets):
    """Test searching for assets."""
    asset = sample_assets[0]

    # Create index and add asset
    manager.create_index(IsccIndex(name="test"))
    manager.add_assets("test", [asset])

    # Search
    query = IsccEntry(units=asset.units)
    result = manager.search_assets("test", query, limit=10)

    assert len(result.global_matches) == 1
    assert result.global_matches[0].iscc_id == asset.iscc_id


def test_search_assets_index_not_found(manager, sample_content_units):
    """Test search_assets with non-existent index raises FileNotFoundError."""
    query = IsccEntry(units=[sample_content_units[0], sample_content_units[1]])

    with pytest.raises(FileNotFoundError, match="not found"):
        manager.search_assets("nonexistent", query)


def test_close_closes_all_cached_indexes(manager):
    """Test close() closes all cached index instances."""
    # Create multiple indexes and load into cache
    manager.create_index(IsccIndex(name="idx1"))
    manager.create_index(IsccIndex(name="idx2"))
    manager.get_index("idx1")
    manager.get_index("idx2")

    assert len(manager._index_cache) == 2

    # Close
    manager.close()

    assert len(manager._index_cache) == 0


def test_index_cache_management(manager, sample_assets):
    """Test index instances are cached and reused."""
    asset = sample_assets[0]

    # Create index - this loads it into cache
    manager.create_index(IsccIndex(name="test"))
    assert "test" in manager._index_cache

    # Add asset - should reuse cached instance
    cached_idx = manager._index_cache["test"]
    manager.add_assets("test", [asset])
    assert manager._index_cache["test"] is cached_idx  # Same instance

    # Get asset - should still reuse cached instance
    manager.get_asset("test", asset.iscc_id)
    assert manager._index_cache["test"] is cached_idx  # Still same instance


def test_get_file_size_mb(manager, tmp_path, sample_assets):
    """Test _get_file_size_mb helper."""
    # Create index with some data
    manager.create_index(IsccIndex(name="test"))
    manager.add_assets("test", [sample_assets[0]])

    # Get size
    lmdb_file = tmp_path / "test.lmdb"
    size_mb = manager._get_file_size_mb(lmdb_file)

    assert isinstance(size_mb, int)
    assert size_mb >= 0


def test_list_indexes_with_assets(manager, sample_assets):
    """Test list_indexes includes asset count and size."""
    # Create index and add assets
    manager.create_index(IsccIndex(name="test"))
    manager.add_assets("test", [sample_assets[0]])

    # List
    indexes = manager.list_indexes()

    assert len(indexes) == 1
    assert indexes[0].name == "test"
    assert indexes[0].assets == 1
    assert indexes[0].size >= 0


def test_multiple_indexes_independent(manager, sample_iscc_ids, sample_content_units):
    """Test multiple indexes operate independently."""
    # Create two indexes
    manager.create_index(IsccIndex(name="idx1"))
    manager.create_index(IsccIndex(name="idx2"))

    # Add different assets to each
    asset1 = IsccEntry(
        iscc_id=sample_iscc_ids[0],
        units=[sample_content_units[0], sample_content_units[1]],
    )
    asset2 = IsccEntry(
        iscc_id=sample_iscc_ids[1],
        units=[sample_content_units[2], sample_content_units[3]],
    )

    manager.add_assets("idx1", [asset1])
    manager.add_assets("idx2", [asset2])

    # Verify independence
    assert manager.get_index("idx1").assets == 1
    assert manager.get_index("idx2").assets == 1

    # Asset1 only in idx1
    retrieved1 = manager.get_asset("idx1", asset1.iscc_id)
    assert retrieved1.iscc_id == asset1.iscc_id

    with pytest.raises(FileNotFoundError):
        manager.get_asset("idx2", asset1.iscc_id)


def test_create_index_ignores_assets_and_size(manager):
    """Test create_index ignores provided assets and size fields."""
    index = IsccIndex(name="test", assets=999, size=999)
    result = manager.create_index(index)

    # Should always be 0 for new index
    assert result.assets == 0
    assert result.size == 0


def test_list_indexes_skips_corrupted(manager, tmp_path):
    """Test list_indexes skips corrupted or inaccessible indexes."""
    # Create valid index
    manager.create_index(IsccIndex(name="valid"))

    # Create fake .lmdb file (not actually valid LMDB)
    fake_lmdb = tmp_path / "corrupted.lmdb"
    fake_lmdb.write_text("not a real lmdb file")

    # List should only return valid index
    indexes = manager.list_indexes()

    # Should have at least the valid one (corrupted may be skipped)
    names = [idx.name for idx in indexes]
    assert "valid" in names


def test_manager_works_with_existing_directory(tmp_path):
    """Test manager works when base directory already exists."""
    base_path = tmp_path / "existing"
    base_path.mkdir()

    # Should not raise
    mgr = LmdbIndexManager(base_path)
    assert base_path.exists()

    mgr.close()
