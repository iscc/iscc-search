"""
Tests for UsearchIndexManager protocol implementation.

Tests full protocol compliance, index lifecycle management, hybrid LMDB/NphdIndex architecture,
and INSTANCE special handling.
"""

import iscc_core as ic
import pytest
from iscc_search.schema import IsccIndex, IsccEntry, IsccQuery, Status
from iscc_search.indexes.usearch import UsearchIndexManager
from iscc_search.protocols.index import IsccIndexProtocol


@pytest.fixture
def manager(tmp_path):
    """Create UsearchIndexManager instance for testing."""
    mgr = UsearchIndexManager(tmp_path)
    yield mgr
    mgr.close()


# Helper fixtures for test-specific ISCC generation


@pytest.fixture
def gen_iscc_id_realm_0():
    """Generator function for creating ISCC-IDs with realm 0."""
    counter = [0]  # Use list for mutable counter in closure

    def _gen():
        iscc_id = ic.gen_iscc_id(timestamp=5000000 + counter[0], hub_id=counter[0] % 100, realm_id=0)["iscc"]
        counter[0] += 1
        return iscc_id

    return _gen


@pytest.fixture
def gen_content_text():
    """Generator function for creating CONTENT-TEXT units from text."""

    def _gen(text):
        # type: (str) -> str
        code = ic.gen_text_code_v0(text)
        return code["iscc"]

    return _gen


@pytest.fixture
def gen_instance():
    """Generator function for creating INSTANCE units from data."""

    def _gen(data):
        # type: (bytes) -> str
        # gen_instance_code_v0 expects file-like object or uses stream if opened with `with open()`
        # For simplicity, use Code.rnd to generate random INSTANCE units
        code = ic.Code.rnd(ic.MT.INSTANCE, bits=128)
        return f"ISCC:{code}"

    return _gen


def test_manager_initialization(tmp_path):
    """Test UsearchIndexManager creates base directory."""
    base_path = tmp_path / "indexes"
    assert not base_path.exists()

    mgr = UsearchIndexManager(base_path)

    assert base_path.exists()
    assert base_path.is_dir()

    mgr.close()


def test_manager_implements_protocol(manager):
    """Test UsearchIndexManager implements IsccIndexProtocol."""
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


def test_create_index_creates_directory(manager, tmp_path):
    """Test create_index creates directory with index.lmdb."""
    index = IsccIndex(name="myindex")
    manager.create_index(index)

    index_dir = tmp_path / "myindex"
    assert index_dir.exists()
    assert index_dir.is_dir()

    lmdb_file = index_dir / "index.lmdb"
    assert lmdb_file.exists()


def test_create_index_invalid_name(manager):
    """Test create_index with invalid name raises ValidationError from Pydantic."""
    from pydantic import ValidationError

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
    manager.create_index(IsccIndex(name="test"))

    result = manager.get_index("test")

    assert result.name == "test"
    assert result.assets == 0
    assert result.size >= 0


def test_get_index_not_found(manager):
    """Test get_index with non-existent index raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="not found"):
        manager.get_index("nonexistent")


def test_list_indexes_with_data(manager):
    """Test list_indexes returns all created indexes."""
    manager.create_index(IsccIndex(name="index1"))
    manager.create_index(IsccIndex(name="index2"))
    manager.create_index(IsccIndex(name="index3"))

    indexes = manager.list_indexes()
    names = [idx.name for idx in indexes]

    assert len(indexes) == 3
    assert set(names) == {"index1", "index2", "index3"}


def test_delete_index_success(manager, tmp_path):
    """Test deleting an index removes directory."""
    manager.create_index(IsccIndex(name="test"))
    index_dir = tmp_path / "test"
    assert index_dir.exists()

    manager.delete_index("test")

    assert not index_dir.exists()


def test_delete_index_not_found(manager):
    """Test delete_index with non-existent index raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="not found"):
        manager.delete_index("nonexistent")


def test_delete_index_with_cached_instance(manager, tmp_path, sample_iscc_ids, sample_content_units):
    """Test delete_index closes and removes cached index instance."""
    manager.create_index(IsccIndex(name="cached"))

    # Add asset to populate cache
    asset = IsccEntry(
        iscc_id=sample_iscc_ids[0],
        units=[sample_content_units[0], sample_content_units[1]],
    )
    manager.add_assets("cached", [asset])

    # Verify index is in cache
    assert "cached" in manager._index_cache

    # Delete should close and remove from cache
    manager.delete_index("cached")

    # Verify removed from cache
    assert "cached" not in manager._index_cache

    # Verify directory deleted
    index_dir = tmp_path / "cached"
    assert not index_dir.exists()


def test_add_assets_basic(manager, sample_iscc_ids, sample_content_units):
    """Test adding assets to index."""
    manager.create_index(IsccIndex(name="test"))

    assets = [
        IsccEntry(
            iscc_id=sample_iscc_ids[0],
            units=[sample_content_units[0], sample_content_units[1]],  # Need min 2 units
        )
    ]

    results = manager.add_assets("test", assets)

    assert len(results) == 1
    assert results[0].iscc_id == sample_iscc_ids[0]
    assert results[0].status == Status.created


def test_add_assets_index_not_found(manager, sample_iscc_ids, sample_content_units):
    """Test add_assets with non-existent index raises FileNotFoundError."""
    assets = [
        IsccEntry(
            iscc_id=sample_iscc_ids[0],
            units=[sample_content_units[0], sample_content_units[1]],
        )
    ]

    with pytest.raises(FileNotFoundError, match="not found"):
        manager.add_assets("nonexistent", assets)


def test_get_asset_success(manager, sample_iscc_ids, sample_content_units):
    """Test retrieving asset by ISCC-ID."""
    manager.create_index(IsccIndex(name="test"))

    original = IsccEntry(
        iscc_id=sample_iscc_ids[0],
        units=[sample_content_units[0], sample_content_units[1]],
        metadata={"title": "Test Asset"},
    )

    manager.add_assets("test", [original])

    retrieved = manager.get_asset("test", sample_iscc_ids[0])

    assert retrieved.iscc_id == sample_iscc_ids[0]
    assert len(retrieved.units) == 2
    assert retrieved.metadata == {"title": "Test Asset"}


def test_get_asset_not_found(manager, sample_iscc_ids):
    """Test get_asset with non-existent ISCC-ID raises FileNotFoundError."""
    manager.create_index(IsccIndex(name="test"))

    with pytest.raises(FileNotFoundError, match="not found"):
        manager.get_asset("test", sample_iscc_ids[0])


def test_get_asset_index_not_found(manager, sample_iscc_ids):
    """Test get_asset with non-existent index raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="not found"):
        manager.get_asset("nonexistent", sample_iscc_ids[0])


def test_search_assets_empty_index(manager, sample_content_units):
    """Test search on empty index returns empty results."""
    manager.create_index(IsccIndex(name="test"))

    query = IsccQuery(units=[sample_content_units[0], sample_content_units[1]])
    result = manager.search_assets("test", query)

    assert result.global_matches == []


def test_search_assets_similarity_matching(manager, gen_iscc_id_realm_0, gen_content_text, gen_instance):
    """Test similarity-based search with CONTENT units."""
    manager.create_index(IsccIndex(name="test"))

    # Common INSTANCE unit for all assets (required for min 2 units)
    instance = gen_instance(b"unused")

    # Add multiple assets with similar CONTENT units
    assets = [
        IsccEntry(
            iscc_id=gen_iscc_id_realm_0(),
            units=[gen_content_text("Hello World"), instance],
        ),
        IsccEntry(
            iscc_id=gen_iscc_id_realm_0(),
            units=[gen_content_text("Hello World!"), instance],  # Similar
        ),
        IsccEntry(
            iscc_id=gen_iscc_id_realm_0(),
            units=[gen_content_text("Completely Different"), instance],
        ),
    ]

    manager.add_assets("test", assets)

    # Search with similar query
    query = IsccQuery(units=[gen_content_text("Hello World"), instance])
    result = manager.search_assets("test", query, limit=10)

    # Should find matches, ordered by similarity
    assert len(result.global_matches) > 0

    # First match should have high score (CONTENT similarity + INSTANCE proportional match)
    # With 128-bit INSTANCE: score = CONTENT(~1.0) + INSTANCE(0.5) = ~1.5
    assert result.global_matches[0].score > 1.4


def test_search_assets_instance_exact_matching(manager, gen_iscc_id_realm_0, gen_instance, gen_content_text):
    """Test exact matching with INSTANCE units via LMDB dupsort."""
    manager.create_index(IsccIndex(name="test"))

    instance_unit = gen_instance(b"unused")  # Arg ignored, generates random
    content_unit = gen_content_text("Test")  # Need 2 units minimum

    # Add asset with INSTANCE unit
    asset = IsccEntry(
        iscc_id=gen_iscc_id_realm_0(),
        units=[instance_unit, content_unit],
    )
    manager.add_assets("test", [asset])

    # Search with same units
    query = IsccQuery(units=[instance_unit, content_unit])
    result = manager.search_assets("test", query)

    # Should find exact match with high score
    # With 128-bit INSTANCE: score = INSTANCE(0.5) + CONTENT(~1.0) = ~1.5
    assert len(result.global_matches) == 1
    assert result.global_matches[0].score >= 1.4
    assert result.global_matches[0].iscc_id == asset.iscc_id


def test_search_assets_hybrid(manager, gen_iscc_id_realm_0, gen_content_text, gen_instance):
    """Test search with both INSTANCE and similarity units."""
    manager.create_index(IsccIndex(name="test"))

    content_unit = gen_content_text("Hybrid test")
    instance_unit = gen_instance(b"unused")

    # Add asset with both unit types
    asset = IsccEntry(
        iscc_id=gen_iscc_id_realm_0(),
        units=[content_unit, instance_unit],
    )
    manager.add_assets("test", [asset])

    # Search with same units
    query = IsccQuery(units=[content_unit, instance_unit])
    result = manager.search_assets("test", query)

    # Should find match with aggregated score
    assert len(result.global_matches) == 1
    # Score should be sum of both matches
    # With 128-bit INSTANCE: score = INSTANCE(0.5) + CONTENT(~1.0) = ~1.5
    assert result.global_matches[0].score >= 1.4


def test_search_assets_index_not_found(manager, sample_content_units):
    """Test search_assets with non-existent index raises FileNotFoundError."""
    query = IsccQuery(units=[sample_content_units[0], sample_content_units[1]])

    with pytest.raises(FileNotFoundError, match="not found"):
        manager.search_assets("nonexistent", query)


def test_close_cleans_up_cache(manager, tmp_path):
    """Test close() cleans up cached index instances."""
    manager.create_index(IsccIndex(name="test"))

    # Access index to populate cache
    manager.get_index("test")
    assert "test" in manager._index_cache

    # Close should clear cache
    manager.close()
    assert len(manager._index_cache) == 0


def test_persistence_after_close(manager, tmp_path, sample_iscc_ids, sample_content_units):
    """Test index persists after close and can be reopened."""
    manager.create_index(IsccIndex(name="test"))

    asset = IsccEntry(
        iscc_id=sample_iscc_ids[0],
        units=[sample_content_units[0], sample_content_units[1]],
    )
    manager.add_assets("test", [asset])

    iscc_id = asset.iscc_id
    manager.close()

    # Reopen with new manager
    manager2 = UsearchIndexManager(tmp_path)
    try:
        # Should still find the asset
        retrieved = manager2.get_asset("test", iscc_id)
        assert retrieved.iscc_id == iscc_id

        # Search should also work
        query = IsccQuery(units=[sample_content_units[0], sample_content_units[1]])
        result = manager2.search_assets("test", query)
        assert len(result.global_matches) > 0
    finally:
        manager2.close()


def test_multiple_indexes_isolation(manager, sample_iscc_ids, sample_content_units):
    """Test multiple indexes are properly isolated."""
    manager.create_index(IsccIndex(name="index1"))
    manager.create_index(IsccIndex(name="index2"))

    asset1 = IsccEntry(
        iscc_id=sample_iscc_ids[0],
        units=[sample_content_units[0], sample_content_units[1]],
    )
    manager.add_assets("index1", [asset1])

    # index2 should be empty
    query = IsccQuery(units=[sample_content_units[0], sample_content_units[1]])
    result = manager.search_assets("index2", query)
    assert result.global_matches == []

    # index1 should have the asset
    result = manager.search_assets("index1", query)
    assert len(result.global_matches) > 0


def test_list_indexes_skips_invalid_entries(tmp_path):
    """Test list_indexes skips non-directories and directories without index.lmdb."""
    manager = UsearchIndexManager(tmp_path)

    # Create a valid index
    manager.create_index(IsccIndex(name="valid"))

    # Create a regular file (should be skipped)
    (tmp_path / "notadir.txt").write_text("not a directory")

    # Create directory without index.lmdb (should be skipped)
    invalid_dir = tmp_path / "invalid"
    invalid_dir.mkdir()

    # Create directory with corrupted index.lmdb (should be skipped)
    corrupt_dir = tmp_path / "corrupt"
    corrupt_dir.mkdir()
    (corrupt_dir / "index.lmdb").write_bytes(b"corrupted data")

    # List should only return valid index
    indexes = manager.list_indexes()
    assert len(indexes) == 1
    assert indexes[0].name == "valid"

    manager.close()


def test_add_assets_missing_iscc_id(manager, sample_content_units):
    """Test add_assets raises ValueError when iscc_id is missing."""
    manager.create_index(IsccIndex(name="test"))

    # Asset without iscc_id should raise ValueError
    assets = [
        IsccEntry(
            units=[sample_content_units[0], sample_content_units[1]],
        )
    ]

    with pytest.raises(ValueError, match="Asset must have iscc_id field"):
        manager.add_assets("test", assets)


def test_add_assets_missing_iscc_id_after_realm_set(manager, sample_iscc_ids, sample_content_units):
    """Test adding asset without iscc_id after realm_id is already set."""
    manager.create_index(IsccIndex(name="test"))

    # First add valid asset to set realm_id
    valid_asset = IsccEntry(
        iscc_id=sample_iscc_ids[0],
        units=[sample_content_units[0], sample_content_units[1]],
    )
    manager.add_assets("test", [valid_asset])

    # Now try to add asset without iscc_id - should raise ValueError
    invalid_asset = IsccEntry(
        units=[sample_content_units[2], sample_content_units[3]],
    )

    with pytest.raises(ValueError, match="Asset must have iscc_id field"):
        manager.add_assets("test", [invalid_asset])


def test_add_assets_realm_mismatch(manager, sample_iscc_ids, sample_content_units):
    """Test add_assets raises ValueError for realm_id mismatch."""
    manager.create_index(IsccIndex(name="test"))

    # Add asset with realm 0
    asset1 = IsccEntry(
        iscc_id=sample_iscc_ids[0],  # realm 0
        units=[sample_content_units[0], sample_content_units[1]],
    )
    manager.add_assets("test", [asset1])

    # Try to add asset with realm 1 - should fail
    realm_1_id = ic.gen_iscc_id(timestamp=9000000, hub_id=1, realm_id=1)["iscc"]
    asset2 = IsccEntry(
        iscc_id=realm_1_id,
        units=[sample_content_units[2], sample_content_units[3]],
    )

    with pytest.raises(ValueError, match="Realm ID mismatch"):
        manager.add_assets("test", [asset2])
