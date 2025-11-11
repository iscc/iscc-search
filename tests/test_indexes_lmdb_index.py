"""
Tests for LmdbIndex class.

Tests core LMDB index functionality including asset storage, search,
realm validation, and auto-resize behavior.
"""

import pytest
import iscc_core as ic
from iscc_search.schema import IsccEntry, Status
from iscc_search.indexes.lmdb.index import LmdbIndex


@pytest.fixture
def temp_lmdb_path(tmp_path):
    """Create temporary LMDB file path."""
    return tmp_path / "test_index.lmdb"


@pytest.fixture
def lmdb_index(temp_lmdb_path):
    """Create LmdbIndex instance for testing."""
    idx = LmdbIndex(temp_lmdb_path)
    yield idx
    idx.close()


def test_lmdb_index_initialization(temp_lmdb_path):
    """Test LmdbIndex creates LMDB file on initialization."""
    idx = LmdbIndex(temp_lmdb_path)

    assert temp_lmdb_path.exists()
    assert idx.get_realm_id() is None  # No assets yet
    assert idx.get_asset_count() == 0

    idx.close()


def test_add_assets_created_status(lmdb_index, sample_assets):
    """Test adding new asset returns created status."""
    results = lmdb_index.add_assets([sample_assets[0]])

    assert len(results) == 1
    assert results[0].iscc_id == sample_assets[0].iscc_id
    assert results[0].status == Status.created


def test_add_assets_updated_status(lmdb_index, sample_assets):
    """Test adding existing asset returns updated status."""
    asset = sample_assets[0]

    # Add first time
    lmdb_index.add_assets([asset])

    # Add again with modified metadata
    modified = IsccEntry(
        iscc_id=asset.iscc_id,
        iscc_code=asset.iscc_code,
        units=asset.units,
        metadata={"title": "Modified Title"},
    )
    results = lmdb_index.add_assets([modified])

    assert len(results) == 1
    assert results[0].status == Status.updated


def test_add_assets_sets_realm_id(lmdb_index, sample_assets):
    """Test adding first asset sets index realm_id."""
    assert lmdb_index.get_realm_id() is None

    lmdb_index.add_assets([sample_assets[0]])

    assert lmdb_index.get_realm_id() == 0


def test_add_assets_realm_id_validation(lmdb_index, sample_assets):
    """Test realm_id validation rejects mixed realms."""
    # Add asset with realm=0 (from sample_assets)
    lmdb_index.add_assets([sample_assets[0]])

    # Create asset with realm=1 (need at least 2 units for schema validation)
    iscc_id_realm1 = ic.gen_iscc_id(timestamp=5000000, hub_id=9, realm_id=1)["iscc"]
    asset_realm1 = IsccEntry(
        iscc_id=iscc_id_realm1,
        units=[sample_assets[0].units[0], sample_assets[0].units[1]],
    )

    # Try to add asset with realm=1 (should fail)
    with pytest.raises(ValueError, match="Realm ID mismatch"):
        lmdb_index.add_assets([asset_realm1])


def test_add_assets_missing_iscc_id(lmdb_index, sample_content_units):
    """Test adding asset without iscc_id raises error."""
    asset = IsccEntry(
        iscc_id=None,
        units=[sample_content_units[0], sample_content_units[1]],
    )

    with pytest.raises(ValueError, match="must have iscc_id"):
        lmdb_index.add_assets([asset])


def test_add_assets_empty_list(lmdb_index):
    """Test adding empty list returns empty results."""
    results = lmdb_index.add_assets([])
    assert results == []


def test_add_assets_without_units(lmdb_index, sample_iscc_ids):
    """Test adding asset without units (only iscc_id)."""
    asset = IsccEntry(iscc_id=sample_iscc_ids[0], units=None)
    results = lmdb_index.add_assets([asset])

    assert len(results) == 1
    assert results[0].status == Status.created


def test_get_asset_success(lmdb_index, sample_assets):
    """Test retrieving asset by ISCC-ID."""
    asset = sample_assets[0]

    # Add asset
    lmdb_index.add_assets([asset])

    # Retrieve
    retrieved = lmdb_index.get_asset(asset.iscc_id)

    assert retrieved.iscc_id == asset.iscc_id
    assert retrieved.iscc_code == asset.iscc_code
    assert retrieved.units == asset.units
    assert retrieved.metadata == asset.metadata


def test_get_asset_not_found(lmdb_index, sample_iscc_ids):
    """Test get_asset raises FileNotFoundError for missing asset."""
    # Use an ISCC-ID that wasn't added
    with pytest.raises(FileNotFoundError, match="not found"):
        lmdb_index.get_asset(sample_iscc_ids[5])


def test_get_asset_empty_index(lmdb_index, sample_iscc_ids):
    """Test get_asset on empty index."""
    with pytest.raises(FileNotFoundError, match="index empty"):
        lmdb_index.get_asset(sample_iscc_ids[0])


def test_get_asset_invalid_iscc_id(lmdb_index):
    """Test get_asset with invalid ISCC-ID format."""
    with pytest.raises(ValueError, match="Invalid ISCC-ID"):
        lmdb_index.get_asset("NOT_VALID")


def test_get_asset_realm_mismatch(lmdb_index, sample_assets):
    """Test get_asset raises ValueError for ISCC-ID with different realm."""
    # Add asset with realm=0
    lmdb_index.add_assets([sample_assets[0]])

    # Try to get asset with realm=1 ISCC-ID
    iscc_id_realm1 = ic.gen_iscc_id(timestamp=9999999, hub_id=99, realm_id=1)["iscc"]

    with pytest.raises(ValueError, match="Realm mismatch"):
        lmdb_index.get_asset(iscc_id_realm1)


def test_search_assets_basic(lmdb_index, sample_assets):
    """Test basic search returns matches with metadata."""
    asset = sample_assets[0]

    # Add asset
    lmdb_index.add_assets([asset])

    # Search with same units
    query = IsccEntry(units=asset.units)
    result = lmdb_index.search_assets(query, limit=10)

    assert len(result.global_matches) == 1
    assert result.global_matches[0].iscc_id == asset.iscc_id
    assert result.global_matches[0].score > 0
    assert result.global_matches[0].metadata.model_dump(exclude_none=True) == asset.metadata


def test_search_assets_no_units(lmdb_index, sample_iscc_ids):
    """Test search without units or iscc_code raises error."""
    query = IsccEntry(iscc_id=sample_iscc_ids[0])

    with pytest.raises(ValueError, match="must have either 'iscc_code' or 'units'"):
        lmdb_index.search_assets(query)


def test_search_assets_empty_index(lmdb_index, sample_content_units):
    """Test search on empty index returns no matches."""
    query = IsccEntry(units=[sample_content_units[0], sample_content_units[1]])
    result = lmdb_index.search_assets(query, limit=10)

    assert len(result.global_matches) == 0


def test_search_assets_limit(lmdb_index, sample_iscc_ids, sample_content_units):
    """Test search respects limit parameter."""
    # Add multiple assets
    assets = []
    for i in range(10):
        asset = IsccEntry(
            iscc_id=sample_iscc_ids[i],
            units=[sample_content_units[0], sample_content_units[1]],  # Same units for all
        )
        assets.append(asset)

    lmdb_index.add_assets(assets)

    # Search with limit=5
    query = IsccEntry(units=[sample_content_units[0], sample_content_units[1]])
    result = lmdb_index.search_assets(query, limit=5)

    assert len(result.global_matches) == 5


def test_search_assets_scoring(lmdb_index, sample_iscc_ids, sample_content_units):
    """Test search results are sorted by score (descending)."""
    # Add assets with different unit overlaps
    # Asset 1: matches both units
    asset1 = IsccEntry(
        iscc_id=sample_iscc_ids[0],
        units=[sample_content_units[0], sample_content_units[1]],
    )
    # Asset 2: matches only one unit
    asset2 = IsccEntry(
        iscc_id=sample_iscc_ids[1],
        units=[sample_content_units[0], sample_content_units[2]],
    )

    lmdb_index.add_assets([asset1, asset2])

    # Search with both units from asset1
    query = IsccEntry(units=[sample_content_units[0], sample_content_units[1]])
    result = lmdb_index.search_assets(query, limit=10)

    assert len(result.global_matches) == 2
    # Asset1 should score higher (matches more units)
    assert result.global_matches[0].iscc_id == asset1.iscc_id
    assert result.global_matches[0].score > result.global_matches[1].score


def test_get_asset_count(lmdb_index, sample_assets):
    """Test asset count tracking."""
    assert lmdb_index.get_asset_count() == 0

    lmdb_index.add_assets([sample_assets[0]])
    assert lmdb_index.get_asset_count() == 1

    # Add another asset
    lmdb_index.add_assets([sample_assets[1]])
    assert lmdb_index.get_asset_count() == 2


def test_get_realm_id_before_and_after(lmdb_index, sample_assets):
    """Test realm_id is None before assets, set after."""
    assert lmdb_index.get_realm_id() is None

    lmdb_index.add_assets([sample_assets[0]])

    assert lmdb_index.get_realm_id() == 0


def test_close_and_reopen(temp_lmdb_path, sample_assets):
    """Test closing and reopening index preserves data."""
    asset = sample_assets[0]

    # Create and add asset
    idx1 = LmdbIndex(temp_lmdb_path)
    idx1.add_assets([asset])
    idx1.close()

    # Reopen and verify
    idx2 = LmdbIndex(temp_lmdb_path)
    assert idx2.get_asset_count() == 1
    assert idx2.get_realm_id() == 0

    retrieved = idx2.get_asset(asset.iscc_id)
    assert retrieved.iscc_id == asset.iscc_id

    idx2.close()


def test_map_size_property(lmdb_index):
    """Test map_size property returns valid size."""
    size = lmdb_index.map_size
    assert isinstance(size, int)
    assert size > 0


def test_auto_resize_on_map_full(temp_lmdb_path, sample_iscc_ids, sample_content_units):
    """Test auto-resize when LMDB map is full."""
    # Create index with very small initial map_size to force resize
    idx = LmdbIndex(temp_lmdb_path, lmdb_options={"map_size": 16 * 1024})  # 16KB - very small
    initial_size = idx.map_size

    # Add many assets with large metadata to trigger MapFullError
    assets = []
    for i in range(min(50, len(sample_iscc_ids))):
        asset = IsccEntry(
            iscc_id=sample_iscc_ids[i % len(sample_iscc_ids)],
            units=[sample_content_units[0], sample_content_units[1]],
            metadata={"data": "x" * 2000, "index": i, "more": "y" * 500},  # Large metadata
        )
        assets.append(asset)

    # This should trigger auto-resize
    idx.add_assets(assets)

    # Verify map_size increased
    final_size = idx.map_size
    assert final_size > initial_size
    # Should have at least doubled
    assert final_size >= initial_size * 2

    idx.close()


def test_del_closes_env(temp_lmdb_path):
    """Test __del__ closes LMDB environment."""
    idx = LmdbIndex(temp_lmdb_path)

    # Delete index (triggers __del__)
    del idx

    # LMDB environment should be closed (can't verify directly without error)


def test_add_assets_with_string_units(lmdb_index, sample_iscc_ids, sample_content_units):
    """Test adding assets when units are plain strings (not Unit objects)."""
    # Create asset with plain string units (bypasses Pydantic Unit conversion)
    # This tests the else branch in add_assets where unit_item is already a string
    asset_dict = {
        "iscc_id": sample_iscc_ids[0],
        "units": [str(sample_content_units[0]), str(sample_content_units[1])],
    }
    asset = IsccEntry(**asset_dict)

    # Units should be Unit objects after validation
    results = lmdb_index.add_assets([asset])
    assert len(results) == 1
    assert results[0].status.value == "created"


def test_search_with_string_units(lmdb_index, sample_assets, sample_content_units):
    """Test searching when query units are plain strings (not Unit objects)."""
    # Add assets first
    lmdb_index.add_assets([sample_assets[0]])

    # Create query with plain string units
    query_dict = {
        "units": [str(sample_content_units[0]), str(sample_content_units[1])],
    }
    query = IsccEntry(**query_dict)

    # Search should work
    result = lmdb_index.search_assets(query, limit=10)
    assert len(result.global_matches) >= 1


def test_get_db_cache_miss(lmdb_index, sample_assets):
    """Test _get_db() when database exists but isn't in cache."""
    # Add assets to create databases
    lmdb_index.add_assets([sample_assets[0]])

    # Clear the cache to force _get_db to open database without cache
    lmdb_index._db_cache.clear()

    # Now get asset - this should call _get_db which will open __assets__ from disk
    retrieved = lmdb_index.get_asset(sample_assets[0].iscc_id)
    assert retrieved.iscc_id == sample_assets[0].iscc_id

    # Verify database was added to cache
    assert "__assets__" in lmdb_index._db_cache


def test_search_unit_no_matches(lmdb_index, sample_assets, sample_data_units):
    """Test searching for units with no matches (cursor.set_range returns False)."""
    # Add assets with content units
    lmdb_index.add_assets([sample_assets[0]])

    # Search for completely different data units (different unit type, no matches)
    query = IsccEntry(units=[sample_data_units[0], sample_data_units[1]])
    result = lmdb_index.search_assets(query, limit=10)

    # Should return empty results
    assert len(result.global_matches) == 0


def test_add_multiple_assets_batch(lmdb_index, sample_assets):
    """Test adding multiple assets in single call."""
    results = lmdb_index.add_assets(sample_assets[:5])

    assert len(results) == 5
    assert all(r.status == Status.created for r in results)
    assert lmdb_index.get_asset_count() == 5
