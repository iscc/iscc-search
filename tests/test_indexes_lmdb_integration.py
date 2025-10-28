"""
Integration tests for LMDB index implementation.

Tests end-to-end workflows, settings integration, and real-world scenarios.
"""

import pytest
import iscc_core as ic
from iscc_search.schema import IsccAsset, IsccIndex
from iscc_search.indexes.lmdb import LmdbIndexManager
from iscc_search.settings import SearchSettings, get_index


@pytest.fixture
def manager(tmp_path):
    """Create LmdbIndexManager for integration tests."""
    mgr = LmdbIndexManager(tmp_path)
    yield mgr
    mgr.close()


def test_full_workflow_create_add_search_get_delete(manager, sample_assets):
    """Test complete workflow: create → add → search → get → delete."""
    # 1. Create index
    index = manager.create_index(IsccIndex(name="workflow"))
    assert index.name == "workflow"
    assert index.assets == 0

    # 2. Add assets
    results = manager.add_assets("workflow", sample_assets[:5])
    assert len(results) == 5

    # 3. Search
    query = IsccAsset(units=sample_assets[0].units)
    search_result = manager.search_assets("workflow", query, limit=10)
    assert len(search_result.matches) >= 1

    # 4. Get specific asset
    retrieved = manager.get_asset("workflow", sample_assets[0].iscc_id)
    assert retrieved.iscc_id == sample_assets[0].iscc_id

    # 5. Verify index metadata
    index_meta = manager.get_index("workflow")
    assert index_meta.assets == 5
    assert index_meta.size > 0

    # 6. Delete index
    manager.delete_index("workflow")

    # Verify deleted
    with pytest.raises(FileNotFoundError):
        manager.get_index("workflow")


def test_multi_index_scenario(manager, sample_iscc_ids, sample_content_units, sample_data_units):
    """Test working with multiple indexes simultaneously."""
    # Create multiple indexes
    manager.create_index(IsccIndex(name="images"))
    manager.create_index(IsccIndex(name="documents"))
    manager.create_index(IsccIndex(name="videos"))

    # Add different assets to each
    image_asset = IsccAsset(
        iscc_id=sample_iscc_ids[0],
        units=[sample_content_units[0], sample_content_units[1]],
        metadata={"type": "image"},
    )
    doc_asset = IsccAsset(
        iscc_id=sample_iscc_ids[1],
        units=[sample_content_units[2], sample_content_units[3]],
        metadata={"type": "document"},
    )
    video_asset = IsccAsset(
        iscc_id=sample_iscc_ids[2],
        units=[sample_data_units[0], sample_data_units[1]],
        metadata={"type": "video"},
    )

    manager.add_assets("images", [image_asset])
    manager.add_assets("documents", [doc_asset])
    manager.add_assets("videos", [video_asset])

    # List all indexes
    indexes = manager.list_indexes()
    assert len(indexes) == 3

    # Verify each index has its assets
    assert manager.get_index("images").assets == 1
    assert manager.get_index("documents").assets == 1
    assert manager.get_index("videos").assets == 1


def test_large_dataset(manager, large_dataset):
    """Test handling larger dataset with auto-resize."""
    ids, units = large_dataset

    manager.create_index(IsccIndex(name="large"))

    # Add many assets
    assets = []
    for i in range(len(ids)):
        # Provide at least 2 units for schema validation
        asset = IsccAsset(
            iscc_id=ids[i],
            units=[units[i % len(units)], units[(i + 1) % len(units)]],
            metadata={"index": i, "data": "x" * 100},
        )
        assets.append(asset)

    # Add in batches
    batch_size = 20
    for i in range(0, len(assets), batch_size):
        batch = assets[i : i + batch_size]
        manager.add_assets("large", batch)

    # Verify all added
    index_meta = manager.get_index("large")
    assert index_meta.assets == len(assets)

    # Search should return results
    query = IsccAsset(units=[units[0], units[1]])
    result = manager.search_assets("large", query, limit=50)
    assert len(result.matches) > 0


def test_update_asset_metadata(manager, sample_iscc_ids, sample_content_units):
    """Test updating asset by re-adding with same ISCC-ID."""
    manager.create_index(IsccIndex(name="updates"))

    # Add original
    original = IsccAsset(
        iscc_id=sample_iscc_ids[0],
        units=[sample_content_units[0], sample_content_units[1]],
        metadata={"version": 1, "title": "Original"},
    )
    result1 = manager.add_assets("updates", [original])
    assert result1[0].status.value == "created"

    # Update with same ISCC-ID
    updated = IsccAsset(
        iscc_id=sample_iscc_ids[0],
        units=[sample_content_units[0], sample_content_units[2]],  # Changed units
        metadata={"version": 2, "title": "Updated"},
    )
    result2 = manager.add_assets("updates", [updated])
    assert result2[0].status.value == "updated"

    # Verify metadata updated
    retrieved = manager.get_asset("updates", sample_iscc_ids[0])
    assert retrieved.metadata["version"] == 2
    assert retrieved.metadata["title"] == "Updated"

    # Still only one asset
    index_meta = manager.get_index("updates")
    assert index_meta.assets == 1


def test_settings_integration_file_path(tmp_path):
    """Test get_index() factory with lmdb:// URI scheme in settings."""
    # Set index_uri to lmdb:// scheme (use file:// format with three slashes)
    settings = SearchSettings(index_uri=f"lmdb:///{tmp_path}")

    # Override global settings (for this test)
    import iscc_search.settings

    original_settings = iscc_search.settings.search_settings
    iscc_search.settings.search_settings = settings

    try:
        # Get index via factory
        idx = get_index()

        # Should be LmdbIndexManager
        assert isinstance(idx, LmdbIndexManager)
        assert idx.base_path == tmp_path

        idx.close()
    finally:
        # Restore original settings
        iscc_search.settings.search_settings = original_settings


def test_persistence_across_manager_instances(tmp_path, sample_assets):
    """Test data persists across manager instances."""
    asset = sample_assets[0]

    # First manager: create and add
    mgr1 = LmdbIndexManager(tmp_path)
    mgr1.create_index(IsccIndex(name="persistent"))
    mgr1.add_assets("persistent", [asset])
    mgr1.close()

    # Second manager: verify data exists
    mgr2 = LmdbIndexManager(tmp_path)
    indexes = mgr2.list_indexes()
    assert len(indexes) == 1
    assert indexes[0].name == "persistent"
    assert indexes[0].assets == 1

    retrieved = mgr2.get_asset("persistent", asset.iscc_id)
    assert retrieved.iscc_id == asset.iscc_id
    assert retrieved.metadata == asset.metadata

    mgr2.close()


def test_search_with_no_matches(manager, sample_iscc_ids, sample_content_units, sample_data_units):
    """Test search returns empty results when no matches."""
    manager.create_index(IsccIndex(name="search"))

    # Add asset with specific content units
    asset = IsccAsset(
        iscc_id=sample_iscc_ids[0],
        units=[sample_content_units[0], sample_content_units[1]],
    )
    manager.add_assets("search", [asset])

    # Search for different data units (no match)
    query = IsccAsset(units=[sample_data_units[0], sample_data_units[1]])
    result = manager.search_assets("search", query)

    assert len(result.matches) == 0


def test_empty_index_operations(manager, sample_iscc_ids, sample_content_units):
    """Test operations on empty index."""
    manager.create_index(IsccIndex(name="empty"))

    # Get metadata
    index_meta = manager.get_index("empty")
    assert index_meta.assets == 0

    # Search returns no results
    query = IsccAsset(units=[sample_content_units[0], sample_content_units[1]])
    result = manager.search_assets("empty", query)
    assert len(result.matches) == 0

    # Get asset raises FileNotFoundError
    with pytest.raises(FileNotFoundError):
        manager.get_asset("empty", sample_iscc_ids[0])


def test_add_assets_without_metadata(manager, sample_iscc_ids, sample_content_units):
    """Test adding assets with minimal data (no metadata)."""
    manager.create_index(IsccIndex(name="minimal"))

    asset = IsccAsset(
        iscc_id=sample_iscc_ids[0],
        units=[sample_content_units[0], sample_content_units[1]],
        # No metadata
    )

    results = manager.add_assets("minimal", [asset])
    assert len(results) == 1

    # Retrieve and verify
    retrieved = manager.get_asset("minimal", asset.iscc_id)
    assert retrieved.metadata is None


def test_concurrent_index_access(manager, sample_iscc_ids, sample_content_units):
    """Test accessing multiple indexes without conflicts."""
    # Create multiple indexes
    for i in range(5):
        manager.create_index(IsccIndex(name=f"idx{i}"))

    # Add different data to each
    for i in range(5):
        asset = IsccAsset(
            iscc_id=sample_iscc_ids[i],
            units=[sample_content_units[0], sample_content_units[1]],
            metadata={"index_num": i},
        )
        manager.add_assets(f"idx{i}", [asset])

    # Verify all indexes have correct data
    for i in range(5):
        index_meta = manager.get_index(f"idx{i}")
        assert index_meta.assets == 1

    # All should be cached
    assert len(manager._index_cache) == 5


def test_realm_consistency_across_updates(manager, sample_iscc_ids, sample_content_units):
    """Test realm_id remains consistent across asset updates."""
    manager.create_index(IsccIndex(name="realm"))

    # Add first asset (realm=0)
    asset1 = IsccAsset(
        iscc_id=sample_iscc_ids[0],
        units=[sample_content_units[0], sample_content_units[1]],
    )
    manager.add_assets("realm", [asset1])

    # Add second asset (also realm=0)
    asset2 = IsccAsset(
        iscc_id=sample_iscc_ids[1],
        units=[sample_content_units[2], sample_content_units[3]],
    )
    manager.add_assets("realm", [asset2])

    # Verify both added successfully
    assert manager.get_index("realm").assets == 2

    # Trying to add realm=1 should still fail
    iscc_id_realm1 = ic.gen_iscc_id(timestamp=9000000, hub_id=99, realm_id=1)["iscc"]
    asset_wrong_realm = IsccAsset(
        iscc_id=iscc_id_realm1,
        units=[sample_content_units[0], sample_content_units[1]],
    )

    with pytest.raises(ValueError, match="Realm ID mismatch"):
        manager.add_assets("realm", [asset_wrong_realm])


@pytest.mark.xfail(
    reason="Known limitation: updating assets does not clean up old unit postings. "
    "When an asset is updated with different units, the old unit→asset mappings "
    "remain in the inverted index, causing stale search results. "
    "See Issue 2 in code review - deferred for future implementation."
)
def test_update_removes_old_unit_postings(manager, sample_iscc_ids, sample_content_units):
    """Test that updating asset units removes old unit postings from inverted index."""
    manager.create_index(IsccIndex(name="cleanup"))

    # Add original asset with units[0, 1]
    original = IsccAsset(
        iscc_id=sample_iscc_ids[0],
        units=[sample_content_units[0], sample_content_units[1]],
        metadata={"version": 1},
    )
    manager.add_assets("cleanup", [original])

    # Verify original units can be searched
    query_unit1 = IsccAsset(units=[sample_content_units[1]])
    result_before = manager.search_assets("cleanup", query_unit1, limit=10)
    assert len(result_before.matches) == 1
    assert result_before.matches[0].iscc_id == sample_iscc_ids[0]

    # Update asset with different units[0, 2] - removes unit[1], adds unit[2]
    updated = IsccAsset(
        iscc_id=sample_iscc_ids[0],
        units=[sample_content_units[0], sample_content_units[2]],
        metadata={"version": 2},
    )
    result = manager.add_assets("cleanup", [updated])
    assert result[0].status.value == "updated"

    # EXPECTED BEHAVIOR: Searching for old unit[1] should return NO matches
    # because the asset no longer contains it
    query_old_unit = IsccAsset(units=[sample_content_units[1]])
    result_after = manager.search_assets("cleanup", query_old_unit, limit=10)
    assert len(result_after.matches) == 0, (
        f"Expected no matches for old unit, but found {len(result_after.matches)}. "
        "Old unit postings should be removed when asset is updated."
    )

    # Searching for new unit[2] should return the asset
    query_new_unit = IsccAsset(units=[sample_content_units[2]])
    result_new = manager.search_assets("cleanup", query_new_unit, limit=10)
    assert len(result_new.matches) == 1
    assert result_new.matches[0].iscc_id == sample_iscc_ids[0]
