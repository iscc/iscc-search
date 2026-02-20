"""M1: Integration tests for UsearchIndex simprint indexing."""

import pytest
from iscc_search.indexes.usearch.index import UsearchIndex
from iscc_search.schema import IsccEntry, IsccSimprint


def test_usearch_simprint_indexing(tmp_path, sample_assets_with_simprints):
    """Test adding assets with simprints creates derived simprint indexes."""
    index = UsearchIndex(path=tmp_path / "test_index")

    # Add assets with simprints
    results = index.add_assets(sample_assets_with_simprints)
    assert len(results) == 5

    # Verify derived simprint indexes created
    assert len(index._simprint_indexes) > 0

    # Close index to persist files to disk
    index.close()

    # Verify SIMPRINT_*/ directories created
    simprint_dirs = list(tmp_path.glob("test_index/SIMPRINT_*/"))
    assert len(simprint_dirs) > 0


def test_usearch_threshold_parameter(tmp_path):
    """Test threshold parameter is stored."""
    index = UsearchIndex(path=tmp_path / "test_index", threshold=0.9)
    assert index.threshold == 0.9
    index.close()

    # Default threshold
    index2 = UsearchIndex(path=tmp_path / "test_index2")
    assert index2.threshold == 0.75
    index2.close()


def test_usearch_simprint_realm_consistency(tmp_path, sample_assets_with_simprints):
    """Test realm ID validation across indexes."""
    import iscc_core as ic

    index = UsearchIndex(path=tmp_path / "test_index")

    # Add realm=0 assets with simprints
    index.add_assets(sample_assets_with_simprints[:2])

    # Create realm=1 asset with valid realm-1 ISCC-ID
    realm1_iscc_id = ic.gen_iscc_id(timestamp=9999999, hub_id=99, realm_id=1)["iscc"]
    realm1_asset = IsccEntry(
        iscc_id=realm1_iscc_id,
        iscc_code="ISCC:KECYCMZIOY36XXGZ7S6QJQ2AEEXPOVEHZYPK6GMSFLU3WF54UPZMTPY",
        simprints={"CONTENT_TEXT_V0": [IsccSimprint(simprint="AXvu3tp2kF8mN9qL4rT1sZ", offset=0, size=500)]},
    )

    # Try to add realm=1 asset - should fail at UsearchIndex realm validation
    with pytest.raises(ValueError, match="Realm ID mismatch"):
        index.add_assets([realm1_asset])

    index.close()


def test_usearch_simprint_close_reopen(tmp_path, sample_assets_with_simprints):
    """Test simprint persistence across close/reopen."""
    # Add assets with simprints
    index = UsearchIndex(path=tmp_path / "test_index")
    index.add_assets(sample_assets_with_simprints)
    index.close()

    # Reopen index
    index2 = UsearchIndex(path=tmp_path / "test_index")

    # Verify derived simprint indexes loaded
    assert len(index2._simprint_indexes) > 0

    # Verify LMDB simprint databases loaded
    assert len(index2._sp_data_dbs) > 0

    index2.close()


def test_usearch_empty_simprints(tmp_path, sample_iscc_ids, sample_content_units):
    """Test assets with empty simprints dict."""
    index = UsearchIndex(path=tmp_path / "test_index")

    # Create asset with empty simprints dict
    asset = IsccEntry(
        iscc_id=sample_iscc_ids[0],
        units=[sample_content_units[0], sample_content_units[1]],
        simprints={},  # Empty dict
    )

    # Add asset - should not create simprint index directories
    results = index.add_assets([asset])
    assert len(results) == 1

    # Verify no simprint directories created
    simprint_dirs = list(tmp_path.glob("test_index/SIMPRINT_*/"))
    assert len(simprint_dirs) == 0

    index.close()


def test_usearch_no_simprints_field(tmp_path, sample_iscc_ids, sample_content_units):
    """Test assets without simprints field (None)."""
    index = UsearchIndex(path=tmp_path / "test_index")

    # Create asset without simprints field
    asset = IsccEntry(
        iscc_id=sample_iscc_ids[0],
        units=[sample_content_units[0], sample_content_units[1]],
        # simprints=None (default)
    )

    # Add asset
    results = index.add_assets([asset])
    assert len(results) == 1

    index.close()
