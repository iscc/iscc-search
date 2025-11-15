"""M1: Integration tests for UsearchIndex simprint indexing."""

import pytest
from iscc_search.indexes.usearch.index import UsearchIndex
from iscc_search.schema import IsccEntry, IsccSimprint


def test_realm_id_int_property(tmp_path, sample_assets_with_simprints):
    """Test realm_id_int property extracts realm correctly."""
    # Test by adding assets to an index, which sets realm_id through normal flow
    index = UsearchIndex(path=tmp_path / "test_index")

    # Before adding assets, realm_id_int should be None
    assert index._simprint_index.realm_id_int is None

    # Add assets with simprints (realm 0)
    index.add_assets(sample_assets_with_simprints[:2])

    # After adding assets, realm_id_int should be 0
    assert index._simprint_index.realm_id_int == 0

    index.close()


def test_usearch_simprint_indexing(tmp_path, sample_assets_with_simprints):
    """Test adding assets with simprints creates simprint index files."""
    index = UsearchIndex(path=tmp_path / "test_index")

    # Add assets with simprints
    results = index.add_assets(sample_assets_with_simprints)
    assert len(results) == 5

    # Verify SIMPRINT_*.lmdb files created in same directory as index.lmdb
    simprint_files = list(tmp_path.glob("test_index/SIMPRINT_*.lmdb"))
    assert len(simprint_files) > 0  # At least one simprint type indexed

    # Verify can load index
    indexed_types = index._simprint_index.get_indexed_types()
    assert "CONTENT_TEXT_V0" in indexed_types or "SEMANTIC_TEXT_V0" in indexed_types

    index.close()


def test_usearch_threshold_parameter(tmp_path):
    """Test threshold parameter is stored."""
    index = UsearchIndex(path=tmp_path / "test_index", threshold=0.9)
    assert index.threshold == 0.9
    index.close()

    # Default threshold
    index2 = UsearchIndex(path=tmp_path / "test_index2")
    assert index2.threshold == 0.0
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

    # Try to add realm=1 asset - should fail at LmdbSimprintIndexMulti validation
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

    # Verify simprint index loaded
    assert index2._simprint_index is not None
    indexed_types = index2._simprint_index.get_indexed_types()
    assert len(indexed_types) > 0

    # Verify realm consistency maintained
    if index2._simprint_index.realm_id_int is not None:
        assert index2._realm_id == index2._simprint_index.realm_id_int

    index2.close()


def test_usearch_simprint_error_handling(tmp_path, sample_assets_with_simprints, monkeypatch):
    """Test that simprint failures propagate and fail the add operation."""
    index = UsearchIndex(path=tmp_path / "test_index")

    # Mock add_raw_multi to raise exception
    def mock_add_raw_multi(*args, **kwargs):
        raise RuntimeError("Simulated simprint add failure")

    monkeypatch.setattr(index._simprint_index, "add_raw_multi", mock_add_raw_multi)

    # Add assets - should fail due to simprint error
    with pytest.raises(RuntimeError, match="Simulated simprint add failure"):
        index.add_assets(sample_assets_with_simprints)

    index.close()


def test_usearch_simprint_realm_mismatch_fails_loudly(tmp_path, sample_assets_with_simprints):
    """
    Test that realm mismatches between SIMPRINT files and LMDB fail loudly.

    Reproduces scenario where:
    1. Directory has SIMPRINT_*.lmdb files (realm 1)
    2. No index.lmdb exists (so _realm_id is None during _load_simprint_index)
    3. First asset added is realm 0
    4. LMDB accepts it and sets _realm_id = 0
    5. But simprint add_raw_multi raises ValueError because simprint index is realm 1
    6. This should propagate and fail the operation (not be silently logged)
    """
    import iscc_core as ic

    # Step 1: Create index with realm 1 assets and simprints
    index1 = UsearchIndex(path=tmp_path / "test_index")

    # Create realm 1 assets by generating proper realm 1 ISCC-IDs
    realm1_assets = []
    for i, asset in enumerate(sample_assets_with_simprints[:2]):
        realm1_id = ic.gen_iscc_id(timestamp=1000000 + i, hub_id=i, realm_id=1)["iscc"]
        realm1_assets.append(
            IsccEntry(
                iscc_id=realm1_id,
                iscc_code=asset.iscc_code,
                units=asset.units,
                simprints=asset.simprints,
                metadata=asset.metadata,
            )
        )

    index1.add_assets(realm1_assets)
    assert index1._realm_id == 1
    assert index1._simprint_index.realm_id_int == 1
    index1.close()

    # Step 2: Delete index.lmdb to simulate missing main index
    # This causes _realm_id to be None, but SIMPRINT_*.lmdb files remain (realm 1)
    (tmp_path / "test_index" / "index.lmdb").unlink()

    # Step 3: Open index again (no index.lmdb, so _realm_id is None)
    # But simprint files exist with realm 1
    index2 = UsearchIndex(path=tmp_path / "test_index")
    assert index2._realm_id is None  # No index.lmdb
    assert index2._simprint_index.realm_id_int == 1  # But simprint files exist

    # Step 4: Try to add realm 0 asset
    # LMDB will accept it and set _realm_id = 0
    # But simprint index will reject it with ValueError (realm mismatch)
    # This should propagate and fail loudly (not be silently logged)
    with pytest.raises(ValueError, match="realm mismatch"):
        index2.add_assets(sample_assets_with_simprints[:1])  # Realm 0 asset

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

    # Add asset - should not create simprint index files
    results = index.add_assets([asset])
    assert len(results) == 1

    # Verify no simprint files created
    simprint_files = list(tmp_path.glob("test_index/SIMPRINT_*.lmdb"))
    assert len(simprint_files) == 0

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


def test_usearch_conversion_helper(tmp_path, sample_assets_with_simprints, sample_content_units):
    """Test _asset_to_simprint_entry conversion."""
    from iscc_search.indexes.usearch.index import UsearchIndex

    index = UsearchIndex(path=tmp_path / "test_index")

    # Test with asset that has simprints
    asset = sample_assets_with_simprints[0]
    entry = index._asset_to_simprint_entry(asset)

    assert entry is not None
    assert len(entry.iscc_id) == 10  # 2-byte header + 8-byte body
    assert len(entry.simprints) > 0

    # Test with asset without simprints
    asset_no_simprints = IsccEntry(
        iscc_id="ISCC:MAAGXFA6YVEB5A4A", units=[sample_content_units[0], sample_content_units[1]]
    )
    entry2 = index._asset_to_simprint_entry(asset_no_simprints)
    assert entry2 is None

    index.close()


def test_usearch_load_simprint_realm_mismatch(tmp_path, sample_assets_with_simprints):
    """Test realm mismatch detection when loading existing simprint index."""
    import iscc_core as ic
    import shutil

    # Create realm-0 index with assets (but no simprints for simplicity)
    assets_realm0 = [
        IsccEntry(
            iscc_id=sample_assets_with_simprints[0].iscc_id,  # realm 0
            iscc_code=sample_assets_with_simprints[0].iscc_code,
            units=sample_assets_with_simprints[0].units,
            # No simprints
        )
    ]
    index0 = UsearchIndex(path=tmp_path / "realm0_index")
    index0.add_assets(assets_realm0)
    index0.close()

    # Create realm-1 index with assets and simprints
    realm1_iscc_id = ic.gen_iscc_id(timestamp=9999999, hub_id=99, realm_id=1)["iscc"]
    assets_realm1 = [
        IsccEntry(
            iscc_id=realm1_iscc_id,
            iscc_code=sample_assets_with_simprints[0].iscc_code,
            units=sample_assets_with_simprints[0].units,
            simprints=sample_assets_with_simprints[0].simprints,
        )
    ]
    index1 = UsearchIndex(path=tmp_path / "realm1_index")
    index1.add_assets(assets_realm1)
    index1.close()

    # Copy realm-1 simprint files to realm-0 index (creating mismatch)
    for simprint_file in (tmp_path / "realm1_index").glob("SIMPRINT_*.lmdb"):
        shutil.copy(simprint_file, tmp_path / "realm0_index" / simprint_file.name)

    # Try to reopen realm-0 index - should detect realm mismatch
    with pytest.raises(ValueError, match="Realm ID mismatch"):
        UsearchIndex(path=tmp_path / "realm0_index")


def test_usearch_load_simprint_error(tmp_path, monkeypatch):
    """Test error handling when simprint index fails to load."""
    import iscc_search.indexes.usearch.index as usearch_module

    # Mock LmdbSimprintIndexMulti to raise exception
    def mock_init(*args, **kwargs):
        raise RuntimeError("Simulated simprint load failure")

    monkeypatch.setattr(usearch_module, "LmdbSimprintIndexMulti", mock_init)

    # Try to create index - should raise exception from simprint loading
    with pytest.raises(RuntimeError, match="Simulated simprint load failure"):
        UsearchIndex(path=tmp_path / "test_index")
