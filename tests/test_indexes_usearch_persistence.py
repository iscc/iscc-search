"""
Tests for UsearchIndex persistence features: save-on-close and auto-rebuild.
"""

import iscc_core as ic
from iscc_search.indexes.usearch.index import UsearchIndex
from iscc_search.schema import IsccEntry


def test_usearch_index_save_on_close(tmp_path, sample_iscc_ids):
    """Test that NphdIndex files are saved on close() and loaded correctly."""
    index_path = tmp_path / "save_on_close"

    # Create index and add assets
    idx = UsearchIndex(index_path, realm_id=0, max_dim=256)
    content_unit = ic.gen_text_code_v0("Test content for save on close")["iscc"]
    instance_unit = f"ISCC:{ic.Code.rnd(ic.MT.INSTANCE, bits=128)}"
    asset = IsccEntry(
        iscc_id=sample_iscc_ids[0],
        units=[instance_unit, content_unit],
    )
    idx.add_assets([asset])

    # Close index (should save NphdIndex files)
    idx.close()

    # Verify .usearch file exists
    usearch_file = index_path / "CONTENT_TEXT_V0.usearch"
    assert usearch_file.exists(), "NphdIndex file should exist after close()"

    # Reopen index and verify data persists
    idx2 = UsearchIndex(index_path, realm_id=0, max_dim=256)
    query = IsccEntry(units=[instance_unit, content_unit])
    result = idx2.search_assets(query, limit=10)

    assert len(result.global_matches) == 1
    assert result.global_matches[0].iscc_id == sample_iscc_ids[0]

    idx2.close()


def test_usearch_index_flush_method(tmp_path, sample_iscc_ids):
    """Test explicit flush() saves NphdIndex files without closing."""
    index_path = tmp_path / "flush_test"

    idx = UsearchIndex(index_path, realm_id=0, max_dim=256)
    content_unit = ic.gen_text_code_v0("Test content for flush")["iscc"]
    instance_unit = f"ISCC:{ic.Code.rnd(ic.MT.INSTANCE, bits=128)}"
    asset = IsccEntry(
        iscc_id=sample_iscc_ids[0],
        units=[instance_unit, content_unit],
    )
    idx.add_assets([asset])

    # Explicit flush (should save without closing)
    idx.flush()

    # Verify .usearch file exists
    usearch_file = index_path / "CONTENT_TEXT_V0.usearch"
    assert usearch_file.exists(), "NphdIndex file should exist after flush()"

    # Index should still be usable
    query = IsccEntry(units=[instance_unit, content_unit])
    result = idx.search_assets(query, limit=10)
    assert len(result.global_matches) == 1

    idx.close()


def test_usearch_index_auto_rebuild_on_corrupted_file(tmp_path, sample_iscc_ids):
    """Test auto-rebuild when .usearch file exists but is corrupted."""
    index_path = tmp_path / "rebuild_corrupted"

    # Create index and add assets (including one without units to cover that branch)
    idx = UsearchIndex(index_path, realm_id=0, max_dim=256)
    content_unit = ic.gen_text_code_v0("Test content for rebuild corrupted")["iscc"]
    instance_unit = f"ISCC:{ic.Code.rnd(ic.MT.INSTANCE, bits=128)}"

    # Add normal asset
    asset1 = IsccEntry(
        iscc_id=sample_iscc_ids[0],
        units=[instance_unit, content_unit],
    )
    idx.add_assets([asset1])
    idx.close()

    # Corrupt .usearch file by writing garbage
    usearch_file = index_path / "CONTENT_TEXT_V0.usearch"
    with open(usearch_file, "wb") as f:
        f.write(b"corrupted data")

    # Reopen index - should detect corruption and auto-rebuild from LMDB
    idx2 = UsearchIndex(index_path, realm_id=0, max_dim=256)

    # Verify data is recovered via rebuild
    query = IsccEntry(units=[instance_unit, content_unit])
    result = idx2.search_assets(query, limit=10)
    assert len(result.global_matches) == 1
    assert result.global_matches[0].iscc_id == sample_iscc_ids[0]

    idx2.close()


def test_usearch_index_auto_rebuild_on_count_mismatch(tmp_path, sample_iscc_ids):
    """Test auto-rebuild when vector count doesn't match metadata."""
    index_path = tmp_path / "rebuild_mismatch"

    # Create index and add assets
    idx = UsearchIndex(index_path, realm_id=0, max_dim=256)
    content_unit_1 = ic.gen_text_code_v0("Content 1 for count mismatch")["iscc"]
    content_unit_2 = ic.gen_text_code_v0("Content 2 for count mismatch")["iscc"]
    instance_unit = f"ISCC:{ic.Code.rnd(ic.MT.INSTANCE, bits=128)}"

    asset1 = IsccEntry(iscc_id=sample_iscc_ids[0], units=[instance_unit, content_unit_1])
    asset2 = IsccEntry(iscc_id=sample_iscc_ids[1], units=[instance_unit, content_unit_2])

    idx.add_assets([asset1, asset2])
    idx.close()

    # Simulate out-of-sync by manually corrupting metadata AFTER close
    # Open just to update metadata, don't load indexes
    import lmdb
    import struct

    env = lmdb.open(str(index_path / "index.lmdb"), subdir=False, max_dbs=3)
    with env.begin(write=True) as txn:
        metadata_db = env.open_db(b"__metadata__", txn=txn)
        key = b"nphd_count:CONTENT_TEXT_V0"
        txn.put(key, struct.pack(">Q", 999), db=metadata_db)
    env.close()

    # Reopen - should detect mismatch (999 != 2) and rebuild
    idx2 = UsearchIndex(index_path, realm_id=0, max_dim=256)

    # Verify both assets are found (rebuild worked)
    result = idx2.search_assets(IsccEntry(units=[instance_unit, content_unit_1, content_unit_2]), limit=10)
    assert len(result.global_matches) == 2

    idx2.close()


def test_usearch_index_metadata_tracking(tmp_path, sample_iscc_ids):
    """Test that vector count metadata is tracked correctly."""
    index_path = tmp_path / "metadata_tracking"

    idx = UsearchIndex(index_path, realm_id=0, max_dim=256)

    # Initially, no metadata
    assert idx._get_nphd_metadata("CONTENT_TEXT_V0") is None

    # Add asset
    content_unit = ic.gen_text_code_v0("Test for metadata tracking")["iscc"]
    instance_unit = f"ISCC:{ic.Code.rnd(ic.MT.INSTANCE, bits=128)}"
    asset = IsccEntry(iscc_id=sample_iscc_ids[0], units=[instance_unit, content_unit])
    idx.add_assets([asset])

    # Metadata should be updated
    count = idx._get_nphd_metadata("CONTENT_TEXT_V0")
    assert count == 1

    # Add another asset
    content_unit2 = ic.gen_text_code_v0("Another test for metadata tracking")["iscc"]
    asset2 = IsccEntry(iscc_id=sample_iscc_ids[1], units=[instance_unit, content_unit2])
    idx.add_assets([asset2])

    # Metadata should be updated again
    count = idx._get_nphd_metadata("CONTENT_TEXT_V0")
    assert count == 2

    idx.close()


def test_usearch_index_rebuild_with_no_vectors(tmp_path, sample_iscc_ids):
    """Test rebuild handles case where no vectors exist for unit_type."""
    index_path = tmp_path / "rebuild_no_vectors"

    idx = UsearchIndex(index_path, realm_id=0, max_dim=256)

    # Add at least one asset to initialize the database
    content_unit = ic.gen_text_code_v0("Test for rebuild with no vectors")["iscc"]
    instance_unit = f"ISCC:{ic.Code.rnd(ic.MT.INSTANCE, bits=128)}"
    asset = IsccEntry(iscc_id=sample_iscc_ids[0], units=[instance_unit, content_unit])
    idx.add_assets([asset])

    # Trigger rebuild for non-existent unit_type
    idx._rebuild_nphd_index("NONEXISTENT_TYPE")

    # Should complete without error, no index created
    assert "NONEXISTENT_TYPE" not in idx._nphd_indexes

    idx.close()


def test_usearch_index_no_save_on_add(tmp_path, sample_iscc_ids):
    """Test that add_assets does NOT immediately save to disk."""
    index_path = tmp_path / "no_save_on_add"

    idx = UsearchIndex(index_path, realm_id=0, max_dim=256)
    content_unit = ic.gen_text_code_v0("Test no save on add")["iscc"]
    instance_unit = f"ISCC:{ic.Code.rnd(ic.MT.INSTANCE, bits=128)}"
    asset = IsccEntry(iscc_id=sample_iscc_ids[0], units=[instance_unit, content_unit])

    # Add asset but don't close
    idx.add_assets([asset])

    # .usearch file should NOT exist yet (save-on-close only)
    usearch_file = index_path / "CONTENT_TEXT_V0.usearch"
    assert not usearch_file.exists(), "NphdIndex file should not exist before close()"

    # Close to save
    idx.close()

    # Now file should exist
    assert usearch_file.exists(), "NphdIndex file should exist after close()"


def test_usearch_index_crash_recovery_rebuild_missing_files(tmp_path, sample_iscc_ids):
    """
    Test crash recovery: rebuild missing .usearch files from LMDB on startup.

    Simulates a crash scenario where vectors were added but never flushed,
    leaving metadata in LMDB but no .usearch file on disk.
    """
    index_path = tmp_path / "crash_recovery"

    # Create index and add assets with multiple unit types
    idx = UsearchIndex(index_path, realm_id=0, max_dim=256)
    content_unit = ic.gen_text_code_v0("Test content for crash recovery")["iscc"]
    data_unit = f"ISCC:{ic.Code.rnd(ic.MT.DATA, bits=128)}"
    instance_unit = f"ISCC:{ic.Code.rnd(ic.MT.INSTANCE, bits=128)}"

    asset = IsccEntry(
        iscc_id=sample_iscc_ids[0],
        units=[instance_unit, content_unit, data_unit],
    )
    idx.add_assets([asset])

    # Verify metadata exists but files don't
    assert idx._get_nphd_metadata("CONTENT_TEXT_V0") == 1
    assert idx._get_nphd_metadata("DATA_NONE_V0") == 1

    content_file = index_path / "CONTENT_TEXT_V0.usearch"
    data_file = index_path / "DATA_NONE_V0.usearch"
    assert not content_file.exists(), "Files should not exist before close()"
    assert not data_file.exists(), "Files should not exist before close()"

    # Simulate crash: close LMDB but DON'T save NphdIndex files
    # This leaves metadata in LMDB but no .usearch files on disk
    idx.env.close()  # Direct env close, bypassing UsearchIndex.close()

    # Verify files still don't exist (simulating crash before flush)
    assert not content_file.exists()
    assert not data_file.exists()

    # Reopen index - should detect missing files and auto-rebuild from LMDB
    idx2 = UsearchIndex(index_path, realm_id=0, max_dim=256)

    # Verify files were created by auto-rebuild
    assert content_file.exists(), "Missing file should be rebuilt on startup"
    assert data_file.exists(), "Missing file should be rebuilt on startup"

    # Verify data is accessible via search (proving rebuild worked)
    query = IsccEntry(units=[instance_unit, content_unit, data_unit])
    result = idx2.search_assets(query, limit=10)
    assert len(result.global_matches) == 1
    assert result.global_matches[0].iscc_id == sample_iscc_ids[0]

    # Verify both unit types are searchable
    assert "CONTENT_TEXT_V0" in result.global_matches[0].matches
    assert "DATA_NONE_V0" in result.global_matches[0].matches

    idx2.close()


def test_usearch_index_get_all_tracked_unit_types(tmp_path, sample_iscc_ids):
    """Test _get_all_tracked_unit_types correctly scans metadata."""
    index_path = tmp_path / "tracked_types"

    idx = UsearchIndex(index_path, realm_id=0, max_dim=256)

    # Initially empty
    tracked = idx._get_all_tracked_unit_types()
    assert tracked == set()

    # Add assets with different unit types
    content_unit = ic.gen_text_code_v0("Test content")["iscc"]
    data_unit = f"ISCC:{ic.Code.rnd(ic.MT.DATA, bits=128)}"
    instance_unit = f"ISCC:{ic.Code.rnd(ic.MT.INSTANCE, bits=128)}"

    asset1 = IsccEntry(
        iscc_id=sample_iscc_ids[0],
        units=[instance_unit, content_unit],
    )
    asset2 = IsccEntry(
        iscc_id=sample_iscc_ids[1],
        units=[instance_unit, data_unit],
    )
    idx.add_assets([asset1, asset2])

    # Check tracked types (INSTANCE is not similarity unit, shouldn't appear)
    tracked = idx._get_all_tracked_unit_types()
    assert tracked == {"CONTENT_TEXT_V0", "DATA_NONE_V0"}

    # Test natural loop exhaustion: delete realm_id so nphd_count keys are last in DB
    # This tests the branch where the for loop completes naturally without breaking
    import struct

    with idx.env.begin(write=True) as txn:
        metadata_db = idx.env.open_db(b"__metadata__", txn=txn)
        txn.delete(b"realm_id", db=metadata_db)

    # Should still find the tracked types (tests 480->491: loop exhausts naturally)
    tracked = idx._get_all_tracked_unit_types()
    assert tracked == {"CONTENT_TEXT_V0", "DATA_NONE_V0"}

    # Test edge case 1: delete nphd_count keys but keep realm_id (add it back first)
    with idx.env.begin(write=True) as txn:
        metadata_db = idx.env.open_db(b"__metadata__", txn=txn)
        # Add realm_id back
        txn.put(b"realm_id", struct.pack(">I", 0), db=metadata_db)

    # Test edge case 2: delete nphd_count keys but keep realm_id
    # This tests the branch where set_range returns True but loop breaks immediately
    with idx.env.begin(write=True) as txn:
        metadata_db = idx.env.open_db(b"__metadata__", txn=txn)
        cursor = txn.cursor(metadata_db)
        # Delete all nphd_count:* keys
        prefix = b"nphd_count:"
        if cursor.set_range(prefix):
            keys_to_delete = []
            for key_bytes, _ in cursor:
                if not key_bytes.startswith(prefix):
                    break
                keys_to_delete.append(key_bytes)
            for key in keys_to_delete:
                txn.delete(key, db=metadata_db)

    # Should return empty set (tests 480->491 branch: loop breaks on first non-matching key)
    tracked = idx._get_all_tracked_unit_types()
    assert tracked == set()

    # Test edge case 2: delete realm_id too so set_range returns False
    with idx.env.begin(write=True) as txn:
        metadata_db = idx.env.open_db(b"__metadata__", txn=txn)
        txn.delete(b"realm_id", db=metadata_db)

    # Should return empty set (tests 479->491 branch: set_range returns False)
    tracked = idx._get_all_tracked_unit_types()
    assert tracked == set()

    idx.close()


def test_usearch_index_crash_recovery_multiple_missing_files(tmp_path, sample_iscc_ids):
    """
    Test crash recovery with multiple missing .usearch files.

    Verifies that all tracked unit_types are rebuilt when their files are missing.
    """
    index_path = tmp_path / "multi_crash_recovery"

    # Create index and add assets with three different similarity unit types
    idx = UsearchIndex(index_path, realm_id=0, max_dim=256)

    assets = []
    for i in range(3):
        content_unit = ic.gen_text_code_v0(f"Content {i}")["iscc"]
        data_unit = f"ISCC:{ic.Code.rnd(ic.MT.DATA, bits=128)}"
        meta_unit = ic.gen_meta_code_v0(f"Asset {i}")["iscc"]
        instance_unit = f"ISCC:{ic.Code.rnd(ic.MT.INSTANCE, bits=128)}"

        assets.append(
            IsccEntry(
                iscc_id=sample_iscc_ids[i],
                units=[instance_unit, content_unit, data_unit, meta_unit],
            )
        )

    idx.add_assets(assets)

    # Verify metadata tracked for all three types
    tracked = idx._get_all_tracked_unit_types()
    assert tracked == {"CONTENT_TEXT_V0", "DATA_NONE_V0", "META_NONE_V0"}

    # Simulate crash
    idx.env.close()

    # Verify no .usearch files exist
    assert not (index_path / "CONTENT_TEXT_V0.usearch").exists()
    assert not (index_path / "DATA_NONE_V0.usearch").exists()
    assert not (index_path / "META_NONE_V0.usearch").exists()

    # Reopen - should rebuild all three missing files
    idx2 = UsearchIndex(index_path, realm_id=0, max_dim=256)

    # Verify all files were created
    assert (index_path / "CONTENT_TEXT_V0.usearch").exists()
    assert (index_path / "DATA_NONE_V0.usearch").exists()
    assert (index_path / "META_NONE_V0.usearch").exists()

    # Verify all three assets are searchable
    for i in range(3):
        asset_query = assets[i]
        result = idx2.search_assets(asset_query, limit=10)
        assert any(m.iscc_id == sample_iscc_ids[i] for m in result.global_matches)

    idx2.close()
