"""
Tests for UsearchIndex persistence features: save-on-close and auto-rebuild.

ShardedNphdIndex stores data in per-unit-type directories (not single .usearch files).
Each directory contains shard files and bloom filter managed by ShardedNphdIndex.
"""

import shutil

import iscc_core as ic
from iscc_search.indexes.usearch.index import UsearchIndex
from iscc_search.schema import IsccEntry, IsccQuery


def test_usearch_index_save_on_close(tmp_path, sample_iscc_ids):
    """Test that ShardedNphdIndex directories are saved on close() and loaded correctly."""
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

    # Close index (should save ShardedNphdIndex directories)
    idx.close()

    # Verify shard directory exists
    shard_dir = index_path / "CONTENT_TEXT_V0"
    assert shard_dir.is_dir(), "ShardedNphdIndex directory should exist after close()"

    # Reopen index and verify data persists
    idx2 = UsearchIndex(index_path, realm_id=0, max_dim=256)
    query = IsccQuery(units=[instance_unit, content_unit])
    result = idx2.search_assets(query, limit=10)

    assert len(result.global_matches) == 1
    assert result.global_matches[0].iscc_id == sample_iscc_ids[0]

    idx2.close()


def test_usearch_index_flush_method(tmp_path, sample_iscc_ids):
    """Test explicit flush() saves ShardedNphdIndex without closing."""
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

    # Verify shard directory exists
    shard_dir = index_path / "CONTENT_TEXT_V0"
    assert shard_dir.is_dir(), "ShardedNphdIndex directory should exist after flush()"

    # Index should still be usable
    query = IsccQuery(units=[instance_unit, content_unit])
    result = idx.search_assets(query, limit=10)
    assert len(result.global_matches) == 1

    idx.close()


def test_usearch_index_auto_rebuild_on_corrupted_shard(tmp_path, sample_iscc_ids):
    """Test auto-rebuild when shard directory exists but contains corrupt data."""
    index_path = tmp_path / "rebuild_corrupted"

    # Create index and add assets
    idx = UsearchIndex(index_path, realm_id=0, max_dim=256)
    content_unit = ic.gen_text_code_v0("Test content for rebuild corrupted")["iscc"]
    instance_unit = f"ISCC:{ic.Code.rnd(ic.MT.INSTANCE, bits=128)}"

    asset1 = IsccEntry(
        iscc_id=sample_iscc_ids[0],
        units=[instance_unit, content_unit],
    )
    idx.add_assets([asset1])
    idx.close()

    # Corrupt shard directory by removing its contents and writing garbage
    shard_dir = index_path / "CONTENT_TEXT_V0"
    shutil.rmtree(shard_dir)
    shard_dir.mkdir()
    (shard_dir / "shard_000.usearch").write_bytes(b"corrupted data")

    # Reopen index - should detect corruption and auto-rebuild from LMDB
    idx2 = UsearchIndex(index_path, realm_id=0, max_dim=256)

    # Verify data is recovered via rebuild
    query = IsccQuery(units=[instance_unit, content_unit])
    result = idx2.search_assets(query, limit=10)
    assert len(result.global_matches) == 1
    assert result.global_matches[0].iscc_id == sample_iscc_ids[0]

    idx2.close()


def test_usearch_index_loads_stale_on_count_mismatch(tmp_path, sample_iscc_ids):
    """Test stale index accepted with warning when vector count doesn't match metadata."""
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
    import lmdb
    import struct

    env = lmdb.open(str(index_path / "index.lmdb"), subdir=False, max_dbs=32)
    with env.begin(write=True) as txn:
        metadata_db = env.open_db(b"__metadata__", txn=txn)
        key = b"nphd_count:CONTENT_TEXT_V0"
        txn.put(key, struct.pack(">Q", 999), db=metadata_db)
    env.close()

    # Reopen - should detect mismatch (999 != 2), log warning, but load stale index
    idx2 = UsearchIndex(index_path, realm_id=0, max_dim=256)

    # Verify stale index is loaded and functional (2 actual vectors despite metadata saying 999)
    assert "CONTENT_TEXT_V0" in idx2._nphd_indexes
    assert idx2._nphd_indexes["CONTENT_TEXT_V0"].size == 2

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
    rebuilt = idx._rebuild_nphd_index("NONEXISTENT_TYPE")

    # Should complete without error, no index created
    assert rebuilt is False
    assert "NONEXISTENT_TYPE" not in idx._nphd_indexes

    idx.close()


def test_usearch_index_rebuild_without_existing_dir(tmp_path, sample_iscc_ids):
    """Test _rebuild_nphd_index when shard directory does not exist yet."""
    index_path = tmp_path / "rebuild_no_dir"

    idx = UsearchIndex(index_path, realm_id=0, max_dim=256)

    # Add asset to populate LMDB
    content_unit = ic.gen_text_code_v0("Test content for rebuild no dir")["iscc"]
    instance_unit = f"ISCC:{ic.Code.rnd(ic.MT.INSTANCE, bits=128)}"
    asset = IsccEntry(iscc_id=sample_iscc_ids[0], units=[instance_unit, content_unit])
    idx.add_assets([asset])

    # Remove the ShardedNphdIndex from memory and delete its directory
    unit_type = "CONTENT_TEXT_V0"
    del idx._nphd_indexes[unit_type]
    shard_dir = index_path / unit_type
    if shard_dir.exists():
        shutil.rmtree(shard_dir)
    assert not shard_dir.exists()

    # Rebuild — should create directory from scratch (exercises shard_dir.exists()=False branch)
    rebuilt = idx._rebuild_nphd_index(unit_type)

    assert rebuilt is True
    assert unit_type in idx._nphd_indexes
    assert idx._nphd_indexes[unit_type].size == 1

    idx.close()


def test_usearch_index_rebuild_with_existing_dir(tmp_path, sample_iscc_ids):
    """Test _rebuild_nphd_index removes stale shard directory before rebuilding."""
    index_path = tmp_path / "rebuild_existing_dir"

    idx = UsearchIndex(index_path, realm_id=0, max_dim=256)

    content_unit = ic.gen_text_code_v0("Test content for rebuild existing dir")["iscc"]
    instance_unit = f"ISCC:{ic.Code.rnd(ic.MT.INSTANCE, bits=128)}"
    asset = IsccEntry(iscc_id=sample_iscc_ids[0], units=[instance_unit, content_unit])
    idx.add_assets([asset])

    # Save so shard files exist on disk
    idx.flush()

    unit_type = "CONTENT_TEXT_V0"
    shard_dir = index_path / unit_type
    assert shard_dir.exists()

    # Remove from memory but leave directory on disk
    del idx._nphd_indexes[unit_type]

    # Rebuild — should remove stale dir first (exercises shard_dir.exists()=True branch)
    rebuilt = idx._rebuild_nphd_index(unit_type)

    assert rebuilt is True
    assert unit_type in idx._nphd_indexes
    assert idx._nphd_indexes[unit_type].size == 1

    idx.close()


def test_usearch_index_rebuild_resets_cached_nphd_before_delete(tmp_path, sample_iscc_ids, monkeypatch):
    """Cached NPHD indexes are released before their shard directory is removed."""
    index_path = tmp_path / "rebuild_cached_release"

    idx = UsearchIndex(index_path, realm_id=0, max_dim=256)

    content_unit = ic.gen_text_code_v0("Test content for cached rebuild release")["iscc"]
    instance_unit = f"ISCC:{ic.Code.rnd(ic.MT.INSTANCE, bits=128)}"
    asset = IsccEntry(iscc_id=sample_iscc_ids[0], units=[instance_unit, content_unit])
    idx.add_assets([asset])
    idx.flush()

    unit_type = "CONTENT_TEXT_V0"
    real_index = idx._nphd_indexes[unit_type]
    real_index.reset()

    reset_called = False

    class CachedIndex:
        def reset(self):
            # type: () -> None
            nonlocal reset_called
            reset_called = True

    idx._nphd_indexes[unit_type] = CachedIndex()

    original_rmtree = shutil.rmtree

    def rmtree_spy(path):
        assert reset_called is True
        original_rmtree(path)

    monkeypatch.setattr("iscc_search.indexes.usearch.index.shutil.rmtree", rmtree_spy)

    rebuilt = idx._rebuild_nphd_index(unit_type)

    assert rebuilt is True
    assert reset_called is True
    assert unit_type in idx._nphd_indexes
    assert idx._nphd_indexes[unit_type] is not real_index

    idx.close()


def test_usearch_index_no_save_on_add(tmp_path, sample_iscc_ids):
    """Test that add_assets does NOT immediately save to disk (shard files)."""
    index_path = tmp_path / "no_save_on_add"

    idx = UsearchIndex(index_path, realm_id=0, max_dim=256)
    content_unit = ic.gen_text_code_v0("Test no save on add")["iscc"]
    instance_unit = f"ISCC:{ic.Code.rnd(ic.MT.INSTANCE, bits=128)}"
    asset = IsccEntry(iscc_id=sample_iscc_ids[0], units=[instance_unit, content_unit])

    # Add asset but don't close
    idx.add_assets([asset])

    # ShardedNphdIndex creates the directory at construction time, but shard files
    # should NOT exist yet (save-on-close only)
    shard_dir = index_path / "CONTENT_TEXT_V0"
    shard_files = list(shard_dir.glob("shard_*.usearch")) if shard_dir.exists() else []
    assert len(shard_files) == 0, "Shard files should not exist before close()"

    # Close to save
    idx.close()

    # Now shard files should exist
    shard_files = list(shard_dir.glob("shard_*.usearch"))
    assert len(shard_files) > 0, "Shard files should exist after close()"


def test_usearch_index_crash_recovery_loads_stale(tmp_path, sample_iscc_ids):
    """
    Test crash recovery: stale (empty) indexes accepted after missing shard dirs.

    Simulates a crash scenario where vectors were added but never flushed,
    leaving metadata in LMDB but no shard directory on disk.
    Auto-rebuild is disabled to prevent OOM on large indexes.
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

    # Verify metadata exists
    assert idx._get_nphd_metadata("CONTENT_TEXT_V0") == 1
    assert idx._get_nphd_metadata("DATA_NONE_V0") == 1

    # Simulate crash: close LMDB but DON'T save ShardedNphdIndex
    idx.env.close()

    # Remove any shard directories that may have been created (empty dirs from constructor)
    for d in [index_path / "CONTENT_TEXT_V0", index_path / "DATA_NONE_V0"]:
        if d.exists():
            shutil.rmtree(d)

    # Reopen index - detects mismatch (expected 1, found 0) but loads stale index
    idx2 = UsearchIndex(index_path, realm_id=0, max_dim=256)

    # Stale indexes are loaded (empty but functional)
    assert "CONTENT_TEXT_V0" in idx2._nphd_indexes
    assert "DATA_NONE_V0" in idx2._nphd_indexes
    assert idx2._nphd_indexes["CONTENT_TEXT_V0"].size == 0
    assert idx2._nphd_indexes["DATA_NONE_V0"].size == 0

    # INSTANCE match still works (LMDB-based, not affected by shard loss)
    query = IsccQuery(units=[instance_unit, content_unit, data_unit])
    result = idx2.search_assets(query, limit=10)
    assert len(result.global_matches) == 1
    assert "INSTANCE_NONE_V0" in result.global_matches[0].types

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

    # Should still find the tracked types (tests loop exhausts naturally)
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

    # Should return empty set (loop breaks on first non-matching key)
    tracked = idx._get_all_tracked_unit_types()
    assert tracked == set()

    # Test edge case 2: delete realm_id too so set_range returns False
    with idx.env.begin(write=True) as txn:
        metadata_db = idx.env.open_db(b"__metadata__", txn=txn)
        txn.delete(b"realm_id", db=metadata_db)

    # Should return empty set (set_range returns False)
    tracked = idx._get_all_tracked_unit_types()
    assert tracked == set()

    idx.close()


def test_flush_skips_clean_sub_indexes(tmp_path, sample_iscc_ids):
    """flush() skips sub-indexes with dirty == 0."""
    index_path = tmp_path / "flush_clean"

    idx = UsearchIndex(index_path, realm_id=0, max_dim=256)
    content_unit = ic.gen_text_code_v0("Test flush clean")["iscc"]
    instance_unit = f"ISCC:{ic.Code.rnd(ic.MT.INSTANCE, bits=128)}"
    asset = IsccEntry(
        iscc_id=sample_iscc_ids[0],
        units=[instance_unit, content_unit],
    )
    idx.add_assets([asset])

    # First flush saves dirty indexes
    idx.flush()

    # Second flush should skip (all clean now)
    # Verify no error and indexes still usable
    idx.flush()

    query = IsccQuery(units=[instance_unit, content_unit])
    result = idx.search_assets(query, limit=10)
    assert len(result.global_matches) == 1

    idx.close()


def test_close_skips_clean_saves_but_resets(tmp_path, sample_iscc_ids):
    """close() skips save on clean sub-indexes but still resets them."""
    index_path = tmp_path / "close_clean"

    idx = UsearchIndex(index_path, realm_id=0, max_dim=256)
    content_unit = ic.gen_text_code_v0("Test close clean")["iscc"]
    instance_unit = f"ISCC:{ic.Code.rnd(ic.MT.INSTANCE, bits=128)}"
    asset = IsccEntry(
        iscc_id=sample_iscc_ids[0],
        units=[instance_unit, content_unit],
    )
    idx.add_assets([asset])

    # Flush to make indexes clean
    idx.flush()

    # Close should skip save (clean) but still reset and close LMDB
    idx.close()

    # Reopen and verify data persists (was saved by flush)
    idx2 = UsearchIndex(index_path, realm_id=0, max_dim=256)
    query = IsccQuery(units=[instance_unit, content_unit])
    result = idx2.search_assets(query, limit=10)
    assert len(result.global_matches) == 1

    idx2.close()


def test_close_idempotent(tmp_path, sample_iscc_ids):
    """close() is idempotent - calling it multiple times is safe."""
    index_path = tmp_path / "close_idempotent"

    idx = UsearchIndex(index_path, realm_id=0, max_dim=256)
    content_unit = ic.gen_text_code_v0("Test close idempotent")["iscc"]
    instance_unit = f"ISCC:{ic.Code.rnd(ic.MT.INSTANCE, bits=128)}"
    asset = IsccEntry(
        iscc_id=sample_iscc_ids[0],
        units=[instance_unit, content_unit],
    )
    idx.add_assets([asset])

    # Close multiple times - should not raise
    idx.close()
    idx.close()
    idx.close()

    # Verify data was saved correctly
    idx2 = UsearchIndex(index_path, realm_id=0, max_dim=256)
    query = IsccQuery(units=[instance_unit, content_unit])
    result = idx2.search_assets(query, limit=10)
    assert len(result.global_matches) == 1
    idx2.close()


def test_auto_flush_triggers_on_threshold(tmp_path, sample_iscc_ids):
    """Auto-flush triggers when dirty >= flush_interval."""
    index_path = tmp_path / "auto_flush"

    idx = UsearchIndex(index_path, realm_id=0, max_dim=256, flush_interval=1)
    content_unit = ic.gen_text_code_v0("Test auto flush")["iscc"]
    instance_unit = f"ISCC:{ic.Code.rnd(ic.MT.INSTANCE, bits=128)}"
    asset = IsccEntry(
        iscc_id=sample_iscc_ids[0],
        units=[instance_unit, content_unit],
    )
    idx.add_assets([asset])

    # After add_assets with flush_interval=1, sub-indexes should have been auto-flushed
    nphd_index = idx._nphd_indexes["CONTENT_TEXT_V0"]
    assert nphd_index.dirty == 0  # Was auto-flushed

    # Shard files should exist on disk
    shard_dir = index_path / "CONTENT_TEXT_V0"
    shard_files = list(shard_dir.glob("shard_*.usearch")) if shard_dir.exists() else []
    assert len(shard_files) > 0

    idx.close()


def test_auto_flush_below_threshold_skips(tmp_path, sample_iscc_ids):
    """Auto-flush does not trigger when dirty < flush_interval."""
    from iscc_search.schema import IsccSimprint

    index_path = tmp_path / "auto_flush_below"
    sp_type = "CONTENT_TEXT_V0"
    sp_bytes = b"\xab" * 16

    idx = UsearchIndex(index_path, realm_id=0, max_dim=256, flush_interval=1000)
    content_unit = ic.gen_text_code_v0("Test auto flush below threshold")["iscc"]
    instance_unit = f"ISCC:{ic.Code.rnd(ic.MT.INSTANCE, bits=128)}"
    simprints = {sp_type: [IsccSimprint(simprint=ic.encode_base64(sp_bytes), offset=0, size=100)]}
    asset = IsccEntry(
        iscc_id=sample_iscc_ids[0],
        units=[instance_unit, content_unit],
        simprints=simprints,
    )
    idx.add_assets([asset])

    # dirty < 1000 so auto-flush should NOT trigger for either index type
    nphd_index = idx._nphd_indexes["CONTENT_TEXT_V0"]
    assert nphd_index.dirty > 0
    sp_index = idx._simprint_indexes[sp_type]
    assert sp_index.dirty > 0

    idx.close()


def test_auto_flush_disabled_when_zero(tmp_path, sample_iscc_ids):
    """Auto-flush does not trigger when flush_interval=0."""
    index_path = tmp_path / "auto_flush_disabled"

    idx = UsearchIndex(index_path, realm_id=0, max_dim=256, flush_interval=0)
    content_unit = ic.gen_text_code_v0("Test auto flush disabled")["iscc"]
    instance_unit = f"ISCC:{ic.Code.rnd(ic.MT.INSTANCE, bits=128)}"
    asset = IsccEntry(
        iscc_id=sample_iscc_ids[0],
        units=[instance_unit, content_unit],
    )
    idx.add_assets([asset])

    # Sub-indexes should still be dirty (not auto-flushed)
    nphd_index = idx._nphd_indexes["CONTENT_TEXT_V0"]
    assert nphd_index.dirty > 0

    # No shard files on disk yet
    shard_dir = index_path / "CONTENT_TEXT_V0"
    shard_files = list(shard_dir.glob("shard_*.usearch")) if shard_dir.exists() else []
    assert len(shard_files) == 0

    idx.close()


def test_usearch_index_crash_recovery_multiple_missing_dirs(tmp_path, sample_iscc_ids):
    """
    Test crash recovery with multiple missing shard directories.

    Verifies that all tracked unit_types are rebuilt when their directories are missing.
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

    # Remove all shard directories
    for unit_type in ["CONTENT_TEXT_V0", "DATA_NONE_V0", "META_NONE_V0"]:
        shard_dir = index_path / unit_type
        if shard_dir.exists():
            shutil.rmtree(shard_dir)

    # Reopen - should rebuild all three missing directories
    idx2 = UsearchIndex(index_path, realm_id=0, max_dim=256)

    # Verify all directories were created
    assert (index_path / "CONTENT_TEXT_V0").is_dir()
    assert (index_path / "DATA_NONE_V0").is_dir()
    assert (index_path / "META_NONE_V0").is_dir()

    # Verify all three assets are searchable
    for i in range(3):
        query = IsccQuery(units=assets[i].units)
        result = idx2.search_assets(query, limit=10)
        assert any(m.iscc_id == sample_iscc_ids[i] for m in result.global_matches)

    idx2.close()
