"""
Tests for UsearchIndex persistence features: save-on-close and auto-rebuild.
"""

import iscc_core as ic
from iscc_search.indexes.usearch.index import UsearchIndex
from iscc_search.schema import IsccAsset


def test_usearch_index_save_on_close(tmp_path, sample_iscc_ids):
    """Test that NphdIndex files are saved on close() and loaded correctly."""
    index_path = tmp_path / "save_on_close"

    # Create index and add assets
    idx = UsearchIndex(index_path, realm_id=0, max_dim=256)
    content_unit = ic.gen_text_code_v0("Test content for save on close")["iscc"]
    instance_unit = f"ISCC:{ic.Code.rnd(ic.MT.INSTANCE, bits=128)}"
    asset = IsccAsset(
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
    query = IsccAsset(units=[instance_unit, content_unit])
    result = idx2.search_assets(query, limit=10)

    assert len(result.matches) == 1
    assert result.matches[0].iscc_id == sample_iscc_ids[0]

    idx2.close()


def test_usearch_index_flush_method(tmp_path, sample_iscc_ids):
    """Test explicit flush() saves NphdIndex files without closing."""
    index_path = tmp_path / "flush_test"

    idx = UsearchIndex(index_path, realm_id=0, max_dim=256)
    content_unit = ic.gen_text_code_v0("Test content for flush")["iscc"]
    instance_unit = f"ISCC:{ic.Code.rnd(ic.MT.INSTANCE, bits=128)}"
    asset = IsccAsset(
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
    query = IsccAsset(units=[instance_unit, content_unit])
    result = idx.search_assets(query, limit=10)
    assert len(result.matches) == 1

    idx.close()


def test_usearch_index_auto_rebuild_on_corrupted_file(tmp_path, sample_iscc_ids):
    """Test auto-rebuild when .usearch file exists but is corrupted."""
    index_path = tmp_path / "rebuild_corrupted"

    # Create index and add assets (including one without units to cover that branch)
    idx = UsearchIndex(index_path, realm_id=0, max_dim=256)
    content_unit = ic.gen_text_code_v0("Test content for rebuild corrupted")["iscc"]
    instance_unit = f"ISCC:{ic.Code.rnd(ic.MT.INSTANCE, bits=128)}"

    # Add normal asset
    asset1 = IsccAsset(
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
    query = IsccAsset(units=[instance_unit, content_unit])
    result = idx2.search_assets(query, limit=10)
    assert len(result.matches) == 1
    assert result.matches[0].iscc_id == sample_iscc_ids[0]

    idx2.close()


def test_usearch_index_auto_rebuild_on_count_mismatch(tmp_path, sample_iscc_ids):
    """Test auto-rebuild when vector count doesn't match metadata."""
    index_path = tmp_path / "rebuild_mismatch"

    # Create index and add assets
    idx = UsearchIndex(index_path, realm_id=0, max_dim=256)
    content_unit_1 = ic.gen_text_code_v0("Content 1 for count mismatch")["iscc"]
    content_unit_2 = ic.gen_text_code_v0("Content 2 for count mismatch")["iscc"]
    instance_unit = f"ISCC:{ic.Code.rnd(ic.MT.INSTANCE, bits=128)}"

    asset1 = IsccAsset(iscc_id=sample_iscc_ids[0], units=[instance_unit, content_unit_1])
    asset2 = IsccAsset(iscc_id=sample_iscc_ids[1], units=[instance_unit, content_unit_2])

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
    result = idx2.search_assets(IsccAsset(units=[instance_unit, content_unit_1, content_unit_2]), limit=10)
    assert len(result.matches) == 2

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
    asset = IsccAsset(iscc_id=sample_iscc_ids[0], units=[instance_unit, content_unit])
    idx.add_assets([asset])

    # Metadata should be updated
    count = idx._get_nphd_metadata("CONTENT_TEXT_V0")
    assert count == 1

    # Add another asset
    content_unit2 = ic.gen_text_code_v0("Another test for metadata tracking")["iscc"]
    asset2 = IsccAsset(iscc_id=sample_iscc_ids[1], units=[instance_unit, content_unit2])
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
    asset = IsccAsset(iscc_id=sample_iscc_ids[0], units=[instance_unit, content_unit])
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
    asset = IsccAsset(iscc_id=sample_iscc_ids[0], units=[instance_unit, content_unit])

    # Add asset but don't close
    idx.add_assets([asset])

    # .usearch file should NOT exist yet (save-on-close only)
    usearch_file = index_path / "CONTENT_TEXT_V0.usearch"
    assert not usearch_file.exists(), "NphdIndex file should not exist before close()"

    # Close to save
    idx.close()

    # Now file should exist
    assert usearch_file.exists(), "NphdIndex file should exist after close()"
