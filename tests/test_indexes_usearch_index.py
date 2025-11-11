"""
Tests for UsearchIndex class - low-level index implementation.

Tests UsearchIndex directly to cover edge cases not exercised through UsearchIndexManager.
"""

import io
import pytest
import iscc_core as ic
from iscc_search.indexes.usearch.index import UsearchIndex
from iscc_search.schema import IsccEntry


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
    asset = IsccEntry(
        iscc_id=sample_iscc_ids[0],
        units=[instance_str, content_unit],
        metadata={"test": "256bit_instance"},
    )
    usearch_index.add_assets([asset])

    # Search with 256-bit query - should match itself and return metadata
    query = IsccEntry(units=[instance_str, content_unit])
    result = usearch_index.search_assets(query, limit=10)

    assert len(result.global_matches) == 1
    assert result.global_matches[0].iscc_id == sample_iscc_ids[0]
    assert result.global_matches[0].metadata.model_dump(exclude_none=True) == {"test": "256bit_instance"}

    # Search with 128-bit - tests the reverse matching code path (query_len == 32 branch)
    query_128 = IsccEntry(units=[instance_128, content_unit])
    _ = usearch_index.search_assets(query_128, limit=10)
    # May or may not match depending on prefix - just testing code path executes

    # Search with 64-bit - tests the reverse matching code path (query_len >= 16 branch)
    query_64 = IsccEntry(units=[instance_64, content_unit])
    _ = usearch_index.search_assets(query_64, limit=10)
    # May or may not match depending on prefix - just testing code path executes


def test_usearch_index_bidirectional_instance_matching_128bit(usearch_index, sample_iscc_ids):
    """Test bidirectional prefix matching with 128-bit INSTANCE codes."""
    # Create 128-bit INSTANCE code (16 bytes)
    instance_128 = ic.Code.rnd(ic.MT.INSTANCE, bits=128)
    instance_str = f"ISCC:{instance_128}"

    # Add asset with 128-bit INSTANCE
    content_unit = ic.gen_text_code_v0("Test content 128")["iscc"]
    asset = IsccEntry(
        iscc_id=sample_iscc_ids[1],
        units=[instance_str, content_unit],
    )
    usearch_index.add_assets([asset])

    # Search with 128-bit query - should match itself
    query = IsccEntry(units=[instance_str, content_unit])
    result = usearch_index.search_assets(query, limit=10)

    assert len(result.global_matches) == 1
    assert result.global_matches[0].iscc_id == sample_iscc_ids[1]

    # Create a different 64-bit code to test the reverse matching path (query_len >= 16 branch)
    instance_64_obj = ic.Code.rnd(ic.MT.INSTANCE, bits=64)
    instance_64 = f"ISCC:{instance_64_obj}"
    query_64 = IsccEntry(units=[instance_64, content_unit])
    _ = usearch_index.search_assets(query_64, limit=10)
    # Just testing the code path executes


def test_usearch_index_instance_proportional_scoring(usearch_index, sample_iscc_ids):
    """Test that INSTANCE matches are scored proportionally to match length."""
    # Create INSTANCE codes of different lengths with shared prefixes
    # We'll use predictable prefixes to ensure matches
    base_bytes = bytes([1, 2, 3, 4, 5, 6, 7, 8])  # 64 bits
    prefix_128 = base_bytes + bytes([9, 10, 11, 12, 13, 14, 15, 16])  # 128 bits
    prefix_256 = prefix_128 + bytes(range(17, 33))  # 256 bits

    # Create ISCC codes from these bytes
    ic_64 = "ISCC:" + ic.encode_component(ic.MT.INSTANCE, ic.ST.NONE, ic.VS.V0, 64, base_bytes)
    ic_128 = "ISCC:" + ic.encode_component(ic.MT.INSTANCE, ic.ST.NONE, ic.VS.V0, 128, prefix_128)
    ic_256 = "ISCC:" + ic.encode_component(ic.MT.INSTANCE, ic.ST.NONE, ic.VS.V0, 256, prefix_256)

    # Add assets with different INSTANCE lengths
    content_unit = ic.gen_text_code_v0("Shared content for scoring test")["iscc"]

    asset_64 = IsccEntry(iscc_id=sample_iscc_ids[0], units=[ic_64, content_unit])
    asset_128 = IsccEntry(iscc_id=sample_iscc_ids[1], units=[ic_128, content_unit])
    asset_256 = IsccEntry(iscc_id=sample_iscc_ids[2], units=[ic_256, content_unit])

    usearch_index.add_assets([asset_64, asset_128, asset_256])

    # Test 1: 64-bit exact match should score 0.25
    query_64 = IsccEntry(units=[ic_64, content_unit])
    result_64 = usearch_index.search_assets(query_64, limit=10)

    # Find the exact match
    match_64 = next(m for m in result_64.global_matches if m.iscc_id == sample_iscc_ids[0])
    assert match_64.types["INSTANCE_NONE_V0"] == 0.25, "64-bit exact match should score 0.25"

    # Test 2: 128-bit exact match should score 0.5
    query_128 = IsccEntry(units=[ic_128, content_unit])
    result_128 = usearch_index.search_assets(query_128, limit=10)

    match_128 = next(m for m in result_128.global_matches if m.iscc_id == sample_iscc_ids[1])
    assert match_128.types["INSTANCE_NONE_V0"] == 0.5, "128-bit exact match should score 0.5"

    # Test 3: 256-bit exact match should score 1.0
    query_256 = IsccEntry(units=[ic_256, content_unit])
    result_256 = usearch_index.search_assets(query_256, limit=10)

    match_256 = next(m for m in result_256.global_matches if m.iscc_id == sample_iscc_ids[2])
    assert match_256.types["INSTANCE_NONE_V0"] == 1.0, "256-bit exact match should score 1.0"

    # Test 4: Forward prefix match - 64-bit query matches 256-bit stored (64-bit overlap)
    # Query with 64-bit should match all three (they all share the 64-bit prefix)
    all_matches_64 = {m.iscc_id for m in result_64.global_matches}
    assert sample_iscc_ids[0] in all_matches_64  # Exact 64-bit match
    assert sample_iscc_ids[1] in all_matches_64  # 128-bit starts with 64-bit prefix
    assert sample_iscc_ids[2] in all_matches_64  # 256-bit starts with 64-bit prefix

    # All should score 0.25 (64-bit overlap)
    for match in result_64.global_matches:
        if match.iscc_id in [sample_iscc_ids[0], sample_iscc_ids[1], sample_iscc_ids[2]]:
            assert match.types["INSTANCE_NONE_V0"] == 0.25

    # Test 5: Reverse prefix match - 256-bit query matches 64-bit stored
    # The 256-bit query contains the 64-bit and 128-bit as prefixes, so should match via reverse search
    all_matches_256 = {m.iscc_id for m in result_256.global_matches}
    assert sample_iscc_ids[2] in all_matches_256  # Exact 256-bit match (score 1.0)
    assert sample_iscc_ids[1] in all_matches_256  # 128-bit is prefix (score 0.5)
    assert sample_iscc_ids[0] in all_matches_256  # 64-bit is prefix (score 0.25)

    # Verify individual scores
    match_256_exact = next(m for m in result_256.global_matches if m.iscc_id == sample_iscc_ids[2])
    match_256_to_128 = next(m for m in result_256.global_matches if m.iscc_id == sample_iscc_ids[1])
    match_256_to_64 = next(m for m in result_256.global_matches if m.iscc_id == sample_iscc_ids[0])

    assert match_256_exact.types["INSTANCE_NONE_V0"] == 1.0
    assert match_256_to_128.types["INSTANCE_NONE_V0"] == 0.5
    assert match_256_to_64.types["INSTANCE_NONE_V0"] == 0.25


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


def test_usearch_index_get_asset_realm_mismatch(tmp_path, sample_assets):
    """Test get_asset raises ValueError for ISCC-ID with different realm."""
    index_path = tmp_path / "realm_index"
    idx = UsearchIndex(index_path, realm_id=0, max_dim=256)

    # Add asset with realm=0
    idx.add_assets([sample_assets[0]])

    # Try to get asset with realm=1 ISCC-ID
    iscc_id_realm1 = ic.gen_iscc_id(timestamp=9999999, hub_id=99, realm_id=1)["iscc"]

    with pytest.raises(ValueError, match="Realm mismatch"):
        idx.get_asset(iscc_id_realm1)

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

    asset = IsccEntry(
        iscc_id=iscc_id,
        units=[instance_unit, content_unit],
    )
    idx.add_assets([asset])

    # Search with CONTENT unit that doesn't exist in index (unit type not in _nphd_indexes)
    # This tests the branch: if unit_type in self._nphd_indexes
    different_content = ic.gen_text_code_v0("Different content completely unique")["iscc"]
    query = IsccEntry(units=[different_content, instance_unit])
    result = idx.search_assets(query, limit=10)

    # Should still find via INSTANCE match even though CONTENT not in index
    assert len(result.global_matches) >= 1
    idx.close()


def test_usearch_index_retry_limits_configured():
    """Test that retry limit constants are properly configured."""
    # Verify constants exist and have sensible values
    assert hasattr(UsearchIndex, "MAX_RESIZE_RETRIES")
    assert hasattr(UsearchIndex, "MAX_MAP_SIZE")

    # Verify they have reasonable values
    assert UsearchIndex.MAX_RESIZE_RETRIES == 10, "MAX_RESIZE_RETRIES should be 10"
    assert UsearchIndex.MAX_MAP_SIZE == 1024 * 1024 * 1024 * 1024, "MAX_MAP_SIZE should be 1TB"

    # Verify they are positive integers
    assert isinstance(UsearchIndex.MAX_RESIZE_RETRIES, int)
    assert isinstance(UsearchIndex.MAX_MAP_SIZE, int)
    assert UsearchIndex.MAX_RESIZE_RETRIES > 0
    assert UsearchIndex.MAX_MAP_SIZE > 0


def test_usearch_index_max_dim_persistence(tmp_path, sample_iscc_ids):
    """Test that max_dim is persisted and loaded correctly on index reopening."""
    index_path = tmp_path / "max_dim_test"

    # Create index with non-default max_dim
    idx = UsearchIndex(index_path, realm_id=0, max_dim=128)
    assert idx.max_dim == 128, "Initial index should have max_dim=128"

    # Add some data to create unit indexes
    instance_unit = f"ISCC:{ic.Code.rnd(ic.MT.INSTANCE, bits=128)}"
    content_unit = ic.gen_text_code_v0("Test content for max_dim")["iscc"]
    asset1 = IsccEntry(
        iscc_id=sample_iscc_ids[0],
        units=[instance_unit, content_unit],
    )
    idx.add_assets([asset1])

    # Verify the created NphdIndex has correct max_dim
    unit_type = "CONTENT_TEXT_V0"
    assert unit_type in idx._nphd_indexes
    assert idx._nphd_indexes[unit_type].max_dim == 128

    # Close the index
    idx.close()

    # Reopen WITHOUT specifying max_dim (should load from metadata)
    idx_reopened = UsearchIndex(index_path, realm_id=0)
    assert idx_reopened.max_dim == 128, "Reopened index should load max_dim=128 from metadata"

    # Add data that creates a NEW unit index to verify it uses loaded max_dim
    instance_unit2 = f"ISCC:{ic.Code.rnd(ic.MT.INSTANCE, bits=128)}"
    data_unit = ic.gen_data_code_v0(io.BytesIO(b"Some binary data for testing"))["iscc"]
    asset2 = IsccEntry(
        iscc_id=sample_iscc_ids[1],
        units=[instance_unit2, data_unit],
    )
    idx_reopened.add_assets([asset2])

    # Verify the new NphdIndex also has correct max_dim
    data_unit_type = "DATA_NONE_V0"
    assert data_unit_type in idx_reopened._nphd_indexes
    assert idx_reopened._nphd_indexes[data_unit_type].max_dim == 128, "New unit index should use loaded max_dim"

    idx_reopened.close()


def test_usearch_index_duplicate_iscc_ids_in_batch(usearch_index, sample_iscc_ids):
    """Test that adding multiple assets with same ISCC-ID in batch handles deduplication."""
    # Create multiple assets with the same ISCC-ID but different content
    content_unit_1 = ic.gen_text_code_v0("First version of content")["iscc"]
    content_unit_2 = ic.gen_text_code_v0("Second version of content")["iscc"]
    content_unit_3 = ic.gen_text_code_v0("Third version of content")["iscc"]

    instance_unit = f"ISCC:{ic.Code.rnd(ic.MT.INSTANCE, bits=128)}"

    # Create three assets with the same ISCC-ID
    asset1 = IsccEntry(
        iscc_id=sample_iscc_ids[0],
        units=[instance_unit, content_unit_1],
        metadata={"version": 1},
    )
    asset2 = IsccEntry(
        iscc_id=sample_iscc_ids[0],  # Same ISCC-ID
        units=[instance_unit, content_unit_2],
        metadata={"version": 2},
    )
    asset3 = IsccEntry(
        iscc_id=sample_iscc_ids[0],  # Same ISCC-ID
        units=[instance_unit, content_unit_3],
        metadata={"version": 3},
    )

    # Add all three in a single batch - should handle deduplication internally
    results = usearch_index.add_assets([asset1, asset2, asset3])

    # Should get 3 results (one for each add operation)
    assert len(results) == 3
    assert results[0].status == "created"
    assert results[1].status == "updated"  # Second overwrites first
    assert results[2].status == "updated"  # Third overwrites second

    # Verify that the last asset's metadata is stored
    retrieved = usearch_index.get_asset(sample_iscc_ids[0])
    assert retrieved.metadata["version"] == 3

    # Verify search works (last content unit should be indexed)
    query = IsccEntry(units=[instance_unit, content_unit_3])
    result = usearch_index.search_assets(query, limit=10)
    assert len(result.global_matches) >= 1
    assert sample_iscc_ids[0] in [m.iscc_id for m in result.global_matches]


def test_usearch_index_infer_realm_from_first_asset(tmp_path, sample_iscc_ids):
    """Test that realm_id is inferred from first asset when not specified."""
    index_path = tmp_path / "infer_realm"

    # Create index WITHOUT specifying realm_id
    idx = UsearchIndex(index_path, realm_id=None, max_dim=256)

    # realm_id should be None before adding assets
    assert idx._realm_id is None

    # Add first asset - should infer realm_id from it
    content_unit = ic.gen_text_code_v0("First asset content")["iscc"]
    instance_unit = f"ISCC:{ic.Code.rnd(ic.MT.INSTANCE, bits=128)}"

    asset = IsccEntry(
        iscc_id=sample_iscc_ids[0],  # Has realm_id=0
        units=[instance_unit, content_unit],
    )

    results = idx.add_assets([asset])
    assert len(results) == 1
    assert results[0].status == "created"

    # realm_id should now be inferred as 0
    assert idx._realm_id == 0

    # Verify asset was stored
    retrieved = idx.get_asset(sample_iscc_ids[0])
    assert retrieved.iscc_id == sample_iscc_ids[0]

    idx.close()


def test_usearch_index_migration_from_legacy_index(tmp_path, sample_iscc_ids):
    """Test migration path for legacy indexes without stored realm_id."""
    import lmdb
    import struct
    from iscc_search.indexes import common
    from iscc_search.models import IsccID

    index_path = tmp_path / "legacy_index"

    # Step 1: Manually create a legacy index structure (without realm_id in metadata)
    # This simulates an old index from before realm_id was stored
    index_path.mkdir(parents=True, exist_ok=True)
    db_path = index_path / "index.lmdb"  # UsearchIndex expects index.lmdb, not metadata.lmdb

    # Create LMDB environment and add assets WITHOUT storing realm_id
    env = lmdb.open(
        str(db_path),
        max_dbs=10,
        map_size=100 * 1024 * 1024,
        subdir=False,
    )

    with env.begin(write=True) as txn:
        # Create metadata database
        metadata_db = env.open_db(b"__metadata__", txn=txn)

        # Store max_dim but NOT realm_id (simulating old index)
        txn.put(b"max_dim", struct.pack(">I", 256), db=metadata_db)
        # Do NOT store realm_id - this is the key difference

        # Create assets database and add an asset
        assets_db = env.open_db(b"__assets__", txn=txn)

        # Create a test asset
        content_unit = ic.gen_text_code_v0("Legacy asset content")["iscc"]
        instance_unit = f"ISCC:{ic.Code.rnd(ic.MT.INSTANCE, bits=128)}"
        asset = IsccEntry(
            iscc_id=sample_iscc_ids[0],
            units=[instance_unit, content_unit],
            metadata={"legacy": True},
        )

        # Serialize and store the asset using the same key format as UsearchIndex
        iscc_id_obj = IsccID(sample_iscc_ids[0])
        asset_key = struct.pack(">Q", int(iscc_id_obj))
        asset_bytes = common.serialize_asset(asset)
        txn.put(asset_key, asset_bytes, db=assets_db)

    env.close()

    # Step 2: Open the legacy index - should trigger migration
    # This should infer realm_id from the first asset
    idx = UsearchIndex(index_path, realm_id=None)

    # Should have inferred realm_id=0 from the stored asset
    assert idx._realm_id == 0

    # Should be able to retrieve the legacy asset
    retrieved = idx.get_asset(sample_iscc_ids[0])
    assert retrieved.iscc_id == sample_iscc_ids[0]
    assert retrieved.metadata["legacy"] is True

    # Should be able to add new assets
    new_content = ic.gen_text_code_v0("New asset after migration")["iscc"]
    new_instance = f"ISCC:{ic.Code.rnd(ic.MT.INSTANCE, bits=128)}"
    new_asset = IsccEntry(
        iscc_id=sample_iscc_ids[1],
        units=[new_instance, new_content],
    )

    results = idx.add_assets([new_asset])
    assert len(results) == 1
    assert results[0].status == "created"

    idx.close()

    # Step 3: Reopen the migrated index - should load realm_id from metadata now
    idx_reopened = UsearchIndex(index_path, realm_id=None)
    assert idx_reopened._realm_id == 0

    # Both assets should be accessible
    assert idx_reopened.get_asset(sample_iscc_ids[0]).metadata["legacy"] is True
    assert idx_reopened.get_asset(sample_iscc_ids[1]).iscc_id == sample_iscc_ids[1]

    idx_reopened.close()


def test_usearch_index_migration_missing_created_at(tmp_path, sample_iscc_ids):
    """Test migration path adds created_at timestamp if missing."""
    import lmdb
    import struct
    from iscc_search.indexes import common
    from iscc_search.models import IsccID

    index_path = tmp_path / "legacy_no_timestamp"
    index_path.mkdir(parents=True, exist_ok=True)
    db_path = index_path / "index.lmdb"

    # Create legacy index without created_at
    env = lmdb.open(
        str(db_path),
        max_dbs=10,
        map_size=100 * 1024 * 1024,
        subdir=False,
    )

    with env.begin(write=True) as txn:
        metadata_db = env.open_db(b"__metadata__", txn=txn)
        assets_db = env.open_db(b"__assets__", txn=txn)

        # Store max_dim but NOT realm_id or created_at
        txn.put(b"max_dim", struct.pack(">I", 256), db=metadata_db)

        # Add an asset
        content_unit = ic.gen_text_code_v0("Asset without timestamp")["iscc"]
        instance_unit = f"ISCC:{ic.Code.rnd(ic.MT.INSTANCE, bits=128)}"
        asset = IsccEntry(
            iscc_id=sample_iscc_ids[0],
            units=[instance_unit, content_unit],
        )

        iscc_id_obj = IsccID(sample_iscc_ids[0])
        asset_key = struct.pack(">Q", int(iscc_id_obj))
        asset_bytes = common.serialize_asset(asset)
        txn.put(asset_key, asset_bytes, db=assets_db)

    env.close()

    # Open the index - should add created_at during migration
    idx = UsearchIndex(index_path, realm_id=None)

    # Verify created_at was added
    with idx.env.begin() as txn:
        metadata_db = idx.env.open_db(b"__metadata__", txn=txn)
        created_at_bytes = txn.get(b"created_at", db=metadata_db)
        assert created_at_bytes is not None, "created_at should be added during migration"

        created_at = struct.unpack(">d", created_at_bytes)[0]
        assert created_at > 0, "created_at should be a valid timestamp"

    idx.close()


def test_usearch_index_migration_missing_max_dim(tmp_path, sample_iscc_ids):
    """Test migration path adds max_dim if missing."""
    import lmdb
    import struct
    from iscc_search.indexes import common
    from iscc_search.models import IsccID

    index_path = tmp_path / "legacy_no_max_dim"
    index_path.mkdir(parents=True, exist_ok=True)
    db_path = index_path / "index.lmdb"

    # Create legacy index without max_dim
    env = lmdb.open(
        str(db_path),
        max_dbs=10,
        map_size=100 * 1024 * 1024,
        subdir=False,
    )

    with env.begin(write=True) as txn:
        metadata_db = env.open_db(b"__metadata__", txn=txn)
        assets_db = env.open_db(b"__assets__", txn=txn)

        # Do NOT store realm_id or max_dim

        # Add an asset
        content_unit = ic.gen_text_code_v0("Asset without max_dim")["iscc"]
        instance_unit = f"ISCC:{ic.Code.rnd(ic.MT.INSTANCE, bits=128)}"
        asset = IsccEntry(
            iscc_id=sample_iscc_ids[0],
            units=[instance_unit, content_unit],
        )

        iscc_id_obj = IsccID(sample_iscc_ids[0])
        asset_key = struct.pack(">Q", int(iscc_id_obj))
        asset_bytes = common.serialize_asset(asset)
        txn.put(asset_key, asset_bytes, db=assets_db)

    env.close()

    # Open the index with explicit max_dim - should use it
    idx = UsearchIndex(index_path, realm_id=None, max_dim=128)

    # Should have stored the provided max_dim
    assert idx.max_dim == 128

    # Verify max_dim was stored in metadata
    with idx.env.begin() as txn:
        metadata_db = idx.env.open_db(b"__metadata__", txn=txn)
        max_dim_bytes = txn.get(b"max_dim", db=metadata_db)
        assert max_dim_bytes is not None, "max_dim should be stored"

        max_dim = struct.unpack(">I", max_dim_bytes)[0]
        assert max_dim == 128, "max_dim should be 128"

    idx.close()


def test_usearch_index_migration_invalid_asset_no_iscc_id(tmp_path):
    """Test migration fails when first asset has no iscc_id."""
    import lmdb
    import struct
    from iscc_search.indexes import common

    index_path = tmp_path / "legacy_invalid"
    index_path.mkdir(parents=True, exist_ok=True)
    db_path = index_path / "index.lmdb"

    # Create legacy index with asset that has no iscc_id
    env = lmdb.open(
        str(db_path),
        max_dbs=10,
        map_size=100 * 1024 * 1024,
        subdir=False,
    )

    with env.begin(write=True) as txn:
        metadata_db = env.open_db(b"__metadata__", txn=txn)
        assets_db = env.open_db(b"__assets__", txn=txn)

        # Store max_dim but NOT realm_id
        txn.put(b"max_dim", struct.pack(">I", 256), db=metadata_db)

        # Add an asset WITHOUT iscc_id (invalid for migration)
        content_unit = ic.gen_text_code_v0("Asset without iscc_id")["iscc"]
        instance_unit = f"ISCC:{ic.Code.rnd(ic.MT.INSTANCE, bits=128)}"
        asset = IsccEntry(
            iscc_id=None,  # No iscc_id!
            units=[instance_unit, content_unit],
        )

        # Use a dummy key since we can't derive it from iscc_id
        asset_key = struct.pack(">Q", 1)
        asset_bytes = common.serialize_asset(asset)
        txn.put(asset_key, asset_bytes, db=assets_db)

    env.close()

    # Opening the index should fail because first asset has no iscc_id
    with pytest.raises(ValueError, match="Cannot infer realm_id"):
        UsearchIndex(index_path, realm_id=None)


def test_usearch_index_explicit_realm_id_first_add(tmp_path, sample_iscc_ids):
    """Test adding assets when index was created with explicit realm_id."""
    index_path = tmp_path / "explicit_realm"

    # Create index WITH explicit realm_id
    idx = UsearchIndex(index_path, realm_id=0, max_dim=256)

    # realm_id should be set in memory
    assert idx._realm_id == 0

    # Add first asset - this covers the branch where realm_id is in memory but not in DB
    content_unit = ic.gen_text_code_v0("First asset with explicit realm")["iscc"]
    instance_unit = f"ISCC:{ic.Code.rnd(ic.MT.INSTANCE, bits=128)}"

    asset = IsccEntry(
        iscc_id=sample_iscc_ids[0],
        units=[instance_unit, content_unit],
    )

    results = idx.add_assets([asset])
    assert len(results) == 1
    assert results[0].status == "created"

    # Verify asset was stored
    retrieved = idx.get_asset(sample_iscc_ids[0])
    assert retrieved.iscc_id == sample_iscc_ids[0]

    idx.close()


def test_usearch_index_migration_with_existing_created_at(tmp_path, sample_iscc_ids):
    """Test migration path when created_at already exists."""
    import lmdb
    import struct
    import time
    from iscc_search.indexes import common
    from iscc_search.models import IsccID

    index_path = tmp_path / "legacy_with_timestamp"
    index_path.mkdir(parents=True, exist_ok=True)
    db_path = index_path / "index.lmdb"

    # Create legacy index with created_at already set
    env = lmdb.open(
        str(db_path),
        max_dbs=10,
        map_size=100 * 1024 * 1024,
        subdir=False,
    )

    with env.begin(write=True) as txn:
        metadata_db = env.open_db(b"__metadata__", txn=txn)
        assets_db = env.open_db(b"__assets__", txn=txn)

        # Store max_dim and created_at but NOT realm_id
        txn.put(b"max_dim", struct.pack(">I", 256), db=metadata_db)
        txn.put(b"created_at", struct.pack(">d", time.time()), db=metadata_db)

        # Add an asset
        content_unit = ic.gen_text_code_v0("Asset with existing timestamp")["iscc"]
        instance_unit = f"ISCC:{ic.Code.rnd(ic.MT.INSTANCE, bits=128)}"
        asset = IsccEntry(
            iscc_id=sample_iscc_ids[0],
            units=[instance_unit, content_unit],
        )

        iscc_id_obj = IsccID(sample_iscc_ids[0])
        asset_key = struct.pack(">Q", int(iscc_id_obj))
        asset_bytes = common.serialize_asset(asset)
        txn.put(asset_key, asset_bytes, db=assets_db)

    env.close()

    # Open the index - should migrate without overwriting created_at
    idx = UsearchIndex(index_path, realm_id=None)

    # Verify migration succeeded
    assert idx._realm_id == 0

    # Verify created_at still exists
    with idx.env.begin() as txn:
        metadata_db = idx.env.open_db(b"__metadata__", txn=txn)
        created_at_bytes = txn.get(b"created_at", db=metadata_db)
        assert created_at_bytes is not None

    idx.close()
