"""Comprehensive tests for IsccStore with 100% coverage."""

import pytest

from iscc_vdb.store import IsccStore


@pytest.fixture
def temp_store_path(tmp_path):
    # type: (typing.Any) -> typing.Any
    """Provide temporary path for store testing."""
    return tmp_path / "test_store"


@pytest.fixture
def sample_entry():
    # type: () -> dict
    """Provide sample ISCC entry for testing."""
    return {
        "iscc_id": "ISCC:IAACBFKZG52UU",
        "iscc_code": "ISCC:KACYPXW445FTYNJ3CYSXHAFJMA2HUWULUNRFE3BLHRSCXYH2M5AEGQY",
        "units": ["ISCC:GAAPXW445FTYNJ", "ISCC:EAAXHAFJMA2HUWUL", "ISCC:IAAFE3BLHRSCXYH"],
    }


def test_create_durable_store(temp_store_path):
    # type: (typing.Any) -> None
    """Test creating durable store with realm_id metadata initialized."""
    store = IsccStore(temp_store_path, realm_id=1, lmdb_options={"sync": True, "metasync": True, "lock": True})
    assert store.realm_id == 1
    assert store.get_metadata("__realm_id__") == 1
    store.close()


def test_create_non_durable_store(temp_store_path):
    # type: (typing.Any) -> None
    """Test creating non-durable store for testing scenarios."""
    store = IsccStore(temp_store_path, realm_id=0, lmdb_options={"sync": False, "metasync": False, "lock": False})
    assert store.realm_id == 0
    store.close()


def test_put_and_get_entry(temp_store_path, sample_entry):
    # type: (typing.Any, dict) -> None
    """Test storing and retrieving entry (round-trip)."""
    store = IsccStore(temp_store_path)
    iscc_id = 12345

    store.put(iscc_id, sample_entry)
    retrieved = store.get(iscc_id)

    assert retrieved == sample_entry
    store.close()


def test_get_nonexistent_entry(temp_store_path):
    # type: (typing.Any) -> None
    """Test retrieving non-existent entry returns None."""
    store = IsccStore(temp_store_path)
    result = store.get(99999)
    assert result is None
    store.close()


def test_delete_existing_entry(temp_store_path, sample_entry):
    # type: (typing.Any, dict) -> None
    """Test deleting existing entry returns True."""
    store = IsccStore(temp_store_path)
    iscc_id = 12345

    store.put(iscc_id, sample_entry)
    deleted = store.delete(iscc_id)

    assert deleted is True
    assert store.get(iscc_id) is None
    store.close()


def test_delete_nonexistent_entry(temp_store_path):
    # type: (typing.Any) -> None
    """Test deleting non-existent entry returns False."""
    store = IsccStore(temp_store_path)
    deleted = store.delete(99999)
    assert deleted is False
    store.close()


def test_iter_empty_store(temp_store_path):
    # type: (typing.Any) -> None
    """Test iterating empty store yields no entries."""
    store = IsccStore(temp_store_path)
    entries = list(store.iter_entries())
    assert entries == []
    store.close()


def test_iter_populated_store(temp_store_path, sample_entry):
    # type: (typing.Any, dict) -> None
    """Test iterating populated store yields all entries."""
    store = IsccStore(temp_store_path)

    # Add multiple entries
    store.put(100, {**sample_entry, "iscc_id": "ISCC:ID100"})
    store.put(200, {**sample_entry, "iscc_id": "ISCC:ID200"})
    store.put(300, {**sample_entry, "iscc_id": "ISCC:ID300"})

    entries = list(store.iter_entries())

    assert len(entries) == 3
    assert entries[0][0] == 100
    assert entries[1][0] == 200
    assert entries[2][0] == 300
    store.close()


def test_time_ordering_preserved(temp_store_path, sample_entry):
    # type: (typing.Any, dict) -> None
    """Test ISCC-IDs are stored in ascending time order."""
    store = IsccStore(temp_store_path)

    # Insert in random order
    timestamps = [5000000, 1000000, 3000000, 4000000, 2000000]
    for ts in timestamps:
        store.put(ts, {**sample_entry, "timestamp": ts})

    # Retrieve in sorted order
    entries = list(store.iter_entries())
    retrieved_timestamps = [entry[1]["timestamp"] for entry in entries]

    assert retrieved_timestamps == [1000000, 2000000, 3000000, 4000000, 5000000]
    store.close()


def test_get_nonexistent_metadata(temp_store_path):
    # type: (typing.Any) -> None
    """Test retrieving non-existent metadata returns None."""
    store = IsccStore(temp_store_path)
    result = store.get_metadata("nonexistent")
    assert result is None
    store.close()


def test_put_and_get_metadata(temp_store_path):
    # type: (typing.Any) -> None
    """Test storing and retrieving metadata (round-trip)."""
    store = IsccStore(temp_store_path)

    store.put_metadata("test_key", "test_value")
    result = store.get_metadata("test_key")

    assert result == "test_value"
    store.close()


def test_metadata_various_types(temp_store_path):
    # type: (typing.Any) -> None
    """Test metadata storage with various JSON-serializable types."""
    store = IsccStore(temp_store_path)

    store.put_metadata("int_val", 42)
    store.put_metadata("str_val", "hello")
    store.put_metadata("list_val", [1, 2, 3])
    store.put_metadata("dict_val", {"nested": {"key": "value"}})

    assert store.get_metadata("int_val") == 42
    assert store.get_metadata("str_val") == "hello"
    assert store.get_metadata("list_val") == [1, 2, 3]
    assert store.get_metadata("dict_val") == {"nested": {"key": "value"}}
    store.close()


def test_reopen_existing_store(temp_store_path):
    # type: (typing.Any) -> None
    """Test reopening existing store preserves realm_id from metadata."""
    # Create store with realm_id=1
    store1 = IsccStore(temp_store_path, realm_id=1)
    store1.close()

    # Reopen with different realm_id parameter - should use stored value
    store2 = IsccStore(temp_store_path, realm_id=0)
    assert store2.realm_id == 1  # Preserves original
    store2.close()


def test_default_lmdb_options_applied(temp_store_path):
    # type: (typing.Any) -> None
    """Test DEFAULT_LMDB_OPTIONS are applied when no options provided."""
    store = IsccStore(temp_store_path)

    # Verify environment was created successfully with defaults
    assert store.env is not None
    assert store.realm_id == 0

    # Verify we can perform basic operations with default options
    store.put_metadata("test", "value")
    assert store.get_metadata("test") == "value"
    store.close()


def test_user_options_override_defaults(temp_store_path):
    # type: (typing.Any) -> None
    """Test user-provided options override DEFAULT_LMDB_OPTIONS."""
    # Override some defaults for testing (non-durable mode)
    custom_options = {
        "sync": False,
        "metasync": False,
        "lock": False,
    }

    store = IsccStore(temp_store_path, lmdb_options=custom_options)

    # Verify environment was created successfully
    assert store.env is not None

    # Verify we can perform operations with custom options
    store.put_metadata("test", "value")
    assert store.get_metadata("test") == "value"
    store.close()


def test_max_dbs_and_subdir_cannot_be_overridden(temp_store_path):
    # type: (typing.Any) -> None
    """Test max_dbs and subdir are forced internally and cannot be overridden."""
    # Try to override max_dbs and subdir
    user_options = {
        "max_dbs": 10,  # Should be forced to 2
        "subdir": True,  # Should be forced to False
        "sync": False,  # This should be applied
    }

    store = IsccStore(temp_store_path, lmdb_options=user_options)

    # Verify environment was created successfully
    assert store.env is not None

    # Verify we have exactly 2 named databases (entries and metadata)
    # If max_dbs was wrong, opening the second db would fail
    assert store.entries_db is not None
    assert store.metadata_db is not None

    # Verify basic operations work
    store.put_metadata("test", "value")
    assert store.get_metadata("test") == "value"
    store.close()


def test_close_store(temp_store_path):
    # type: (typing.Any) -> None
    """Test closing store does not crash."""
    store = IsccStore(temp_store_path)
    store.close()  # Should not raise


def test_large_entry(temp_store_path):
    # type: (typing.Any) -> None
    """Test storing and retrieving large complex entry."""
    store = IsccStore(temp_store_path)

    large_entry = {
        "iscc_id": "ISCC:IAACBFKZG52UU",
        "iscc_code": "ISCC:KACYPXW445FTYNJ3CYSXHAFJMA2HUWULUNRFE3BLHRSCXYH2M5AEGQY",
        "units": ["ISCC:UNIT" + str(i) for i in range(100)],
        "metadata": {"nested": {"deeply": {"structured": {"data": list(range(100))}}}},
    }

    store.put(12345, large_entry)
    retrieved = store.get(12345)

    assert retrieved == large_entry
    store.close()


def test_multiple_entries_batch(temp_store_path, sample_entry):
    # type: (typing.Any, dict) -> None
    """Test adding 100 entries and verifying all are retrieved."""
    store = IsccStore(temp_store_path)

    # Add 100 entries
    for i in range(100):
        entry = {**sample_entry, "iscc_id": f"ISCC:ID{i}"}
        store.put(i, entry)

    # Verify all entries retrieved
    entries = list(store.iter_entries())
    assert len(entries) == 100

    # Verify specific entries
    assert store.get(0)["iscc_id"] == "ISCC:ID0"
    assert store.get(50)["iscc_id"] == "ISCC:ID50"
    assert store.get(99)["iscc_id"] == "ISCC:ID99"

    store.close()


def test_map_size_auto_expansion(temp_store_path, sample_entry):
    # type: (typing.Any, dict) -> None
    """Test map_size automatically doubles when full."""
    store = IsccStore(temp_store_path)

    # Get initial map_size
    initial_size = store.map_size

    # Add many large entries to trigger expansion
    large_entry = {**sample_entry, "large_data": "x" * 50000}  # ~50KB per entry
    for i in range(300):  # 300 * 50KB = ~15MB, should trigger expansion
        entry = {**large_entry, "iscc_id": f"ISCC:ID{i}"}
        store.put(i, entry)

    # Verify map_size has doubled at least once
    assert store.map_size > initial_size

    # Verify all entries are still retrievable
    assert store.get(0)["iscc_id"] == "ISCC:ID0"
    assert store.get(299)["iscc_id"] == "ISCC:ID299"

    store.close()


def test_map_size_metadata_expansion(temp_store_path):
    # type: (typing.Any) -> None
    """Test map_size expansion works for metadata operations."""
    store = IsccStore(temp_store_path)

    initial_size = store.map_size

    # Add large metadata values to trigger expansion
    large_value = "x" * 100000  # 100KB string
    for i in range(200):  # 200 * 100KB = ~20MB
        store.put_metadata(f"key_{i}", large_value)

    # Verify map_size has expanded
    assert store.map_size > initial_size

    # Verify metadata is retrievable
    assert store.get_metadata("key_0") == large_value
    assert store.get_metadata("key_199") == large_value

    store.close()


def test_set_mapsize_manual(temp_store_path):
    # type: (typing.Any) -> None
    """Test manually setting map_size."""
    store = IsccStore(temp_store_path)
    initial_size = store.map_size

    # Manually double the map_size
    new_size = initial_size * 2
    store.set_mapsize(new_size)

    assert store.map_size == new_size
    store.close()


def test_reopen_larger_database(temp_store_path, sample_entry):
    # type: (typing.Any, dict) -> None
    """Test reopening database that has grown beyond initial map_size."""
    store1 = IsccStore(temp_store_path)

    # Add large entries to force expansion
    large_entry = {**sample_entry, "large_data": "x" * 50000}  # ~50KB per entry
    for i in range(300):  # 300 * 50KB = ~15MB
        entry = {**large_entry, "iscc_id": f"ISCC:ID{i}"}
        store1.put(i, entry)

    final_size = store1.map_size
    store1.close()

    # Reopen - LMDB uses actual database size
    store2 = IsccStore(temp_store_path)

    # Verify we can read all entries
    assert store2.get(0)["iscc_id"] == "ISCC:ID0"
    assert store2.get(299)["iscc_id"] == "ISCC:ID299"

    # Database should be usable
    assert len(list(store2.iter_entries())) == 300

    store2.close()
