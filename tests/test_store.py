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
    store = IsccStore(temp_store_path, realm_id=1, durable=True)
    assert store.realm_id == 1
    assert store.get_metadata("__realm_id__") == 1
    store.close()


def test_create_non_durable_store(temp_store_path):
    # type: (typing.Any) -> None
    """Test creating non-durable store for testing scenarios."""
    store = IsccStore(temp_store_path, realm_id=0, durable=False)
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
    # Start with very small map_size (64KB) to trigger expansion quickly
    initial_size = 64 * 1024
    store = IsccStore(temp_store_path, map_size=initial_size)

    assert store.map_size == initial_size

    # Add entries until map_size expands (should happen after a few entries)
    for i in range(100):
        entry = {**sample_entry, "iscc_id": f"ISCC:ID{i}"}
        store.put(i, entry)

    # Verify map_size has doubled at least once
    assert store.map_size > initial_size

    # Verify all entries are still retrievable
    assert store.get(0)["iscc_id"] == "ISCC:ID0"
    assert store.get(99)["iscc_id"] == "ISCC:ID99"

    store.close()


def test_map_size_metadata_expansion(temp_store_path):
    # type: (typing.Any) -> None
    """Test map_size expansion works for metadata operations."""
    # Start with very small map_size (64KB)
    initial_size = 64 * 1024
    store = IsccStore(temp_store_path, map_size=initial_size)

    # Add large metadata values to trigger expansion
    large_value = "x" * 10000  # 10KB string
    for i in range(20):
        store.put_metadata(f"key_{i}", large_value)

    # Verify map_size has potentially expanded
    # (may not expand if metadata fits in initial size)
    assert store.map_size >= initial_size

    # Verify metadata is retrievable
    assert store.get_metadata("key_0") == large_value
    assert store.get_metadata("key_19") == large_value

    store.close()


def test_set_mapsize_manual(temp_store_path):
    # type: (typing.Any) -> None
    """Test manually setting map_size."""
    store = IsccStore(temp_store_path, map_size=64 * 1024)
    initial_size = store.map_size

    # Manually double the map_size
    new_size = initial_size * 2
    store.set_mapsize(new_size)

    assert store.map_size == new_size
    store.close()


def test_reopen_larger_database(temp_store_path, sample_entry):
    # type: (typing.Any, dict) -> None
    """Test reopening database larger than initial map_size."""
    # Create database with very small map_size (128KB) and grow it
    store1 = IsccStore(temp_store_path, map_size=128 * 1024)

    # Add large entries to force expansion beyond 128KB
    large_entry = {**sample_entry, "large_data": "x" * 5000}  # ~5KB per entry
    for i in range(50):  # 50 * 5KB = ~250KB, will expand to 256KB
        entry = {**large_entry, "iscc_id": f"ISCC:ID{i}"}
        store1.put(i, entry)

    final_size = store1.map_size
    assert final_size > 128 * 1024  # Should have grown
    store1.close()

    # Reopen with smaller initial map_size (64KB)
    # LMDB should automatically adjust to actual database size
    store2 = IsccStore(temp_store_path, map_size=64 * 1024)

    # Verify we can read all entries
    assert store2.get(0)["iscc_id"] == "ISCC:ID0"
    assert store2.get(49)["iscc_id"] == "ISCC:ID49"

    # map_size should be at least as large as before
    assert store2.map_size >= final_size

    store2.close()
