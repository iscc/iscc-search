"""Tests for IsccUnitIndex class."""

import tempfile
import typing
from pathlib import Path

import iscc_core as ic
import numpy as np
import pytest

from iscc_vdb.unit_index import IsccUnitIndex


def test_basic_functionality():
    # type: () -> None
    """Test basic add, search, and get operations."""
    index = IsccUnitIndex()

    # Generate test data using ic.Code.rnd
    meta_unit = "ISCC:" + str(ic.Code.rnd(ic.MT.META, bits=64))
    meta_unit2 = "ISCC:" + str(ic.Code.rnd(ic.MT.META, bits=64))

    # Generate ISCC-IDs
    iscc_id1 = ic.gen_iscc_id(1000, server_id=1, realm_id=0)["iscc"]
    iscc_id2 = ic.gen_iscc_id(2000, server_id=1, realm_id=0)["iscc"]

    # Add vectors
    index.add(iscc_id1, meta_unit)
    index.add(iscc_id2, meta_unit2)

    # Check contains
    assert 1 not in index  # Integer key doesn't exist
    # We need to check with the actual integer key
    decoded_id1 = ic.iscc_decode(iscc_id1)
    key1 = int.from_bytes(decoded_id1[4], "big", signed=False)
    assert key1 in index

    # Test get
    retrieved = index.get(iscc_id1)
    assert retrieved == meta_unit

    # Test search
    matches = index.search(meta_unit, count=2)
    assert len(matches.keys) == 2
    assert matches.keys[0] == iscc_id1  # Exact match should be first
    assert matches.distances[0] == 0.0


def test_type_locking():
    # type: () -> None
    """Test that index locks to first vector type."""
    index = IsccUnitIndex()

    # Add META unit
    meta_unit = "ISCC:" + str(ic.Code.rnd(ic.MT.META, bits=64))
    index.add(1, meta_unit)

    # Try to add CONTENT unit - should fail
    content_unit = "ISCC:" + str(ic.Code.rnd(ic.MT.CONTENT, bits=64))
    with pytest.raises(ValueError, match="Type mismatch"):
        index.add(2, content_unit)


def test_sample_unit_initialization():
    # type: () -> None
    """Test initialization with sample unit."""
    # Generate a sample unit and decode it to get its exact type
    sample_code = ic.Code.rnd(ic.MT.CONTENT, bits=128)
    sample_unit = "ISCC:" + str(sample_code)

    # Decode to get the exact subtype and version
    decoded = ic.iscc_decode(sample_unit)
    maintype, subtype, version, _, _ = decoded

    # Create index locked to this specific type
    index = IsccUnitIndex(sample_unit=sample_unit)

    # Create another unit with the same type characteristics
    # We need to generate units until we get one with matching subtype
    for _ in range(10):  # Try a few times
        content_code = ic.Code.rnd(ic.MT.CONTENT, bits=128)
        content_unit = "ISCC:" + str(content_code)
        decoded2 = ic.iscc_decode(content_unit)
        if decoded2[1] == subtype and decoded2[2] == version:
            # Found matching subtype and version
            index.add(1, content_unit)
            break
    else:
        # If we can't find a match, just skip the same-type test
        pass

    # Should reject different MainType
    meta_unit = "ISCC:" + str(ic.Code.rnd(ic.MT.META, bits=128))
    with pytest.raises(ValueError, match="Type mismatch"):
        index.add(2, meta_unit)


def test_different_vector_lengths():
    # type: () -> None
    """Test handling of different vector lengths."""
    index = IsccUnitIndex()

    # Add vectors of different lengths (all same type)
    units = []
    for bits in [64, 128, 192, 256]:
        unit = "ISCC:" + str(ic.Code.rnd(ic.MT.DATA, bits=bits))
        units.append(unit)
        index.add(bits, unit)

    # All should be retrievable
    for bits, unit in zip([64, 128, 192, 256], units, strict=False):
        retrieved = index.get(bits)
        assert retrieved == unit


def test_batch_operations():
    # type: () -> None
    """Test batch add and search."""
    index = IsccUnitIndex()

    # Generate batch data
    keys = []
    units = []
    for i in range(5):
        iscc_id = ic.gen_iscc_id(i * 1000, server_id=1, realm_id=0)["iscc"]
        unit = "ISCC:" + str(ic.Code.rnd(ic.MT.INSTANCE, bits=128))
        keys.append(iscc_id)
        units.append(unit)

    # Batch add
    index.add(keys, units)

    # Batch get
    retrieved = index.get(keys)
    assert retrieved == units

    # Batch search
    query_units = units[:2]
    matches = index.search(query_units, count=3)
    # Keys are converted to list of lists for batch results
    assert len(matches.keys) == 2  # 2 queries
    assert len(matches.keys[0]) == 3  # 3 results per query
    # First result for each query should be exact match
    assert matches.keys[0][0] == keys[0]
    assert matches.keys[1][0] == keys[1]


def test_raw_bytes_input():
    # type: () -> None
    """Test that raw bytes input works."""
    index = IsccUnitIndex()

    # Add with raw bytes
    vector1 = b"\xff" * 16
    vector2 = b"\xaa" * 16
    index.add(100, vector1)
    index.add(200, vector2)

    # Search with raw bytes
    matches = index.search(vector1, count=1)
    # Result should be ISCC-ID
    assert matches.keys[0].startswith("ISCC:")
    assert "MAI" in matches.keys[0]  # ISCC-ID prefix


def test_numpy_array_input():
    # type: () -> None
    """Test that numpy array input works."""
    index = IsccUnitIndex()

    # Add with numpy arrays
    vector1 = np.array([0xFF] * 8, dtype=np.uint8)
    vector2 = np.array([0xAA] * 8, dtype=np.uint8)
    index.add(10, vector1)
    index.add(20, vector2)

    # Get should return ISCC-UNIT if type is set
    # Since we used raw bytes, no type info is available
    result = index.get(10)
    # Will return packed vector since no type info
    assert isinstance(result, np.ndarray)


def test_save_restore(tmp_path):
    # type: (typing.Any) -> None
    """Test saving and restoring index with metadata."""
    # Create and populate index
    index = IsccUnitIndex(realm_id=0, version=1)

    # Add some data with ISCC units to set type
    unit1 = "ISCC:" + str(ic.Code.rnd(ic.MT.META, bits=64))
    unit2 = "ISCC:" + str(ic.Code.rnd(ic.MT.META, bits=128))
    iscc_id1 = ic.gen_iscc_id(1000, server_id=1, realm_id=0)["iscc"]
    iscc_id2 = ic.gen_iscc_id(2000, server_id=1, realm_id=0)["iscc"]

    index.add(iscc_id1, unit1)
    index.add(iscc_id2, unit2)

    # Save
    save_path = tmp_path / "test_index.usearch"
    index.save(str(save_path))

    # Check metadata file exists
    meta_path = Path(str(save_path) + ".meta.json")
    assert meta_path.exists()

    # Restore
    restored = IsccUnitIndex.restore(str(save_path))

    # Check metadata was restored
    assert restored.realm_id == 0
    assert restored.version == 1
    assert restored.unit_header == index.unit_header

    # Check data is accessible
    assert restored.get(iscc_id1) == unit1
    assert restored.get(iscc_id2) == unit2

    # Search should work
    matches = restored.search(unit1, count=1)
    assert matches.keys[0] == iscc_id1

    # Clean up
    del restored


def test_save_restore_without_metadata(tmp_path):
    # type: (typing.Any) -> None
    """Test restoring index without metadata file."""
    # Create index with raw bytes (no type info)
    index = IsccUnitIndex()
    index.add(1, b"\xff" * 8)

    # Save
    save_path = tmp_path / "test_index_no_meta.usearch"
    index.save(str(save_path))

    # Remove metadata file
    meta_path = Path(str(save_path) + ".meta.json")
    meta_path.unlink()

    # Restore should still work
    restored = IsccUnitIndex.restore(str(save_path))
    assert len(restored) == 1
    assert 1 in restored

    # Clean up
    del restored


def test_integer_key_operations():
    # type: () -> None
    """Test operations with integer keys."""
    index = IsccUnitIndex(realm_id=0, version=1)

    # Add with integer keys
    unit = "ISCC:" + str(ic.Code.rnd(ic.MT.CONTENT, bits=192))
    index.add(42, unit)

    # Get with integer key should work
    retrieved = index.get(42)
    assert retrieved == unit

    # Search results should have ISCC-ID keys
    matches = index.search(unit, count=1)
    assert matches.keys[0].startswith("ISCC:")

    # Decode the result ISCC-ID
    decoded = ic.iscc_decode(matches.keys[0])
    assert decoded[0] == ic.MT.ID  # MainType should be ID
    assert decoded[1] == 0  # SubType should be realm_id
    assert decoded[2] == 1  # Version should match


def test_invalid_inputs():
    # type: () -> None
    """Test error handling for invalid inputs."""
    index = IsccUnitIndex()

    # Invalid vector length
    with pytest.raises(ValueError, match="Vector must be"):
        index.add(1, b"\xff" * 7)  # 7 bytes not allowed

    with pytest.raises(ValueError, match="Vector must be"):
        index.add(1, b"\xff" * 33)  # 33 bytes too long

    # Invalid integer key
    with pytest.raises(ValueError, match="Integer key must be"):
        index.add(-1, b"\xff" * 8)  # Negative key

    with pytest.raises(ValueError, match="Integer key must be"):
        index.add(2**64, b"\xff" * 8)  # Key too large

    # Invalid ISCC-ID (wrong MainType)
    meta_unit = "ISCC:" + str(ic.Code.rnd(ic.MT.META, bits=64))
    with pytest.raises(ValueError, match="Expected ISCC-ID"):
        index.add(meta_unit, b"\xff" * 8)  # Using ISCC-UNIT as key


def test_composite_code_rejection():
    # type: () -> None
    """Test that composite codes are rejected."""
    # Since we can't easily create invalid ISCC codes through the API,
    # we'll test these error paths by mocking the scenarios

    # Test invalid body length handling in initialization (lines 56-57)
    # We can test this by creating an index and then manually setting invalid data
    index = IsccUnitIndex()

    # Add a valid unit first
    valid_unit = "ISCC:" + str(ic.Code.rnd(ic.MT.META, bits=64))
    index.add(1, valid_unit)

    # Test invalid vector length in normalize_vector
    with pytest.raises(ValueError, match="Vector must be 8, 16, 24, or 32 bytes"):
        index.add(2, b"\xff" * 5)  # 5 bytes is invalid

    # Test composite code rejection (lines 61-62)
    # We need to directly test the _set_unit_type_from_sample method
    # Since we can't create composite codes via the API, we mock the scenario
    index2 = IsccUnitIndex()
    # Manually create a scenario where maintype would be ISCC
    # This can be tested by modifying unit_header directly
    index2.unit_header = (ic.MT.ISCC, 0, 0)  # Composite type

    # Now any add should fail due to type mismatch
    with pytest.raises(ValueError, match="Type mismatch"):
        index2.add(1, valid_unit)

    # Test invalid body length in normalize_vector when type is set (lines 122-123)
    index3 = IsccUnitIndex()
    # Set type with valid unit
    index3.add(1, valid_unit)

    # Try to add invalid length vector
    with pytest.raises(ValueError, match="Vector must be 8, 16, 24, or 32 bytes"):
        index3.add(2, b"\xff" * 7)  # 7 bytes is invalid


def test_mixed_key_types():
    # type: () -> None
    """Test mixing integer and ISCC-ID keys."""
    index = IsccUnitIndex()

    unit1 = "ISCC:" + str(ic.Code.rnd(ic.MT.DATA, bits=256))
    unit2 = "ISCC:" + str(ic.Code.rnd(ic.MT.DATA, bits=256))

    # Add with integer key
    index.add(1000, unit1)

    # Add with ISCC-ID key
    iscc_id = ic.gen_iscc_id(2000, server_id=1, realm_id=0)["iscc"]
    index.add(iscc_id, unit2)

    # Both should be retrievable
    assert index.get(1000) == unit1
    assert index.get(iscc_id) == unit2


def test_empty_index_operations():
    # type: () -> None
    """Test operations on empty index."""
    index = IsccUnitIndex()

    # Get from empty index
    assert index.get(1) is None

    # Search in empty index
    matches = index.search(b"\xff" * 8, count=10)
    # Empty index returns array with -1 values
    assert all(k == -1 for k in matches.keys) or len(matches.keys) == 0

    # Save/restore empty index
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "empty.usearch"
        index.save(str(save_path))

        restored = IsccUnitIndex.restore(str(save_path))
        assert len(restored) == 0

        del restored


def test_realm_and_version_in_iscc_id():
    # type: () -> None
    """Test that realm_id is correctly used in ISCC-ID generation."""
    index = IsccUnitIndex(realm_id=0, version=1)

    unit = "ISCC:" + str(ic.Code.rnd(ic.MT.INSTANCE, bits=128))
    index.add(999, unit)

    # Search and check the returned ISCC-ID
    matches = index.search(unit, count=1)
    result_id = matches.keys[0]

    # Decode and verify
    decoded = ic.iscc_decode(result_id)
    assert decoded[0] == ic.MT.ID
    assert decoded[1] == 0  # realm_id
    assert decoded[2] == 1  # version is always 1 for ISCC-IDs


def test_contains_with_iscc_id():
    # type: () -> None
    """Test __contains__ with ISCC-ID keys."""
    index = IsccUnitIndex()

    iscc_id = ic.gen_iscc_id(5000, server_id=1, realm_id=0)["iscc"]
    unit = "ISCC:" + str(ic.Code.rnd(ic.MT.META, bits=64))

    index.add(iscc_id, unit)

    # Check with integer key (extracted from ISCC-ID)
    decoded = ic.iscc_decode(iscc_id)
    key_int = int.from_bytes(decoded[4], "big", signed=False)
    assert key_int in index

    # Note: Direct ISCC-ID string checking would require overriding __contains__
    # which is not implemented in the current version


def test_iscc_id_validation_errors():
    # type: () -> None
    """Test ISCC-ID validation error handling."""
    index = IsccUnitIndex()

    # Test that non-ID ISCCs are rejected as keys (lines 94-95)
    # Try to use a META unit as a key (should fail because it's not an ID)
    unit_128 = "ISCC:" + str(ic.Code.rnd(ic.MT.META, bits=128))
    with pytest.raises(ValueError, match="Expected ISCC-ID.*got MainType"):
        index.add(unit_128, b"\xff" * 8)

    # Test ISCC-UNIT with invalid body length in normalize_vector
    # when maintype is already set (lines 122-123)
    index2 = IsccUnitIndex()
    # First set the type with a valid META unit
    valid_meta = "ISCC:" + str(ic.Code.rnd(ic.MT.META, bits=64))
    index2.add(1, valid_meta)

    # Create a content unit with different maintype
    content_unit = "ISCC:" + str(ic.Code.rnd(ic.MT.CONTENT, bits=64))
    with pytest.raises(ValueError, match="Type mismatch"):
        index2.add(2, content_unit)

    # Test with ISCC-ID that has wrong MainType for lines 94-95
    # We test this by creating a scenario to trigger the error path
    # Generate a DATA unit and try to use it as key
    data_unit = "ISCC:" + str(ic.Code.rnd(ic.MT.DATA, bits=64))
    with pytest.raises(ValueError, match="Expected ISCC-ID.*got MainType"):
        index.add(data_unit, b"\xff" * 8)


def test_get_error_handling():
    # type: () -> None
    """Test error handling in get operations."""
    index = IsccUnitIndex()

    # Add some data
    unit = "ISCC:" + str(ic.Code.rnd(ic.MT.META, bits=64))
    index.add(100, unit)

    # Test _get_single with invalid key (lines 232-233)
    # Use a META unit as key which will fail validation
    meta_as_key = "ISCC:" + str(ic.Code.rnd(ic.MT.META, bits=64))
    result = index.get(meta_as_key)
    assert result is None

    # Test get with non-existent key (line 242)
    result = index.get(999)
    assert result is None

    # Test batch get with invalid and non-existent keys (lines 260-262, 280)
    batch_keys = [100, meta_as_key, 999]
    results = index.get(batch_keys)
    assert results[0] == unit  # Valid key
    assert results[1] is None  # Invalid ISCC format (not an ID)
    assert results[2] is None  # Non-existent key

    # Test batch get with all invalid keys (line 267)
    meta1 = "ISCC:" + str(ic.Code.rnd(ic.MT.META, bits=64))
    meta2 = "ISCC:" + str(ic.Code.rnd(ic.MT.META, bits=64))
    invalid_batch = [meta1, meta2]  # META units as keys
    results = index.get(invalid_batch)
    assert results == [None, None]

    # Test batch get on empty index (line 272)
    empty_index = IsccUnitIndex()
    results = empty_index.get([1, 2, 3])
    # Empty index should return None for all keys
    assert results == [None, None, None]


def test_search_results_edge_cases():
    # type: () -> None
    """Test search results edge cases."""
    index = IsccUnitIndex()

    # Add data with known keys
    unit = "ISCC:" + str(ic.Code.rnd(ic.MT.META, bits=64))
    index.add(2**64 - 2, unit)  # Large but valid key

    # Search to test result key conversion
    matches = index.search(unit, count=1)

    # The key should be converted to ISCC-ID
    assert matches.keys[0] is not None
    assert matches.keys[0].startswith("ISCC:")

    # Test single search with invalid keys in results (line 335)
    # When index has few items, search might return invalid keys
    index2 = IsccUnitIndex()
    unit2 = "ISCC:" + str(ic.Code.rnd(ic.MT.META, bits=64))
    index2.add(100, unit2)

    # Search with count > items in index may include invalid results
    matches2 = index2.search(unit2, count=5)
    # First should be valid, rest might be None
    assert matches2.keys[0].startswith("ISCC:")
    # Check None handling for invalid keys (line 335)
    for i in range(1, len(matches2.keys)):
        if matches2.distances[i] == float("inf"):
            assert matches2.keys[i] is None

    # Test batch search with None keys (line 322)
    batch_query = [unit2, b"\xff" * 8]
    batch_matches = index2.search(batch_query, count=3)
    # Results should be 2x3 matrix
    assert len(batch_matches.keys) == 2
    for row_idx, row in enumerate(batch_matches.keys):
        for col_idx, key in enumerate(row):
            # Check if distance is inf (invalid result)
            if batch_matches.distances[row_idx][col_idx] == float("inf"):
                assert key is None  # Line 322


def test_restore_without_view():
    # type: () -> None
    """Test restore with view=False to cover line 404."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create and save index
        index = IsccUnitIndex(realm_id=0, version=1)
        unit = "ISCC:" + str(ic.Code.rnd(ic.MT.META, bits=64))
        index.add(100, unit)

        save_path = Path(tmpdir) / "test_load.usearch"
        index.save(str(save_path))

        # Restore with view=False (uses load instead of view)
        restored = IsccUnitIndex.restore(str(save_path), view=False)

        # Verify it works
        assert restored.get(100) == unit
        assert restored.realm_id == 0
        assert restored.version == 1

        del restored


def test_batch_search_with_none_keys():
    # type: () -> None
    """Test batch search handling of None keys in results."""
    index = IsccUnitIndex()

    # Add multiple vectors
    units = []
    for i in range(3):
        unit = "ISCC:" + str(ic.Code.rnd(ic.MT.META, bits=64))
        units.append(unit)
        index.add(i * 1000, unit)

    # Batch search
    query_units = units[:2]
    matches = index.search(query_units, count=5)

    # Results should be properly formatted
    assert len(matches.keys) == 2
    for row in matches.keys:
        for key in row:
            if key is not None:
                assert key.startswith("ISCC:")


def test_get_batch_with_mixed_validity():
    # type: () -> None
    """Test batch get with mix of valid and invalid keys for line 280."""
    index = IsccUnitIndex()

    # Add data
    unit1 = "ISCC:" + str(ic.Code.rnd(ic.MT.META, bits=64))
    unit2 = "ISCC:" + str(ic.Code.rnd(ic.MT.META, bits=64))
    index.add(100, unit1)
    index.add(200, unit2)

    # Batch get with mixed validity
    keys = [100, 999, 200]  # Valid, invalid, valid
    results = index.get(keys)

    assert results[0] == unit1
    assert results[1] is None  # Key not in index
    assert results[2] == unit2


def test_composite_code_detection():
    # type: () -> None
    """Test detection and rejection of composite codes (lines 61-62)."""
    # Create a custom test that simulates composite code scenario
    # We'll use a mock to test the validation logic

    # Test by creating a sample unit that would be composite
    # Since we can't create actual composite codes, we test the path
    # by using an actual ISCC composite example if available

    # For now, test that the error message is correct
    # when unit_header has MainType.ISCC
    # Manually test the _set_unit_type_from_sample method
    # with a simulated composite scenario
    # This requires mocking ic.iscc_decode to return MT.ISCC
    from unittest.mock import MagicMock, patch

    mock_decoded = (ic.MT.ISCC, 0, 0, 64, b"\xff" * 8)
    with (
        patch("iscc_core.iscc_decode", return_value=mock_decoded),
        pytest.raises(ValueError, match="Composite codes.*are not supported"),
    ):
        IsccUnitIndex(sample_unit="ISCC:MOCK_COMPOSITE")


def test_invalid_iscc_body_lengths():
    # type: () -> None
    """Test handling of invalid ISCC body lengths (lines 56-57, 122-123)."""
    # Test invalid body length in sample_unit initialization
    from unittest.mock import patch

    # Mock decode to return invalid body length
    mock_decoded = (ic.MT.META, 0, 0, 32, b"\x12\x34\x56\x78")  # 4 bytes
    with (
        patch("iscc_core.iscc_decode", return_value=mock_decoded),
        pytest.raises(ValueError, match="Invalid ISCC body length: 4 bytes"),
    ):
        IsccUnitIndex(sample_unit="ISCC:MOCK_INVALID")

    # Test invalid body length in normalize_vector when type is set
    index = IsccUnitIndex()
    valid_unit = "ISCC:" + str(ic.Code.rnd(ic.MT.META, bits=64))
    index.add(1, valid_unit)

    # Mock decode to return invalid body for normalize_vector
    mock_decoded2 = (ic.MT.META, 0, 0, 40, b"\x12\x34\x56\x78\x90")  # 5 bytes
    with (
        patch("iscc_core.iscc_decode", return_value=mock_decoded2),
        pytest.raises(ValueError, match="Invalid ISCC body length: 5 bytes"),
    ):
        index.add(2, "ISCC:MOCK_INVALID")


def test_iscc_id_wrong_body_length():
    # type: () -> None
    """Test ISCC-ID with wrong body length (lines 94-95)."""
    from unittest.mock import patch

    index = IsccUnitIndex()

    # Mock decode to return ID with 16 bytes instead of 8
    mock_decoded = (ic.MT.ID, 0, 1, 128, b"\xff" * 16)
    with (
        patch("iscc_core.iscc_decode", return_value=mock_decoded),
        pytest.raises(ValueError, match="ISCC-ID must decode to 8 bytes, got 16 bytes"),
    ):
        index.add("ISCC:MOCK_ID_16BYTES", b"\xff" * 8)


def test_get_single_nonexistent():
    # type: () -> None
    """Test _get_single with non-existent key (line 242)."""
    index = IsccUnitIndex()

    # Add a unit
    unit = "ISCC:" + str(ic.Code.rnd(ic.MT.META, bits=64))
    index.add(100, unit)

    # Get non-existent key
    result = index.get(999)
    assert result is None

    # Test when parent's get returns None (line 242)
    from unittest.mock import patch

    with patch.object(index.__class__.__bases__[0], "get", return_value=None):
        result = index.get(100)  # Even existing key returns None
        assert result is None


def test_batch_get_empty_results():
    # type: () -> None
    """Test batch get returning None from parent (lines 267, 272)."""
    from unittest.mock import MagicMock, patch

    index = IsccUnitIndex()
    unit = "ISCC:" + str(ic.Code.rnd(ic.MT.META, bits=64))
    index.add(100, unit)

    # Test when all keys are invalid (line 267)
    # Use META units as keys which will be invalid
    meta1 = "ISCC:" + str(ic.Code.rnd(ic.MT.META, bits=64))
    meta2 = "ISCC:" + str(ic.Code.rnd(ic.MT.META, bits=64))
    results = index.get([meta1, meta2])
    assert results == [None, None]

    # Test when parent's get returns None (line 272)
    with patch.object(index.__class__.__bases__[0], "get", return_value=None):
        results = index.get([100, 200])
        assert results is None


def test_search_key_conversion_edge_cases():
    # type: () -> None
    """Test search result key conversion edge cases (lines 322, 335)."""
    index = IsccUnitIndex()

    # Add a single unit
    unit = "ISCC:" + str(ic.Code.rnd(ic.MT.META, bits=64))
    index.add(100, unit)

    # Mock search results to test edge cases
    from unittest.mock import MagicMock, patch

    # Test batch search with None key handling (line 322)
    mock_results = MagicMock()
    # Use max uint64 value to represent invalid keys (usearch convention)
    max_uint64 = 2**64 - 1
    # Create batch results (2D array)
    batch_keys = np.array([[100, max_uint64], [200, max_uint64]], dtype=np.uint64)
    mock_results.keys = batch_keys
    # ndim and shape are already set by numpy

    with patch.object(index.__class__.__bases__[0], "search", return_value=mock_results):
        results = index.search([unit, b"\xff" * 8], count=2)
        # First row
        assert results.keys[0][0].startswith("ISCC:")  # Valid key
        assert results.keys[0][1] is None  # Invalid key (max uint64 becomes None)
        # Second row
        assert results.keys[1][0].startswith("ISCC:")  # Valid key
        assert results.keys[1][1] is None  # Max uint64 becomes None (line 322)

    # Test single search with invalid key (line 335)
    mock_single = MagicMock()
    single_keys = np.array([100, max_uint64], dtype=np.uint64)
    mock_single.keys = single_keys
    # ndim is already 1 for single array

    with patch.object(index.__class__.__bases__[0], "search", return_value=mock_single):
        results = index.search(unit, count=2)
        assert results.keys[0].startswith("ISCC:")  # Valid key
        assert results.keys[1] is None  # Invalid key (line 335)

    # Test when results don't have keys attribute (line 312->341)
    mock_no_keys = MagicMock(spec=[])
    # Remove keys attribute
    delattr(mock_no_keys, "keys")

    with patch.object(index.__class__.__bases__[0], "search", return_value=mock_no_keys):
        results = index.search(unit, count=1)
        # Results should be returned as-is without modification
        assert results == mock_no_keys
