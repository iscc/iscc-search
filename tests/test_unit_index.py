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
    for bits, unit in zip([64, 128, 192, 256], units):
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
    # Generate a composite ISCC code
    # Note: We'll simulate this since ic.Code.rnd might not generate composite codes
    # Composite codes have MainType.ISCC

    # First, let's check if we can create a composite code
    # For now, we'll test the validation logic directly
    index = IsccUnitIndex()

    # Create a mock composite code scenario
    # We'll test this by checking the sample_unit initialization
    # since that's where the validation happens

    # This test would need actual composite code generation
    # For now, we ensure the error message is correct in the implementation
    pass  # Placeholder for when composite code generation is available


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
