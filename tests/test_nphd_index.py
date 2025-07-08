"""Test the NphdIndex subclass implementation."""

import os
import tempfile
import typing

import numpy as np
import pytest

from iscc_vdb.nphd_index import NphdIndex


def test_basic_functionality():
    # type: () -> None
    """Test basic add, search, and get operations."""
    # Create index
    index = NphdIndex(connectivity=16, expansion_add=40, expansion_search=16)

    # Add vectors of different lengths
    vectors = [
        np.array([0xFF, 0xAA, 0x55, 0x00, 0xFF, 0xAA, 0x55, 0x00], dtype=np.uint8),  # 8 bytes
        np.array([0xFF] * 16, dtype=np.uint8),  # 16 bytes
        np.array([0xAA] * 24, dtype=np.uint8),  # 24 bytes
        np.array([0x55] * 32, dtype=np.uint8),  # 32 bytes
    ]
    keys = [1, 2, 3, 4]

    # Add vectors
    index.add(keys, vectors)

    # Search with 8-byte query
    query = np.array([0xFF, 0xAA, 0x55, 0x00, 0xFF, 0xAA, 0x55, 0x01], dtype=np.uint8)
    matches = index.search(query, count=2)

    # Check we got results
    assert len(matches.keys) == 2
    assert 1 in matches.keys  # First vector should be similar

    # Test batch search
    queries = [
        np.array([0xFF] * 8, dtype=np.uint8),
        np.array([0xAA] * 16, dtype=np.uint8),
    ]
    batch_matches = index.search(queries, count=2)
    assert batch_matches.keys.shape == (2, 2)  # 2 queries, 2 results each

    # Test get (returns packed vectors)
    packed = index.get(1)
    assert packed is not None
    assert len(packed) == 33  # 1 byte length + 32 bytes max data
    assert packed[0] == 8  # Length signal for 8-byte vector

    # Test contains
    assert 1 in index
    assert 5 not in index

    # Test size
    assert len(index) == 4


def test_single_vector_operations():
    # type: () -> None
    """Test operations with single vectors."""
    index = NphdIndex()

    # Add single vector as bytes
    vector = b"\xff" * 16
    index.add(42, vector)

    assert 42 in index
    assert len(index) == 1

    # Search with single vector
    matches = index.search(vector, count=1)
    assert matches.keys[0] == 42
    assert matches.distances[0] == 0.0  # Exact match


def test_memory_mapped_save_restore(tmp_path):
    # type: (typing.Any) -> None
    """Test saving and restoring index."""
    # Create and populate index
    index = NphdIndex()
    index.add([1, 2], [b"\xff" * 8, b"\xaa" * 16])

    # Save to temporary file
    tmp_file = tmp_path / "test_index.usearch"
    index.save(str(tmp_file))

    # Restore and verify
    restored = NphdIndex.restore(str(tmp_file))
    assert len(restored) == 2
    assert 1 in restored
    assert 2 in restored

    # Search should work
    matches = restored.search(b"\xff" * 8, count=1)
    assert matches.keys[0] == 1

    # Important: Close the memory-mapped view before pytest cleans up
    del restored


def test_max_bits_configuration():
    # type: () -> None
    """Test index with different max_bits settings."""
    # Create index with smaller max_bits
    index = NphdIndex(max_bits=128)
    assert index.max_bits == 128
    assert index.max_bytes == 16
    assert index.ndim == 136  # 128 + 8 bits for length signal

    # Add vectors up to max size
    index.add(1, b"\xff" * 8)  # OK - 8 bytes < 16
    index.add(2, b"\xff" * 16)  # OK - exactly max size

    # Vector larger than max_bytes should fail
    try:
        index.add(3, b"\xff" * 17)  # Should fail - 17 > 16
        msg = "Should have raised ValueError"
        raise AssertionError(msg)
    except ValueError as e:
        assert "exceeds max_bytes" in str(e)

    assert len(index) == 2

    # Test get returns proper packed size
    packed = index.get(1)
    assert len(packed) == 17  # 1 byte length + 16 bytes max data (not 33!)


def test_inherited_functionality():
    # type: () -> None
    """Test that inherited usearch.Index methods work correctly."""
    index = NphdIndex()

    # Add some data
    index.add([1, 2, 3], [b"\xff" * 8, b"\xaa" * 16, b"\x55" * 24])

    # Test properties
    assert index.size == 3
    assert index.capacity >= 3
    assert index.ndim == 264  # 256 + 8 bits
    assert hasattr(index, "connectivity")
    assert hasattr(index, "expansion_add")
    assert hasattr(index, "expansion_search")

    # Test clear
    index.clear()
    assert len(index) == 0

    # Re-add and test remove
    index.add(10, b"\x11" * 8)
    assert 10 in index
    index.remove(10)
    assert 10 not in index


def test_numpy_2d_array_operations():
    # type: () -> None
    """Test add and search with numpy 2D arrays."""
    index = NphdIndex()

    # Test add with numpy 2D array (covers lines 81-83)
    # Create 2D array with same-length vectors
    vectors_2d = np.array(
        [
            [0xFF, 0xAA, 0xFF, 0xAA, 0xFF, 0xAA, 0xFF, 0xAA],  # 8 bytes
            [0x55, 0x00, 0x55, 0x00, 0x55, 0x00, 0x55, 0x00],  # 8 bytes
        ],
        dtype=np.uint8,
    )
    index.add([1, 2], vectors_2d)
    assert len(index) == 2

    # Test search with numpy 2D array (covers lines 109-111)
    queries_2d = np.array(
        [
            [0xFF, 0xAA, 0xFF, 0xAA, 0xFF, 0xAA, 0xFF, 0xAA],
            [0x55, 0x00, 0x55, 0x00, 0x55, 0x00, 0x55, 0x00],
        ],
        dtype=np.uint8,
    )
    matches = index.search(queries_2d, count=1)
    assert matches.keys.shape == (2, 1)
    assert matches.keys[0, 0] == 1
    assert matches.keys[1, 0] == 2


def test_single_vector_unexpected_format():
    # type: () -> None
    """Test add and search with unexpected single vector formats."""
    index = NphdIndex()

    # Test add with unexpected format (covers lines 85-87)
    # Use a tuple - not a list, not bytes, not a numpy array
    vector_tuple = (0xFF, 0xAA, 0xFF, 0xAA, 0xFF, 0xAA, 0xFF, 0xAA)
    index.add(1, vector_tuple)
    assert 1 in index

    # Test search with unexpected format (covers lines 113-115)
    matches = index.search(vector_tuple, count=1)
    assert matches.keys[0] == 1


def test_restore_metadata_failure():
    # type: () -> None
    """Test restore with invalid file that has no metadata."""
    # Test with non-existent file (metadata returns None)
    non_existent_path = os.path.join(tempfile.gettempdir(), "non_existent_file_12345.usearch")

    # Test metadata failure (covers lines 145-146)
    with pytest.raises(RuntimeError, match="Failed to read metadata"):
        NphdIndex.restore(non_existent_path)


def test_restore_without_view(tmp_path):
    # type: (typing.Any) -> None
    """Test restore with view=False."""
    # Create and save index
    index = NphdIndex()
    index.add(1, b"\xff" * 8)

    tmp_file = tmp_path / "test_index_no_view.usearch"
    index.save(str(tmp_file))

    # Restore without view (covers line 158)
    restored = NphdIndex.restore(str(tmp_file), view=False)
    assert len(restored) == 1
    assert 1 in restored

    # No need to worry about file locking when view=False
    del restored
