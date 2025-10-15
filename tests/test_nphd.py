"""Tests for NPHD index functionality."""

import numpy as np
import pytest
from usearch.index import ScalarKind

from iscc_vdb.nphd import NphdIndex, pad_vectors, unpad_vectors


def test_nphd_index_default_initialization():
    """Initialize NphdIndex with default max_dim."""
    index = NphdIndex()

    assert index.max_dim == 256
    assert index.max_bytes == 32
    assert index.ndim == 264  # 256 + 8 bits for length signal
    assert index.dtype == ScalarKind.B1


def test_nphd_index_custom_max_dim():
    """Initialize NphdIndex with custom max_dim values."""
    index64 = NphdIndex(max_dim=64)
    assert index64.max_dim == 64
    assert index64.max_bytes == 8
    assert index64.ndim == 72

    index128 = NphdIndex(max_dim=128)
    assert index128.max_dim == 128
    assert index128.max_bytes == 16
    assert index128.ndim == 136


def test_nphd_index_rejects_ndim_kwarg():
    """NphdIndex should reject ndim in kwargs."""
    with pytest.raises(AssertionError, match="`ndim` is calculated from `max_dim`"):
        NphdIndex(ndim=256)


def test_nphd_index_rejects_metric_kwarg():
    """NphdIndex should reject metric in kwargs."""
    with pytest.raises(AssertionError, match="`metric` is set automatically"):
        NphdIndex(metric="hamming")


def test_nphd_index_rejects_dtype_kwarg():
    """NphdIndex should reject dtype in kwargs."""
    with pytest.raises(AssertionError, match="`dtype` is set automatically"):
        NphdIndex(dtype=ScalarKind.F32)


def test_nphd_index_add_one_with_explicit_key():
    """Add a single vector with explicit key."""
    index = NphdIndex(max_dim=128)
    vector = np.array([1, 2, 3, 4, 5], dtype=np.uint8)

    key = index.add_one(42, vector)

    assert key == 42
    assert len(index) == 1


def test_nphd_index_add_one_with_auto_key():
    """Add a single vector with auto-generated key."""
    index = NphdIndex(max_dim=128)
    vector = np.array([1, 2, 3, 4, 5], dtype=np.uint8)

    key = index.add_one(None, vector)

    assert isinstance(key, int)
    assert len(index) == 1


def test_nphd_index_add_one_auto_keys_increasing():
    """Auto-generated keys should increase with multiple add_one calls."""
    index = NphdIndex(max_dim=128)

    key1 = index.add_one(None, np.array([1, 2, 3], dtype=np.uint8))
    key2 = index.add_one(None, np.array([4, 5, 6], dtype=np.uint8))
    key3 = index.add_one(None, np.array([7, 8, 9], dtype=np.uint8))

    assert key1 < key2 < key3
    assert len(index) == 3


def test_nphd_index_add_one_multiple_vectors():
    """Add multiple vectors one at a time."""
    index = NphdIndex(max_dim=128)

    key1 = index.add_one(10, np.array([1, 2, 3], dtype=np.uint8))
    key2 = index.add_one(20, np.array([4, 5, 6, 7], dtype=np.uint8))
    key3 = index.add_one(30, np.array([8, 9], dtype=np.uint8))

    assert key1 == 10
    assert key2 == 20
    assert key3 == 30
    assert len(index) == 3


def test_nphd_index_add_one_variable_lengths():
    """Add vectors of different lengths within max_dim."""
    index = NphdIndex(max_dim=256)

    # 8 bytes = 64 bits
    key1 = index.add_one(1, np.array([1] * 8, dtype=np.uint8))
    # 16 bytes = 128 bits
    key2 = index.add_one(2, np.array([2] * 16, dtype=np.uint8))
    # 32 bytes = 256 bits (max)
    key3 = index.add_one(3, np.array([3] * 32, dtype=np.uint8))

    assert len(index) == 3



def test_nphd_index_add_many_with_explicit_keys():
    """Add multiple vectors with explicit keys."""
    index = NphdIndex(max_dim=128)
    keys = [10, 20, 30]
    vectors = [
        np.array([1, 2, 3], dtype=np.uint8),
        np.array([4, 5], dtype=np.uint8),
        np.array([6, 7, 8, 9], dtype=np.uint8),
    ]

    result_keys = index.add_many(keys, vectors)

    assert isinstance(result_keys, np.ndarray)
    assert len(result_keys) == 3
    np.testing.assert_array_equal(result_keys, [10, 20, 30])
    assert len(index) == 3


def test_nphd_index_add_many_with_auto_keys():
    """Add multiple vectors with auto-generated keys."""
    index = NphdIndex(max_dim=128)
    vectors = [
        np.array([1, 2, 3], dtype=np.uint8),
        np.array([4, 5, 6], dtype=np.uint8),
        np.array([7, 8, 9], dtype=np.uint8),
    ]

    result_keys = index.add_many(None, vectors)

    assert isinstance(result_keys, np.ndarray)
    assert len(result_keys) == 3
    # Auto-generated keys should be increasing
    assert result_keys[0] < result_keys[1] < result_keys[2]
    assert len(index) == 3


def test_nphd_index_add_many_with_2d_array():
    """Add multiple vectors as 2D numpy array."""
    index = NphdIndex(max_dim=128)
    keys = [100, 101, 102]
    vectors = np.array(
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        dtype=np.uint8,
    )

    result_keys = index.add_many(keys, vectors)

    np.testing.assert_array_equal(result_keys, [100, 101, 102])
    assert len(index) == 3


def test_nphd_index_add_many_variable_lengths():
    """Add multiple vectors of different lengths."""
    index = NphdIndex(max_dim=256)
    keys = [1, 2, 3, 4]
    vectors = [
        np.array([1] * 8, dtype=np.uint8),   # 8 bytes = 64 bits
        np.array([2] * 16, dtype=np.uint8),  # 16 bytes = 128 bits
        np.array([3] * 24, dtype=np.uint8),  # 24 bytes = 192 bits
        np.array([4] * 32, dtype=np.uint8),  # 32 bytes = 256 bits
    ]

    result_keys = index.add_many(keys, vectors)

    assert len(result_keys) == 4
    assert len(index) == 4


def test_pad_vectors_with_list():
    """Pad a list of variable-length vectors."""
    vectors = [
        np.array([1, 2, 3], dtype=np.uint8),
        np.array([4, 5], dtype=np.uint8),
        np.array([6, 7, 8, 9], dtype=np.uint8),
    ]
    result = pad_vectors(vectors, nbytes=4)

    assert result.shape == (3, 5)  # 3 vectors, 4 bytes + 1 length byte
    assert result.dtype == np.uint8

    # First vector: length=3, data=[1,2,3], padding=[0]
    np.testing.assert_array_equal(result[0], [3, 1, 2, 3, 0])

    # Second vector: length=2, data=[4,5], padding=[0,0]
    np.testing.assert_array_equal(result[1], [2, 4, 5, 0, 0])

    # Third vector: length=4, data=[6,7,8,9], padding=[]
    np.testing.assert_array_equal(result[2], [4, 6, 7, 8, 9])


def test_pad_vectors_with_2d_array():
    """Pad a 2D numpy array of uniform-length vectors."""
    vectors = np.array(
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        dtype=np.uint8,
    )
    result = pad_vectors(vectors, nbytes=4)

    assert result.shape == (3, 5)
    np.testing.assert_array_equal(result[0], [3, 1, 2, 3, 0])
    np.testing.assert_array_equal(result[1], [3, 4, 5, 6, 0])
    np.testing.assert_array_equal(result[2], [3, 7, 8, 9, 0])


def test_pad_vectors_truncates_long_vectors():
    """Vectors longer than nbytes are truncated."""
    vectors = [np.array([1, 2, 3, 4, 5, 6], dtype=np.uint8)]
    result = pad_vectors(vectors, nbytes=3)

    assert result.shape == (1, 4)
    # Length byte shows original length 6, but only first 3 bytes stored
    np.testing.assert_array_equal(result[0], [6, 1, 2, 3])


def test_pad_vectors_with_single_byte_vectors():
    """Handle single-byte vectors."""
    vectors = [np.array([255], dtype=np.uint8), np.array([128], dtype=np.uint8)]
    result = pad_vectors(vectors, nbytes=2)

    assert result.shape == (2, 3)
    np.testing.assert_array_equal(result[0], [1, 255, 0])
    np.testing.assert_array_equal(result[1], [1, 128, 0])


def test_unpad_vectors_variable_lengths():
    """Unpad vectors of different lengths."""
    padded = np.array(
        [[3, 1, 2, 3, 0], [2, 4, 5, 0, 0], [4, 6, 7, 8, 9]],
        dtype=np.uint8,
    )
    result = unpad_vectors(padded)

    assert len(result) == 3
    np.testing.assert_array_equal(result[0], [1, 2, 3])
    np.testing.assert_array_equal(result[1], [4, 5])
    np.testing.assert_array_equal(result[2], [6, 7, 8, 9])


def test_unpad_vectors_single_byte():
    """Unpad single-byte vectors."""
    padded = np.array([[1, 255, 0], [1, 128, 0]], dtype=np.uint8)
    result = unpad_vectors(padded)

    assert len(result) == 2
    np.testing.assert_array_equal(result[0], [255])
    np.testing.assert_array_equal(result[1], [128])


def test_pad_unpad_roundtrip():
    """Verify pad_vectors and unpad_vectors are inverse operations."""
    original = [
        np.array([1, 2, 3], dtype=np.uint8),
        np.array([4, 5], dtype=np.uint8),
        np.array([6, 7, 8, 9], dtype=np.uint8),
    ]

    padded = pad_vectors(original, nbytes=4)
    recovered = unpad_vectors(padded)

    assert len(recovered) == len(original)
    for i in range(len(original)):
        np.testing.assert_array_equal(recovered[i], original[i])


def test_nphd_index_get_key_single_vector():
    """Retrieve a single vector that was added to the index."""
    index = NphdIndex(max_dim=128)
    vector = np.array([1, 2, 3, 4, 5], dtype=np.uint8)

    index.add_one(42, vector)
    retrieved = index.get_key(42)

    assert retrieved is not None
    np.testing.assert_array_equal(retrieved, vector)


def test_nphd_index_get_key_after_add_many():
    """Retrieve individual vectors after adding multiple vectors."""
    index = NphdIndex(max_dim=128)
    keys = [10, 20, 30]
    vectors = [
        np.array([1, 2, 3], dtype=np.uint8),
        np.array([4, 5, 6, 7], dtype=np.uint8),
        np.array([8], dtype=np.uint8),
    ]

    index.add_many(keys, vectors)

    for i, key in enumerate(keys):
        retrieved = index.get_key(key)
        assert retrieved is not None
        np.testing.assert_array_equal(retrieved, vectors[i])


def test_nphd_index_get_key_variable_lengths():
    """Retrieve vectors of different lengths."""
    index = NphdIndex(max_dim=256)

    # Add vectors of various lengths
    vec_short = np.array([1, 2], dtype=np.uint8)  # 2 bytes
    vec_medium = np.array([3] * 16, dtype=np.uint8)  # 16 bytes
    vec_long = np.array([4] * 32, dtype=np.uint8)  # 32 bytes (max)

    index.add_one(1, vec_short)
    index.add_one(2, vec_medium)
    index.add_one(3, vec_long)

    np.testing.assert_array_equal(index.get_key(1), vec_short)
    np.testing.assert_array_equal(index.get_key(2), vec_medium)
    np.testing.assert_array_equal(index.get_key(3), vec_long)


def test_nphd_index_get_key_single_byte_vector():
    """Retrieve a single-byte vector."""
    index = NphdIndex(max_dim=64)
    vector = np.array([255], dtype=np.uint8)

    index.add_one(100, vector)
    retrieved = index.get_key(100)

    assert retrieved is not None
    assert len(retrieved) == 1
    np.testing.assert_array_equal(retrieved, vector)


def test_nphd_index_get_key_preserves_values():
    """Retrieved vector values match exactly what was stored."""
    index = NphdIndex(max_dim=256)

    # Use various byte values to ensure no data corruption
    vector = np.array([0, 1, 127, 128, 255, 42, 200], dtype=np.uint8)

    index.add_one(50, vector)
    retrieved = index.get_key(50)

    assert retrieved is not None
    assert retrieved.dtype == np.uint8
    np.testing.assert_array_equal(retrieved, vector)


def test_nphd_index_get_key_roundtrip_all_lengths():
    """Verify get_key roundtrip for all possible lengths up to max_dim."""
    index = NphdIndex(max_dim=64)  # 8 bytes max

    # Test all lengths from 1 to max_bytes
    for length in range(1, 9):
        vector = np.array([length] * length, dtype=np.uint8)
        key = length * 10

        index.add_one(key, vector)
        retrieved = index.get_key(key)

        assert retrieved is not None
        assert len(retrieved) == length
        np.testing.assert_array_equal(retrieved, vector)
