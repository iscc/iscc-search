"""
Confirm the expected behavior of usearch Index.add() with

- metric=MetricKind.Hamming
- dtype=ScalarKind.B1
- multi=False (single vector per key)
- multi=True (multiple vectors per key)
"""

import numpy as np
import pytest
from numpy.testing import assert_array_equal
from usearch.index import Index, ScalarKind, MetricKind


# Tests for Index.add() with multi=False (single vector per key)


def test_add_single_key_returns_array_with_key():
    """Adding a single vector returns numpy array containing the key."""
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1, multi=False)

    result = idx.add(1, np.array([178, 204, 60, 240], dtype=np.uint8))

    assert isinstance(result, np.ndarray)
    assert result.dtype == np.uint64
    assert result.shape == (1,)
    assert result[0] == 1

    # Verify vector was stored
    stored = idx.get(1)
    assert_array_equal(stored, np.array([178, 204, 60, 240], dtype=np.uint8))


def test_add_multiple_different_keys_returns_respective_keys():
    """Adding vectors to different keys returns their keys."""
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1, multi=False)

    result1 = idx.add(1, np.array([178, 204, 60, 240], dtype=np.uint8))
    result2 = idx.add(2, np.array([100, 150, 200, 250], dtype=np.uint8))
    result3 = idx.add(3, np.array([1, 2, 3, 4], dtype=np.uint8))

    assert result1[0] == 1
    assert result2[0] == 2
    assert result3[0] == 3


def test_add_duplicate_key_raises_runtime_error():
    """Adding to same key twice with multi=False raises RuntimeError."""
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1, multi=False)

    idx.add(1, np.array([178, 204, 60, 240], dtype=np.uint8))

    with pytest.raises(RuntimeError, match="Duplicate keys not allowed"):
        idx.add(1, np.array([100, 150, 200, 250], dtype=np.uint8))


def test_add_with_key_none_generates_key():
    """Adding with key=None auto-generates a key starting from 0."""
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1, multi=False)

    result = idx.add(None, np.array([178, 204, 60, 240], dtype=np.uint8))

    assert isinstance(result, np.ndarray)
    assert result.shape == (1,)
    assert result[0] == 0  # First auto-generated key is 0

    # Verify vector was stored with generated key
    stored = idx.get(0)
    assert_array_equal(stored, np.array([178, 204, 60, 240], dtype=np.uint8))


def test_add_batch_with_explicit_keys_returns_all_keys():
    """Batch add with explicit keys returns array of all keys."""
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1, multi=False)

    keys = [1, 2, 3]
    vectors = np.array([
        [178, 204, 60, 240],
        [100, 150, 200, 250],
        [1, 2, 3, 4],
    ], dtype=np.uint8)

    result = idx.add(keys, vectors)

    assert isinstance(result, np.ndarray)
    assert result.dtype == np.uint64
    assert result.shape == (3,)
    assert_array_equal(result, np.array([1, 2, 3], dtype=np.uint64))

    # Verify all vectors were stored
    assert_array_equal(idx.get(1), vectors[0])
    assert_array_equal(idx.get(2), vectors[1])
    assert_array_equal(idx.get(3), vectors[2])


def test_add_batch_with_key_none_generates_sequential_keys():
    """Batch add with key=None auto-generates sequential keys starting from 0."""
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1, multi=False)

    vectors = np.array([
        [178, 204, 60, 240],
        [100, 150, 200, 250],
        [1, 2, 3, 4],
    ], dtype=np.uint8)

    result = idx.add(None, vectors)

    assert isinstance(result, np.ndarray)
    assert result.shape == (3,)
    assert_array_equal(result, np.array([0, 1, 2], dtype=np.uint64))

    # Verify all vectors were stored with generated keys
    assert_array_equal(idx.get(0), vectors[0])
    assert_array_equal(idx.get(1), vectors[1])
    assert_array_equal(idx.get(2), vectors[2])


# Tests for Index.add() with multi=True (multiple vectors per key)


def test_add_single_key_multi_returns_array_with_key():
    """Adding a single vector with multi=True returns array containing the key."""
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1, multi=True)

    result = idx.add(1, np.array([178, 204, 60, 240], dtype=np.uint8))

    assert isinstance(result, np.ndarray)
    assert result.dtype == np.uint64
    assert result.shape == (1,)
    assert result[0] == 1

    # Verify vector was stored
    stored = idx.get(1)
    assert isinstance(stored, np.ndarray)
    assert stored.ndim == 2
    assert stored.shape[0] == 1
    assert_array_equal(stored[0], np.array([178, 204, 60, 240], dtype=np.uint8))


def test_add_multiple_vectors_same_key_returns_same_key():
    """Adding multiple vectors to same key with multi=True returns the same key each time."""
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1, multi=True)

    result1 = idx.add(1, np.array([178, 204, 60, 240], dtype=np.uint8))
    result2 = idx.add(1, np.array([100, 150, 200, 250], dtype=np.uint8))
    result3 = idx.add(1, np.array([1, 2, 3, 4], dtype=np.uint8))

    assert result1[0] == 1
    assert result2[0] == 1
    assert result3[0] == 1

    # Verify all vectors were stored
    stored = idx.get(1)
    assert stored.shape == (3, 4)
    assert_array_equal(stored[0], np.array([178, 204, 60, 240], dtype=np.uint8))
    assert_array_equal(stored[1], np.array([100, 150, 200, 250], dtype=np.uint8))
    assert_array_equal(stored[2], np.array([1, 2, 3, 4], dtype=np.uint8))


def test_add_with_key_none_multi_generates_key():
    """Adding with key=None and multi=True auto-generates a key."""
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1, multi=True)

    result = idx.add(None, np.array([178, 204, 60, 240], dtype=np.uint8))

    assert isinstance(result, np.ndarray)
    assert result.shape == (1,)
    assert result[0] == 0

    # Verify vector was stored with generated key
    stored = idx.get(0)
    assert stored.shape == (1, 4)
    assert_array_equal(stored[0], np.array([178, 204, 60, 240], dtype=np.uint8))


def test_add_batch_with_explicit_keys_multi_returns_all_keys():
    """Batch add with explicit keys and multi=True returns array of all keys."""
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1, multi=True)

    keys = [1, 2, 3]
    vectors = np.array([
        [178, 204, 60, 240],
        [100, 150, 200, 250],
        [1, 2, 3, 4],
    ], dtype=np.uint8)

    result = idx.add(keys, vectors)

    assert isinstance(result, np.ndarray)
    assert result.dtype == np.uint64
    assert result.shape == (3,)
    assert_array_equal(result, np.array([1, 2, 3], dtype=np.uint64))

    # Verify all vectors were stored
    assert_array_equal(idx.get(1)[0], vectors[0])
    assert_array_equal(idx.get(2)[0], vectors[1])
    assert_array_equal(idx.get(3)[0], vectors[2])


def test_add_batch_with_key_none_multi_generates_sequential_keys():
    """Batch add with key=None and multi=True auto-generates sequential keys."""
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1, multi=True)

    vectors = np.array([
        [178, 204, 60, 240],
        [100, 150, 200, 250],
        [1, 2, 3, 4],
    ], dtype=np.uint8)

    result = idx.add(None, vectors)

    assert isinstance(result, np.ndarray)
    assert result.shape == (3,)
    assert_array_equal(result, np.array([0, 1, 2], dtype=np.uint64))

    # Verify all vectors were stored with generated keys
    assert_array_equal(idx.get(0)[0], vectors[0])
    assert_array_equal(idx.get(1)[0], vectors[1])
    assert_array_equal(idx.get(2)[0], vectors[2])


def test_add_batch_with_duplicate_keys_multi_stores_all_vectors():
    """Batch add with duplicate keys and multi=True stores all vectors for each key."""
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1, multi=True)

    # Add multiple vectors to key 1 in a single batch
    keys = [1, 1, 1]
    vectors = np.array([
        [178, 204, 60, 240],
        [100, 150, 200, 250],
        [1, 2, 3, 4],
    ], dtype=np.uint8)

    result = idx.add(keys, vectors)

    assert result.shape == (3,)
    assert_array_equal(result, np.array([1, 1, 1], dtype=np.uint64))

    # Verify all three vectors are stored for key 1
    stored = idx.get(1)
    assert stored.shape == (3, 4)
    assert_array_equal(stored[0], vectors[0])
    assert_array_equal(stored[1], vectors[1])
    assert_array_equal(stored[2], vectors[2])


# Tests for auto-key generation behavior


def test_autokey_equals_current_index_size():
    """
    Auto-generated keys (key=None) equal the current size of the index.

    The auto-key generation algorithm uses len(index) as the next key.
    This is independent of what explicit keys have been used.
    """
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1, multi=False)

    # Empty index: size=0, autokey should be 0
    result1 = idx.add(None, np.array([1, 1, 1, 1], dtype=np.uint8))
    assert result1[0] == 0
    assert len(idx) == 1

    # Add explicit key 100: size becomes 2
    idx.add(100, np.array([2, 2, 2, 2], dtype=np.uint8))
    assert len(idx) == 2

    # Autokey should now be 2 (not 1!)
    result2 = idx.add(None, np.array([3, 3, 3, 3], dtype=np.uint8))
    assert result2[0] == 2
    assert len(idx) == 3

    # Add explicit key 50: size becomes 4
    idx.add(50, np.array([4, 4, 4, 4], dtype=np.uint8))
    assert len(idx) == 4

    # Autokey should now be 4
    result3 = idx.add(None, np.array([5, 5, 5, 5], dtype=np.uint8))
    assert result3[0] == 4
    assert len(idx) == 5


def test_autokey_with_non_contiguous_explicit_keys():
    """
    Auto-generated keys use index size, not highest existing key.

    When explicit keys like 5, 10, 100 are used, auto-generated keys
    start from the index size, not from 101.
    """
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1, multi=False)

    # Add non-contiguous explicit keys
    idx.add(5, np.array([5, 5, 5, 5], dtype=np.uint8))
    idx.add(10, np.array([10, 10, 10, 10], dtype=np.uint8))
    idx.add(100, np.array([100, 100, 100, 100], dtype=np.uint8))

    assert len(idx) == 3

    # Autokey should be 3 (not 101!)
    result = idx.add(None, np.array([50, 60, 70, 80], dtype=np.uint8))
    assert result[0] == 3


def test_autokey_can_collide_with_explicit_keys():
    """
    Auto-generated keys can collide with explicitly assigned keys.

    Since autokey = len(index), if an explicit key equals a future size,
    a collision will occur and raise RuntimeError.
    """
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1, multi=False)

    # Add explicit keys at positions that will collide with future autokeys
    idx.add(0, np.array([0, 0, 0, 0], dtype=np.uint8))  # size=1
    idx.add(2, np.array([2, 2, 2, 2], dtype=np.uint8))  # size=2
    idx.add(4, np.array([4, 4, 4, 4], dtype=np.uint8))  # size=3

    # Autokey should be 3 (size=3), which doesn't collide
    result = idx.add(None, np.array([10, 10, 10, 10], dtype=np.uint8))
    assert result[0] == 3
    assert len(idx) == 4

    # Next autokey would be 4, but key 4 already exists!
    with pytest.raises(RuntimeError, match="Duplicate keys not allowed"):
        idx.add(None, np.array([11, 11, 11, 11], dtype=np.uint8))


def test_autokey_batch_uses_sequential_sizes():
    """
    Batch add with key=None generates keys based on sequential sizes.

    When adding N vectors with key=None, the keys are:
    [current_size, current_size+1, ..., current_size+N-1]
    """
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1, multi=False)

    # Add explicit keys first
    idx.add(5, np.array([5, 5, 5, 5], dtype=np.uint8))
    idx.add(20, np.array([20, 20, 20, 20], dtype=np.uint8))

    assert len(idx) == 2

    # Batch add 3 vectors with key=None
    vectors = np.array([
        [10, 10, 10, 10],
        [11, 11, 11, 11],
        [12, 12, 12, 12],
    ], dtype=np.uint8)

    result = idx.add(None, vectors)

    # Should get keys [2, 3, 4] (sizes during the batch operation)
    assert_array_equal(result, np.array([2, 3, 4], dtype=np.uint64))
    assert len(idx) == 5
