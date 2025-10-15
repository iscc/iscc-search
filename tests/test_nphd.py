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
