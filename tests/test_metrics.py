"""Tests for custom ISCC metrics."""

import numpy as np
import pytest
from usearch.index import Index, ScalarKind

from iscc_vdb.metrics import (
    create_nphd_metric,
    pack_binary_vector,
    unpack_binary_vector,
)


def test_pack_unpack_binary_vector():
    # type: () -> None
    """Test packing and unpacking of binary vectors with length signals."""
    # Test various lengths from 1 to 255 bytes
    test_cases = [
        (b"\x00", 1),
        (b"\xff" * 5, 5),
        (b"\xaa" * 10, 10),
        (b"\x55" * 32, 32),
        (b"\x12" * 100, 100),
        (b"\x34" * 255, 255),
    ]

    for vector_bytes, expected_length in test_cases:
        # Pack the vector with appropriate max_bytes
        max_bytes = max(32, expected_length)  # Ensure max_bytes >= length
        packed = pack_binary_vector(vector_bytes, max_bytes)

        # Check packed vector properties
        assert packed.dtype == np.uint8
        assert len(packed) == max_bytes + 1  # 1 byte signal + max_bytes data

        # Check length signal
        assert packed[0] == expected_length

        # Unpack and verify
        unpacked = unpack_binary_vector(packed)
        assert unpacked == vector_bytes
        assert len(unpacked) == expected_length


def test_pack_binary_vector_invalid_length():
    # type: () -> None
    """Test that pack_binary_vector raises error for invalid lengths."""
    # Test length 0
    with pytest.raises(ValueError, match="Vector must be 1-255 bytes"):
        pack_binary_vector(b"")

    # Test length > 255
    with pytest.raises(ValueError, match="Vector must be 1-255 bytes"):
        pack_binary_vector(b"\x00" * 256, max_bytes=256)

    # Test length > max_bytes
    with pytest.raises(ValueError, match="Vector length 33 exceeds max_bytes 32"):
        pack_binary_vector(b"\x00" * 33, max_bytes=32)


def test_pack_unpack_iscc_vector():
    # type: () -> None
    """Test packing and unpacking of ISCC vectors with length signals."""
    # Test different ISCC lengths
    test_cases = [
        (b"\x00" * 8, 8),
        (b"\xff" * 16, 16),
        (b"\xaa" * 24, 24),
        (b"\x55" * 32, 32),
    ]

    for iscc_bytes, expected_length in test_cases:
        # Pack the vector
        packed = pack_binary_vector(iscc_bytes)

        # Check packed vector properties
        assert packed.dtype == np.uint8
        assert len(packed) == 33  # 1 byte signal + 32 bytes max data

        # Check length signal (now stores actual byte count)
        assert packed[0] == expected_length

        # Unpack and verify
        unpacked = unpack_binary_vector(packed)
        assert unpacked == iscc_bytes
        assert len(unpacked) == expected_length


def test_pack_iscc_vector_invalid_length():
    # type: () -> None
    """Test that pack_binary_vector validates ISCC-specific lengths."""
    # Note: generic pack_binary_vector doesn't enforce ISCC-specific lengths
    # These tests now just verify general length constraints
    with pytest.raises(ValueError, match="Vector must be 1-255 bytes"):
        pack_binary_vector(b"")

    with pytest.raises(ValueError, match="Vector must be 1-255 bytes"):
        pack_binary_vector(b"\x00" * 256, max_bytes=32)


def test_nphd_metric_creation():
    # type: () -> None
    """Test creation of NPHD metric."""
    metric = create_nphd_metric()

    # Verify metric properties
    assert metric.pointer != 0
    assert metric.kind is not None
    assert metric.signature is not None


def test_nphd_with_usearch_index():
    # type: () -> None
    """Test NPHD metric integration with usearch Index."""
    # Create NPHD metric
    nphd_metric = create_nphd_metric()

    # Create index with NPHD metric
    # Using 264 bits (33 bytes) to accommodate length signal + max ISCC
    index = Index(
        ndim=264,  # 33 bytes * 8 bits
        metric=nphd_metric,
        dtype=ScalarKind.B1,
    )

    # Create test ISCC vectors of different lengths
    iscc1_8 = b"\x11" * 8
    iscc2_8 = b"\x22" * 8
    iscc3_16 = b"\x33" * 16
    iscc4_24 = b"\x44" * 24
    iscc5_32 = b"\x55" * 32

    # Pack vectors
    packed1 = pack_binary_vector(iscc1_8)
    packed2 = pack_binary_vector(iscc2_8)
    packed3 = pack_binary_vector(iscc3_16)
    packed4 = pack_binary_vector(iscc4_24)
    packed5 = pack_binary_vector(iscc5_32)

    # Convert to bit arrays for usearch
    bit_packed1 = np.packbits(np.unpackbits(packed1))
    bit_packed2 = np.packbits(np.unpackbits(packed2))
    bit_packed3 = np.packbits(np.unpackbits(packed3))
    bit_packed4 = np.packbits(np.unpackbits(packed4))
    bit_packed5 = np.packbits(np.unpackbits(packed5))

    # Add vectors to index
    index.add(0, bit_packed1)
    index.add(1, bit_packed2)
    index.add(2, bit_packed3)
    index.add(3, bit_packed4)
    index.add(4, bit_packed5)

    assert index.size == 5

    # Search for exact matches
    matches = index.search(bit_packed1, 5)
    assert len(matches) == 5
    assert matches[0].key == 0  # Should find itself first
    assert matches[0].distance == 0.0  # Exact match

    # Verify retrieval
    retrieved = index.get(0)
    assert np.array_equal(retrieved, bit_packed1)


def test_nphd_distance_calculations():
    # type: () -> None
    """Test NPHD distance calculations for various vector pairs."""
    nphd_metric = create_nphd_metric()

    # Create index
    index = Index(
        ndim=264,
        metric=nphd_metric,
        dtype=ScalarKind.B1,
    )

    # Test case 1: Identical vectors (8 bytes)
    vec1 = pack_binary_vector(b"\xff" * 8)
    vec1_bits = np.packbits(np.unpackbits(vec1))
    index.add(0, vec1_bits)

    matches = index.search(vec1_bits, 1)
    assert matches[0].distance == 0.0  # Identical vectors

    # Test case 2: Completely different vectors (8 bytes)
    vec2 = pack_binary_vector(b"\x00" * 8)
    vec2_bits = np.packbits(np.unpackbits(vec2))
    index.add(1, vec2_bits)

    matches = index.search(vec1_bits, 2)
    # Should find vec2 as second match with distance 1.0 (completely different)
    assert len(matches) == 2
    assert matches[1].key == 1
    assert matches[1].distance == 1.0

    # Test case 3: Partially similar vectors
    vec3 = pack_binary_vector(b"\xf0" * 8)  # Half bits match with vec1
    vec3_bits = np.packbits(np.unpackbits(vec3))
    index.add(2, vec3_bits)

    matches = index.search(vec3_bits, 3)
    assert matches[0].key == 2  # Should find itself first
    assert matches[0].distance == 0.0
    # Distance to vec1 should be ~0.5 (half bits differ)
    vec1_match = next(m for m in matches if m.key == 0)
    assert 0.4 < vec1_match.distance < 0.6


def test_nphd_variable_length_comparison():
    # type: () -> None
    """Test NPHD behavior with variable-length vectors."""
    nphd_metric = create_nphd_metric()

    index = Index(
        ndim=264,
        metric=nphd_metric,
        dtype=ScalarKind.B1,
    )

    # Add vectors of different lengths
    # 8-byte vector: all ones
    vec_8 = pack_binary_vector(b"\xff" * 8)
    vec_8_bits = np.packbits(np.unpackbits(vec_8))
    index.add(0, vec_8_bits)

    # 16-byte vector: first 8 bytes all ones, next 8 all zeros
    vec_16 = pack_binary_vector(b"\xff" * 8 + b"\x00" * 8)
    vec_16_bits = np.packbits(np.unpackbits(vec_16))
    index.add(1, vec_16_bits)

    # 32-byte vector: first 8 bytes all ones, rest all zeros
    vec_32 = pack_binary_vector(b"\xff" * 8 + b"\x00" * 24)
    vec_32_bits = np.packbits(np.unpackbits(vec_32))
    index.add(2, vec_32_bits)

    # Search with the 8-byte vector
    matches = index.search(vec_8_bits, 3, exact=True)

    # All vectors should be found
    assert len(matches) == 3

    # Debug: print all matches
    for i, match in enumerate(matches):
        print(f"Match {i}: key={match.key}, distance={match.distance}")

    # All vectors should have distance 0.0 because they share the same 8-byte prefix
    # and the 8-byte vector is compared only on its length
    keys_found = {match.key for match in matches}
    assert keys_found == {0, 1, 2}

    # All should have distance 0.0
    for match in matches:
        assert match.distance == 0.0


def test_nphd_metric_properties():
    # type: () -> None
    """Test that NPHD satisfies metric properties."""
    nphd_metric = create_nphd_metric()

    index = Index(
        ndim=264,
        metric=nphd_metric,
        dtype=ScalarKind.B1,
    )

    # Create test vectors
    vec_a = pack_binary_vector(b"\xaa" * 8)
    vec_b = pack_binary_vector(b"\xbb" * 8)
    vec_c = pack_binary_vector(b"\xcc" * 8)

    vec_a_bits = np.packbits(np.unpackbits(vec_a))
    vec_b_bits = np.packbits(np.unpackbits(vec_b))
    vec_c_bits = np.packbits(np.unpackbits(vec_c))

    # Add vectors
    index.add(0, vec_a_bits)
    index.add(1, vec_b_bits)
    index.add(2, vec_c_bits)

    # Test non-negativity: d(a,b) >= 0
    matches = index.search(vec_a_bits, 3)
    for match in matches:
        assert match.distance >= 0.0

    # Test identity: d(a,a) = 0
    self_match = next(m for m in matches if m.key == 0)
    assert self_match.distance == 0.0

    # Test symmetry: d(a,b) = d(b,a)
    matches_from_a = index.search(vec_a_bits, 3)
    matches_from_b = index.search(vec_b_bits, 3)

    dist_a_to_b = next(m for m in matches_from_a if m.key == 1).distance
    dist_b_to_a = next(m for m in matches_from_b if m.key == 0).distance

    assert abs(dist_a_to_b - dist_b_to_a) < 1e-6  # Allow small floating point errors
