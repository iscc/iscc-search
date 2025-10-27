"""
Tests for custom ISCC metrics module.
"""

import numpy as np
import pytest
from usearch.index import Index, MetricKind, ScalarKind

from iscc_search.metrics import MAX_BYTES, create_nphd_metric


def test_nphd_identical_vectors():
    # type: () -> None
    """NPHD distance between identical vectors should be 0.0."""
    metric = create_nphd_metric()
    index = Index(ndim=264, metric=metric, dtype=ScalarKind.B1)

    # Create identical 8-byte (64-bit) vectors: length signal + data
    # First byte = 8 (length), followed by 8 bytes of data
    vec1 = np.array([8] + [255] * 8 + [0] * (MAX_BYTES - 9), dtype=np.uint8)
    vec2 = np.array([8] + [255] * 8 + [0] * (MAX_BYTES - 9), dtype=np.uint8)

    index.add(0, vec1)
    index.add(1, vec2)

    matches = index.search(vec1, 1)
    # When vectors are identical, either could be returned first
    assert matches.keys[0] in [0, 1]
    assert matches.distances[0] == pytest.approx(0.0, abs=1e-6)


def test_nphd_completely_different_vectors():
    # type: () -> None
    """NPHD distance between completely different vectors should be ~1.0."""
    metric = create_nphd_metric()
    index = Index(ndim=264, metric=metric, dtype=ScalarKind.B1)

    # Create completely different 8-byte vectors
    vec1 = np.array([8] + [255] * 8 + [0] * (MAX_BYTES - 9), dtype=np.uint8)
    vec2 = np.array([8] + [0] * 8 + [0] * (MAX_BYTES - 9), dtype=np.uint8)

    index.add(0, vec1)
    index.add(1, vec2)

    matches = index.search(vec1, 2)
    # Distance should be 1.0 (all bits different)
    if matches.keys[0] == 1:
        assert matches.distances[0] == pytest.approx(1.0, abs=1e-6)
    else:
        assert matches.distances[1] == pytest.approx(1.0, abs=1e-6)


def test_nphd_single_bit_difference():
    # type: () -> None
    """NPHD with single bit difference should be 1/num_bits."""
    metric = create_nphd_metric()
    index = Index(ndim=264, metric=metric, dtype=ScalarKind.B1)

    # Create 8-byte vectors with single bit difference
    # vec1: all zeros, vec2: one bit set in last byte
    vec1 = np.array([8] + [0] * 8 + [0] * (MAX_BYTES - 9), dtype=np.uint8)
    vec2 = np.array([8] + [0] * 7 + [1] + [0] * (MAX_BYTES - 9), dtype=np.uint8)

    index.add(0, vec1)
    index.add(1, vec2)

    matches = index.search(vec1, 2)
    # Distance should be 1/64 = 0.015625
    expected_distance = 1.0 / 64.0
    if matches.keys[0] == 1:
        assert matches.distances[0] == pytest.approx(expected_distance, abs=1e-6)
    else:
        assert matches.distances[1] == pytest.approx(expected_distance, abs=1e-6)


def test_nphd_prefix_compatibility_different_lengths():
    # type: () -> None
    """NPHD should use shorter vector length for comparison."""
    metric = create_nphd_metric()
    index = Index(ndim=264, metric=metric, dtype=ScalarKind.B1)

    # Create vectors of different lengths with matching prefix
    # Short vector: 4 bytes
    vec_short = np.array([4] + [255] * 4 + [0] * (MAX_BYTES - 5), dtype=np.uint8)
    # Long vector: 8 bytes, same prefix but different suffix
    vec_long = np.array([8] + [255] * 4 + [170] * 4 + [0] * (MAX_BYTES - 9), dtype=np.uint8)

    index.add(0, vec_short)
    index.add(1, vec_long)

    # Search with short vector - should match perfectly (distance 0.0)
    matches = index.search(vec_short, 2)
    # Both vectors should have 0.0 distance because they share the same 4-byte prefix
    assert all(d == pytest.approx(0.0, abs=1e-6) for d in matches.distances)
    # Both keys should be present
    assert set(matches.keys) == {0, 1}


def test_nphd_prefix_compatibility_partial_match():
    # type: () -> None
    """NPHD should normalize by shorter vector when prefixes differ."""
    metric = create_nphd_metric()
    index = Index(ndim=264, metric=metric, dtype=ScalarKind.B1)

    # Short vector: 4 bytes with one bit set
    vec_short = np.array([4] + [0] * 3 + [1] + [0] * (MAX_BYTES - 5), dtype=np.uint8)
    # Long vector: 8 bytes, different in the 4-byte prefix
    vec_long = np.array([8] + [0] * 8 + [0] * (MAX_BYTES - 9), dtype=np.uint8)

    index.add(0, vec_short)
    index.add(1, vec_long)

    matches = index.search(vec_short, 2)
    # Distance should be 1/32 (1 bit difference in 32 bits = 4 bytes)
    expected_distance = 1.0 / 32.0
    if matches.keys[0] == 1:
        assert matches.distances[0] == pytest.approx(expected_distance, abs=1e-6)
    else:
        assert matches.distances[1] == pytest.approx(expected_distance, abs=1e-6)


def test_nphd_zero_length_vectors():
    # type: () -> None
    """NPHD with zero-length vectors should return 0.0."""
    metric = create_nphd_metric()
    index = Index(ndim=264, metric=metric, dtype=ScalarKind.B1)

    # Create zero-length vectors (length signal = 0)
    vec1 = np.array([0] + [255] * (MAX_BYTES - 1), dtype=np.uint8)
    vec2 = np.array([0] + [170] * (MAX_BYTES - 1), dtype=np.uint8)

    index.add(0, vec1)
    index.add(1, vec2)

    matches = index.search(vec1, 2)
    # Both zero-length vectors should have 0.0 distance
    assert all(d == pytest.approx(0.0, abs=1e-6) for d in matches.distances)
    assert set(matches.keys) == {0, 1}


def test_nphd_maximum_length_vectors():
    # type: () -> None
    """NPHD should work with maximum length (32-byte) vectors."""
    metric = create_nphd_metric()
    index = Index(ndim=264, metric=metric, dtype=ScalarKind.B1)

    # Create maximum length vectors (32 bytes)
    vec1 = np.array([32] + [255] * 32, dtype=np.uint8)
    vec2 = np.array([32] + [170] * 32, dtype=np.uint8)

    index.add(0, vec1)
    index.add(1, vec2)

    matches = index.search(vec1, 1)
    assert matches.keys[0] == 0
    # Should find itself with 0.0 distance


def test_nphd_single_byte_vectors():
    # type: () -> None
    """NPHD should work with minimal (1-byte) vectors."""
    metric = create_nphd_metric()
    index = Index(ndim=264, metric=metric, dtype=ScalarKind.B1)

    # Create single-byte vectors
    vec1 = np.array([1] + [255] + [0] * (MAX_BYTES - 2), dtype=np.uint8)
    vec2 = np.array([1] + [0] + [0] * (MAX_BYTES - 2), dtype=np.uint8)

    index.add(0, vec1)
    index.add(1, vec2)

    matches = index.search(vec1, 2)
    # Distance should be 1.0 (all 8 bits different)
    if matches.keys[0] == 1:
        assert matches.distances[0] == pytest.approx(1.0, abs=1e-6)
    else:
        assert matches.distances[1] == pytest.approx(1.0, abs=1e-6)


def test_nphd_multiple_bit_differences():
    # type: () -> None
    """NPHD should correctly calculate distance with multiple bit differences."""
    metric = create_nphd_metric()
    index = Index(ndim=264, metric=metric, dtype=ScalarKind.B1)

    # Create 4-byte vectors with 4 bits different (15 has 4 bits set)
    vec1 = np.array([4] + [0] * 4 + [0] * (MAX_BYTES - 5), dtype=np.uint8)
    vec2 = np.array([4] + [0] * 3 + [15] + [0] * (MAX_BYTES - 5), dtype=np.uint8)

    index.add(0, vec1)
    index.add(1, vec2)

    matches = index.search(vec1, 2)
    # Distance should be 4/32 = 0.125 (4 bits different in 32 bits)
    expected_distance = 4.0 / 32.0
    if matches.keys[0] == 1:
        assert matches.distances[0] == pytest.approx(expected_distance, abs=1e-6)
    else:
        assert matches.distances[1] == pytest.approx(expected_distance, abs=1e-6)


def test_create_nphd_metric_returns_compiled_metric():
    # type: () -> None
    """create_nphd_metric should return a valid CompiledMetric instance."""
    metric = create_nphd_metric()

    assert metric is not None
    assert hasattr(metric, "pointer")
    assert hasattr(metric, "kind")
    assert hasattr(metric, "signature")
    assert metric.kind == MetricKind.Hamming


def test_nphd_metric_integration_with_index():
    # type: () -> None
    """NPHD metric should integrate properly with usearch Index."""
    metric = create_nphd_metric()

    # Should be able to create an index with the metric
    index = Index(ndim=264, metric=metric, dtype=ScalarKind.B1)

    # Should be able to add vectors
    vec = np.array([8] + [255] * 8 + [0] * (MAX_BYTES - 9), dtype=np.uint8)
    index.add(0, vec)

    # Should be able to search
    matches = index.search(vec, 1)
    assert len(matches.keys) == 1
    assert matches.keys[0] == 0
