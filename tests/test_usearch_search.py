"""
Confirm the expected behavior of usearch Index.search() with

- metric=MetricKind.Hamming
- dtype=ScalarKind.B1
- single query vector (returns Matches)
- batch query vectors (returns BatchMatches)
- different count values
- exact=True vs exact=False
"""

import numpy as np
import pytest
from numpy.testing import assert_array_equal
from usearch.index import Index, ScalarKind, MetricKind


# Tests for Index.search() with single query vector (returns Matches)


def test_search_single_vector_returns_matches_object():
    """Searching with single vector returns Matches object."""
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1)
    idx.add(1, np.array([178, 204, 60, 240], dtype=np.uint8))
    idx.add(2, np.array([100, 150, 200, 250], dtype=np.uint8))
    idx.add(3, np.array([1, 2, 3, 4], dtype=np.uint8))

    query = np.array([178, 204, 60, 240], dtype=np.uint8)
    result = idx.search(query, count=3)

    # Verify result is a Matches object (not BatchMatches)
    assert hasattr(result, "keys")
    assert hasattr(result, "distances")
    assert hasattr(result, "visited_members")
    assert hasattr(result, "computed_distances")
    assert len(result) == 3


def test_search_single_vector_keys_and_distances_are_1d_arrays():
    """Matches object has 1D numpy arrays for keys and distances."""
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1)
    idx.add(1, np.array([178, 204, 60, 240], dtype=np.uint8))
    idx.add(2, np.array([100, 150, 200, 250], dtype=np.uint8))
    idx.add(3, np.array([1, 2, 3, 4], dtype=np.uint8))

    query = np.array([178, 204, 60, 240], dtype=np.uint8)
    result = idx.search(query, count=3)

    assert isinstance(result.keys, np.ndarray)
    assert result.keys.ndim == 1
    assert result.keys.shape == (3,)

    assert isinstance(result.distances, np.ndarray)
    assert result.distances.ndim == 1
    assert result.distances.shape == (3,)


def test_search_single_vector_exact_match_has_zero_distance():
    """Searching for exact match returns distance of 0."""
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1)
    idx.add(1, np.array([178, 204, 60, 240], dtype=np.uint8))
    idx.add(2, np.array([100, 150, 200, 250], dtype=np.uint8))

    query = np.array([178, 204, 60, 240], dtype=np.uint8)
    result = idx.search(query, count=1)

    expected_key = np.array([1], dtype=np.uint64)
    expected_distance = np.array([0.0], dtype=np.float32)

    assert result.keys[0] == expected_key[0]
    assert result.distances[0] == expected_distance[0]


def test_search_single_vector_results_ordered_by_distance():
    """Search results are ordered by increasing distance."""
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1)
    # Add vectors with known Hamming distances to query
    idx.add(1, np.array([255, 255, 255, 255], dtype=np.uint8))  # Will be furthest
    idx.add(2, np.array([178, 204, 60, 240], dtype=np.uint8))  # Exact match (distance 0)
    idx.add(3, np.array([178, 204, 60, 241], dtype=np.uint8))  # 1 bit difference

    query = np.array([178, 204, 60, 240], dtype=np.uint8)
    result = idx.search(query, count=3)

    expected_keys = np.array([2, 3, 1], dtype=np.uint64)

    assert_array_equal(result.keys, expected_keys)
    # Verify distances are in ascending order
    assert result.distances[0] < result.distances[1] < result.distances[2]
    assert result.distances[0] == 0.0  # Exact match


def test_search_single_vector_count_limits_results():
    """count parameter limits number of results returned."""
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1)
    idx.add(1, np.array([178, 204, 60, 240], dtype=np.uint8))
    idx.add(2, np.array([100, 150, 200, 250], dtype=np.uint8))
    idx.add(3, np.array([1, 2, 3, 4], dtype=np.uint8))
    idx.add(4, np.array([10, 20, 30, 40], dtype=np.uint8))
    idx.add(5, np.array([50, 60, 70, 80], dtype=np.uint8))

    query = np.array([178, 204, 60, 240], dtype=np.uint8)

    # Request only 2 results
    result = idx.search(query, count=2)
    expected_length = 2

    assert len(result) == expected_length
    assert result.keys.shape == (2,)
    assert result.distances.shape == (2,)


def test_search_single_vector_default_count_is_10():
    """Default count parameter is 10."""
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1)
    # Add only 3 vectors
    idx.add(1, np.array([178, 204, 60, 240], dtype=np.uint8))
    idx.add(2, np.array([100, 150, 200, 250], dtype=np.uint8))
    idx.add(3, np.array([1, 2, 3, 4], dtype=np.uint8))

    query = np.array([178, 204, 60, 240], dtype=np.uint8)
    result = idx.search(query)  # No count specified

    # Should return all 3 vectors (limited by index size, not default count)
    expected_length = 3

    assert len(result) == expected_length


def test_search_single_vector_count_exceeds_index_size():
    """Requesting more results than index size returns all vectors."""
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1)
    idx.add(1, np.array([178, 204, 60, 240], dtype=np.uint8))
    idx.add(2, np.array([100, 150, 200, 250], dtype=np.uint8))

    query = np.array([178, 204, 60, 240], dtype=np.uint8)
    result = idx.search(query, count=100)

    expected_length = 2  # Only 2 vectors in index

    assert len(result) == expected_length


def test_search_single_vector_empty_index_returns_empty_matches():
    """Searching empty index returns Matches with length 0."""
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1)

    query = np.array([178, 204, 60, 240], dtype=np.uint8)
    result = idx.search(query, count=10)

    expected_length = 0

    assert len(result) == expected_length
    assert result.keys.shape == (0,)
    assert result.distances.shape == (0,)


def test_search_single_vector_accessing_individual_matches():
    """Individual Match objects can be accessed by index."""
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1)
    idx.add(1, np.array([178, 204, 60, 240], dtype=np.uint8))
    idx.add(2, np.array([100, 150, 200, 250], dtype=np.uint8))

    query = np.array([178, 204, 60, 240], dtype=np.uint8)
    result = idx.search(query, count=2)

    # Access first match
    first_match = result[0]
    assert hasattr(first_match, "key")
    assert hasattr(first_match, "distance")
    assert first_match.key == result.keys[0]
    assert first_match.distance == result.distances[0]

    # Access second match
    second_match = result[1]
    assert second_match.key == result.keys[1]
    assert second_match.distance == result.distances[1]


def test_search_single_vector_to_list_returns_tuples():
    """Matches.to_list() returns list of (key, distance) tuples."""
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1)
    idx.add(1, np.array([178, 204, 60, 240], dtype=np.uint8))
    idx.add(2, np.array([100, 150, 200, 250], dtype=np.uint8))

    query = np.array([178, 204, 60, 240], dtype=np.uint8)
    result = idx.search(query, count=2)

    result_list = result.to_list()

    assert isinstance(result_list, list)
    assert len(result_list) == 2
    assert all(isinstance(item, tuple) and len(item) == 2 for item in result_list)
    # Verify first tuple matches first result
    assert result_list[0][0] == result.keys[0]
    assert result_list[0][1] == result.distances[0]


# Tests for Index.search() with batch query vectors (returns BatchMatches)


def test_search_batch_vectors_returns_batch_matches_object():
    """Searching with multiple vectors returns BatchMatches object."""
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1)
    idx.add(1, np.array([178, 204, 60, 240], dtype=np.uint8))
    idx.add(2, np.array([100, 150, 200, 250], dtype=np.uint8))
    idx.add(3, np.array([1, 2, 3, 4], dtype=np.uint8))

    queries = np.array(
        [
            [178, 204, 60, 240],
            [100, 150, 200, 250],
        ],
        dtype=np.uint8,
    )
    result = idx.search(queries, count=3)

    # Verify result is a BatchMatches object
    assert hasattr(result, "keys")
    assert hasattr(result, "distances")
    assert hasattr(result, "counts")
    assert hasattr(result, "visited_members")
    assert hasattr(result, "computed_distances")
    assert len(result) == 2  # Two queries


def test_search_batch_vectors_keys_and_distances_are_2d_arrays():
    """BatchMatches object has 2D numpy arrays for keys and distances."""
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1)
    idx.add(1, np.array([178, 204, 60, 240], dtype=np.uint8))
    idx.add(2, np.array([100, 150, 200, 250], dtype=np.uint8))
    idx.add(3, np.array([1, 2, 3, 4], dtype=np.uint8))

    queries = np.array(
        [
            [178, 204, 60, 240],
            [100, 150, 200, 250],
        ],
        dtype=np.uint8,
    )
    result = idx.search(queries, count=3)

    assert isinstance(result.keys, np.ndarray)
    assert result.keys.ndim == 2
    assert result.keys.shape == (2, 3)  # 2 queries, up to 3 results each

    assert isinstance(result.distances, np.ndarray)
    assert result.distances.ndim == 2
    assert result.distances.shape == (2, 3)


def test_search_batch_vectors_each_query_has_own_results():
    """Each query in batch has independent results."""
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1)
    idx.add(1, np.array([178, 204, 60, 240], dtype=np.uint8))
    idx.add(2, np.array([100, 150, 200, 250], dtype=np.uint8))
    idx.add(3, np.array([1, 2, 3, 4], dtype=np.uint8))

    queries = np.array(
        [
            [178, 204, 60, 240],  # Exact match to key 1
            [100, 150, 200, 250],  # Exact match to key 2
        ],
        dtype=np.uint8,
    )
    result = idx.search(queries, count=1)

    # First query should find key 1 with distance 0
    expected_keys_query_0 = np.array([1], dtype=np.uint64)
    expected_distance_query_0 = np.array([0.0], dtype=np.float32)

    assert result.keys[0][0] == expected_keys_query_0[0]
    assert result.distances[0][0] == expected_distance_query_0[0]

    # Second query should find key 2 with distance 0
    expected_keys_query_1 = np.array([2], dtype=np.uint64)
    expected_distance_query_1 = np.array([0.0], dtype=np.float32)

    assert result.keys[1][0] == expected_keys_query_1[0]
    assert result.distances[1][0] == expected_distance_query_1[0]


def test_search_batch_vectors_counts_array_shows_results_per_query():
    """BatchMatches.counts array shows number of valid results per query."""
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1)
    idx.add(1, np.array([178, 204, 60, 240], dtype=np.uint8))
    idx.add(2, np.array([100, 150, 200, 250], dtype=np.uint8))

    queries = np.array(
        [
            [178, 204, 60, 240],
            [100, 150, 200, 250],
        ],
        dtype=np.uint8,
    )
    # Request more results than available
    result = idx.search(queries, count=10)

    expected_counts = np.array([2, 2], dtype=np.uint64)

    assert isinstance(result.counts, np.ndarray)
    assert_array_equal(result.counts, expected_counts)


def test_search_batch_vectors_accessing_individual_query_matches():
    """Individual Matches objects for each query can be accessed by index."""
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1)
    idx.add(1, np.array([178, 204, 60, 240], dtype=np.uint8))
    idx.add(2, np.array([100, 150, 200, 250], dtype=np.uint8))
    idx.add(3, np.array([1, 2, 3, 4], dtype=np.uint8))

    queries = np.array(
        [
            [178, 204, 60, 240],
            [100, 150, 200, 250],
        ],
        dtype=np.uint8,
    )
    result = idx.search(queries, count=2)

    # Access Matches object for first query
    first_query_matches = result[0]
    assert hasattr(first_query_matches, "keys")
    assert hasattr(first_query_matches, "distances")
    assert len(first_query_matches) == 2

    # Access Matches object for second query
    second_query_matches = result[1]
    assert len(second_query_matches) == 2


def test_search_batch_vectors_accessing_nested_match_objects():
    """Match objects can be accessed from BatchMatches via query index."""
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1)
    idx.add(1, np.array([178, 204, 60, 240], dtype=np.uint8))
    idx.add(2, np.array([100, 150, 200, 250], dtype=np.uint8))

    queries = np.array(
        [
            [178, 204, 60, 240],
            [100, 150, 200, 250],
        ],
        dtype=np.uint8,
    )
    result = idx.search(queries, count=2)

    # Access first Match of first query
    first_query_first_match = result[0][0]
    assert hasattr(first_query_first_match, "key")
    assert hasattr(first_query_first_match, "distance")
    assert first_query_first_match.key == result.keys[0][0]
    assert first_query_first_match.distance == result.distances[0][0]


def test_search_batch_vectors_to_list_returns_flattened_tuples():
    """BatchMatches.to_list() returns flattened list of (key, distance) tuples."""
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1)
    idx.add(1, np.array([178, 204, 60, 240], dtype=np.uint8))
    idx.add(2, np.array([100, 150, 200, 250], dtype=np.uint8))

    queries = np.array(
        [
            [178, 204, 60, 240],
            [100, 150, 200, 250],
        ],
        dtype=np.uint8,
    )
    result = idx.search(queries, count=2)

    result_list = result.to_list()

    # Should be flattened: 2 queries * 2 results each = 4 tuples
    expected_length = 4

    assert isinstance(result_list, list)
    assert len(result_list) == expected_length
    assert all(isinstance(item, tuple) and len(item) == 2 for item in result_list)


# Tests for exact parameter (exact=True vs exact=False)


def test_search_exact_false_uses_approximate_search():
    """exact=False (default) uses approximate HNSW search."""
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1)
    idx.add(1, np.array([178, 204, 60, 240], dtype=np.uint8))
    idx.add(2, np.array([100, 150, 200, 250], dtype=np.uint8))

    query = np.array([178, 204, 60, 240], dtype=np.uint8)
    result = idx.search(query, count=2, exact=False)

    expected_length = 2

    # Approximate search should still work correctly
    assert len(result) == expected_length
    # First result should be exact match with distance 0
    assert result.keys[0] == 1
    assert result.distances[0] == 0.0


def test_search_exact_true_uses_exhaustive_search():
    """exact=True uses exhaustive linear-time search."""
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1)
    idx.add(1, np.array([178, 204, 60, 240], dtype=np.uint8))
    idx.add(2, np.array([100, 150, 200, 250], dtype=np.uint8))

    query = np.array([178, 204, 60, 240], dtype=np.uint8)
    result = idx.search(query, count=2, exact=True)

    expected_length = 2

    # Exact search guarantees true nearest neighbors
    assert len(result) == expected_length
    # First result should be exact match with distance 0
    assert result.keys[0] == 1
    assert result.distances[0] == 0.0


def test_search_exact_true_and_false_same_results_small_dataset():
    """For small datasets, exact and approximate search return same results."""
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1)
    idx.add(1, np.array([178, 204, 60, 240], dtype=np.uint8))
    idx.add(2, np.array([100, 150, 200, 250], dtype=np.uint8))
    idx.add(3, np.array([1, 2, 3, 4], dtype=np.uint8))

    query = np.array([178, 204, 60, 240], dtype=np.uint8)

    result_approximate = idx.search(query, count=3, exact=False)
    result_exact = idx.search(query, count=3, exact=True)

    # For small datasets, results should be identical
    assert_array_equal(result_approximate.keys, result_exact.keys)
    assert_array_equal(result_approximate.distances, result_exact.distances)


# Tests for visited_members and computed_distances statistics


def test_search_provides_statistics():
    """Matches object provides search statistics."""
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1)
    idx.add(1, np.array([178, 204, 60, 240], dtype=np.uint8))
    idx.add(2, np.array([100, 150, 200, 250], dtype=np.uint8))
    idx.add(3, np.array([1, 2, 3, 4], dtype=np.uint8))

    query = np.array([178, 204, 60, 240], dtype=np.uint8)
    result = idx.search(query, count=2)

    # Statistics should be non-negative integers
    assert isinstance(result.visited_members, int)
    assert isinstance(result.computed_distances, int)
    assert result.visited_members >= 0
    assert result.computed_distances >= 0


def test_search_batch_provides_statistics():
    """BatchMatches object provides search statistics across all queries."""
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1)
    idx.add(1, np.array([178, 204, 60, 240], dtype=np.uint8))
    idx.add(2, np.array([100, 150, 200, 250], dtype=np.uint8))

    queries = np.array(
        [
            [178, 204, 60, 240],
            [100, 150, 200, 250],
        ],
        dtype=np.uint8,
    )
    result = idx.search(queries, count=2)

    # Statistics should be non-negative integers
    assert isinstance(result.visited_members, int)
    assert isinstance(result.computed_distances, int)
    assert result.visited_members >= 0
    assert result.computed_distances >= 0


# Tests for edge cases and special scenarios


def test_search_with_count_zero_causes_segfault():
    """
    Searching with count=0 causes segmentation fault (usearch bug).

    This test documents the known issue that usearch crashes when count=0.
    The test is skipped to prevent test suite crashes.

    Related: This should be reported as a bug to unum-cloud/usearch
    """
    pytest.skip("count=0 causes segmentation fault in usearch")


def test_search_with_count_one_returns_single_best_match():
    """Searching with count=1 returns only the best match."""
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1)
    idx.add(1, np.array([178, 204, 60, 240], dtype=np.uint8))
    idx.add(2, np.array([100, 150, 200, 250], dtype=np.uint8))
    idx.add(3, np.array([1, 2, 3, 4], dtype=np.uint8))

    query = np.array([178, 204, 60, 240], dtype=np.uint8)
    result = idx.search(query, count=1)

    expected_length = 1
    expected_key = np.array([1], dtype=np.uint64)
    expected_distance = np.array([0.0], dtype=np.float32)

    assert len(result) == expected_length
    assert result.keys[0] == expected_key[0]
    assert result.distances[0] == expected_distance[0]


def test_search_binary_vectors_hamming_distance_is_bit_count():
    """Hamming distance for binary vectors equals number of differing bits."""
    idx = Index(ndim=8, metric=MetricKind.Hamming, dtype=ScalarKind.B1)
    # Add vector: 11111111 in binary (255 in decimal)
    idx.add(1, np.array([255], dtype=np.uint8))
    # Add vector: 00000000 in binary (0 in decimal)
    idx.add(2, np.array([0], dtype=np.uint8))

    # Query: 11111110 in binary (254 in decimal) - differs from 255 by 1 bit
    query = np.array([254], dtype=np.uint8)
    result = idx.search(query, count=2)

    # Key 1 (255) should be closest: 1 bit difference
    # Key 2 (0) should be further: 7 bit differences
    expected_keys = np.array([1, 2], dtype=np.uint64)
    expected_distance_to_key1 = 1.0
    expected_distance_to_key2 = 7.0

    assert_array_equal(result.keys, expected_keys)
    assert result.distances[0] == expected_distance_to_key1
    assert result.distances[1] == expected_distance_to_key2
