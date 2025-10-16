"""Test UnitIndex.search() method."""

import iscc_core as ic
import numpy as np
import pytest
from numpy.testing import assert_array_equal

from iscc_vdb.unit import UnitIndex


def test_search_single_iscc_unit_returns_matches(sample_iscc_ids, sample_meta_units):
    """Search with single ISCC-UNIT returns Matches object with ISCC-ID keys."""
    idx = UnitIndex()

    # Add units (use first 4 IDs for 4 units)
    idx.add(sample_iscc_ids[:4], sample_meta_units)

    # Search with single ISCC-UNIT
    result = idx.search(sample_meta_units[0], count=2)

    # Should return Matches (not BatchMatches)
    assert hasattr(result, "keys")
    assert hasattr(result, "distances")
    assert result.keys.ndim == 1

    # Keys should be ISCC-IDs (strings)
    assert isinstance(result.keys[0], str)
    assert result.keys[0].startswith("ISCC:")


def test_search_multiple_iscc_units_returns_batch_matches(sample_iscc_ids, sample_meta_units):
    """Search with multiple ISCC-UNITs returns BatchMatches object with ISCC-ID keys."""
    idx = UnitIndex()

    # Add units
    idx.add(sample_iscc_ids[:4], sample_meta_units)

    # Search with multiple ISCC-UNITs
    result = idx.search(sample_meta_units[:2], count=2)

    # Should return BatchMatches
    assert hasattr(result, "keys")
    assert hasattr(result, "distances")
    assert hasattr(result, "counts")
    assert result.keys.ndim == 2

    # Keys should be ISCC-IDs (strings)
    assert isinstance(result.keys[0][0], str)
    assert result.keys[0][0].startswith("ISCC:")


def test_search_exact_match_returns_distance_zero(sample_iscc_ids, sample_meta_units):
    """Search for exact match returns distance 0 and correct ISCC-ID."""
    idx = UnitIndex()

    # Add units
    idx.add(sample_iscc_ids[:4], sample_meta_units)

    # Search for exact match
    result = idx.search(sample_meta_units[0], count=1)

    # Should find exact match with distance 0
    assert result.distances[0] == 0.0
    assert result.keys[0] == sample_iscc_ids[0]


def test_search_results_ordered_by_distance(similar_units):
    """Search results are ordered by increasing distance."""
    base_unit, similar_unit, dissimilar_unit = similar_units

    idx = UnitIndex()

    # Generate ISCC-IDs for the units
    iscc_ids = [ic.gen_iscc_id(timestamp=1000000 + i, hub_id=i, realm_id=0)["iscc"] for i in range(3)]

    # Add units in mixed order
    idx.add(iscc_ids[0], dissimilar_unit)
    idx.add(iscc_ids[1], base_unit)
    idx.add(iscc_ids[2], similar_unit)

    # Search for base unit
    result = idx.search(base_unit, count=3)

    # Results should be ordered: exact match (0), similar (1), dissimilar (many)
    assert result.distances[0] < result.distances[1] < result.distances[2]
    assert result.distances[0] == 0.0
    assert result.keys[0] == iscc_ids[1]


def test_search_with_count_limits_results(sample_iscc_ids):
    """count parameter limits number of results returned."""
    idx = UnitIndex()

    # Generate and add 10 units
    units = [f"ISCC:{ic.Code.rnd(ic.MT.META, bits=128)}" for _ in range(10)]
    idx.add(sample_iscc_ids, units)

    # Search with count=3
    result = idx.search(units[0], count=3)

    assert len(result.keys) == 3
    assert len(result.distances) == 3


def test_search_empty_index_returns_empty_matches():
    """Search in empty index returns Matches with empty arrays."""
    idx = UnitIndex()

    # Generate a test unit
    test_unit = f"ISCC:{ic.Code.rnd(ic.MT.META, bits=64)}"

    result = idx.search(test_unit, count=10)

    assert len(result.keys) == 0
    assert len(result.distances) == 0


def test_search_batch_counts_show_results_per_query(sample_iscc_ids, sample_meta_units):
    """BatchMatches.counts shows actual results per query."""
    idx = UnitIndex()

    # Add 4 units
    idx.add(sample_iscc_ids[:4], sample_meta_units)

    # Search with 2 queries, count=10 (more than available)
    result = idx.search(sample_meta_units[:2], count=10)

    # Should return BatchMatches
    assert result.counts[0] == 4  # First query found all 4 results
    assert result.counts[1] == 4  # Second query found all 4 results


def test_search_returns_iscc_ids_matching_added_keys(sample_iscc_ids, sample_meta_units):
    """Search returns ISCC-IDs that match the keys used in add()."""
    idx = UnitIndex()

    # Add units with specific ISCC-IDs
    idx.add(sample_iscc_ids[:4], sample_meta_units)

    # Search for each unit
    for i in range(4):
        result = idx.search(sample_meta_units[i], count=1)
        # Best match should be the same ISCC-ID we added
        assert result.keys[0] == sample_iscc_ids[i]
        assert result.distances[0] == 0.0


def test_search_single_provides_statistics(sample_iscc_ids, sample_meta_units):
    """Matches object provides search statistics."""
    idx = UnitIndex()
    idx.add(sample_iscc_ids[:4], sample_meta_units)

    result = idx.search(sample_meta_units[0], count=2)

    assert isinstance(result.visited_members, int)
    assert isinstance(result.computed_distances, int)
    assert result.visited_members >= 0
    assert result.computed_distances >= 0


def test_search_batch_provides_statistics(sample_iscc_ids, sample_meta_units):
    """BatchMatches object provides search statistics."""
    idx = UnitIndex()
    idx.add(sample_iscc_ids[:4], sample_meta_units)

    result = idx.search(sample_meta_units[:2], count=2)

    assert isinstance(result.visited_members, int)
    assert isinstance(result.computed_distances, int)
    assert result.visited_members >= 0
    assert result.computed_distances >= 0


def test_search_with_count_one_returns_single_match(sample_iscc_ids, sample_meta_units):
    """Search with count=1 returns single best match."""
    idx = UnitIndex()
    idx.add(sample_iscc_ids[:4], sample_meta_units)

    result = idx.search(sample_meta_units[0], count=1)

    assert len(result.keys) == 1
    assert result.keys[0] == sample_iscc_ids[0]
    assert result.distances[0] == 0.0


def test_search_with_exact_parameter(sample_iscc_ids, sample_meta_units):
    """Search with exact=True uses exhaustive search."""
    idx = UnitIndex()
    idx.add(sample_iscc_ids[:4], sample_meta_units)

    result_approx = idx.search(sample_meta_units[0], count=2, exact=False)
    result_exact = idx.search(sample_meta_units[0], count=2, exact=True)

    # For small datasets, results should be identical
    assert_array_equal(result_approx.keys, result_exact.keys)
    assert_array_equal(result_approx.distances, result_exact.distances)


def test_search_batch_each_query_has_own_results(sample_iscc_ids, sample_meta_units):
    """Each query in batch has independent results."""
    idx = UnitIndex()

    # Generate more unique units for this test
    units = [f"ISCC:{ic.Code.rnd(ic.MT.META, bits=128)}" for _ in range(4)]

    # Add different units
    idx.add(sample_iscc_ids[:4], units)

    # Search for first two units
    result = idx.search(units[:2], count=1)

    # Each query should find its exact match with distance 0
    assert result.keys[0][0] == sample_iscc_ids[0]
    assert result.distances[0][0] == 0.0
    assert result.keys[1][0] == sample_iscc_ids[1]
    assert result.distances[1][0] == 0.0


def test_search_with_threads_parameter(sample_iscc_ids, sample_meta_units):
    """Search with threads parameter works correctly."""
    idx = UnitIndex()
    idx.add(sample_iscc_ids[:4], sample_meta_units)

    result = idx.search(sample_meta_units[0], count=2, threads=2)

    # Should return valid results
    assert len(result.keys) == 2
    assert result.keys[0] == sample_iscc_ids[0]
    assert result.distances[0] == 0.0


def test_search_count_exceeds_index_size(sample_iscc_ids, sample_meta_units):
    """Requesting more results than index size returns all vectors."""
    idx = UnitIndex()
    idx.add(sample_iscc_ids[:2], sample_meta_units[:2])

    result = idx.search(sample_meta_units[0], count=100)

    # Should return only 2 results (all vectors in index)
    assert len(result.keys) == 2


# Tests for UnitMatches wrapper class methods


def test_unit_matches_to_list_works_with_string_keys(sample_iscc_ids, sample_meta_units):
    """UnitMatches.to_list() works with ISCC-ID string keys."""
    idx = UnitIndex()
    idx.add(sample_iscc_ids[:4], sample_meta_units)

    result = idx.search(sample_meta_units[0], count=2)

    # Should return list of tuples with string keys
    result_list = result.to_list()
    assert isinstance(result_list, list)
    assert len(result_list) == 2
    assert isinstance(result_list[0], tuple)
    assert len(result_list[0]) == 2
    assert isinstance(result_list[0][0], str)
    assert result_list[0][0].startswith("ISCC:")
    assert isinstance(result_list[0][1], float)


def test_unit_matches_getitem_returns_unit_match(sample_iscc_ids, sample_meta_units):
    """UnitMatches[i] returns UnitMatch with string key."""
    idx = UnitIndex()
    idx.add(sample_iscc_ids[:4], sample_meta_units)

    result = idx.search(sample_meta_units[0], count=2)

    # Access individual match
    match = result[0]
    assert hasattr(match, "key")
    assert hasattr(match, "distance")
    assert isinstance(match.key, str)
    assert match.key.startswith("ISCC:")
    assert isinstance(match.distance, float)


def test_unit_match_to_tuple_works(sample_iscc_ids, sample_meta_units):
    """UnitMatch.to_tuple() returns (key, distance) tuple."""
    idx = UnitIndex()
    idx.add(sample_iscc_ids[:4], sample_meta_units)

    result = idx.search(sample_meta_units[0], count=1)
    match = result[0]

    # Convert to tuple
    match_tuple = match.to_tuple()
    assert isinstance(match_tuple, tuple)
    assert len(match_tuple) == 2
    assert match_tuple[0] == match.key
    assert match_tuple[1] == match.distance


def test_unit_matches_len_returns_match_count(sample_iscc_ids, sample_meta_units):
    """len(UnitMatches) returns number of matches."""
    idx = UnitIndex()
    idx.add(sample_iscc_ids[:4], sample_meta_units)

    result = idx.search(sample_meta_units[0], count=3)

    assert len(result) == 3


def test_unit_matches_repr(sample_iscc_ids, sample_meta_units):
    """UnitMatches.__repr__() returns informative string."""
    idx = UnitIndex()
    idx.add(sample_iscc_ids[:4], sample_meta_units)

    result = idx.search(sample_meta_units[0], count=2)

    repr_str = repr(result)
    assert "UnitMatches" in repr_str
    assert "2" in repr_str


# Tests for UnitBatchMatches wrapper class methods


def test_unit_batch_matches_to_list_works_with_string_keys(sample_iscc_ids, sample_meta_units):
    """UnitBatchMatches.to_list() works with ISCC-ID string keys."""
    idx = UnitIndex()
    idx.add(sample_iscc_ids[:4], sample_meta_units)

    result = idx.search(sample_meta_units[:2], count=2)

    # Should return list of lists of tuples with string keys
    result_list = result.to_list()
    assert isinstance(result_list, list)
    assert len(result_list) == 4  # 2 queries × 2 matches each
    assert isinstance(result_list[0], tuple)
    assert len(result_list[0]) == 2
    assert isinstance(result_list[0][0], str)
    assert result_list[0][0].startswith("ISCC:")
    assert isinstance(result_list[0][1], float)


def test_unit_batch_matches_getitem_returns_unit_matches(sample_iscc_ids, sample_meta_units):
    """UnitBatchMatches[i] returns UnitMatches object."""
    idx = UnitIndex()
    idx.add(sample_iscc_ids[:4], sample_meta_units)

    result = idx.search(sample_meta_units[:2], count=2)

    # Access individual query results
    first_query = result[0]
    assert hasattr(first_query, "keys")
    assert hasattr(first_query, "distances")
    assert len(first_query) == 2
    assert isinstance(first_query.keys[0], str)
    assert first_query.keys[0].startswith("ISCC:")


def test_unit_batch_matches_len_returns_query_count(sample_iscc_ids, sample_meta_units):
    """len(UnitBatchMatches) returns number of queries."""
    idx = UnitIndex()
    idx.add(sample_iscc_ids[:4], sample_meta_units)

    result = idx.search(sample_meta_units[:3], count=2)

    assert len(result) == 3


def test_unit_batch_matches_repr(sample_iscc_ids, sample_meta_units):
    """UnitBatchMatches.__repr__() returns informative string."""
    idx = UnitIndex()
    idx.add(sample_iscc_ids[:4], sample_meta_units)

    result = idx.search(sample_meta_units[:2], count=2)

    repr_str = repr(result)
    assert "UnitBatchMatches" in repr_str
    assert "2" in repr_str  # number of queries


def test_unit_batch_matches_mean_recall_works(sample_iscc_ids, sample_meta_units):
    """UnitBatchMatches.mean_recall() works with ISCC-ID strings."""
    idx = UnitIndex()
    idx.add(sample_iscc_ids[:4], sample_meta_units)

    # Search for first two units
    result = idx.search(sample_meta_units[:2], count=1)

    # Expected keys are the ISCC-IDs we added
    expected = np.array([sample_iscc_ids[0], sample_iscc_ids[1]])

    # Should find both exact matches
    recall = result.mean_recall(expected, count=1)
    assert recall == 1.0


def test_unit_batch_matches_count_matches_works(sample_iscc_ids, sample_meta_units):
    """UnitBatchMatches.count_matches() works with ISCC-ID strings."""
    idx = UnitIndex()
    idx.add(sample_iscc_ids[:4], sample_meta_units)

    # Search for first two units
    result = idx.search(sample_meta_units[:2], count=1)

    # Expected keys are the ISCC-IDs we added
    expected = np.array([sample_iscc_ids[0], sample_iscc_ids[1]])

    # Should find both exact matches
    count = result.count_matches(expected, count=1)
    assert count == 2


def test_unit_batch_matches_getitem_chaining(sample_iscc_ids, sample_meta_units):
    """Can chain indexing: batch[i][j] returns UnitMatch."""
    idx = UnitIndex()
    idx.add(sample_iscc_ids[:4], sample_meta_units)

    result = idx.search(sample_meta_units[:2], count=2)

    # Access first match of first query
    match = result[0][0]
    assert hasattr(match, "key")
    assert hasattr(match, "distance")
    assert isinstance(match.key, str)
    assert match.key.startswith("ISCC:")


def test_unit_batch_matches_getitem_index_error(sample_iscc_ids, sample_meta_units):
    """UnitBatchMatches[i] raises IndexError for invalid index."""
    idx = UnitIndex()
    idx.add(sample_iscc_ids[:4], sample_meta_units)

    result = idx.search(sample_meta_units[:2], count=2)

    # Try to access invalid index
    with pytest.raises(IndexError):
        _ = result[10]


def test_unit_batch_matches_count_matches_with_none_count(sample_iscc_ids, sample_meta_units):
    """UnitBatchMatches.count_matches() with count=None uses all results."""
    idx = UnitIndex()
    idx.add(sample_iscc_ids[:4], sample_meta_units)

    # Search for first two units, get all results
    result = idx.search(sample_meta_units[:2], count=4)

    # Expected keys are the ISCC-IDs we added
    expected = np.array([sample_iscc_ids[0], sample_iscc_ids[1]])

    # Should find both exact matches using default count (all results)
    count = result.count_matches(expected, count=None)
    assert count == 2


def test_unit_batch_matches_count_matches_with_count_greater_than_one(sample_iscc_ids, sample_meta_units):
    """UnitBatchMatches.count_matches() works with count > 1."""
    idx = UnitIndex()
    idx.add(sample_iscc_ids[:4], sample_meta_units)

    # Search for first two units, get top 3 results each
    result = idx.search(sample_meta_units[:2], count=3)

    # Expected keys are the ISCC-IDs we added
    expected = np.array([sample_iscc_ids[0], sample_iscc_ids[1]])

    # Should find both exact matches within top 3
    count = result.count_matches(expected, count=3)
    assert count == 2
