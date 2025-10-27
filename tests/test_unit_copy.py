"""Tests for UnitIndex.copy() method."""

from iscc_search.unit import UnitIndex


def test_copy_preserves_metadata(large_dataset):
    """Verify unit_type and realm_id are preserved in copy."""
    iscc_ids, iscc_units = large_dataset
    index1 = UnitIndex(max_dim=256, unit_type="META_NONE_V0", realm_id=1)
    index1.add(iscc_ids[:10], iscc_units[:10])

    index2 = index1.copy()

    assert index2.unit_type == "META_NONE_V0"
    assert index2.realm_id == 1


def test_copy_preserves_data(large_dataset):
    """Verify vectors are preserved in copy."""
    iscc_ids, iscc_units = large_dataset
    index1 = UnitIndex(max_dim=256)
    index1.add(iscc_ids[:10], iscc_units[:10])

    index2 = index1.copy()

    # Verify same number of vectors
    assert len(index2) == len(index1)

    # Verify vectors match
    for i, iscc_id in enumerate(iscc_ids[:10]):
        retrieved = index2.get(iscc_id)
        assert retrieved == iscc_units[i]


def test_copy_preserves_configuration(large_dataset):
    """Verify index configuration is preserved in copy."""
    iscc_ids, iscc_units = large_dataset
    index1 = UnitIndex(
        max_dim=256,
        connectivity=32,
        expansion_add=100,
        expansion_search=50,
    )
    index1.add(iscc_ids[:10], iscc_units[:10])

    index2 = index1.copy()

    assert index2.max_dim == 256
    assert index2.connectivity == 32
    assert index2.expansion_add == 100
    assert index2.expansion_search == 50


def test_copy_is_independent(large_dataset):
    """Verify copied index is independent of original."""
    iscc_ids, iscc_units = large_dataset
    index1 = UnitIndex(max_dim=256)
    index1.add(iscc_ids[:5], iscc_units[:5])

    index2 = index1.copy()

    # Add more to original
    index1.add(iscc_ids[5:10], iscc_units[5:10])

    # Copy should not be affected
    assert len(index1) == 10
    assert len(index2) == 5


def test_copy_search_works(large_dataset):
    """Verify search works on copied index."""
    iscc_ids, iscc_units = large_dataset
    index1 = UnitIndex(max_dim=256)
    index1.add(iscc_ids[:10], iscc_units[:10])

    index2 = index1.copy()

    # Search should work on copy
    results = index2.search(iscc_units[0])
    assert len(results) > 0
    assert results[0].key == iscc_ids[0]


def test_copy_with_none_metadata(large_dataset):
    """Test copy with unit_type=None and realm_id=None."""
    iscc_ids, iscc_units = large_dataset
    index1 = UnitIndex(max_dim=256)
    index1.add(iscc_ids[:10], iscc_units[:10])

    index2 = index1.copy()

    # Metadata gets set from first ISCC-UNIT/ID
    assert index2.unit_type is not None
    assert index2.realm_id is not None
