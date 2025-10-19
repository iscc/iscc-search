"""Tests for UnitIndex.restore() method."""

import os
import tempfile

from iscc_vdb.unit import UnitIndex


def test_restore_from_file(large_dataset):
    """Verify index can be restored from saved file."""
    iscc_ids, iscc_units = large_dataset
    # Create and save index
    index1 = UnitIndex(max_dim=256, unit_type="META-NONE-V0", realm_id=1)
    index1.add(iscc_ids[:10], iscc_units[:10])

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test.usearch")
        index1.save(path)

        # Restore from file
        index2 = UnitIndex.restore(path)

        assert index2 is not None
        assert len(index2) == 10


def test_restore_with_metadata(large_dataset):
    """Verify metadata is restored via restore()."""
    iscc_ids, iscc_units = large_dataset
    # Create and save index
    index1 = UnitIndex(max_dim=256, unit_type="META-NONE-V0", realm_id=1)
    index1.add(iscc_ids[:10], iscc_units[:10])

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test.usearch")
        index1.save(path)

        # Restore from file
        index2 = UnitIndex.restore(path)

        assert index2.unit_type == "META-NONE-V0"
        assert index2.realm_id == 1


def test_restore_without_metadata(large_dataset):
    """Handle missing .meta file gracefully."""
    iscc_ids, iscc_units = large_dataset
    # Create and save index
    index1 = UnitIndex(max_dim=256)
    index1.add(iscc_ids[:10], iscc_units[:10])

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test.usearch")
        index1.save(path)

        # Delete metadata file
        meta_path = path + ".meta"
        os.remove(meta_path)

        # Restore should work without metadata
        index2 = UnitIndex.restore(path)

        assert index2 is not None
        assert len(index2) == 10
        # Metadata will be None since file is missing
        assert index2.unit_type is None
        assert index2.realm_id is None


def test_restore_invalid_file():
    """Handle invalid file gracefully."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "nonexistent.usearch")

        # Restore should return None for missing file
        index = UnitIndex.restore(path)

        assert index is None


def test_restore_with_view(large_dataset):
    """Verify restore() with view=True works."""
    iscc_ids, iscc_units = large_dataset
    # Create and save index
    index1 = UnitIndex(max_dim=256, unit_type="META-NONE-V0", realm_id=1)
    index1.add(iscc_ids[:10], iscc_units[:10])

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test.usearch")
        index1.save(path)

        # Restore with view=True
        index2 = UnitIndex.restore(path, view=True)

        try:
            assert index2 is not None
            assert index2.unit_type == "META-NONE-V0"
            assert index2.realm_id == 1
            assert len(index2) == 10
        finally:
            if index2:
                index2.reset()


def test_restore_preserves_vectors(large_dataset):
    """Verify vectors are correctly restored."""
    iscc_ids, iscc_units = large_dataset
    # Create and save index
    index1 = UnitIndex(max_dim=256)
    index1.add(iscc_ids[:10], iscc_units[:10])

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test.usearch")
        index1.save(path)

        # Restore from file
        index2 = UnitIndex.restore(path)

        # Verify vectors match
        for i, iscc_id in enumerate(iscc_ids[:10]):
            retrieved = index2.get(iscc_id)
            assert retrieved == iscc_units[i]


def test_restore_search_works(large_dataset):
    """Verify search works on restored index."""
    iscc_ids, iscc_units = large_dataset
    # Create and save index
    index1 = UnitIndex(max_dim=256)
    index1.add(iscc_ids[:10], iscc_units[:10])

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test.usearch")
        index1.save(path)

        # Restore from file
        index2 = UnitIndex.restore(path)

        # Search should work
        results = index2.search(iscc_units[0])
        assert len(results) > 0
        assert results[0].key == iscc_ids[0]


def test_restore_with_kwargs(large_dataset):
    """Verify restore() accepts additional constructor arguments."""
    iscc_ids, iscc_units = large_dataset
    # Create and save index
    index1 = UnitIndex(max_dim=256, connectivity=16)
    index1.add(iscc_ids[:10], iscc_units[:10])

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test.usearch")
        index1.save(path)

        # Restore with custom connectivity (should be ignored in favor of saved value)
        index2 = UnitIndex.restore(path, connectivity=32)

        assert index2 is not None
        # Connectivity comes from saved index
        assert index2.connectivity == 16
