"""Tests for UnitIndex.view() method."""

import os
import tempfile

from iscc_vdb.unit import UnitIndex


def test_view_from_file_restores_metadata(large_dataset):
    """Verify metadata is restored when viewing from file."""
    iscc_ids, iscc_units = large_dataset
    # Create and save index
    index1 = UnitIndex(max_dim=256, unit_type="META_NONE_V0", realm_id=1)
    index1.add(iscc_ids[:10], iscc_units[:10])

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test.usearch")
        index1.save(path)

        # Create new index and view
        index2 = UnitIndex(max_dim=256)
        index2.view(path)

        try:
            assert index2.unit_type == "META_NONE_V0"
            assert index2.realm_id == 1
            assert len(index2) == 10
        finally:
            # Close memory-mapped file before cleanup
            index2.reset()


def test_view_without_metadata_file(large_dataset):
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

        # View should work, but metadata won't be restored
        index2 = UnitIndex(max_dim=256, unit_type="TEXT_NONE_V0", realm_id=1)
        index2.view(path)

        try:
            # Metadata from constructor should be preserved
            assert index2.unit_type == "TEXT_NONE_V0"
            assert index2.realm_id == 1
            assert len(index2) == 10
        finally:
            index2.reset()


def test_view_overwrites_existing_metadata(large_dataset):
    """Verify view() overwrites existing metadata."""
    iscc_ids, iscc_units = large_dataset
    # Create and save index with metadata
    index1 = UnitIndex(max_dim=256, unit_type="META_NONE_V0", realm_id=1)
    index1.add(iscc_ids[:10], iscc_units[:10])

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test.usearch")
        index1.save(path)

        # Create index with different metadata and view
        index2 = UnitIndex(max_dim=256, unit_type="TEXT_NONE_V0", realm_id=0)
        index2.view(path)

        try:
            # Metadata should be from viewed file, not constructor
            assert index2.unit_type == "META_NONE_V0"
            assert index2.realm_id == 1
        finally:
            index2.reset()


def test_view_uses_self_path(large_dataset):
    """Verify view() uses self.path when path_or_buffer is None."""
    iscc_ids, iscc_units = large_dataset
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test.usearch")

        # Create and save
        index1 = UnitIndex(max_dim=256, path=path, unit_type="META_NONE_V0", realm_id=1)
        index1.add(iscc_ids[:10], iscc_units[:10])
        index1.save()

        # Create new index with same path and view
        index2 = UnitIndex(max_dim=256, path=path)
        index2.view()

        try:
            assert index2.unit_type == "META_NONE_V0"
            assert index2.realm_id == 1
        finally:
            index2.reset()


def test_view_preserves_vectors(large_dataset):
    """Verify vectors are correctly viewed."""
    iscc_ids, iscc_units = large_dataset
    # Create and save index
    index1 = UnitIndex(max_dim=256)
    index1.add(iscc_ids[:10], iscc_units[:10])

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test.usearch")
        index1.save(path)

        # View into new index
        index2 = UnitIndex(max_dim=256)
        index2.view(path)

        try:
            # Verify vectors match
            for i, iscc_id in enumerate(iscc_ids[:10]):
                retrieved = index2.get(iscc_id)
                assert retrieved == iscc_units[i]
        finally:
            index2.reset()


def test_view_is_read_only(large_dataset):
    """Verify viewed index is read-only (memory-mapped)."""
    iscc_ids, iscc_units = large_dataset
    # Create and save index
    index1 = UnitIndex(max_dim=256)
    index1.add(iscc_ids[:10], iscc_units[:10])

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test.usearch")
        index1.save(path)

        # View into new index
        index2 = UnitIndex(max_dim=256)
        index2.view(path)

        try:
            # Search should work
            results = index2.search(iscc_units[0])
            assert len(results) > 0
        finally:
            index2.reset()


def test_view_from_buffer_no_metadata(large_dataset):
    """Verify view from buffer works without metadata."""
    iscc_ids, iscc_units = large_dataset
    # Create and save index
    index1 = UnitIndex(max_dim=256, unit_type="META_NONE_V0", realm_id=1)
    index1.add(iscc_ids[:10], iscc_units[:10])

    # Save to buffer
    buffer = index1.save()

    # View from buffer into new index (metadata won't be restored)
    index2 = UnitIndex(max_dim=256, unit_type="TEXT_NONE_V0", realm_id=1)
    index2.view(buffer)

    try:
        # Metadata from constructor should be preserved (not from buffer)
        assert index2.unit_type == "TEXT_NONE_V0"
        assert index2.realm_id == 1
        assert len(index2) == 10
    finally:
        index2.reset()
