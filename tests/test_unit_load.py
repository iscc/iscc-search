"""Tests for UnitIndex.load() method."""

import json
import os
import tempfile

from iscc_vdb.unit import UnitIndex


def test_load_from_file_restores_metadata(large_dataset):
    """Verify metadata is restored when loading from file."""
    iscc_ids, iscc_units = large_dataset
    # Create and save index
    index1 = UnitIndex(max_dim=256, unit_type="META-NONE-V0", realm_id=1)
    index1.add(iscc_ids[:10], iscc_units[:10])

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test.usearch")
        index1.save(path)

        # Create new index and load
        index2 = UnitIndex(max_dim=256)
        index2.load(path)

        assert index2.unit_type == "META-NONE-V0"
        assert index2.realm_id == 1
        assert len(index2) == 10


def test_load_without_metadata_file(large_dataset):
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

        # Load should work, but metadata won't be restored
        index2 = UnitIndex(max_dim=256, unit_type="TEXT-NONE-V0", realm_id=1)
        index2.load(path)

        # Metadata from constructor should be preserved
        assert index2.unit_type == "TEXT-NONE-V0"
        assert index2.realm_id == 1
        assert len(index2) == 10


def test_load_overwrites_existing_metadata(large_dataset):
    """Verify load() overwrites existing metadata."""
    iscc_ids, iscc_units = large_dataset
    # Create and save index with metadata
    index1 = UnitIndex(max_dim=256, unit_type="META-NONE-V0", realm_id=1)
    index1.add(iscc_ids[:10], iscc_units[:10])

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test.usearch")
        index1.save(path)

        # Create index with different metadata and load
        index2 = UnitIndex(max_dim=256, unit_type="TEXT-NONE-V0", realm_id=0)
        index2.load(path)

        # Metadata should be from loaded file, not constructor
        assert index2.unit_type == "META-NONE-V0"
        assert index2.realm_id == 1


def test_load_uses_self_path(large_dataset):
    """Verify load() uses self.path when path_or_buffer is None."""
    iscc_ids, iscc_units = large_dataset
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test.usearch")

        # Create and save
        index1 = UnitIndex(max_dim=256, path=path, unit_type="META-NONE-V0", realm_id=1)
        index1.add(iscc_ids[:10], iscc_units[:10])
        index1.save()

        # Create new index with same path and load
        index2 = UnitIndex(max_dim=256, path=path)
        index2.load()

        assert index2.unit_type == "META-NONE-V0"
        assert index2.realm_id == 1


def test_load_preserves_vectors(large_dataset):
    """Verify vectors are correctly loaded."""
    iscc_ids, iscc_units = large_dataset
    # Create and save index
    index1 = UnitIndex(max_dim=256)
    index1.add(iscc_ids[:10], iscc_units[:10])

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test.usearch")
        index1.save(path)

        # Load into new index
        index2 = UnitIndex(max_dim=256)
        index2.load(path)

        # Verify vectors match
        for i, iscc_id in enumerate(iscc_ids[:10]):
            retrieved = index2.get(iscc_id)
            assert retrieved == iscc_units[i]


def test_load_from_buffer_no_metadata(large_dataset):
    """Verify load from buffer works without metadata."""
    iscc_ids, iscc_units = large_dataset
    # Create and save index
    index1 = UnitIndex(max_dim=256, unit_type="META-NONE-V0", realm_id=1)
    index1.add(iscc_ids[:10], iscc_units[:10])

    # Save to buffer
    buffer = index1.save()

    # Load from buffer into new index (metadata won't be restored)
    index2 = UnitIndex(max_dim=256, unit_type="TEXT-NONE-V0", realm_id=1)
    index2.load(buffer)

    # Metadata from constructor should be preserved (not from buffer)
    assert index2.unit_type == "TEXT-NONE-V0"
    assert index2.realm_id == 1
    assert len(index2) == 10
