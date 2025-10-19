"""Tests for UnitIndex.save() method."""

import json
import os
import tempfile

import numpy as np

from iscc_vdb.unit import UnitIndex


def test_save_to_file_creates_metadata(large_dataset):
    """Verify .meta file is created when saving to file path."""
    iscc_ids, iscc_units = large_dataset
    index = UnitIndex(max_dim=256)
    index.add(iscc_ids[:10], iscc_units[:10])

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test.usearch")
        index.save(path)

        # Verify index file exists
        assert os.path.exists(path)

        # Verify metadata file exists
        meta_path = path + ".meta"
        assert os.path.exists(meta_path)


def test_save_metadata_content(large_dataset):
    """Verify metadata content is correct."""
    iscc_ids, iscc_units = large_dataset
    index = UnitIndex(max_dim=256, unit_type="META-NONE-V0", realm_id=1)
    index.add(iscc_ids[:10], iscc_units[:10])

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test.usearch")
        index.save(path)

        meta_path = path + ".meta"
        with open(meta_path, "r") as f:
            meta = json.load(f)

        assert meta["unit_type"] == "META-NONE-V0"
        assert meta["realm_id"] == 1


def test_save_with_none_values(large_dataset):
    """Test saving with unit_type=None and realm_id=None."""
    iscc_ids, iscc_units = large_dataset
    index = UnitIndex(max_dim=256)
    index.add(iscc_ids[:10], iscc_units[:10])

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test.usearch")
        index.save(path)

        meta_path = path + ".meta"
        with open(meta_path, "r") as f:
            meta = json.load(f)

        # unit_type gets set from first ISCC-UNIT, realm_id from first ISCC-ID or defaults to 0
        assert meta["unit_type"] is not None
        assert meta["realm_id"] is not None


def test_save_to_buffer_no_metadata(large_dataset):
    """Verify no .meta file is created for buffer saves."""
    iscc_ids, iscc_units = large_dataset
    index = UnitIndex(max_dim=256)
    index.add(iscc_ids[:10], iscc_units[:10])

    # Save to buffer
    buffer = index.save()

    # Should return buffer (bytes or bytearray)
    assert isinstance(buffer, (bytes, bytearray))
    assert len(buffer) > 0


def test_save_uses_self_path(large_dataset):
    """Verify save() uses self.path when path_or_buffer is None."""
    iscc_ids, iscc_units = large_dataset
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test.usearch")
        index = UnitIndex(max_dim=256, path=path)
        index.add(iscc_ids[:10], iscc_units[:10])

        # Save without specifying path
        index.save()

        # Verify files exist
        assert os.path.exists(path)
        assert os.path.exists(path + ".meta")


def test_save_returns_none_for_file(large_dataset):
    """Verify save() returns None when saving to file."""
    iscc_ids, iscc_units = large_dataset
    index = UnitIndex(max_dim=256)
    index.add(iscc_ids[:10], iscc_units[:10])

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test.usearch")
        result = index.save(path)

        assert result is None
