"""Tests for IsccIndex class."""

import os
import tempfile

import pytest

from iscc_vdb.instance import InstanceIndex
from iscc_vdb.iscc import IsccIndex


@pytest.fixture
def temp_index():
    """Create temporary IsccIndex for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test_index")
        idx = IsccIndex(path=path)
        yield idx
        if idx.instance_index is not None:
            idx.instance_index.close()


def test_init_creates_directory(temp_index):
    """Constructor creates index directory when path is provided."""
    assert os.path.exists(temp_index.path)
    assert os.path.isdir(temp_index.path)


def test_init_default_parameters(temp_index):
    """Constructor sets default parameters correctly."""
    assert temp_index.realm_id == 0
    assert temp_index.max_dim == 256
    assert temp_index.unit_index_kwargs == {}


def test_init_custom_parameters():
    """Constructor accepts custom parameters."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test_index")
        idx = IsccIndex(path=path, realm_id=1, max_dim=128, connectivity=16)

        assert idx.path == path
        assert idx.realm_id == 1
        assert idx.max_dim == 128
        assert idx.unit_index_kwargs == {"connectivity": 16}

        idx.instance_index.close()


def test_init_creates_instance_index(temp_index):
    """Constructor creates InstanceIndex when path is provided."""
    assert hasattr(temp_index, "instance_index")
    assert isinstance(temp_index.instance_index, InstanceIndex)

    instance_path = os.path.join(temp_index.path, "instance")
    assert os.path.exists(instance_path)


def test_init_creates_empty_unit_indexes_dict(temp_index):
    """Constructor initializes empty unit_indexes dictionary."""
    assert hasattr(temp_index, "unit_indexes")
    assert isinstance(temp_index.unit_indexes, dict)
    assert len(temp_index.unit_indexes) == 0


def test_init_instance_index_has_correct_realm_id(temp_index):
    """InstanceIndex is initialized with correct realm_id when path is provided."""
    assert temp_index.instance_index.realm_id == temp_index.realm_id


def test_init_with_pathlike_object():
    """Constructor accepts PathLike objects."""
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test_index"
        idx = IsccIndex(path=path)

        assert idx.path == os.fspath(path)
        assert os.path.exists(idx.path)

        idx.instance_index.close()


def test_init_without_path():
    """Constructor works without path parameter."""
    idx = IsccIndex()

    assert idx.path is None
    assert idx.realm_id == 0
    assert idx.max_dim == 256
    assert idx.unit_index_kwargs == {}
    assert isinstance(idx.unit_indexes, dict)
    assert len(idx.unit_indexes) == 0
    assert idx.instance_index is None


def test_init_path_none_explicit():
    """Constructor handles explicit path=None."""
    idx = IsccIndex(path=None, realm_id=1, max_dim=128)

    assert idx.path is None
    assert idx.realm_id == 1
    assert idx.max_dim == 128
    assert idx.instance_index is None
