"""Tests for NphdIndex.load() method."""

import numpy as np

from iscc_search.nphd import NphdIndex


def test_load_from_file(tmp_path):
    """Load index from file path."""
    index = NphdIndex(max_dim=128)
    vector = np.random.randint(0, 256, 16, dtype=np.uint8)
    index.add(42, vector)

    file_path = tmp_path / "index.usearch"
    index.save(str(file_path))

    loaded = NphdIndex(max_dim=128)
    loaded.load(str(file_path))

    assert 42 in loaded
    np.testing.assert_array_equal(loaded.get(42), vector)


def test_load_from_buffer():
    """Load index from buffer."""
    index = NphdIndex(max_dim=128)
    vector = np.random.randint(0, 256, 16, dtype=np.uint8)
    index.add(42, vector)

    buffer = index.save()

    loaded = NphdIndex(max_dim=128)
    loaded.load(buffer)

    assert 42 in loaded
    np.testing.assert_array_equal(loaded.get(42), vector)


def test_load_restores_max_dim(tmp_path):
    """Load correctly restores max_dim from saved ndim."""
    index = NphdIndex(max_dim=192)
    vector = np.random.randint(0, 256, 24, dtype=np.uint8)
    index.add(1, vector)

    file_path = tmp_path / "index.usearch"
    index.save(str(file_path))

    loaded = NphdIndex(max_dim=128)
    loaded.load(str(file_path))

    assert loaded.max_dim == 192
    assert loaded.max_bytes == 24


def test_load_with_different_max_dims(tmp_path):
    """Load works with different max_dim values."""
    for max_dim in [64, 128, 192, 256]:
        max_bytes = max_dim // 8
        index = NphdIndex(max_dim=max_dim)
        vector = np.random.randint(0, 256, max_bytes, dtype=np.uint8)
        index.add(1, vector)

        file_path = tmp_path / f"index_{max_dim}.usearch"
        index.save(str(file_path))

        loaded = NphdIndex(max_dim=64)
        loaded.load(str(file_path))

        assert loaded.max_dim == max_dim
        assert loaded.max_bytes == max_bytes
        np.testing.assert_array_equal(loaded.get(1), vector)


def test_loaded_index_supports_operations(tmp_path):
    """Loaded index can perform add and search operations."""
    index = NphdIndex(max_dim=128)
    vector1 = np.random.randint(0, 256, 16, dtype=np.uint8)
    index.add(1, vector1)

    file_path = tmp_path / "index.usearch"
    index.save(str(file_path))

    loaded = NphdIndex(max_dim=128)
    loaded.load(str(file_path))

    vector2 = np.random.randint(0, 256, 16, dtype=np.uint8)
    loaded.add(2, vector2)
    assert 2 in loaded

    matches = loaded.search(vector1, count=1)
    assert matches.keys[0] == 1
