"""Tests for NphdIndex.view() method."""

import numpy as np

from iscc_search.nphd import NphdIndex


def test_view_from_file(tmp_path):
    """View index from file path."""
    index = NphdIndex(max_dim=128)
    vector = np.random.randint(0, 256, 16, dtype=np.uint8)
    index.add(42, vector)

    file_path = tmp_path / "index.usearch"
    index.save(str(file_path))

    viewed = NphdIndex(max_dim=128)
    viewed.view(str(file_path))

    assert 42 in viewed
    np.testing.assert_array_equal(viewed.get(42), vector)


def test_view_from_buffer():
    """View index from buffer."""
    index = NphdIndex(max_dim=128)
    vector = np.random.randint(0, 256, 16, dtype=np.uint8)
    index.add(42, vector)

    buffer = index.save()

    viewed = NphdIndex(max_dim=128)
    viewed.view(buffer)

    assert 42 in viewed
    np.testing.assert_array_equal(viewed.get(42), vector)


def test_view_restores_max_dim(tmp_path):
    """View correctly restores max_dim from saved ndim."""
    index = NphdIndex(max_dim=192)
    vector = np.random.randint(0, 256, 24, dtype=np.uint8)
    index.add(1, vector)

    file_path = tmp_path / "index.usearch"
    index.save(str(file_path))

    viewed = NphdIndex(max_dim=128)
    viewed.view(str(file_path))

    assert viewed.max_dim == 192
    assert viewed.max_bytes == 24


def test_view_with_different_max_dims(tmp_path):
    """View works with different max_dim values."""
    for max_dim in [64, 128, 192, 256]:
        max_bytes = max_dim // 8
        index = NphdIndex(max_dim=max_dim)
        vector = np.random.randint(0, 256, max_bytes, dtype=np.uint8)
        index.add(1, vector)

        file_path = tmp_path / f"index_{max_dim}.usearch"
        index.save(str(file_path))

        viewed = NphdIndex(max_dim=64)
        viewed.view(str(file_path))

        assert viewed.max_dim == max_dim
        assert viewed.max_bytes == max_bytes
        np.testing.assert_array_equal(viewed.get(1), vector)


def test_viewed_index_supports_search(tmp_path):
    """Viewed index can perform search operations."""
    index = NphdIndex(max_dim=128)
    vector = np.random.randint(0, 256, 16, dtype=np.uint8)
    index.add(1, vector)

    file_path = tmp_path / "index.usearch"
    index.save(str(file_path))

    viewed = NphdIndex(max_dim=128)
    viewed.view(str(file_path))

    matches = viewed.search(vector, count=1)
    assert matches.keys[0] == 1
