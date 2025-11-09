"""Tests for simprint package initialization."""


def test_simprint_package_imports():
    # type: () -> None
    """Test that simprint package exports LmdbSimprintIndex."""
    from iscc_search.indexes.simprint import LmdbSimprintIndex

    assert LmdbSimprintIndex is not None
    assert LmdbSimprintIndex.__name__ == "LmdbSimprintIndex"
