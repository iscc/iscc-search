"""Tests for simprint package initialization."""


def test_simprint_package_imports():
    # type: () -> None
    """Test that simprint package exports LmdbSimprintIndex64."""
    from iscc_search.indexes.simprint import LmdbSimprintIndex64

    assert LmdbSimprintIndex64 is not None
    assert LmdbSimprintIndex64.__name__ == "LmdbSimprintIndex64"
