"""Test settings management for ISCC-Search."""

from iscc_search.settings import SearchSettings, search_settings, get_index
from iscc_search.indexes.memory import MemoryIndex


def test_search_settings_default_initialization():
    """Test SearchSettings with default values."""
    settings = SearchSettings()
    assert isinstance(settings.index_uri, str)
    assert len(settings.index_uri) > 0


def test_search_settings_custom_index_uri():
    """Test SearchSettings with custom index_uri."""
    custom_uri = "/tmp/custom_iscc_data"
    settings = SearchSettings(index_uri=custom_uri)
    assert settings.index_uri == custom_uri


def test_search_settings_postgres_uri():
    """Test SearchSettings with PostgreSQL connection string."""
    postgres_uri = "postgresql://user:pass@localhost/isccdb"
    settings = SearchSettings(index_uri=postgres_uri)
    assert settings.index_uri == postgres_uri
    assert settings.index_uri.startswith("postgresql://")


def test_search_settings_memory_uri():
    """Test SearchSettings with memory:// URI."""
    memory_uri = "memory://"
    settings = SearchSettings(index_uri=memory_uri)
    assert settings.index_uri == memory_uri


def test_search_settings_override_with_dict():
    """Test override method with dict parameter."""
    settings = SearchSettings()
    original_uri = settings.index_uri

    new_uri = "/tmp/new_data_dir"
    updated = settings.override({"index_uri": new_uri})

    # Original should be unchanged
    assert settings.index_uri == original_uri
    # Updated should have new value
    assert updated.index_uri == new_uri
    # Should be different instances
    assert settings is not updated


def test_search_settings_override_with_none():
    """Test override method with None parameter."""
    settings = SearchSettings()
    original_uri = settings.index_uri

    updated = settings.override(None)

    # Should return a deep copy with same values
    assert updated.index_uri == original_uri
    # Should be different instances
    assert settings is not updated


def test_search_settings_override_without_args():
    """Test override method without arguments."""
    settings = SearchSettings()
    original_uri = settings.index_uri

    updated = settings.override()

    # Should return a deep copy with same values
    assert updated.index_uri == original_uri
    # Should be different instances
    assert settings is not updated


def test_search_settings_override_multiple_updates():
    """Test that override handles field updates correctly."""
    settings = SearchSettings()

    # Valid URI should work
    new_uri = "/tmp/valid_path"
    updated = settings.override({"index_uri": new_uri})
    assert updated.index_uri == new_uri
    assert isinstance(updated, SearchSettings)


def test_module_level_search_settings():
    """Test that module-level search_settings is initialized."""
    assert isinstance(search_settings, SearchSettings)
    assert isinstance(search_settings.index_uri, str)


def test_search_settings_extra_fields_ignored():
    """Test that extra fields are ignored per model_config."""
    # Should not raise error due to extra='ignore'
    settings = SearchSettings(unknown_field="value")
    assert not hasattr(settings, "unknown_field")


def test_get_index_default(tmp_path):
    """Test get_index() factory function with default settings using lmdb:// scheme."""
    import iscc_search.settings
    from iscc_search.indexes.lmdb import LmdbIndexManager

    original_uri = iscc_search.settings.search_settings.index_uri
    try:
        # Override to tmp_path with lmdb:// scheme
        iscc_search.settings.search_settings.index_uri = f"lmdb://{tmp_path}"
        index = get_index()
        assert isinstance(index, LmdbIndexManager)
        index.close()
    finally:
        # Restore original
        iscc_search.settings.search_settings.index_uri = original_uri


def test_get_index_memory_uri():
    """Test get_index() factory with memory:// URI."""
    import iscc_search.settings

    original_uri = iscc_search.settings.search_settings.index_uri
    try:
        # Override settings temporarily
        iscc_search.settings.search_settings.index_uri = "memory://"
        index = get_index()
        assert isinstance(index, MemoryIndex)
    finally:
        # Restore original
        iscc_search.settings.search_settings.index_uri = original_uri


def test_get_index_custom_path(tmp_path):
    """Test get_index() factory rejects plain paths without URI scheme."""
    import iscc_search.settings
    import pytest

    original_uri = iscc_search.settings.search_settings.index_uri
    try:
        # Plain paths without scheme should raise ValueError
        custom_path = str(tmp_path / "custom_indexes")
        iscc_search.settings.search_settings.index_uri = custom_path
        with pytest.raises(ValueError, match="requires explicit scheme"):
            get_index()
    finally:
        # Restore original
        iscc_search.settings.search_settings.index_uri = original_uri


def test_get_index_unsupported_uri():
    """Test get_index() factory with unsupported URI scheme."""
    import iscc_search.settings
    import pytest

    original_uri = iscc_search.settings.search_settings.index_uri
    try:
        # PostgreSQL URI is not yet supported
        iscc_search.settings.search_settings.index_uri = "postgresql://user:pass@localhost/isccdb"
        with pytest.raises(ValueError, match="Unsupported ISCC_SEARCH_INDEX_URI scheme"):
            get_index()
    finally:
        # Restore original
        iscc_search.settings.search_settings.index_uri = original_uri


def test_get_index_lmdb_uri(tmp_path):
    """Test get_index() factory with lmdb:// URI scheme."""
    import iscc_search.settings
    from iscc_search.indexes.lmdb import LmdbIndexManager

    original_uri = iscc_search.settings.search_settings.index_uri
    try:
        # lmdb:// URI should work
        lmdb_uri = f"lmdb://{tmp_path / 'lmdb_test'}"
        iscc_search.settings.search_settings.index_uri = lmdb_uri
        index = get_index()
        assert isinstance(index, LmdbIndexManager)
        index.close()
    finally:
        # Restore original
        iscc_search.settings.search_settings.index_uri = original_uri


def test_get_index_usearch_uri():
    """Test get_index() factory with usearch:// URI scheme returns UsearchIndexManager."""
    import iscc_search.settings
    from iscc_search.indexes.usearch import UsearchIndexManager

    original_uri = iscc_search.settings.search_settings.index_uri
    try:
        # usearch:// URI should return UsearchIndexManager
        iscc_search.settings.search_settings.index_uri = "usearch:///tmp/usearch_test"
        index = get_index()
        assert isinstance(index, UsearchIndexManager)
        index.close()
    finally:
        # Restore original
        iscc_search.settings.search_settings.index_uri = original_uri


def test_cors_origins_default():
    """Test cors_origins field has correct default value."""
    settings = SearchSettings()
    assert settings.cors_origins == ["*"]
    assert isinstance(settings.cors_origins, list)


def test_cors_origins_custom_list():
    """Test cors_origins field with custom list."""
    custom_origins = ["https://example.com", "https://app.example.com"]
    settings = SearchSettings(cors_origins=custom_origins)
    assert settings.cors_origins == custom_origins


def test_cors_origins_comma_separated_string():
    """Test cors_origins field parses comma-separated string."""
    cors_string = "https://example.com,https://app.example.com,http://localhost:3000"
    settings = SearchSettings(cors_origins=cors_string)
    expected = ["https://example.com", "https://app.example.com", "http://localhost:3000"]
    assert settings.cors_origins == expected


def test_cors_origins_comma_separated_with_spaces():
    """Test cors_origins field handles spaces in comma-separated string."""
    cors_string = "https://example.com, https://app.example.com,  http://localhost:3000"
    settings = SearchSettings(cors_origins=cors_string)
    expected = ["https://example.com", "https://app.example.com", "http://localhost:3000"]
    assert settings.cors_origins == expected


def test_cors_origins_single_wildcard_string():
    """Test cors_origins field handles single wildcard as string."""
    settings = SearchSettings(cors_origins="*")
    assert settings.cors_origins == ["*"]
