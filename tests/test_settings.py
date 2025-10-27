"""Test settings management for ISCC-VDB."""

from iscc_search.settings import VdbSettings, vdb_settings, get_index
from iscc_search.indexes.memory import MemoryIndex


def test_vdb_settings_default_initialization():
    """Test VdbSettings with default values."""
    settings = VdbSettings()
    assert isinstance(settings.indexes_uri, str)
    assert len(settings.indexes_uri) > 0


def test_vdb_settings_custom_indexes_uri():
    """Test VdbSettings with custom indexes_uri."""
    custom_uri = "/tmp/custom_iscc_data"
    settings = VdbSettings(indexes_uri=custom_uri)
    assert settings.indexes_uri == custom_uri


def test_vdb_settings_postgres_uri():
    """Test VdbSettings with PostgreSQL connection string."""
    postgres_uri = "postgresql://user:pass@localhost/isccdb"
    settings = VdbSettings(indexes_uri=postgres_uri)
    assert settings.indexes_uri == postgres_uri
    assert settings.indexes_uri.startswith("postgresql://")


def test_vdb_settings_memory_uri():
    """Test VdbSettings with memory:// URI."""
    memory_uri = "memory://"
    settings = VdbSettings(indexes_uri=memory_uri)
    assert settings.indexes_uri == memory_uri


def test_vdb_settings_override_with_dict():
    """Test override method with dict parameter."""
    settings = VdbSettings()
    original_uri = settings.indexes_uri

    new_uri = "/tmp/new_data_dir"
    updated = settings.override({"indexes_uri": new_uri})

    # Original should be unchanged
    assert settings.indexes_uri == original_uri
    # Updated should have new value
    assert updated.indexes_uri == new_uri
    # Should be different instances
    assert settings is not updated


def test_vdb_settings_override_with_none():
    """Test override method with None parameter."""
    settings = VdbSettings()
    original_uri = settings.indexes_uri

    updated = settings.override(None)

    # Should return a deep copy with same values
    assert updated.indexes_uri == original_uri
    # Should be different instances
    assert settings is not updated


def test_vdb_settings_override_without_args():
    """Test override method without arguments."""
    settings = VdbSettings()
    original_uri = settings.indexes_uri

    updated = settings.override()

    # Should return a deep copy with same values
    assert updated.indexes_uri == original_uri
    # Should be different instances
    assert settings is not updated


def test_vdb_settings_override_multiple_updates():
    """Test that override handles field updates correctly."""
    settings = VdbSettings()

    # Valid URI should work
    new_uri = "/tmp/valid_path"
    updated = settings.override({"indexes_uri": new_uri})
    assert updated.indexes_uri == new_uri
    assert isinstance(updated, VdbSettings)


def test_module_level_vdb_settings():
    """Test that module-level vdb_settings is initialized."""
    assert isinstance(vdb_settings, VdbSettings)
    assert isinstance(vdb_settings.indexes_uri, str)


def test_vdb_settings_extra_fields_ignored():
    """Test that extra fields are ignored per model_config."""
    # Should not raise error due to extra='ignore'
    settings = VdbSettings(unknown_field="value")
    assert not hasattr(settings, "unknown_field")


def test_get_index_default(tmp_path):
    """Test get_index() factory function with default settings."""
    import iscc_search.settings
    from iscc_search.indexes.lmdb import LmdbIndexManager

    original_uri = iscc_search.settings.vdb_settings.indexes_uri
    try:
        # Override to tmp_path to avoid touching real user data directory
        iscc_search.settings.vdb_settings.indexes_uri = str(tmp_path)
        # Default URI from platformdirs is a file path, now supported via LMDB
        index = get_index()
        assert isinstance(index, LmdbIndexManager)
        index.close()
    finally:
        # Restore original
        iscc_search.settings.vdb_settings.indexes_uri = original_uri


def test_get_index_memory_uri():
    """Test get_index() factory with memory:// URI."""
    import iscc_search.settings

    original_uri = iscc_search.settings.vdb_settings.indexes_uri
    try:
        # Override settings temporarily
        iscc_search.settings.vdb_settings.indexes_uri = "memory://"
        index = get_index()
        assert isinstance(index, MemoryIndex)
    finally:
        # Restore original
        iscc_search.settings.vdb_settings.indexes_uri = original_uri


def test_get_index_custom_path(tmp_path):
    """Test get_index() factory with custom file path (now implemented via LMDB)."""
    import iscc_search.settings
    from iscc_search.indexes.lmdb import LmdbIndexManager

    original_uri = iscc_search.settings.vdb_settings.indexes_uri
    try:
        # Override settings temporarily
        custom_path = str(tmp_path / "custom_indexes")
        iscc_search.settings.vdb_settings.indexes_uri = custom_path
        # File paths are now supported via LmdbIndexManager
        index = get_index()
        assert isinstance(index, LmdbIndexManager)
        index.close()
    finally:
        # Restore original
        iscc_search.settings.vdb_settings.indexes_uri = original_uri


def test_get_index_unsupported_uri():
    """Test get_index() factory with unsupported URI scheme."""
    import iscc_search.settings
    import pytest

    original_uri = iscc_search.settings.vdb_settings.indexes_uri
    try:
        # PostgreSQL URI is not yet supported
        iscc_search.settings.vdb_settings.indexes_uri = "postgresql://user:pass@localhost/isccdb"
        with pytest.raises(ValueError, match="Unsupported ISCC_VDB_INDEXES_URI"):
            get_index()
    finally:
        # Restore original
        iscc_search.settings.vdb_settings.indexes_uri = original_uri


def test_get_index_file_uri(tmp_path):
    """Test get_index() factory with file:// URI scheme."""
    import iscc_search.settings
    from iscc_search.indexes.lmdb import LmdbIndexManager

    original_uri = iscc_search.settings.vdb_settings.indexes_uri
    try:
        # file:// URI should work
        file_uri = f"file://{tmp_path / 'file_uri_test'}"
        iscc_search.settings.vdb_settings.indexes_uri = file_uri
        index = get_index()
        assert isinstance(index, LmdbIndexManager)
        index.close()
    finally:
        # Restore original
        iscc_search.settings.vdb_settings.indexes_uri = original_uri
