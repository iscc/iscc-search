"""Test options management for ISCC-Search."""

from iscc_search.options import SearchOptions, search_opts, get_index
from iscc_search.indexes.memory import MemoryIndex


def test_search_options_default_initialization():
    """Test SearchOptions with default values."""
    options = SearchOptions()
    assert isinstance(options.index_uri, str)
    assert len(options.index_uri) > 0


def test_search_options_custom_index_uri():
    """Test SearchOptions with custom index_uri."""
    custom_uri = "/tmp/custom_iscc_data"
    options = SearchOptions(index_uri=custom_uri)
    assert options.index_uri == custom_uri


def test_search_options_memory_uri():
    """Test SearchOptions with memory:// URI."""
    memory_uri = "memory://"
    options = SearchOptions(index_uri=memory_uri)
    assert options.index_uri == memory_uri


def test_search_options_override_with_dict():
    """Test override method with dict parameter."""
    options = SearchOptions()
    original_uri = options.index_uri

    new_uri = "/tmp/new_data_dir"
    updated = options.override({"index_uri": new_uri})

    # Original should be unchanged
    assert options.index_uri == original_uri
    # Updated should have new value
    assert updated.index_uri == new_uri
    # Should be different instances
    assert options is not updated


def test_search_options_override_with_none():
    """Test override method with None parameter."""
    options = SearchOptions()
    original_uri = options.index_uri

    updated = options.override(None)

    # Should return a deep copy with same values
    assert updated.index_uri == original_uri
    # Should be different instances
    assert options is not updated


def test_search_options_override_without_args():
    """Test override method without arguments."""
    options = SearchOptions()
    original_uri = options.index_uri

    updated = options.override()

    # Should return a deep copy with same values
    assert updated.index_uri == original_uri
    # Should be different instances
    assert options is not updated


def test_search_options_override_multiple_updates():
    """Test that override handles field updates correctly."""
    options = SearchOptions()

    # Valid URI should work
    new_uri = "/tmp/valid_path"
    updated = options.override({"index_uri": new_uri})
    assert updated.index_uri == new_uri
    assert isinstance(updated, SearchOptions)


def test_module_level_search_opts():
    """Test that module-level search_opts is initialized."""
    assert isinstance(search_opts, SearchOptions)
    assert isinstance(search_opts.index_uri, str)


def test_search_options_extra_fields_ignored():
    """Test that extra fields are ignored per model_config."""
    # Should not raise error due to extra='ignore'
    options = SearchOptions(unknown_field="value")
    assert not hasattr(options, "unknown_field")


def test_get_index_default(tmp_path):
    """Test get_index() factory function with default options using lmdb:// scheme."""
    import iscc_search.options
    from iscc_search.indexes.lmdb import LmdbIndexManager

    original_uri = iscc_search.options.search_opts.index_uri
    try:
        # Override to tmp_path with lmdb:// scheme
        iscc_search.options.search_opts.index_uri = f"lmdb://{tmp_path}"
        index = get_index()
        assert isinstance(index, LmdbIndexManager)
        index.close()
    finally:
        # Restore original
        iscc_search.options.search_opts.index_uri = original_uri


def test_get_index_memory_uri():
    """Test get_index() factory with memory:// URI."""
    import iscc_search.options

    original_uri = iscc_search.options.search_opts.index_uri
    try:
        # Override options temporarily
        iscc_search.options.search_opts.index_uri = "memory://"
        index = get_index()
        assert isinstance(index, MemoryIndex)
    finally:
        # Restore original
        iscc_search.options.search_opts.index_uri = original_uri


def test_get_index_custom_path(tmp_path):
    """Test get_index() factory rejects plain paths without URI scheme."""
    import iscc_search.options
    import pytest

    original_uri = iscc_search.options.search_opts.index_uri
    try:
        # Plain paths without scheme should raise ValueError
        custom_path = str(tmp_path / "custom_indexes")
        iscc_search.options.search_opts.index_uri = custom_path
        with pytest.raises(ValueError, match="requires explicit scheme"):
            get_index()
    finally:
        # Restore original
        iscc_search.options.search_opts.index_uri = original_uri


def test_get_index_unsupported_uri():
    """Test get_index() factory with unsupported URI scheme."""
    import iscc_search.options
    import pytest

    original_uri = iscc_search.options.search_opts.index_uri
    try:
        iscc_search.options.search_opts.index_uri = "redis://localhost:6379/0"
        with pytest.raises(ValueError, match="Unsupported ISCC_SEARCH_INDEX_URI scheme"):
            get_index()
    finally:
        # Restore original
        iscc_search.options.search_opts.index_uri = original_uri


def test_get_index_lmdb_uri(tmp_path):
    """Test get_index() factory with lmdb:// URI scheme."""
    import iscc_search.options
    from iscc_search.indexes.lmdb import LmdbIndexManager

    original_uri = iscc_search.options.search_opts.index_uri
    try:
        # lmdb:// URI should work
        lmdb_uri = f"lmdb://{tmp_path / 'lmdb_test'}"
        iscc_search.options.search_opts.index_uri = lmdb_uri
        index = get_index()
        assert isinstance(index, LmdbIndexManager)
        index.close()
    finally:
        # Restore original
        iscc_search.options.search_opts.index_uri = original_uri


def test_get_index_usearch_uri():
    """Test get_index() factory with usearch:// URI scheme returns UsearchIndexManager."""
    import iscc_search.options
    from iscc_search.indexes.usearch import UsearchIndexManager

    original_uri = iscc_search.options.search_opts.index_uri
    try:
        # usearch:// URI should return UsearchIndexManager
        iscc_search.options.search_opts.index_uri = "usearch:///tmp/usearch_test"
        index = get_index()
        assert isinstance(index, UsearchIndexManager)
        index.close()
    finally:
        # Restore original
        iscc_search.options.search_opts.index_uri = original_uri


def test_cors_origins_default():
    """Test cors_origins field has correct default value."""
    options = SearchOptions()
    assert options.cors_origins == "*"
    assert options.cors_origins_list == ["*"]


def test_cors_origins_comma_separated_string():
    """Test cors_origins_list parses comma-separated string."""
    cors_string = "https://example.com,https://app.example.com,http://localhost:3000"
    options = SearchOptions(cors_origins=cors_string)
    expected = ["https://example.com", "https://app.example.com", "http://localhost:3000"]
    assert options.cors_origins_list == expected


def test_cors_origins_comma_separated_with_spaces():
    """Test cors_origins_list handles spaces in comma-separated string."""
    cors_string = "https://example.com, https://app.example.com,  http://localhost:3000"
    options = SearchOptions(cors_origins=cors_string)
    expected = ["https://example.com", "https://app.example.com", "http://localhost:3000"]
    assert options.cors_origins_list == expected


def test_cors_origins_single_wildcard_string():
    """Test cors_origins_list handles single wildcard as string."""
    options = SearchOptions(cors_origins="*")
    assert options.cors_origins_list == ["*"]


def test_host_default():
    """Test host field has correct default value."""
    options = SearchOptions()
    assert options.host == "0.0.0.0"


def test_port_default():
    """Test port field has correct default value."""
    options = SearchOptions()
    assert options.port == 8000


def test_workers_default():
    """Test workers field has correct default value."""
    options = SearchOptions()
    assert options.workers is None


def test_host_custom():
    """Test host field with custom value."""
    options = SearchOptions(host="127.0.0.1")
    assert options.host == "127.0.0.1"


def test_port_custom():
    """Test port field with custom value."""
    options = SearchOptions(port=9000)
    assert options.port == 9000


def test_workers_custom():
    """Test workers field with custom value."""
    options = SearchOptions(workers=4)
    assert options.workers == 4
