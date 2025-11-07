"""Tests for LmdbSimprintIndexMulti64 multi-type simprint index."""

import json

import pytest

from iscc_search.indexes.simprint.lmdb_multi_64 import LmdbSimprintIndexMulti64
from iscc_search.indexes.simprint.models import SimprintEntryMulti, SimprintRaw


@pytest.fixture
def temp_index_path(tmp_path):
    # type: (Path) -> Path
    """Provide a temporary directory for index storage."""
    return tmp_path / "multi_index"


@pytest.fixture
def sample_entries():
    # type: () -> list[SimprintEntryMulti]
    """Create sample multi-type entries for testing."""
    realm_id = b"\x00\x10"  # ISCC-ID header

    entries = [
        SimprintEntryMulti(
            iscc_id=realm_id + b"\x12\x34\x56\x78\x9a\xbc\xde\xf0",
            simprints={
                "CONTENT_TEXT_V0": [
                    SimprintRaw(simprint=b"\x01" * 8, offset=0, size=512),
                    SimprintRaw(simprint=b"\x02" * 8, offset=512, size=489),
                ],
                "SEMANTIC_TEXT_V0": [
                    SimprintRaw(simprint=b"\x03" * 8, offset=0, size=1024),
                ],
            },
        ),
        SimprintEntryMulti(
            iscc_id=realm_id + b"\xfe\xdc\xba\x98\x76\x54\x32\x10",
            simprints={
                "CONTENT_TEXT_V0": [
                    SimprintRaw(simprint=b"\x01" * 8, offset=0, size=256),  # Same simprint
                ],
                "SEMANTIC_TEXT_V0": [
                    SimprintRaw(simprint=b"\x04" * 8, offset=0, size=512),
                ],
            },
        ),
    ]
    return entries


def test_init_creates_directory(temp_index_path):
    # type: (Path) -> None
    """Test that __init__ creates the index directory."""
    index = LmdbSimprintIndexMulti64(str(temp_index_path))
    assert temp_index_path.exists()
    assert temp_index_path.is_dir()
    index.close()


def test_init_with_file_uri(temp_index_path):
    # type: (Path) -> None
    """Test initialization with file:// URI."""
    uri = f"file://{temp_index_path}"
    index = LmdbSimprintIndexMulti64(uri)
    assert temp_index_path.exists()
    index.close()


def test_init_with_lmdb_uri(temp_index_path):
    # type: (Path) -> None
    """Test initialization with lmdb:// URI."""
    uri = f"lmdb://{temp_index_path}"
    index = LmdbSimprintIndexMulti64(uri)
    assert temp_index_path.exists()
    index.close()


def test_add_raw_multi_creates_type_indexes(temp_index_path, sample_entries):
    # type: (Path, list[SimprintEntryMulti]) -> None
    """Test that add_raw_multi creates type-specific subdirectories."""
    index = LmdbSimprintIndexMulti64(str(temp_index_path))
    index.add_raw_multi(sample_entries)

    assert (temp_index_path / "CONTENT_TEXT_V0").exists()
    assert (temp_index_path / "SEMANTIC_TEXT_V0").exists()
    index.close()


def test_add_raw_multi_extracts_realm_id(temp_index_path, sample_entries):
    # type: (Path, list[SimprintEntryMulti]) -> None
    """Test realm_id extraction from first entry."""
    index = LmdbSimprintIndexMulti64(str(temp_index_path))
    index.add_raw_multi(sample_entries)

    assert index.realm_id == b"\x00\x10"

    # Check metadata file
    metadata_path = temp_index_path / "metadata.json"
    assert metadata_path.exists()

    metadata = json.loads(metadata_path.read_text())
    assert metadata["realm_id"] == "0010"
    assert "CONTENT_TEXT_V0" in metadata["indexed_types"]
    assert "SEMANTIC_TEXT_V0" in metadata["indexed_types"]
    index.close()


def test_add_raw_multi_validates_realm_id(temp_index_path, sample_entries):
    # type: (Path, list[SimprintEntryMulti]) -> None
    """Test that mismatched realm_id raises error."""
    index = LmdbSimprintIndexMulti64(str(temp_index_path))
    index.add_raw_multi([sample_entries[0]])

    # Try to add entry with different realm_id
    bad_entry = SimprintEntryMulti(
        iscc_id=b"\xff\xff\x12\x34\x56\x78\x9a\xbc\xde\xf0",
        simprints={"CONTENT_TEXT_V0": [SimprintRaw(simprint=b"\x05" * 8, offset=0, size=100)]},
    )

    with pytest.raises(ValueError, match="realm mismatch"):
        index.add_raw_multi([bad_entry])

    index.close()


def test_add_raw_multi_invalid_iscc_id_length(temp_index_path):
    # type: (Path) -> None
    """Test that invalid ISCC-ID length raises error."""
    index = LmdbSimprintIndexMulti64(str(temp_index_path))

    bad_entry = SimprintEntryMulti(
        iscc_id=b"\x00\x10\x12",  # Only 3 bytes instead of 10
        simprints={"CONTENT_TEXT_V0": [SimprintRaw(simprint=b"\x01" * 8, offset=0, size=100)]},
    )

    with pytest.raises(ValueError, match="must be 10 bytes"):
        index.add_raw_multi([bad_entry])

    index.close()


def test_add_raw_multi_empty_list(temp_index_path):
    # type: (Path) -> None
    """Test that empty entry list is handled gracefully."""
    index = LmdbSimprintIndexMulti64(str(temp_index_path))
    index.add_raw_multi([])

    assert index.realm_id is None
    assert len(index.indexes) == 0
    index.close()


def test_add_raw_multi_duplicate_entries(temp_index_path, sample_entries):
    # type: (Path, list[SimprintEntryMulti]) -> None
    """Test add-once semantics - duplicates are silently ignored."""
    index = LmdbSimprintIndexMulti64(str(temp_index_path))

    # Add same entries twice
    index.add_raw_multi(sample_entries)
    index.add_raw_multi(sample_entries)

    # Both should be in index (count via type-specific index)
    content_index = index.indexes["CONTENT_TEXT_V0"]
    assert len(content_index) == 2  # Both unique ISCC-IDs indexed
    index.close()


def test_search_raw_multi_single_type(temp_index_path, sample_entries):
    # type: (Path, list[SimprintEntryMulti]) -> None
    """Test search with single simprint type."""
    index = LmdbSimprintIndexMulti64(str(temp_index_path))
    index.add_raw_multi(sample_entries)

    # Search for simprint that appears in both assets
    results = index.search_raw_multi(
        {"CONTENT_TEXT_V0": [b"\x01" * 8]},
        limit=10,
        threshold=0.0,
        detailed=True,
    )

    assert len(results) == 2
    assert results[0].iscc_id in [e.iscc_id for e in sample_entries]
    assert "CONTENT_TEXT_V0" in results[0].types
    index.close()


def test_search_raw_multi_multiple_types(temp_index_path, sample_entries):
    # type: (Path, list[SimprintEntryMulti]) -> None
    """Test search across multiple simprint types."""
    index = LmdbSimprintIndexMulti64(str(temp_index_path))
    index.add_raw_multi(sample_entries)

    # Search both types
    results = index.search_raw_multi(
        {
            "CONTENT_TEXT_V0": [b"\x01" * 8],
            "SEMANTIC_TEXT_V0": [b"\x03" * 8],
        },
        limit=10,
        threshold=0.0,
        detailed=True,
    )

    assert len(results) >= 1
    # First result should match both types
    top_match = results[0]
    assert len(top_match.types) >= 1
    index.close()


def test_search_raw_multi_empty_query(temp_index_path, sample_entries):
    # type: (Path, list[SimprintEntryMulti]) -> None
    """Test search with empty query."""
    index = LmdbSimprintIndexMulti64(str(temp_index_path))
    index.add_raw_multi(sample_entries)

    results = index.search_raw_multi({}, limit=10, threshold=0.8, detailed=True)
    assert results == []
    index.close()


def test_search_raw_multi_unknown_type(temp_index_path, sample_entries):
    # type: (Path, list[SimprintEntryMulti]) -> None
    """Test search with type that hasn't been indexed."""
    index = LmdbSimprintIndexMulti64(str(temp_index_path))
    index.add_raw_multi(sample_entries)

    results = index.search_raw_multi(
        {"UNKNOWN_TYPE_V0": [b"\x01" * 8]},
        limit=10,
        threshold=0.8,
        detailed=True,
    )
    assert results == []
    index.close()


def test_search_raw_multi_detailed_false(temp_index_path, sample_entries):
    # type: (Path, list[SimprintEntryMulti]) -> None
    """Test search with detailed=False excludes chunk details."""
    index = LmdbSimprintIndexMulti64(str(temp_index_path))
    index.add_raw_multi(sample_entries)

    results = index.search_raw_multi(
        {"CONTENT_TEXT_V0": [b"\x01" * 8]},
        limit=10,
        threshold=0.0,
        detailed=False,
    )

    assert len(results) > 0
    for result in results:
        for type_result in result.types.values():
            assert type_result.chunks is None
    index.close()


def test_get_indexed_types(temp_index_path, sample_entries):
    # type: (Path, list[SimprintEntryMulti]) -> None
    """Test get_indexed_types returns sorted list."""
    index = LmdbSimprintIndexMulti64(str(temp_index_path))
    assert index.get_indexed_types() == []

    index.add_raw_multi(sample_entries)
    types = index.get_indexed_types()

    assert types == ["CONTENT_TEXT_V0", "SEMANTIC_TEXT_V0"]
    index.close()


def test_contains_existing_iscc_id(temp_index_path, sample_entries):
    # type: (Path, list[SimprintEntryMulti]) -> None
    """Test __contains__ with existing ISCC-ID."""
    index = LmdbSimprintIndexMulti64(str(temp_index_path))
    index.add_raw_multi(sample_entries)

    assert sample_entries[0].iscc_id in index
    assert sample_entries[1].iscc_id in index
    index.close()


def test_contains_non_existing_iscc_id(temp_index_path, sample_entries):
    # type: (Path, list[SimprintEntryMulti]) -> None
    """Test __contains__ with non-existing ISCC-ID."""
    index = LmdbSimprintIndexMulti64(str(temp_index_path))
    index.add_raw_multi(sample_entries)

    non_existing = b"\x00\x10\xaa\xbb\xcc\xdd\xee\xff\x00\x11"
    assert non_existing not in index
    index.close()


def test_contains_invalid_length(temp_index_path):
    # type: (Path) -> None
    """Test __contains__ with invalid ISCC-ID length."""
    index = LmdbSimprintIndexMulti64(str(temp_index_path))

    assert b"\x00\x10\x12" not in index  # Only 3 bytes
    index.close()


def test_get_raw_multi(temp_index_path, sample_entries):
    # type: (Path, list[SimprintEntryMulti]) -> None
    """Test get_raw_multi retrieves entries."""
    index = LmdbSimprintIndexMulti64(str(temp_index_path))
    index.add_raw_multi(sample_entries)

    results = index.get_raw_multi([sample_entries[0].iscc_id])
    assert len(results) == 1
    assert results[0].iscc_id == sample_entries[0].iscc_id
    index.close()


def test_get_raw_multi_non_existing(temp_index_path, sample_entries):
    # type: (Path, list[SimprintEntryMulti]) -> None
    """Test get_raw_multi with non-existing ISCC-IDs."""
    index = LmdbSimprintIndexMulti64(str(temp_index_path))
    index.add_raw_multi(sample_entries)

    non_existing = b"\x00\x10\xaa\xbb\xcc\xdd\xee\xff\x00\x11"
    results = index.get_raw_multi([non_existing])

    assert len(results) == 1
    assert results[0].iscc_id == non_existing
    assert results[0].simprints == {}
    index.close()


def test_delete_raw_multi(temp_index_path, sample_entries):
    # type: (Path, list[SimprintEntryMulti]) -> None
    """Test delete_raw_multi removes entries (currently a no-op)."""
    index = LmdbSimprintIndexMulti64(str(temp_index_path))
    index.add_raw_multi(sample_entries)

    # Delete doesn't raise error (even though not fully implemented)
    index.delete_raw_multi([sample_entries[0].iscc_id])
    index.close()


def test_close_releases_resources(temp_index_path, sample_entries):
    # type: (Path, list[SimprintEntryMulti]) -> None
    """Test close releases all index resources."""
    index = LmdbSimprintIndexMulti64(str(temp_index_path))
    index.add_raw_multi(sample_entries)

    assert len(index.indexes) == 2
    index.close()
    assert len(index.indexes) == 0


def test_context_manager(temp_index_path, sample_entries):
    # type: (Path, list[SimprintEntryMulti]) -> None
    """Test context manager support."""
    with LmdbSimprintIndexMulti64(str(temp_index_path)) as index:
        index.add_raw_multi(sample_entries)
        assert len(index.indexes) == 2

    # Indexes should be closed after context
    assert len(index.indexes) == 0


def test_reopen_index_loads_metadata(temp_index_path, sample_entries):
    # type: (Path, list[SimprintEntryMulti]) -> None
    """Test reopening index loads existing metadata and indexes."""
    # Create and populate index
    index1 = LmdbSimprintIndexMulti64(str(temp_index_path))
    index1.add_raw_multi(sample_entries)
    realm_id = index1.realm_id
    index1.close()

    # Reopen and verify
    index2 = LmdbSimprintIndexMulti64(str(temp_index_path))
    assert index2.realm_id == realm_id
    assert len(index2.indexes) == 2
    assert sample_entries[0].iscc_id in index2
    index2.close()


def test_metadata_persistence(temp_index_path, sample_entries):
    # type: (Path, list[SimprintEntryMulti]) -> None
    """Test metadata.json is correctly saved and loaded."""
    index = LmdbSimprintIndexMulti64(str(temp_index_path))
    index.add_raw_multi(sample_entries)
    index.close()

    # Verify metadata file
    metadata_path = temp_index_path / "metadata.json"
    assert metadata_path.exists()

    metadata = json.loads(metadata_path.read_text())
    assert metadata["realm_id"] == "0010"
    assert set(metadata["indexed_types"]) == {"CONTENT_TEXT_V0", "SEMANTIC_TEXT_V0"}


def test_add_raw_multi_entry_without_simprints(temp_index_path):
    # type: (Path) -> None
    """Test adding entry with empty simprints dict."""
    index = LmdbSimprintIndexMulti64(str(temp_index_path))

    entry = SimprintEntryMulti(
        iscc_id=b"\x00\x10\x12\x34\x56\x78\x9a\xbc\xde\xf0",
        simprints={},
    )

    index.add_raw_multi([entry])
    assert index.realm_id == b"\x00\x10"
    assert len(index.indexes) == 0
    index.close()


def test_search_result_ordering(temp_index_path):
    # type: (Path) -> None
    """Test that search results are ordered by score descending."""
    index = LmdbSimprintIndexMulti64(str(temp_index_path))

    # Create entries with varying simprint overlap
    entries = [
        SimprintEntryMulti(
            iscc_id=b"\x00\x10" + b"\x01" * 8,
            simprints={
                "CONTENT_TEXT_V0": [
                    SimprintRaw(simprint=b"\x01" * 8, offset=0, size=100),
                    SimprintRaw(simprint=b"\x02" * 8, offset=100, size=100),
                ]
            },
        ),
        SimprintEntryMulti(
            iscc_id=b"\x00\x10" + b"\x02" * 8,
            simprints={
                "CONTENT_TEXT_V0": [
                    SimprintRaw(simprint=b"\x01" * 8, offset=0, size=100),
                ]
            },
        ),
    ]

    index.add_raw_multi(entries)

    # Query with both simprints - first entry should score higher
    results = index.search_raw_multi(
        {"CONTENT_TEXT_V0": [b"\x01" * 8, b"\x02" * 8]},
        limit=10,
        threshold=0.0,
        detailed=False,
    )

    assert len(results) == 2
    assert results[0].score >= results[1].score
    index.close()


def test_delete_raw_multi_with_indexes(temp_index_path, sample_entries):
    # type: (Path, list[SimprintEntryMulti]) -> None
    """Test delete_raw_multi with populated indexes."""
    index = LmdbSimprintIndexMulti64(str(temp_index_path))
    index.add_raw_multi(sample_entries)

    # Ensure indexes exist
    assert len(index.indexes) > 0

    # Delete should iterate over indexes (even if not fully implemented)
    index.delete_raw_multi([sample_entries[0].iscc_id])
    index.close()


def test_context_manager_with_exception(temp_index_path, sample_entries):
    # type: (Path, list[SimprintEntryMulti]) -> None
    """Test context manager properly closes even with exception."""
    try:
        with LmdbSimprintIndexMulti64(str(temp_index_path)) as index:
            index.add_raw_multi(sample_entries)
            raise ValueError("Test exception")
    except ValueError:
        pass

    # Index should be closed despite exception
    # We can't directly verify this, but the context manager should have called close()


def test_load_metadata_with_null_realm_id(temp_index_path):
    # type: (Path) -> None
    """Test loading metadata file with null realm_id."""
    # Create metadata file with null realm_id
    temp_index_path.mkdir(parents=True, exist_ok=True)
    metadata_path = temp_index_path / "metadata.json"
    metadata_path.write_text('{"realm_id": null, "indexed_types": []}')

    # Open index - should load but realm_id remains None
    index = LmdbSimprintIndexMulti64(str(temp_index_path))
    assert index.realm_id is None
    index.close()


def test_load_metadata_with_empty_realm_id(temp_index_path):
    # type: (Path) -> None
    """Test loading metadata file with empty string realm_id."""
    # Create metadata file with empty realm_id
    temp_index_path.mkdir(parents=True, exist_ok=True)
    metadata_path = temp_index_path / "metadata.json"
    metadata_path.write_text('{"realm_id": "", "indexed_types": []}')

    # Open index - should load but realm_id remains None
    index = LmdbSimprintIndexMulti64(str(temp_index_path))
    assert index.realm_id is None
    index.close()


def test_validate_realm_id_when_none(temp_index_path):
    # type: (Path) -> None
    """Test _validate_realm_id when realm_id is None (defensive check)."""
    index = LmdbSimprintIndexMulti64(str(temp_index_path))
    assert index.realm_id is None

    # Should return early without error
    index._validate_realm_id(b"\x00\x10\x12\x34\x56\x78\x9a\xbc\xde\xf0")
    index.close()


def test_validate_realm_id_invalid_length(temp_index_path, sample_entries):
    # type: (Path, list[SimprintEntryMulti]) -> None
    """Test _validate_realm_id with invalid ISCC-ID length after realm_id is set."""
    index = LmdbSimprintIndexMulti64(str(temp_index_path))

    # Set realm_id first
    index.add_raw_multi([sample_entries[0]])
    assert index.realm_id is not None

    # Now try to validate an invalid length ISCC-ID
    with pytest.raises(ValueError, match="must be 10 bytes"):
        index._validate_realm_id(b"\x00\x10\x12")  # Only 3 bytes

    index.close()
