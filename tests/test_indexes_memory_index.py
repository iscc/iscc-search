"""Test MemoryIndex implementation."""

import pytest
from iscc_search.indexes.memory import MemoryIndex
from iscc_search.protocols.index import IsccIndexProtocol
from iscc_search.schema import IsccAddResult, IsccEntry, IsccIndex, IsccSearchResult, Metric, Status


def test_memory_index_implements_protocol():
    """Test that MemoryIndex implements IsccIndexProtocol."""
    index = MemoryIndex()
    assert isinstance(index, IsccIndexProtocol)


def test_memory_index_initialization():
    """Test that MemoryIndex initializes with empty storage."""
    index = MemoryIndex()
    assert index._indexes == {}
    assert index.list_indexes() == []


def test_create_index_success():
    """Test creating a new index."""
    index = MemoryIndex()
    result = index.create_index(IsccIndex(name="testindex"))

    assert result.name == "testindex"
    assert result.assets == 0
    assert result.size == 0

    # Verify index appears in list
    indexes = index.list_indexes()
    assert len(indexes) == 1
    assert indexes[0].name == "testindex"


def test_create_index_validates_name():
    """Test that create_index validates index names."""
    from pydantic import ValidationError

    # Valid names
    for valid_name in ["test", "test123", "abc", "a1b2c3"]:
        idx = MemoryIndex()
        result = idx.create_index(IsccIndex(name=valid_name))
        assert result.name == valid_name

    # Invalid names - Pydantic validates these before our code
    invalid_names = [
        "Test",  # Uppercase
        "test-name",  # Hyphen
        "test_name",  # Underscore
        "test.name",  # Dot
        "test name",  # Space
        "123test",  # Starts with digit
        "",  # Empty
        "test-",  # Ends with special char
    ]

    for invalid_name in invalid_names:
        idx = MemoryIndex()
        with pytest.raises((ValueError, ValidationError)):
            idx.create_index(IsccIndex(name=invalid_name))


def test_create_index_duplicate():
    """Test that creating duplicate index raises FileExistsError."""
    index = MemoryIndex()
    index.create_index(IsccIndex(name="testindex"))

    with pytest.raises(FileExistsError, match="Index 'testindex' already exists"):
        index.create_index(IsccIndex(name="testindex"))


def test_get_index_success():
    """Test getting index metadata."""
    index = MemoryIndex()
    index.create_index(IsccIndex(name="testindex"))

    result = index.get_index("testindex")
    assert result.name == "testindex"
    assert result.assets == 0
    assert result.size == 0


def test_get_index_not_found():
    """Test that getting non-existent index raises FileNotFoundError."""
    index = MemoryIndex()

    with pytest.raises(FileNotFoundError, match="Index 'nonexistent' not found"):
        index.get_index("nonexistent")


def test_list_indexes_empty():
    """Test listing indexes when none exist."""
    index = MemoryIndex()
    assert index.list_indexes() == []


def test_list_indexes_multiple():
    """Test listing multiple indexes."""
    index = MemoryIndex()
    index.create_index(IsccIndex(name="index1"))
    index.create_index(IsccIndex(name="index2"))
    index.create_index(IsccIndex(name="index3"))

    indexes = index.list_indexes()
    assert len(indexes) == 3

    names = {idx.name for idx in indexes}
    assert names == {"index1", "index2", "index3"}


def test_delete_index_success():
    """Test deleting an index."""
    index = MemoryIndex()
    index.create_index(IsccIndex(name="testindex"))

    # Verify it exists
    assert len(index.list_indexes()) == 1

    # Delete it
    index.delete_index("testindex")

    # Verify it's gone
    assert len(index.list_indexes()) == 0

    with pytest.raises(FileNotFoundError):
        index.get_index("testindex")


def test_delete_index_not_found():
    """Test that deleting non-existent index raises FileNotFoundError."""
    index = MemoryIndex()

    with pytest.raises(FileNotFoundError, match="Index 'nonexistent' not found"):
        index.delete_index("nonexistent")


def test_add_assets_success(sample_iscc_ids, sample_content_units):
    """Test adding assets to index."""
    index = MemoryIndex()
    index.create_index(IsccIndex(name="testindex"))

    assets = [
        IsccEntry(
            iscc_id=sample_iscc_ids[0],
            units=[sample_content_units[0], sample_content_units[1]],
        ),
        IsccEntry(
            iscc_id=sample_iscc_ids[1],
            units=[sample_content_units[2], sample_content_units[3]],
        ),
    ]

    results = index.add_assets("testindex", assets)

    assert len(results) == 2
    assert all(isinstance(r, IsccAddResult) for r in results)
    assert results[0].iscc_id == sample_iscc_ids[0]
    assert results[0].status == Status.created
    assert results[1].iscc_id == sample_iscc_ids[1]
    assert results[1].status == Status.created

    # Verify index metadata updated
    idx_info = index.get_index("testindex")
    assert idx_info.assets == 2


def test_add_assets_duplicate(sample_iscc_ids, sample_content_units):
    """Test that adding duplicate assets updates instead of creating."""
    index = MemoryIndex()
    index.create_index(IsccIndex(name="testindex"))

    asset = IsccEntry(
        iscc_id=sample_iscc_ids[0],
        units=[sample_content_units[0], sample_content_units[1]],
    )

    # Add first time
    results1 = index.add_assets("testindex", [asset])
    assert results1[0].status == Status.created

    # Add again (same iscc_id)
    results2 = index.add_assets("testindex", [asset])
    assert results2[0].status == Status.updated

    # Verify only one asset in index
    idx_info = index.get_index("testindex")
    assert idx_info.assets == 1


def test_add_assets_index_not_found(sample_iscc_ids):
    """Test that adding assets to non-existent index raises FileNotFoundError."""
    index = MemoryIndex()

    asset = IsccEntry(iscc_id=sample_iscc_ids[0])

    with pytest.raises(FileNotFoundError, match="Index 'nonexistent' not found"):
        index.add_assets("nonexistent", [asset])


def test_add_assets_missing_iscc_id(sample_content_units):
    """Test that adding assets without iscc_id raises ValueError from backend."""
    index = MemoryIndex()
    index.create_index(IsccIndex(name="testindex"))

    # Asset without iscc_id (allowed by schema for search, but required for add)
    asset = IsccEntry(units=[sample_content_units[0], sample_content_units[1]])

    # Backend validates that iscc_id is required when adding
    with pytest.raises(ValueError, match="Asset must have iscc_id field when adding to index"):
        index.add_assets("testindex", [asset])


def test_get_asset_success(sample_iscc_ids, sample_iscc_codes, sample_content_units):
    """Test successfully retrieving an asset by ISCC-ID."""
    index = MemoryIndex()
    index.create_index(IsccIndex(name="testindex"))

    # Add an asset
    code = sample_iscc_codes[0]
    asset = IsccEntry(
        iscc_id=sample_iscc_ids[0],
        iscc_code=code,
        units=[sample_content_units[0], sample_content_units[1]],
        metadata={"title": "Test Document"},
    )
    index.add_assets("testindex", [asset])

    # Get the asset
    retrieved = index.get_asset("testindex", sample_iscc_ids[0])

    assert retrieved.iscc_id == sample_iscc_ids[0]
    assert retrieved.iscc_code == code
    # Units are stored as plain strings
    assert len(retrieved.units) == 2
    assert retrieved.units[0] == sample_content_units[0]
    assert retrieved.units[1] == sample_content_units[1]
    assert retrieved.metadata == {"title": "Test Document"}


def test_get_asset_index_not_found(sample_iscc_ids):
    """Test that getting asset from non-existent index raises FileNotFoundError."""
    index = MemoryIndex()

    with pytest.raises(FileNotFoundError, match="Index 'nonexistent' not found"):
        index.get_asset("nonexistent", sample_iscc_ids[0])


def test_get_asset_not_found(sample_iscc_ids):
    """Test that getting non-existent asset raises FileNotFoundError."""
    index = MemoryIndex()
    index.create_index(IsccIndex(name="testindex"))

    # Add one asset
    asset = IsccEntry(iscc_id=sample_iscc_ids[0])
    index.add_assets("testindex", [asset])

    # Try to get different asset
    with pytest.raises(
        FileNotFoundError,
        match=f"Asset '{sample_iscc_ids[1]}' not found in index 'testindex'",
    ):
        index.get_asset("testindex", sample_iscc_ids[1])


def test_search_assets_by_iscc_code(sample_iscc_ids, sample_iscc_codes):
    """Test searching assets by iscc_code."""
    index = MemoryIndex()
    index.create_index(IsccIndex(name="testindex"))

    # Use sample ISCC-CODEs
    code1 = sample_iscc_codes[0]
    code2 = sample_iscc_codes[1]

    # Add assets
    assets = [
        IsccEntry(
            iscc_id=sample_iscc_ids[0],
            iscc_code=code1,
        ),
        IsccEntry(
            iscc_id=sample_iscc_ids[1],
            iscc_code=code2,
        ),
    ]
    index.add_assets("testindex", assets)

    # Search for matching iscc_code
    query = IsccEntry(iscc_code=code1)
    result = index.search_assets("testindex", query)

    assert isinstance(result, IsccSearchResult)
    # Query is normalized (units derived from iscc_code)
    assert result.query.iscc_code == code1
    assert result.query.units is not None  # Units were derived
    assert result.metric == Metric.bitlength
    assert len(result.global_matches) == 1
    assert result.global_matches[0].iscc_id == sample_iscc_ids[0]
    assert result.global_matches[0].score == 1.0


def test_search_assets_by_iscc_id(sample_iscc_ids, sample_iscc_codes):
    """Test searching assets by iscc_id with iscc_code."""
    index = MemoryIndex()
    index.create_index(IsccIndex(name="testindex"))

    # Add assets with iscc_code (required for search)
    asset = IsccEntry(iscc_id=sample_iscc_ids[0], iscc_code=sample_iscc_codes[0])
    index.add_assets("testindex", [asset])

    # Search by iscc_id and iscc_code
    query = IsccEntry(iscc_id=sample_iscc_ids[0], iscc_code=sample_iscc_codes[0])
    result = index.search_assets("testindex", query)

    assert len(result.global_matches) == 1
    assert result.global_matches[0].iscc_id == sample_iscc_ids[0]


def test_search_assets_no_matches(sample_iscc_ids, sample_iscc_codes):
    """Test searching when no assets match."""
    index = MemoryIndex()
    index.create_index(IsccIndex(name="testindex"))

    # Use two different ISCC-CODEs
    code1 = sample_iscc_codes[0]
    code2 = sample_iscc_codes[1]

    # Add assets with code1
    asset = IsccEntry(
        iscc_id=sample_iscc_ids[0],
        iscc_code=code1,
    )
    index.add_assets("testindex", [asset])

    # Search for non-matching code2
    query = IsccEntry(iscc_code=code2)
    result = index.search_assets("testindex", query)

    assert len(result.global_matches) == 0


def test_search_assets_limit(sample_iscc_ids, sample_iscc_codes):
    """Test that search respects limit parameter."""
    index = MemoryIndex()
    index.create_index(IsccIndex(name="testindex"))

    # Use same iscc_code for all assets
    code = sample_iscc_codes[0]

    # Add 10 assets with same iscc_code but different iscc_ids
    assets = [IsccEntry(iscc_id=sample_iscc_ids[i], iscc_code=code) for i in range(10)]
    index.add_assets("testindex", assets)

    # Search with limit
    query = IsccEntry(iscc_code=code)
    result = index.search_assets("testindex", query, limit=5)

    assert len(result.global_matches) == 5


def test_search_assets_index_not_found(sample_iscc_codes):
    """Test that searching non-existent index raises FileNotFoundError."""
    index = MemoryIndex()

    query = IsccEntry(iscc_code=sample_iscc_codes[0])

    with pytest.raises(FileNotFoundError, match="Index 'nonexistent' not found"):
        index.search_assets("nonexistent", query)


def test_close():
    """Test that close() is a no-op and can be called multiple times."""
    index = MemoryIndex()
    index.create_index(IsccIndex(name="testindex"))

    # Close should be no-op
    index.close()

    # Should be safe to call multiple times
    index.close()
    index.close()

    # Index should still be accessible (since it's in-memory)
    # Note: In production, after close(), the index shouldn't be used
    # But for MemoryIndex, close() does nothing


def test_multiple_indexes_isolation(sample_iscc_ids, sample_iscc_codes):
    """Test that multiple indexes are isolated from each other."""
    index = MemoryIndex()
    index.create_index(IsccIndex(name="index1"))
    index.create_index(IsccIndex(name="index2"))

    # Use two different ISCC-CODEs
    code1 = sample_iscc_codes[0]
    code2 = sample_iscc_codes[1]

    # Add assets to index1
    asset1 = IsccEntry(iscc_id=sample_iscc_ids[0], iscc_code=code1)
    index.add_assets("index1", [asset1])

    # Add assets to index2
    asset2 = IsccEntry(iscc_id=sample_iscc_ids[1], iscc_code=code2)
    index.add_assets("index2", [asset2])

    # Verify isolation
    result1 = index.search_assets("index1", IsccEntry(iscc_code=code1))
    assert len(result1.global_matches) == 1
    assert result1.global_matches[0].iscc_id == sample_iscc_ids[0]

    result2 = index.search_assets("index2", IsccEntry(iscc_code=code2))
    assert len(result2.global_matches) == 1
    assert result2.global_matches[0].iscc_id == sample_iscc_ids[1]

    # Cross-search should return no results
    result_cross = index.search_assets("index1", IsccEntry(iscc_code=code2))
    assert len(result_cross.global_matches) == 0


def test_metadata_field(sample_iscc_ids, sample_iscc_codes):
    """Test that metadata field is preserved."""
    index = MemoryIndex()
    index.create_index(IsccIndex(name="testindex"))

    code = sample_iscc_codes[0]
    asset = IsccEntry(
        iscc_id=sample_iscc_ids[0],
        iscc_code=code,
        metadata={"source": "test", "tags": ["tag1", "tag2"]},
    )
    index.add_assets("testindex", [asset])

    # Search and verify metadata is preserved
    query = IsccEntry(iscc_code=code)
    result = index.search_assets("testindex", query)
    assert len(result.global_matches) == 1

    # Retrieve the asset and verify metadata is preserved
    retrieved = index.get_asset("testindex", sample_iscc_ids[0])
    assert retrieved.metadata == {"source": "test", "tags": ["tag1", "tag2"]}


def test_search_assets_no_matching_iscc_id(sample_iscc_ids, sample_iscc_codes):
    """Test searching by different iscc_code when no match exists."""
    index = MemoryIndex()
    index.create_index(IsccIndex(name="testindex"))

    # Add an asset with one iscc_code
    asset = IsccEntry(iscc_id=sample_iscc_ids[0], iscc_code=sample_iscc_codes[0])
    index.add_assets("testindex", [asset])

    # Search with different iscc_id and iscc_code
    query = IsccEntry(iscc_id=sample_iscc_ids[1], iscc_code=sample_iscc_codes[1])
    result = index.search_assets("testindex", query)

    # Should not match (different iscc_code)
    assert len(result.global_matches) == 0


def test_search_assets_no_iscc_code_in_asset(sample_iscc_ids, sample_iscc_codes):
    """Test searching when asset has no iscc_code (covers branch)."""
    index = MemoryIndex()
    index.create_index(IsccIndex(name="testindex"))

    # Add an asset without iscc_code (only has iscc_id)
    asset = IsccEntry(iscc_id=sample_iscc_ids[0])
    index.add_assets("testindex", [asset])

    # Search with iscc_code (won't match asset without code)
    query = IsccEntry(iscc_code=sample_iscc_codes[0])
    result = index.search_assets("testindex", query)

    # Should not match (asset has no iscc_code)
    assert len(result.global_matches) == 0


def test_search_assets_by_units_only(sample_iscc_ids, sample_iscc_codes):
    """Test searching assets by units only (no iscc_code in query)."""
    from iscc_search.models import IsccCode

    index = MemoryIndex()
    index.create_index(IsccIndex(name="testindex"))

    # Use sample ISCC-CODE
    code1 = sample_iscc_codes[0]

    # Add asset with iscc_code
    asset = IsccEntry(
        iscc_id=sample_iscc_ids[0],
        iscc_code=code1,
    )
    index.add_assets("testindex", [asset])

    # Derive units from code1
    code_obj = IsccCode(code1)
    units = [str(u) for u in code_obj.units]

    # Search with units only (no iscc_code)
    query = IsccEntry(units=units)
    result = index.search_assets("testindex", query)

    assert isinstance(result, IsccSearchResult)
    # Query is normalized (iscc_code derived from units)
    assert result.query.units == units
    assert result.query.iscc_code is not None  # iscc_code was derived
    assert result.metric == Metric.bitlength
    # Should find the matching asset
    assert len(result.global_matches) == 1
    assert result.global_matches[0].iscc_id == sample_iscc_ids[0]
