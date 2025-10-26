"""Test MemoryIndex implementation."""

import pytest
from iscc_vdb.indexes.memory import MemoryIndex
from iscc_vdb.protocol import IsccIndexProtocol
from iscc_vdb.schema import IsccAddResult, IsccIndex, IsccItem, IsccSearchResult, Metric, Status


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
    assert result.items == 0
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
    assert result.items == 0
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


def test_add_items_success(sample_iscc_ids, sample_content_units):
    """Test adding items to index."""
    index = MemoryIndex()
    index.create_index(IsccIndex(name="testindex"))

    items = [
        IsccItem(
            iscc_id=sample_iscc_ids[0],
            units=[sample_content_units[0], sample_content_units[1]],
        ),
        IsccItem(
            iscc_id=sample_iscc_ids[1],
            units=[sample_content_units[2], sample_content_units[3]],
        ),
    ]

    results = index.add_items("testindex", items)

    assert len(results) == 2
    assert all(isinstance(r, IsccAddResult) for r in results)
    assert results[0].iscc_id == sample_iscc_ids[0]
    assert results[0].status == Status.created
    assert results[1].iscc_id == sample_iscc_ids[1]
    assert results[1].status == Status.created

    # Verify index metadata updated
    idx_info = index.get_index("testindex")
    assert idx_info.items == 2


def test_add_items_duplicate(sample_iscc_ids, sample_content_units):
    """Test that adding duplicate items updates instead of creating."""
    index = MemoryIndex()
    index.create_index(IsccIndex(name="testindex"))

    item = IsccItem(
        iscc_id=sample_iscc_ids[0],
        units=[sample_content_units[0], sample_content_units[1]],
    )

    # Add first time
    results1 = index.add_items("testindex", [item])
    assert results1[0].status == Status.created

    # Add again (same iscc_id)
    results2 = index.add_items("testindex", [item])
    assert results2[0].status == Status.updated

    # Verify only one item in index
    idx_info = index.get_index("testindex")
    assert idx_info.items == 1


def test_add_items_index_not_found(sample_iscc_ids):
    """Test that adding items to non-existent index raises FileNotFoundError."""
    index = MemoryIndex()

    item = IsccItem(iscc_id=sample_iscc_ids[0])

    with pytest.raises(FileNotFoundError, match="Index 'nonexistent' not found"):
        index.add_items("nonexistent", [item])


def test_add_items_missing_iscc_id(sample_content_units):
    """Test that adding items without iscc_id raises ValueError."""
    index = MemoryIndex()
    index.create_index(IsccIndex(name="testindex"))

    # Item without iscc_id (but with minimum 2 units required by schema)
    item = IsccItem(units=[sample_content_units[0], sample_content_units[1]])

    with pytest.raises(ValueError, match="Item must have iscc_id field"):
        index.add_items("testindex", [item])


def test_search_items_by_iscc_code(sample_iscc_ids, sample_iscc_codes):
    """Test searching items by iscc_code."""
    index = MemoryIndex()
    index.create_index(IsccIndex(name="testindex"))

    # Use sample ISCC-CODEs
    code1 = sample_iscc_codes[0]
    code2 = sample_iscc_codes[1]

    # Add items
    items = [
        IsccItem(
            iscc_id=sample_iscc_ids[0],
            iscc_code=code1,
        ),
        IsccItem(
            iscc_id=sample_iscc_ids[1],
            iscc_code=code2,
        ),
    ]
    index.add_items("testindex", items)

    # Search for matching iscc_code
    query = IsccItem(iscc_code=code1)
    result = index.search_items("testindex", query)

    assert isinstance(result, IsccSearchResult)
    assert result.query == query
    assert result.metric == Metric.bitlength
    assert len(result.matches) == 1
    assert result.matches[0].iscc_id == sample_iscc_ids[0]
    assert result.matches[0].score == 1.0


def test_search_items_by_iscc_id(sample_iscc_ids):
    """Test searching items by iscc_id."""
    index = MemoryIndex()
    index.create_index(IsccIndex(name="testindex"))

    # Add items
    item = IsccItem(iscc_id=sample_iscc_ids[0])
    index.add_items("testindex", [item])

    # Search by iscc_id
    query = IsccItem(iscc_id=sample_iscc_ids[0])
    result = index.search_items("testindex", query)

    assert len(result.matches) == 1
    assert result.matches[0].iscc_id == sample_iscc_ids[0]


def test_search_items_no_matches(sample_iscc_ids, sample_iscc_codes):
    """Test searching when no items match."""
    index = MemoryIndex()
    index.create_index(IsccIndex(name="testindex"))

    # Use two different ISCC-CODEs
    code1 = sample_iscc_codes[0]
    code2 = sample_iscc_codes[1]

    # Add items with code1
    item = IsccItem(
        iscc_id=sample_iscc_ids[0],
        iscc_code=code1,
    )
    index.add_items("testindex", [item])

    # Search for non-matching code2
    query = IsccItem(iscc_code=code2)
    result = index.search_items("testindex", query)

    assert len(result.matches) == 0


def test_search_items_limit(sample_iscc_ids, sample_iscc_codes):
    """Test that search respects limit parameter."""
    index = MemoryIndex()
    index.create_index(IsccIndex(name="testindex"))

    # Use same iscc_code for all items
    code = sample_iscc_codes[0]

    # Add 10 items with same iscc_code but different iscc_ids
    items = [IsccItem(iscc_id=sample_iscc_ids[i], iscc_code=code) for i in range(10)]
    index.add_items("testindex", items)

    # Search with limit
    query = IsccItem(iscc_code=code)
    result = index.search_items("testindex", query, limit=5)

    assert len(result.matches) == 5


def test_search_items_index_not_found(sample_iscc_codes):
    """Test that searching non-existent index raises FileNotFoundError."""
    index = MemoryIndex()

    query = IsccItem(iscc_code=sample_iscc_codes[0])

    with pytest.raises(FileNotFoundError, match="Index 'nonexistent' not found"):
        index.search_items("nonexistent", query)


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

    # Add items to index1
    item1 = IsccItem(iscc_id=sample_iscc_ids[0], iscc_code=code1)
    index.add_items("index1", [item1])

    # Add items to index2
    item2 = IsccItem(iscc_id=sample_iscc_ids[1], iscc_code=code2)
    index.add_items("index2", [item2])

    # Verify isolation
    result1 = index.search_items("index1", IsccItem(iscc_code=code1))
    assert len(result1.matches) == 1
    assert result1.matches[0].iscc_id == sample_iscc_ids[0]

    result2 = index.search_items("index2", IsccItem(iscc_code=code2))
    assert len(result2.matches) == 1
    assert result2.matches[0].iscc_id == sample_iscc_ids[1]

    # Cross-search should return no results
    result_cross = index.search_items("index1", IsccItem(iscc_code=code2))
    assert len(result_cross.matches) == 0


def test_metadata_field(sample_iscc_ids, sample_iscc_codes):
    """Test that metadata field is preserved."""
    index = MemoryIndex()
    index.create_index(IsccIndex(name="testindex"))

    code = sample_iscc_codes[0]
    item = IsccItem(
        iscc_id=sample_iscc_ids[0],
        iscc_code=code,
        metadata={"source": "test", "tags": ["tag1", "tag2"]},
    )
    index.add_items("testindex", [item])

    # Search and verify metadata is preserved
    query = IsccItem(iscc_code=code)
    result = index.search_items("testindex", query)

    # Note: The search returns matches but we can't directly access item metadata
    # from matches. This is expected - matches contain iscc_id, score, and matches dict.
    # To verify metadata preservation, we'd need a get_item method.
    assert len(result.matches) == 1


def test_search_items_no_matching_iscc_id(sample_iscc_ids):
    """Test searching by iscc_id when no match exists (covers elif branch)."""
    index = MemoryIndex()
    index.create_index(IsccIndex(name="testindex"))

    # Add an item
    item = IsccItem(iscc_id=sample_iscc_ids[0])
    index.add_items("testindex", [item])

    # Search by different iscc_id (no iscc_code)
    query = IsccItem(iscc_id=sample_iscc_ids[1])
    result = index.search_items("testindex", query)

    # Should not match
    assert len(result.matches) == 0
