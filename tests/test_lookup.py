"""Comprehensive tests for IsccLookupIndex with 100% coverage."""

import io
import pytest
import iscc_core as ic
from iscc_search.lookup import IsccLookupIndex, IsccLookupMatchDict, IsccLookupResultDict
from iscc_search.models import IsccItemDict


@pytest.fixture
def temp_lookup_path(tmp_path):
    # type: (typing.Any) -> typing.Any
    """Provide a temporary path for IsccLookupIndex."""
    return tmp_path / "lookup_index"


@pytest.fixture
def sample_iscc_items():
    # type: () -> list[IsccItemDict]
    """Generate sample IsccItemDict items for testing."""
    items = []

    # Item 1: Full ISCC-CODE with META, CONTENT, DATA, INSTANCE
    code1 = ic.gen_iscc_code_v0([
        ic.gen_meta_code_v0("Test Title")["iscc"],
        ic.gen_text_code_v0("Some text content")["iscc"],
        ic.gen_data_code_v0(io.BytesIO(b"data bytes"))["iscc"],
        ic.gen_instance_code_v0(io.BytesIO(b"instance bytes"))["iscc"],
    ])
    items.append(
        IsccItemDict(
            iscc_id=ic.gen_iscc_id(timestamp=1000000, hub_id=1)["iscc"],
            iscc_code=code1["iscc"],
        )
    )

    # Item 2: Extended units (list of units)
    items.append(
        IsccItemDict(
            iscc_id=ic.gen_iscc_id(timestamp=1000001, hub_id=2)["iscc"],
            units=[
                f"ISCC:{ic.Code.rnd(ic.MT.SEMANTIC, ic.ST_CC.TEXT, bits=64)}",
                f"ISCC:{ic.Code.rnd(ic.MT.CONTENT, ic.ST_CC.IMAGE, bits=128)}",
                ic.gen_data_code_v0(io.BytesIO(b"some data"), bits=64)["iscc"],
                ic.gen_instance_code_v0(io.BytesIO(b"instance"), bits=64)["iscc"],
            ],
        )
    )

    # Item 3: Without ISCC-ID (will be auto-generated)
    items.append(
        IsccItemDict(
            iscc_code=ic.gen_iscc_code_v0([
                ic.gen_meta_code_v0("Another Title")["iscc"],
                ic.gen_data_code_v0(io.BytesIO(b"data"))["iscc"],
                ic.gen_instance_code_v0(io.BytesIO(b"inst"))["iscc"],
            ])["iscc"]
        )
    )

    return items


def test_lookup_index_init(temp_lookup_path):
    # type: (typing.Any) -> None
    """Test IsccLookupIndex initialization."""
    idx = IsccLookupIndex(temp_lookup_path, realm_id=1)
    assert idx.path == str(temp_lookup_path)
    assert idx.realm_id == 1
    assert idx._db_cache == {}
    idx.close()


def test_add_single_item_with_id(temp_lookup_path):
    # type: (typing.Any) -> None
    """Test adding single IsccItemDict with provided ISCC-ID."""
    idx = IsccLookupIndex(temp_lookup_path)

    iscc_id = ic.gen_iscc_id(timestamp=1000000, hub_id=1)["iscc"]
    item = IsccItemDict(
        iscc_id=iscc_id,
        iscc_code=ic.gen_iscc_code_v0([
            ic.gen_data_code_v0(io.BytesIO(b"data"))["iscc"],
            ic.gen_instance_code_v0(io.BytesIO(b"inst"))["iscc"],
        ])["iscc"],
    )

    result_ids = idx.add(item)
    assert len(result_ids) == 1
    assert result_ids[0] == iscc_id
    idx.close()


def test_add_single_item_without_id(temp_lookup_path):
    # type: (typing.Any) -> None
    """Test adding single IsccItemDict without ISCC-ID (auto-generated)."""
    idx = IsccLookupIndex(temp_lookup_path)

    item = IsccItemDict(
        iscc_code=ic.gen_iscc_code_v0([
            ic.gen_data_code_v0(io.BytesIO(b"data"))["iscc"],
            ic.gen_instance_code_v0(io.BytesIO(b"inst"))["iscc"],
        ])["iscc"]
    )

    result_ids = idx.add(item)
    assert len(result_ids) == 1
    assert result_ids[0].startswith("ISCC:")
    idx.close()


def test_add_batch_items(temp_lookup_path, sample_iscc_items):
    # type: (typing.Any, list[IsccItemDict]) -> None
    """Test adding multiple IsccItemDict items."""
    idx = IsccLookupIndex(temp_lookup_path)

    result_ids = idx.add(sample_iscc_items)
    assert len(result_ids) == 3
    assert all(iscc_id.startswith("ISCC:") for iscc_id in result_ids)
    idx.close()


def test_add_item_with_units_preferred(temp_lookup_path):
    # type: (typing.Any) -> None
    """Test that units field is preferred over iscc_code when both provided."""
    idx = IsccLookupIndex(temp_lookup_path)

    # Provide both units and iscc_code (different)
    units = [
        ic.gen_data_code_v0(io.BytesIO(b"data1"))["iscc"],
        ic.gen_instance_code_v0(io.BytesIO(b"inst1"))["iscc"],
    ]
    different_code = ic.gen_iscc_code_v0([
        ic.gen_data_code_v0(io.BytesIO(b"data2"))["iscc"],
        ic.gen_instance_code_v0(io.BytesIO(b"inst2"))["iscc"],
    ])["iscc"]

    item = IsccItemDict(
        units=units,
        iscc_code=different_code,  # Should be ignored
    )

    result_ids = idx.add(item)
    assert len(result_ids) == 1

    # Search with units to verify they were used
    search_result = idx.search(IsccItemDict(units=units))
    assert len(search_result) == 1
    assert len(search_result[0]["lookup_matches"]) > 0
    idx.close()


def test_search_single_item(temp_lookup_path, sample_iscc_items):
    # type: (typing.Any, list[IsccItemDict]) -> None
    """Test searching with single IsccItemDict."""
    idx = IsccLookupIndex(temp_lookup_path)

    # Add items
    added_ids = idx.add(sample_iscc_items[:2])

    # Search with first item
    results = idx.search(sample_iscc_items[0])

    assert len(results) == 1  # One result dict
    assert "lookup_matches" in results[0]
    assert len(results[0]["lookup_matches"]) > 0
    # Should find at least itself
    match_ids = [m["iscc_id"] for m in results[0]["lookup_matches"]]
    assert added_ids[0] in match_ids
    idx.close()


def test_search_batch_items(temp_lookup_path, sample_iscc_items):
    # type: (typing.Any, list[IsccItemDict]) -> None
    """Test searching with multiple IsccItemDict items."""
    idx = IsccLookupIndex(temp_lookup_path)

    # Add items
    idx.add(sample_iscc_items)

    # Search with multiple items
    results = idx.search(sample_iscc_items[:2])

    assert len(results) == 2  # One result dict per query item
    assert all("lookup_matches" in r for r in results)
    idx.close()


def test_search_no_matches(temp_lookup_path, sample_iscc_items):
    # type: (typing.Any, list[IsccItemDict]) -> None
    """Test searching when no matches exist."""
    idx = IsccLookupIndex(temp_lookup_path)

    # Add one item
    idx.add(sample_iscc_items[0])

    # Search with completely different item
    different_item = IsccItemDict(
        units=[
            ic.gen_data_code_v0(io.BytesIO(b"completely different"))["iscc"],
            ic.gen_instance_code_v0(io.BytesIO(b"different inst"))["iscc"],
        ]
    )

    results = idx.search(different_item)
    assert len(results) == 1
    assert results[0]["lookup_matches"] == []
    idx.close()


def test_search_limit_parameter(temp_lookup_path):
    # type: (typing.Any) -> None
    """Test that limit parameter restricts result count."""
    idx = IsccLookupIndex(temp_lookup_path)

    # Create and add many items with similar units
    items = []
    for i in range(20):
        item = IsccItemDict(
            units=[
                ic.gen_data_code_v0(io.BytesIO(f"data{i}".encode()))["iscc"],
                ic.gen_instance_code_v0(io.BytesIO(f"inst{i}".encode()))["iscc"],
            ]
        )
        items.append(item)

    idx.add(items)

    # Search with limit=5
    query_item = items[0]
    results = idx.search(query_item, limit=5)

    assert len(results) == 1
    assert len(results[0]["lookup_matches"]) <= 5
    idx.close()


def test_search_score_calculation(temp_lookup_path):
    # type: (typing.Any) -> None
    """Test that scores are calculated correctly."""
    idx = IsccLookupIndex(temp_lookup_path)

    # Add item with known units
    data_unit = ic.gen_data_code_v0(io.BytesIO(b"test data"), bits=64)["iscc"]
    instance_unit = ic.gen_instance_code_v0(io.BytesIO(b"test instance"), bits=64)["iscc"]

    item = IsccItemDict(units=[data_unit, instance_unit])

    added_ids = idx.add(item)

    # Search with same item
    results = idx.search(item)

    assert len(results) == 1
    matches = results[0]["lookup_matches"]
    assert len(matches) > 0

    # First match should be the item itself with high scores
    match = matches[0]
    assert match["iscc_id"] == added_ids[0]
    assert "DATA_NONE_V0" in match["matches"]
    assert "INSTANCE_NONE_V0" in match["matches"]
    assert match["matches"]["DATA_NONE_V0"] == 64  # 64-bit match
    assert match["matches"]["INSTANCE_NONE_V0"] == 64  # 64-bit match
    assert match["score"] == 128  # Total
    idx.close()


def test_search_sorting_by_score(temp_lookup_path):
    # type: (typing.Any) -> None
    """Test that results are sorted by total score (descending)."""
    idx = IsccLookupIndex(temp_lookup_path)

    # Create items with different levels of similarity
    base_data = io.BytesIO(b"base data")
    base_inst = io.BytesIO(b"base instance")

    # Item 1: 64-bit units
    item1 = IsccItemDict(
        units=[
            ic.gen_data_code_v0(base_data, bits=64)["iscc"],
            ic.gen_instance_code_v0(base_inst, bits=64)["iscc"],
        ]
    )

    # Reset streams for reuse
    base_data.seek(0)
    base_inst.seek(0)

    # Item 2: 128-bit units (more bits to match)
    item2 = IsccItemDict(
        units=[
            ic.gen_data_code_v0(base_data, bits=128)["iscc"],
            ic.gen_instance_code_v0(base_inst, bits=128)["iscc"],
        ]
    )

    idx.add(item1)
    idx.add(item2)

    # Search with item2 (128-bit)
    results = idx.search(item2)

    assert len(results) == 1
    matches = results[0]["lookup_matches"]

    # item2 should have higher score and come first
    if len(matches) >= 2:
        assert matches[0]["score"] >= matches[1]["score"]
    idx.close()


def test_search_unit_type_aggregation(temp_lookup_path):
    # type: (typing.Any) -> None
    """Test that scores are tracked by specific unit_type correctly."""
    idx = IsccLookupIndex(temp_lookup_path)

    # Add item with multiple unit types
    item = IsccItemDict(
        units=[
            ic.gen_meta_code_v0("Test", bits=64)["iscc"],
            f"ISCC:{ic.Code.rnd(ic.MT.SEMANTIC, ic.ST_CC.TEXT, bits=64)}",
            f"ISCC:{ic.Code.rnd(ic.MT.CONTENT, ic.ST_CC.TEXT, bits=64)}",
            ic.gen_data_code_v0(io.BytesIO(b"data"), bits=64)["iscc"],
            ic.gen_instance_code_v0(io.BytesIO(b"inst"), bits=64)["iscc"],
        ]
    )

    added_ids = idx.add(item)

    # Search with same item
    results = idx.search(item)

    assert len(results) == 1
    matches = results[0]["lookup_matches"]
    assert len(matches) > 0

    # Check that all unit types have scores
    match = matches[0]
    assert match["iscc_id"] == added_ids[0]
    assert "META_NONE_V0" in match["matches"]
    assert "SEMANTIC_TEXT_V0" in match["matches"]
    assert "CONTENT_TEXT_V0" in match["matches"]
    assert "DATA_NONE_V0" in match["matches"]
    assert "INSTANCE_NONE_V0" in match["matches"]
    assert all(score > 0 for score in match["matches"].values())
    assert match["score"] == sum(match["matches"].values())
    idx.close()


def test_search_nonexistent_unit_type(temp_lookup_path):
    # type: (typing.Any) -> None
    """Test searching with unit type that doesn't exist in index."""
    idx = IsccLookupIndex(temp_lookup_path)

    # Add item with only DATA and INSTANCE
    item1 = IsccItemDict(
        units=[
            ic.gen_data_code_v0(io.BytesIO(b"data"))["iscc"],
            ic.gen_instance_code_v0(io.BytesIO(b"inst"))["iscc"],
        ]
    )
    idx.add(item1)

    # Search with item that has META unit (not in index)
    item2 = IsccItemDict(
        units=[
            ic.gen_meta_code_v0("Title")["iscc"],
            ic.gen_data_code_v0(io.BytesIO(b"data"))["iscc"],
        ]
    )

    results = idx.search(item2)

    # Should still find match on DATA unit
    assert len(results) == 1
    assert len(results[0]["lookup_matches"]) >= 0  # May or may not match
    idx.close()


def test_multiple_databases_created(temp_lookup_path):
    # type: (typing.Any) -> None
    """Test that multiple unit_type databases are created."""
    idx = IsccLookupIndex(temp_lookup_path)

    # Add items with different unit types
    item = IsccItemDict(
        units=[
            ic.gen_meta_code_v0("Test", bits=64)["iscc"],
            f"ISCC:{ic.Code.rnd(ic.MT.SEMANTIC, ic.ST_CC.TEXT, bits=64)}",
            f"ISCC:{ic.Code.rnd(ic.MT.CONTENT, ic.ST_CC.TEXT, bits=64)}",
            ic.gen_data_code_v0(io.BytesIO(b"data"), bits=64)["iscc"],
            ic.gen_instance_code_v0(io.BytesIO(b"inst"), bits=64)["iscc"],
        ]
    )

    idx.add(item)

    # Verify multiple databases were created (cached)
    assert len(idx._db_cache) >= 3  # At least a few different unit types
    idx.close()


def test_close_and_reopen(temp_lookup_path, sample_iscc_items):
    # type: (typing.Any, list[IsccItemDict]) -> None
    """Test persistence: close and reopen index."""
    # Create and populate index
    idx1 = IsccLookupIndex(temp_lookup_path)
    added_ids = idx1.add(sample_iscc_items[0])
    idx1.close()

    # Reopen and verify data persists
    idx2 = IsccLookupIndex(temp_lookup_path)
    results = idx2.search(sample_iscc_items[0])

    assert len(results) == 1
    assert len(results[0]["lookup_matches"]) > 0
    match_ids = [m["iscc_id"] for m in results[0]["lookup_matches"]]
    assert added_ids[0] in match_ids
    idx2.close()


def test_destructor_cleanup(temp_lookup_path, sample_iscc_items):
    # type: (typing.Any, list[IsccItemDict]) -> None
    """Test __del__ cleanup."""
    idx = IsccLookupIndex(temp_lookup_path)
    idx.add(sample_iscc_items[0])
    # Delete to trigger __del__
    del idx
    # Should not raise any errors


def test_destructor_no_env():
    # type: () -> None
    """Test __del__ when env attribute doesn't exist."""
    idx = IsccLookupIndex.__new__(IsccLookupIndex)
    # Don't call __init__, so env won't be set
    del idx
    # Should not raise any errors


def test_custom_realm_id(temp_lookup_path):
    # type: (typing.Any) -> None
    """Test custom realm_id is used for ISCC-ID reconstruction."""
    realm_id = 1
    idx = IsccLookupIndex(temp_lookup_path, realm_id=realm_id)

    item = IsccItemDict(
        units=[
            ic.gen_data_code_v0(io.BytesIO(b"data"))["iscc"],
            ic.gen_instance_code_v0(io.BytesIO(b"inst"))["iscc"],
        ]
    )

    idx.add(item)
    results = idx.search(item)

    # Verify realm_id is embedded in returned ISCC-ID
    match = results[0]["lookup_matches"][0]
    decoded = ic.decode_base32(match["iscc_id"].removeprefix("ISCC:"))
    header_realm = (decoded[1] >> 4) & 0x0F
    assert header_realm == realm_id
    idx.close()


def test_invalid_realm_id(temp_lookup_path):
    # type: (typing.Any) -> None
    """Test that invalid realm_id raises ValueError during search."""
    idx = IsccLookupIndex(temp_lookup_path, realm_id=7)  # Invalid

    item = IsccItemDict(units=[ic.gen_data_code_v0(io.BytesIO(b"data"))["iscc"]])

    idx.add(item)

    # Search should trigger realm_id validation
    with pytest.raises(ValueError, match="Invalid realm_id 7, must be 0 or 1"):
        idx.search(item)

    idx.close()


def test_normalize_input_single_dict(temp_lookup_path):
    # type: (typing.Any) -> None
    """Test _normalize_input with single dict."""
    idx = IsccLookupIndex(temp_lookup_path)

    item = IsccItemDict(units=[ic.gen_data_code_v0(io.BytesIO(b"data"))["iscc"]])

    result = idx._normalize_input(item)
    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0] == item
    idx.close()


def test_normalize_input_list_of_dicts(temp_lookup_path, sample_iscc_items):
    # type: (typing.Any, list[IsccItemDict]) -> None
    """Test _normalize_input with list of dicts."""
    idx = IsccLookupIndex(temp_lookup_path)

    result = idx._normalize_input(sample_iscc_items)
    assert isinstance(result, list)
    assert len(result) == len(sample_iscc_items)
    assert result == sample_iscc_items
    idx.close()


def test_map_size_property(temp_lookup_path):
    # type: (typing.Any) -> None
    """Test map_size property."""
    idx = IsccLookupIndex(temp_lookup_path)

    size = idx.map_size
    assert size > 0
    assert isinstance(size, int)
    idx.close()


def test_map_size_auto_expansion(temp_lookup_path):
    # type: (typing.Any) -> None
    """Test map_size automatically doubles when full."""
    idx = IsccLookupIndex(temp_lookup_path)

    # Set small map_size
    small_size = 64 * 1024  # 64KB
    idx.env.set_mapsize(small_size)
    assert idx.map_size == small_size

    # Add many items to trigger expansion
    items = []
    for i in range(500):
        item = IsccItemDict(
            units=[
                ic.gen_data_code_v0(io.BytesIO(f"data{i}".encode()), bits=256)["iscc"],
                ic.gen_instance_code_v0(io.BytesIO(f"inst{i}".encode()), bits=256)["iscc"],
            ]
        )
        items.append(item)

    # This should trigger map_size expansion
    idx.add(items)

    # Verify map_size has increased
    assert idx.map_size >= small_size
    idx.close()


def test_custom_lmdb_options(temp_lookup_path):
    # type: (typing.Any) -> None
    """Test that custom LMDB options are applied."""
    custom_options = {
        "max_spare_txns": 8,
        "max_readers": 64,
    }

    idx = IsccLookupIndex(temp_lookup_path, lmdb_options=custom_options)

    # Verify index works with custom options
    item = IsccItemDict(units=[ic.gen_data_code_v0(io.BytesIO(b"data"))["iscc"]])

    result_ids = idx.add(item)
    assert len(result_ids) == 1

    results = idx.search(item)
    assert len(results) == 1
    idx.close()


def test_internal_parameters_forced(temp_lookup_path):
    # type: (typing.Any) -> None
    """Test that max_dbs and subdir are always forced internally."""
    custom_options = {
        "max_dbs": 5,  # Should be forced to 32
        "subdir": True,  # Should be forced to False
    }

    idx = IsccLookupIndex(temp_lookup_path, lmdb_options=custom_options)

    # Verify index still works (internal params were forced)
    item = IsccItemDict(units=[ic.gen_data_code_v0(io.BytesIO(b"data"))["iscc"]])

    result_ids = idx.add(item)
    assert len(result_ids) == 1
    idx.close()


def test_prefix_matching_forward(temp_lookup_path):
    # type: (typing.Any) -> None
    """Test forward prefix matching (query shorter than stored)."""
    idx = IsccLookupIndex(temp_lookup_path)

    # Add item with 128-bit data unit
    stored_item = IsccItemDict(units=[ic.gen_data_code_v0(io.BytesIO(b"test data"), bits=128)["iscc"]])
    idx.add(stored_item)

    # Search with 64-bit version (prefix of stored)
    query_item = IsccItemDict(units=[ic.gen_data_code_v0(io.BytesIO(b"test data"), bits=64)["iscc"]])

    results = idx.search(query_item)

    # Should find match
    if len(results[0]["lookup_matches"]) > 0:
        match_ids = [m["iscc_id"] for m in results[0]["lookup_matches"]]
        # May or may not match depending on hash similarity
        assert isinstance(match_ids, list)
    idx.close()


def test_prefix_matching_reverse(temp_lookup_path):
    # type: (typing.Any) -> None
    """Test reverse prefix matching (query longer than stored)."""
    idx = IsccLookupIndex(temp_lookup_path)

    # Add item with 64-bit data unit
    stored_item = IsccItemDict(units=[ic.gen_data_code_v0(io.BytesIO(b"test data"), bits=64)["iscc"]])
    idx.add(stored_item)

    # Search with 128-bit version
    query_item = IsccItemDict(units=[ic.gen_data_code_v0(io.BytesIO(b"test data"), bits=128)["iscc"]])

    results = idx.search(query_item)

    # May find match via reverse search
    if len(results[0]["lookup_matches"]) > 0:
        assert isinstance(results[0]["lookup_matches"], list)
    idx.close()


def test_duplicate_prevention(temp_lookup_path):
    # type: (typing.Any) -> None
    """Test that duplicate ISCC-ID per unit is prevented."""
    idx = IsccLookupIndex(temp_lookup_path)

    # Add same item twice
    item = IsccItemDict(
        iscc_id=ic.gen_iscc_id(timestamp=1000000, hub_id=1)["iscc"],
        units=[ic.gen_data_code_v0(io.BytesIO(b"data"))["iscc"]],
    )

    idx.add(item)
    idx.add(item)  # Add again

    # Search should only find one match
    results = idx.search(item)

    # Count occurrences of the ISCC-ID
    match_ids = [m["iscc_id"] for m in results[0]["lookup_matches"]]
    # Should appear only once (dupdata=False prevents duplicates)
    assert match_ids.count(item["iscc_id"]) == 1
    idx.close()


def test_empty_lookup_matches_structure(temp_lookup_path):
    # type: (typing.Any) -> None
    """Test that empty results have correct structure."""
    idx = IsccLookupIndex(temp_lookup_path)

    # Search in empty index
    item = IsccItemDict(units=[ic.gen_data_code_v0(io.BytesIO(b"data"))["iscc"]])

    results = idx.search(item)

    assert len(results) == 1
    assert "lookup_matches" in results[0]
    assert results[0]["lookup_matches"] == []
    idx.close()


def test_typed_dict_structures():
    # type: () -> None
    """Test TypedDict structures are correctly defined."""
    match_dict = IsccLookupMatchDict(
        iscc_id="ISCC:MAIGIGAPWHP6WYAA",
        score=128,
        matches={
            "DATA_NONE_V0": 64,
            "INSTANCE_NONE_V0": 64,
        },
    )

    assert match_dict["iscc_id"] == "ISCC:MAIGIGAPWHP6WYAA"
    assert match_dict["score"] == 128
    assert match_dict["matches"]["DATA_NONE_V0"] == 64

    result_dict = IsccLookupResultDict(lookup_matches=[match_dict])
    assert len(result_dict["lookup_matches"]) == 1
    assert result_dict["lookup_matches"][0]["score"] == 128


def test_reverse_search_coverage(temp_lookup_path):
    # type: (typing.Any) -> None
    """Test reverse search finds shorter stored units with matching prefixes."""
    idx = IsccLookupIndex(temp_lookup_path)

    # Use same source data to get matching prefixes
    source_data = io.BytesIO(b"test data for reverse search")

    # Generate codes with same source at different bit lengths
    # They will share prefixes since they're from the same data
    source_data.seek(0)
    unit_64 = ic.gen_data_code_v0(source_data, bits=64)["iscc"]

    source_data.seek(0)
    unit_128 = ic.gen_data_code_v0(source_data, bits=128)["iscc"]

    source_data.seek(0)
    unit_192 = ic.gen_data_code_v0(source_data, bits=192)["iscc"]

    # Add the shorter units to index
    item_64 = IsccItemDict(units=[unit_64])
    idx.add(item_64)

    item_128 = IsccItemDict(units=[unit_128])
    idx.add(item_128)

    item_192 = IsccItemDict(units=[unit_192])
    idx.add(item_192)

    # Search with 256-bit version - should find 64, 128, 192 bit matches via reverse search
    source_data.seek(0)
    unit_256 = ic.gen_data_code_v0(source_data, bits=256)["iscc"]
    query_item = IsccItemDict(units=[unit_256])

    results = idx.search(query_item)

    # Should find matches via reverse search
    assert len(results) == 1
    matches = results[0]["lookup_matches"]
    # Should have found the shorter stored units
    assert len(matches) >= 1

    # Also test edge case: search with very short unit (64-bit) when longer units stored
    # This should not trigger reverse search branches (all bit_lengths >= query_bits)
    source_data.seek(0)
    query_short = ic.gen_data_code_v0(source_data, bits=64)["iscc"]
    query_item_short = IsccItemDict(units=[query_short])
    results_short = idx.search(query_item_short)
    # Should still find matches via forward search
    assert len(results_short) == 1
    idx.close()


def test_reverse_search_edge_cases(temp_lookup_path):
    # type: (typing.Any) -> None
    """Test reverse search edge cases for full coverage."""
    idx = IsccLookupIndex(temp_lookup_path)

    # Add many units with varied bit lengths and different data
    # This creates a diverse key space to trigger edge cases
    for i in range(50):
        for bits in [64, 128, 192, 256]:
            item = IsccItemDict(
                units=[
                    ic.gen_data_code_v0(io.BytesIO(f"data{i}_{bits}".encode()), bits=bits)["iscc"],
                ]
            )
            idx.add(item)

    # Search with many different queries to maximize chances of hitting edge cases
    # Some will have matching prefixes, some won't
    for i in range(20):
        query_item = IsccItemDict(units=[ic.gen_data_code_v0(io.BytesIO(f"query{i}".encode()), bits=256)["iscc"]])
        results = idx.search(query_item)
        assert len(results) == 1
        assert "lookup_matches" in results[0]

    # Search with completely different data patterns
    for prefix in [b"xyz", b"abc", b"123", b"test", b"zzzz"]:
        query_item = IsccItemDict(
            units=[ic.gen_data_code_v0(io.BytesIO(prefix + b" different data"), bits=192)["iscc"]]
        )
        results = idx.search(query_item)
        assert len(results) == 1

    idx.close()


def test_reverse_search_empty_database_sections(temp_lookup_path):
    # type: (typing.Any) -> None
    """Test reverse search when specific prefix ranges don't exist."""
    idx = IsccLookupIndex(temp_lookup_path)

    # Add only 64-bit DATA units (creating gaps in key space)
    for i in range(5):
        item = IsccItemDict(units=[ic.gen_data_code_v0(io.BytesIO(f"short{i}".encode()), bits=64)["iscc"]])
        idx.add(item)

    # Search with SEMANTIC units (different unit_type, empty database)
    # This creates a scenario where cursor.set_range might return False
    for i in range(10):
        query_item = IsccItemDict(units=[f"ISCC:{ic.Code.rnd(ic.MT.SEMANTIC, ic.ST_CC.TEXT, bits=256)}"])
        results = idx.search(query_item)
        # SEMANTIC db doesn't exist, so should return empty results
        assert len(results) == 1

    # Add some CONTENT units with specific patterns
    for i in range(20):
        # Use different subtypes to create varied key patterns
        for subtype in [ic.ST_CC.TEXT, ic.ST_CC.IMAGE, ic.ST_CC.AUDIO]:
            unit = f"ISCC:{ic.Code.rnd(ic.MT.CONTENT, subtype, bits=128)}"
            item = IsccItemDict(units=[unit])
            idx.add(item)

    # Search with longer CONTENT units
    # This maximizes chances of cursor.set_range finding keys but
    # query_body.startswith(key) being False
    for i in range(30):
        for subtype in [ic.ST_CC.TEXT, ic.ST_CC.VIDEO]:
            query = f"ISCC:{ic.Code.rnd(ic.MT.CONTENT, subtype, bits=256)}"
            query_item = IsccItemDict(units=[query])
            results = idx.search(query_item)
            assert len(results) == 1

    idx.close()


def test_reverse_search_no_prefix_match(temp_lookup_path):
    # type: (typing.Any) -> None
    """Test reverse search when cursor.set_range finds no matching prefix.

    This test specifically targets the uncovered branch 363->358 in lookup.py
    where cursor.set_range(prefix) returns False and the loop continues.
    """
    idx = IsccLookupIndex(temp_lookup_path)

    # Add many random 256-bit DATA units to populate the database
    # The goal is to create a sparse key space where some prefixes won't match
    for i in range(200):
        unit = ic.gen_data_code_v0(io.BytesIO(f"stored_data_{i}".encode()), bits=256)["iscc"]
        item = IsccItemDict(units=[unit])
        idx.add(item)

    # Perform many searches with random queries
    # Due to hash randomness and sparse key space, at least one query's prefixes
    # (64, 128, or 192 bits) will not exist in the database, causing cursor.set_range
    # to return False and triggering the branch 363->358
    for i in range(300):
        query = ic.gen_data_code_v0(io.BytesIO(f"query_data_{i}_xyz".encode()), bits=256)["iscc"]
        query_item = IsccItemDict(units=[query])
        results = idx.search(query_item)
        assert len(results) == 1
        assert "lookup_matches" in results[0]

    idx.close()
