"""Tests for IsccItem class."""

import iscc_core as ic
import msgspec
import pytest
from io import BytesIO

from iscc_vdb.types import IsccItem, IsccID, IsccCode, IsccUnit, split_iscc_sequence


def test_direct_construction_with_bytes():
    # type: () -> None
    """Test direct construction with id_data and units_data as bytes."""
    # Create ISCC-ID digest
    iscc_id_str = ic.gen_iscc_id(timestamp=1000000, hub_id=5, realm_id=0)["iscc"]
    id_data = ic.decode_base32(iscc_id_str.removeprefix("ISCC:"))

    # Create units data
    meta = ic.gen_meta_code("Test")
    data = ic.gen_data_code(BytesIO(b"Test"))
    instance = ic.gen_instance_code(BytesIO(b"Test"))

    meta_digest = ic.decode_base32(meta["iscc"].removeprefix("ISCC:"))
    data_digest = ic.decode_base32(data["iscc"].removeprefix("ISCC:"))
    instance_digest = ic.decode_base32(instance["iscc"].removeprefix("ISCC:"))
    units_data = meta_digest + data_digest + instance_digest

    # Direct construction
    item = IsccItem(id_data, units_data)

    assert isinstance(item, IsccItem)
    assert item.id_data == id_data
    assert item.units_data == units_data


def test_new_with_iscc_id_and_iscc_code_strings():
    # type: () -> None
    """Test factory method new() with iscc_id + iscc_code as strings."""
    # Create ISCC components
    iscc_id_str = ic.gen_iscc_id(timestamp=1000000, hub_id=5, realm_id=0)["iscc"]
    meta = ic.gen_meta_code("Test Title")
    data = ic.gen_data_code(BytesIO(b"Test Data"))
    instance = ic.gen_instance_code(BytesIO(b"Test Data"))
    code = ic.gen_iscc_code([meta["iscc"], data["iscc"], instance["iscc"]])

    # Create item
    item = IsccItem.new(iscc_id_str, iscc_code=code["iscc"])

    assert isinstance(item, IsccItem)
    assert item.iscc_id == iscc_id_str
    # Units should match decomposed code
    expected_units = ic.iscc_decompose(code["iscc"])
    assert len(item.units) == len(expected_units)


def test_new_with_iscc_id_and_units_strings():
    # type: () -> None
    """Test factory method new() with iscc_id + units as list of strings."""
    # Create ISCC components
    iscc_id_str = ic.gen_iscc_id(timestamp=1000000, hub_id=5, realm_id=0)["iscc"]
    meta = ic.gen_meta_code("Test Title")
    data = ic.gen_data_code(BytesIO(b"Test Data"))
    instance = ic.gen_instance_code(BytesIO(b"Test Data"))

    units = [meta["iscc"], data["iscc"], instance["iscc"]]

    # Create item
    item = IsccItem.new(iscc_id_str, units=units)

    assert isinstance(item, IsccItem)
    assert item.iscc_id == iscc_id_str
    assert len(item.units) == 3


def test_new_with_bytes_inputs():
    # type: () -> None
    """Test factory method new() with bytes inputs."""
    # Create ISCC components as bytes
    iscc_id_str = ic.gen_iscc_id(timestamp=1000000, hub_id=5, realm_id=0)["iscc"]
    id_bytes = ic.decode_base32(iscc_id_str.removeprefix("ISCC:"))

    meta = ic.gen_meta_code("Test")
    data = ic.gen_data_code(BytesIO(b"Test"))
    instance = ic.gen_instance_code(BytesIO(b"Test"))
    code = ic.gen_iscc_code([meta["iscc"], data["iscc"], instance["iscc"]])
    code_bytes = ic.decode_base32(code["iscc"].removeprefix("ISCC:"))

    # Create item with bytes
    item = IsccItem.new(id_bytes, iscc_code=code_bytes)

    assert isinstance(item, IsccItem)
    assert item.iscc_id == iscc_id_str


def test_new_with_mixed_string_and_bytes():
    # type: () -> None
    """Test factory method new() with mixed string and bytes inputs."""
    # String ISCC-ID, bytes units
    iscc_id_str = ic.gen_iscc_id(timestamp=1000000, hub_id=5, realm_id=0)["iscc"]
    meta = ic.gen_meta_code("Test")
    data = ic.gen_data_code(BytesIO(b"Test"))
    instance = ic.gen_instance_code(BytesIO(b"Test"))

    units_bytes = [
        ic.decode_base32(meta["iscc"].removeprefix("ISCC:")),
        ic.decode_base32(data["iscc"].removeprefix("ISCC:")),
        ic.decode_base32(instance["iscc"].removeprefix("ISCC:")),
    ]

    item = IsccItem.new(iscc_id_str, units=units_bytes)

    assert isinstance(item, IsccItem)
    assert item.iscc_id == iscc_id_str
    assert len(item.units) == 3


def test_new_raises_value_error_without_code_or_units():
    # type: () -> None
    """Test new() raises ValueError when neither code nor units provided."""
    iscc_id_str = ic.gen_iscc_id(timestamp=1000000, hub_id=5, realm_id=0)["iscc"]

    with pytest.raises(ValueError, match="Either iscc_code or iscc_units must be provided"):
        IsccItem.new(iscc_id_str)


def test_new_raises_value_error_with_none_values():
    # type: () -> None
    """Test new() raises ValueError when both code and units are None."""
    iscc_id_str = ic.gen_iscc_id(timestamp=1000000, hub_id=5, realm_id=0)["iscc"]

    with pytest.raises(ValueError, match="Either iscc_code or iscc_units must be provided"):
        IsccItem.new(iscc_id_str, iscc_code=None, units=None)


def test_iscc_id_property_returns_canonical_string():
    # type: () -> None
    """Test iscc_id property returns canonical string with ISCC: prefix."""
    iscc_id_str = ic.gen_iscc_id(timestamp=1000000, hub_id=5, realm_id=0)["iscc"]
    meta = ic.gen_meta_code("Test")
    data = ic.gen_data_code(BytesIO(b"Test"))
    instance = ic.gen_instance_code(BytesIO(b"Test"))

    item = IsccItem.new(iscc_id_str, units=[meta["iscc"], data["iscc"], instance["iscc"]])

    assert item.iscc_id == iscc_id_str
    assert item.iscc_id.startswith("ISCC:")


def test_iscc_code_property_computes_wide_code():
    # type: () -> None
    """Test iscc_code property computes wide code from units."""
    iscc_id_str = ic.gen_iscc_id(timestamp=1000000, hub_id=5, realm_id=0)["iscc"]
    meta = ic.gen_meta_code("Test Title")
    data = ic.gen_data_code(BytesIO(b"Test Data"))
    instance = ic.gen_instance_code(BytesIO(b"Test Data"))
    code = ic.gen_iscc_code([meta["iscc"], data["iscc"], instance["iscc"]])

    item = IsccItem.new(iscc_id_str, iscc_code=code["iscc"])

    # Computed iscc_code should be wide version
    expected_units = ic.iscc_decompose(code["iscc"])
    expected_wide_code = ic.gen_iscc_code_v0(expected_units, wide=True)["iscc"]

    assert item.iscc_code == expected_wide_code
    assert item.iscc_code.startswith("ISCC:")


def test_units_property_returns_list_of_strings():
    # type: () -> None
    """Test units property returns list of ISCC unit strings."""
    iscc_id_str = ic.gen_iscc_id(timestamp=1000000, hub_id=5, realm_id=0)["iscc"]
    meta = ic.gen_meta_code("Test")
    data = ic.gen_data_code(BytesIO(b"Test"))
    instance = ic.gen_instance_code(BytesIO(b"Test"))

    units_list = [meta["iscc"], data["iscc"], instance["iscc"]]
    item = IsccItem.new(iscc_id_str, units=units_list)

    units = item.units

    assert isinstance(units, list)
    assert len(units) == 3
    for unit in units:
        assert isinstance(unit, str)
        assert unit.startswith("ISCC:")


def test_units_property_decomposes_correctly():
    # type: () -> None
    """Test units property correctly decomposes units_data."""
    iscc_id_str = ic.gen_iscc_id(timestamp=1000000, hub_id=5, realm_id=0)["iscc"]
    meta = ic.gen_meta_code("Test")
    data = ic.gen_data_code(BytesIO(b"Test"))
    instance = ic.gen_instance_code(BytesIO(b"Test"))

    original_units = [meta["iscc"], data["iscc"], instance["iscc"]]
    item = IsccItem.new(iscc_id_str, units=original_units)

    # Get decomposed units
    decomposed = item.units

    # Should match original (order and content)
    assert len(decomposed) == len(original_units)
    for i, unit in enumerate(decomposed):
        assert unit == original_units[i]


def test_properties_return_consistent_values():
    # type: () -> None
    """Test properties return consistent values on multiple calls."""
    iscc_id_str = ic.gen_iscc_id(timestamp=1000000, hub_id=5, realm_id=0)["iscc"]
    meta = ic.gen_meta_code("Test")
    data = ic.gen_data_code(BytesIO(b"Test"))
    instance = ic.gen_instance_code(BytesIO(b"Test"))

    item = IsccItem.new(iscc_id_str, units=[meta["iscc"], data["iscc"], instance["iscc"]])

    # Call properties multiple times
    id1 = item.iscc_id
    id2 = item.iscc_id
    code1 = item.iscc_code
    code2 = item.iscc_code
    units1 = item.units
    units2 = item.units

    # Properties should return same values (no caching for memory efficiency)
    assert id1 == id2
    assert code1 == code2
    assert units1 == units2


def test_dict_property_returns_typed_dict():
    # type: () -> None
    """Test dict property returns IsccItemDict with all fields."""
    iscc_id_str = ic.gen_iscc_id(timestamp=1000000, hub_id=5, realm_id=0)["iscc"]
    meta = ic.gen_meta_code("Test Title")
    data = ic.gen_data_code(BytesIO(b"Test Data"))
    instance = ic.gen_instance_code(BytesIO(b"Test Data"))

    item = IsccItem.new(iscc_id_str, units=[meta["iscc"], data["iscc"], instance["iscc"]])

    item_dict = item.dict

    assert isinstance(item_dict, dict)
    assert "iscc_id" in item_dict
    assert "iscc_code" in item_dict
    assert "units" in item_dict
    assert item_dict["iscc_id"] == item.iscc_id
    assert item_dict["iscc_code"] == item.iscc_code
    assert item_dict["units"] == item.units


def test_json_property_returns_bytes():
    # type: () -> None
    """Test json property returns msgspec JSON bytes."""
    iscc_id_str = ic.gen_iscc_id(timestamp=1000000, hub_id=5, realm_id=0)["iscc"]
    meta = ic.gen_meta_code("Test")
    data = ic.gen_data_code(BytesIO(b"Test"))
    instance = ic.gen_instance_code(BytesIO(b"Test"))

    item = IsccItem.new(iscc_id_str, units=[meta["iscc"], data["iscc"], instance["iscc"]])

    json_bytes = item.json

    assert isinstance(json_bytes, bytes)
    # Should be valid JSON
    decoded = msgspec.json.decode(json_bytes)
    assert isinstance(decoded, dict)


def test_json_roundtrip_decodes_correctly():
    # type: () -> None
    """Test JSON can be decoded back to dict with correct data."""
    iscc_id_str = ic.gen_iscc_id(timestamp=1000000, hub_id=5, realm_id=0)["iscc"]
    meta = ic.gen_meta_code("Test")
    data = ic.gen_data_code(BytesIO(b"Test"))
    instance = ic.gen_instance_code(BytesIO(b"Test"))

    item = IsccItem.new(iscc_id_str, units=[meta["iscc"], data["iscc"], instance["iscc"]])

    # Encode to JSON
    json_bytes = item.json

    # Decode back
    decoded = msgspec.json.decode(json_bytes)

    # Should match dict property
    assert decoded["iscc_id"] == item.iscc_id
    assert decoded["iscc_code"] == item.iscc_code
    assert decoded["units"] == item.units


def test_roundtrip_dict_to_new_to_dict():
    # type: () -> None
    """Test roundtrip: dict -> new() -> dict produces equivalent result."""
    # Create original item
    iscc_id_str = ic.gen_iscc_id(timestamp=1000000, hub_id=5, realm_id=0)["iscc"]
    meta = ic.gen_meta_code("Test Title")
    data = ic.gen_data_code(BytesIO(b"Test Data"))
    instance = ic.gen_instance_code(BytesIO(b"Test Data"))

    original_item = IsccItem.new(iscc_id_str, units=[meta["iscc"], data["iscc"], instance["iscc"]])
    original_dict = original_item.dict

    # Reconstruct from dict data
    reconstructed_item = IsccItem.new(original_dict["iscc_id"], units=original_dict["units"])
    reconstructed_dict = reconstructed_item.dict

    # Dicts should be equivalent
    assert reconstructed_dict["iscc_id"] == original_dict["iscc_id"]
    assert reconstructed_dict["units"] == original_dict["units"]
    # Note: iscc_code may differ (standard vs wide) but units are preserved


def test_frozen_struct_is_immutable():
    # type: () -> None
    """Test IsccItem is frozen (immutable)."""
    iscc_id_str = ic.gen_iscc_id(timestamp=1000000, hub_id=5, realm_id=0)["iscc"]
    meta = ic.gen_meta_code("Test")
    data = ic.gen_data_code(BytesIO(b"Test"))
    instance = ic.gen_instance_code(BytesIO(b"Test"))

    item = IsccItem.new(iscc_id_str, units=[meta["iscc"], data["iscc"], instance["iscc"]])

    # Attempting to modify should raise AttributeError
    with pytest.raises(AttributeError):
        item.id_data = b"new_data"  # type: ignore

    with pytest.raises(AttributeError):
        item.units_data = b"new_units"  # type: ignore


def test_variable_length_units_64_bits():
    # type: () -> None
    """Test with 64-bit units."""
    iscc_id_str = ic.gen_iscc_id(timestamp=1000000, hub_id=5, realm_id=0)["iscc"]
    meta = ic.gen_meta_code("Test", bits=64)
    data = ic.gen_data_code(BytesIO(b"Test"), bits=64)
    instance = ic.gen_instance_code(BytesIO(b"Test"), bits=64)

    item = IsccItem.new(iscc_id_str, units=[meta["iscc"], data["iscc"], instance["iscc"]])

    # Verify units
    units = item.units
    assert len(units) == 3
    for unit in units:
        unit_obj = IsccUnit(unit)
        assert len(unit_obj) == 64


def test_variable_length_units_128_bits():
    # type: () -> None
    """Test with 128-bit units."""
    iscc_id_str = ic.gen_iscc_id(timestamp=1000000, hub_id=5, realm_id=0)["iscc"]
    meta = ic.gen_meta_code("Test", bits=128)
    data = ic.gen_data_code(BytesIO(b"Test"), bits=128)
    instance = ic.gen_instance_code(BytesIO(b"Test"), bits=128)

    item = IsccItem.new(iscc_id_str, units=[meta["iscc"], data["iscc"], instance["iscc"]])

    # Verify units
    units = item.units
    assert len(units) == 3
    for unit in units:
        unit_obj = IsccUnit(unit)
        assert len(unit_obj) == 128


def test_variable_length_units_mixed():
    # type: () -> None
    """Test with mixed-length units (64, 128, 192, 256 bits)."""
    iscc_id_str = ic.gen_iscc_id(timestamp=1000000, hub_id=5, realm_id=0)["iscc"]
    meta_64 = ic.gen_meta_code("Test", bits=64)
    data_128 = ic.gen_data_code(BytesIO(b"Test"), bits=128)
    instance_256 = ic.gen_instance_code(BytesIO(b"Test"), bits=256)

    item = IsccItem.new(iscc_id_str, units=[meta_64["iscc"], data_128["iscc"], instance_256["iscc"]])

    # Verify units
    units = item.units
    assert len(units) == 3

    # Check individual lengths
    assert len(IsccUnit(units[0])) == 64
    assert len(IsccUnit(units[1])) == 128
    assert len(IsccUnit(units[2])) == 256


def test_binary_data_is_compact():
    # type: () -> None
    """Test binary data (id_data, units_data) is compact."""
    iscc_id_str = ic.gen_iscc_id(timestamp=1000000, hub_id=5, realm_id=0)["iscc"]
    meta = ic.gen_meta_code("Test", bits=64)
    data = ic.gen_data_code(BytesIO(b"Test"), bits=64)
    instance = ic.gen_instance_code(BytesIO(b"Test"), bits=64)

    item = IsccItem.new(iscc_id_str, units=[meta["iscc"], data["iscc"], instance["iscc"]])

    # Check id_data size (2-byte header + 8-byte body = 10 bytes)
    assert len(item.id_data) == 10

    # Check units_data size
    # 3 units x (2-byte header + 8-byte body) = 30 bytes
    assert len(item.units_data) == 30


def test_with_sample_iscc_ids_fixture(sample_iscc_ids):
    # type: (list[str]) -> None
    """Test with sample_iscc_ids fixture."""
    meta = ic.gen_meta_code("Test")
    data = ic.gen_data_code(BytesIO(b"Test"))
    instance = ic.gen_instance_code(BytesIO(b"Test"))

    for iscc_id_str in sample_iscc_ids:
        item = IsccItem.new(iscc_id_str, units=[meta["iscc"], data["iscc"], instance["iscc"]])

        assert item.iscc_id == iscc_id_str
        assert len(item.units) == 3


def test_with_meta_units_fixture(sample_meta_units):
    # type: (list[str]) -> None
    """Test with sample_meta_units fixture (variable lengths)."""
    iscc_id_str = ic.gen_iscc_id(timestamp=1000000, hub_id=5, realm_id=0)["iscc"]

    # Use first 3 meta units of different lengths
    data = ic.gen_data_code(BytesIO(b"Test"))
    instance = ic.gen_instance_code(BytesIO(b"Test"))
    units = [sample_meta_units[0], data["iscc"], instance["iscc"]]

    item = IsccItem.new(iscc_id_str, units=units)

    assert len(item.units) == 3
    assert item.units[0] == sample_meta_units[0]


def test_with_iscc_code_from_fixture():
    # type: () -> None
    """Test construction with ISCC-CODE from multiple unit types."""
    iscc_id_str = ic.gen_iscc_id(timestamp=1000000, hub_id=5, realm_id=0)["iscc"]

    # Create full ISCC-CODE with multiple components
    meta = ic.gen_meta_code("Test Title")
    semantic = ic.gen_text_code("Test content for semantic hashing")
    data = ic.gen_data_code(BytesIO(b"Test Data"))
    instance = ic.gen_instance_code(BytesIO(b"Test Data"))

    code = ic.gen_iscc_code([meta["iscc"], semantic["iscc"], data["iscc"], instance["iscc"]])

    item = IsccItem.new(iscc_id_str, iscc_code=code["iscc"])

    # Should decompose to 4 units
    assert len(item.units) == 4


def test_split_iscc_sequence_helper():
    # type: () -> None
    """Test split_iscc_sequence helper function."""
    # Create concatenated ISCC digests
    meta = ic.gen_meta_code("Test")
    data = ic.gen_data_code(BytesIO(b"Test"))
    instance = ic.gen_instance_code(BytesIO(b"Test"))

    meta_digest = ic.decode_base32(meta["iscc"].removeprefix("ISCC:"))
    data_digest = ic.decode_base32(data["iscc"].removeprefix("ISCC:"))
    instance_digest = ic.decode_base32(instance["iscc"].removeprefix("ISCC:"))

    combined = meta_digest + data_digest + instance_digest

    # Split
    units = split_iscc_sequence(combined)

    assert len(units) == 3
    assert units[0] == meta_digest
    assert units[1] == data_digest
    assert units[2] == instance_digest


def test_new_prefers_units_over_code():
    # type: () -> None
    """Test new() uses units when both code and units are provided."""
    iscc_id_str = ic.gen_iscc_id(timestamp=1000000, hub_id=5, realm_id=0)["iscc"]

    # Create different units and code
    meta1 = ic.gen_meta_code("Title 1")
    data1 = ic.gen_data_code(BytesIO(b"Data 1"))
    instance1 = ic.gen_instance_code(BytesIO(b"Data 1"))
    code1 = ic.gen_iscc_code([meta1["iscc"], data1["iscc"], instance1["iscc"]])

    meta2 = ic.gen_meta_code("Title 2")
    data2 = ic.gen_data_code(BytesIO(b"Data 2"))
    instance2 = ic.gen_instance_code(BytesIO(b"Data 2"))
    units2 = [meta2["iscc"], data2["iscc"], instance2["iscc"]]

    # Create item with both (units should win)
    item = IsccItem.new(iscc_id_str, iscc_code=code1["iscc"], units=units2)

    # Should use units2, not code1
    assert item.units == units2


def test_empty_prefix_handling():
    # type: () -> None
    """Test handling of ISCC strings without prefix."""
    iscc_id_str = ic.gen_iscc_id(timestamp=1000000, hub_id=5, realm_id=0)["iscc"]
    iscc_id_no_prefix = iscc_id_str.removeprefix("ISCC:")

    meta = ic.gen_meta_code("Test")
    data = ic.gen_data_code(BytesIO(b"Test"))
    instance = ic.gen_instance_code(BytesIO(b"Test"))

    # Create with no prefix
    item = IsccItem.new(iscc_id_no_prefix, units=[meta["iscc"], data["iscc"], instance["iscc"]])

    # Should still produce canonical form with prefix
    assert item.iscc_id == iscc_id_str
    assert item.iscc_id.startswith("ISCC:")


def test_msgspec_struct_properties():
    # type: () -> None
    """Test msgspec.Struct properties (frozen, array_like)."""
    iscc_id_str = ic.gen_iscc_id(timestamp=1000000, hub_id=5, realm_id=0)["iscc"]
    meta = ic.gen_meta_code("Test")
    data = ic.gen_data_code(BytesIO(b"Test"))
    instance = ic.gen_instance_code(BytesIO(b"Test"))

    item = IsccItem.new(iscc_id_str, units=[meta["iscc"], data["iscc"], instance["iscc"]])

    # Verify it's a msgspec.Struct
    assert isinstance(item, msgspec.Struct)

    # Test that it can be encoded/decoded with msgspec
    encoded = msgspec.msgpack.encode(item)
    decoded = msgspec.msgpack.decode(encoded, type=IsccItem)

    assert decoded.id_data == item.id_data
    assert decoded.units_data == item.units_data


def test_edge_case_minimum_units():
    # type: () -> None
    """Test with minimum required units (DATA + INSTANCE)."""
    iscc_id_str = ic.gen_iscc_id(timestamp=1000000, hub_id=5, realm_id=0)["iscc"]
    data = ic.gen_data_code(BytesIO(b"Test"))
    instance = ic.gen_instance_code(BytesIO(b"Test"))

    item = IsccItem.new(iscc_id_str, units=[data["iscc"], instance["iscc"]])

    assert len(item.units) == 2


def test_edge_case_maximum_components():
    # type: () -> None
    """Test with maximum components (META + SEMANTIC + CONTENT + DATA + INSTANCE)."""
    iscc_id_str = ic.gen_iscc_id(timestamp=1000000, hub_id=5, realm_id=0)["iscc"]

    # Note: ISCC-CODE typically supports META + CONTENT OR SEMANTIC, but not both
    # Let's test with META + CONTENT + DATA + INSTANCE
    meta = ic.gen_meta_code("Title")
    content = ic.gen_text_code("Content text")
    data = ic.gen_data_code(BytesIO(b"Data"))
    instance = ic.gen_instance_code(BytesIO(b"Data"))

    code = ic.gen_iscc_code([meta["iscc"], content["iscc"], data["iscc"], instance["iscc"]])
    item = IsccItem.new(iscc_id_str, iscc_code=code["iscc"])

    # Should have 4 units
    assert len(item.units) == 4


def test_dict_keys_match_typed_dict():
    # type: () -> None
    """Test dict property returns exactly the keys defined in IsccItemDict."""
    iscc_id_str = ic.gen_iscc_id(timestamp=1000000, hub_id=5, realm_id=0)["iscc"]
    meta = ic.gen_meta_code("Test")
    data = ic.gen_data_code(BytesIO(b"Test"))
    instance = ic.gen_instance_code(BytesIO(b"Test"))

    item = IsccItem.new(iscc_id_str, units=[meta["iscc"], data["iscc"], instance["iscc"]])
    item_dict = item.dict

    # Should have exactly these keys
    expected_keys = {"iscc_id", "iscc_code", "units"}
    assert set(item_dict.keys()) == expected_keys


def test_multiple_items_with_same_data():
    # type: () -> None
    """Test creating multiple IsccItem instances with same data."""
    iscc_id_str = ic.gen_iscc_id(timestamp=1000000, hub_id=5, realm_id=0)["iscc"]
    meta = ic.gen_meta_code("Test")
    data = ic.gen_data_code(BytesIO(b"Test"))
    instance = ic.gen_instance_code(BytesIO(b"Test"))

    # Create two items with same data
    item1 = IsccItem.new(iscc_id_str, units=[meta["iscc"], data["iscc"], instance["iscc"]])
    item2 = IsccItem.new(iscc_id_str, units=[meta["iscc"], data["iscc"], instance["iscc"]])

    # Should be independent objects
    assert item1 is not item2

    # But have same content
    assert item1.id_data == item2.id_data
    assert item1.units_data == item2.units_data
    assert item1.dict == item2.dict


def test_from_dict_with_iscc_id_and_units():
    # type: () -> None
    """Test from_dict with iscc_id and units provided."""
    iscc_id_str = ic.gen_iscc_id(timestamp=1000000, hub_id=5, realm_id=0)["iscc"]
    meta = ic.gen_meta_code("Test")
    data = ic.gen_data_code(BytesIO(b"Test"))
    instance = ic.gen_instance_code(BytesIO(b"Test"))

    data_dict = {"iscc_id": iscc_id_str, "units": [meta["iscc"], data["iscc"], instance["iscc"]]}

    item = IsccItem.from_dict(data_dict)

    assert item.iscc_id == iscc_id_str
    assert len(item.units) == 3


def test_from_dict_without_iscc_id():
    # type: () -> None
    """Test from_dict generates random iscc_id when not provided."""
    meta = ic.gen_meta_code("Test")
    data = ic.gen_data_code(BytesIO(b"Test"))
    instance = ic.gen_instance_code(BytesIO(b"Test"))

    data_dict = {"units": [meta["iscc"], data["iscc"], instance["iscc"]]}

    item = IsccItem.from_dict(data_dict)

    assert item.iscc_id.startswith("ISCC:")
    assert len(item.units) == 3


def test_from_dict_with_iscc_code():
    # type: () -> None
    """Test from_dict with iscc_code instead of units."""
    iscc_id_str = ic.gen_iscc_id(timestamp=1000000, hub_id=5, realm_id=0)["iscc"]
    meta = ic.gen_meta_code("Test")
    data = ic.gen_data_code(BytesIO(b"Test"))
    instance = ic.gen_instance_code(BytesIO(b"Test"))
    code = ic.gen_iscc_code([meta["iscc"], data["iscc"], instance["iscc"]])

    data_dict = {"iscc_id": iscc_id_str, "iscc_code": code["iscc"]}

    item = IsccItem.from_dict(data_dict)

    assert item.iscc_id == iscc_id_str
    assert len(item.units) == 3


def test_from_dict_raises_without_code_or_units():
    # type: () -> None
    """Test from_dict raises ValueError when neither code nor units provided."""
    iscc_id_str = ic.gen_iscc_id(timestamp=1000000, hub_id=5, realm_id=0)["iscc"]

    data_dict = {"iscc_id": iscc_id_str}

    with pytest.raises(ValueError, match="Either iscc_code or iscc_units must be provided"):
        IsccItem.from_dict(data_dict)
