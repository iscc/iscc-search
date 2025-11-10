"""
Tests for common index utilities.

Tests serialization, ISCC parsing, validation, and error handling.
"""

import pytest
import iscc_core as ic
from iscc_search.schema import IsccEntry
from iscc_search.indexes import common


def test_serialize_deserialize_asset_roundtrip(sample_assets):
    """Test IsccEntry serialization and deserialization roundtrip."""
    asset = sample_assets[0]

    # Serialize
    data = common.serialize_asset(asset)
    assert isinstance(data, bytes)

    # Deserialize
    restored = common.deserialize_asset(data)

    # Verify equality
    assert restored.iscc_id == asset.iscc_id
    assert restored.iscc_code == asset.iscc_code
    assert restored.units == asset.units
    assert restored.metadata == asset.metadata


def test_serialize_asset_minimal(sample_iscc_ids):
    """Test serialization with minimal asset (only iscc_id)."""
    asset = IsccEntry(iscc_id=sample_iscc_ids[0])

    data = common.serialize_asset(asset)
    restored = common.deserialize_asset(data)

    assert restored.iscc_id == asset.iscc_id
    assert restored.iscc_code is None
    assert restored.units is None
    assert restored.metadata is None


def test_deserialize_asset_invalid_json():
    """Test deserialization with invalid JSON."""
    with pytest.raises(Exception):  # JSONDecodeError or ValueError
        common.deserialize_asset(b"not valid json{")


def test_extract_iscc_id_body(sample_iscc_ids):
    """Test extracting 8-byte body from ISCC-ID."""
    iscc_id = sample_iscc_ids[0]
    body = common.extract_iscc_id_body(iscc_id)

    assert isinstance(body, bytes)
    assert len(body) == 8  # ISCC-ID body is always 8 bytes


def test_extract_iscc_id_body_invalid():
    """Test extracting body with invalid ISCC-ID."""
    with pytest.raises(ValueError, match="Invalid ISCC-ID"):
        common.extract_iscc_id_body("NOT_AN_ISCC")


def test_extract_realm_id_realm0(sample_iscc_ids):
    """Test extracting realm_id=0 from ISCC-ID."""
    # sample_iscc_ids from conftest use realm=0
    realm = common.extract_realm_id(sample_iscc_ids[0])
    assert realm == 0


def test_extract_realm_id_realm1():
    """Test extracting realm_id=1 from ISCC-ID."""
    # Generate ISCC-ID with realm=1
    iscc_id = ic.gen_iscc_id(timestamp=1000000, hub_id=1, realm_id=1)["iscc"]
    realm = common.extract_realm_id(iscc_id)
    assert realm == 1


def test_extract_realm_id_invalid():
    """Test extracting realm_id with invalid ISCC-ID."""
    with pytest.raises(ValueError, match="Invalid ISCC-ID"):
        common.extract_realm_id("INVALID")


def test_reconstruct_iscc_id_realm0(sample_iscc_ids):
    """Test reconstructing ISCC-ID from body and realm_id=0."""
    # Extract body from existing ISCC-ID
    iscc_id = sample_iscc_ids[0]
    body = common.extract_iscc_id_body(iscc_id)

    # Reconstruct
    reconstructed = common.reconstruct_iscc_id(body, 0)

    assert reconstructed.startswith("ISCC:")
    # Verify can be decoded
    code_bytes = ic.decode_base32(reconstructed.split(":")[-1])
    assert len(code_bytes) == 10
    assert code_bytes[2:] == body


def test_reconstruct_iscc_id_realm1():
    """Test reconstructing ISCC-ID from body and realm_id=1."""
    # Generate ISCC-ID with realm=1
    iscc_id_realm1 = ic.gen_iscc_id(timestamp=2000000, hub_id=5, realm_id=1)["iscc"]
    body = common.extract_iscc_id_body(iscc_id_realm1)

    # Reconstruct
    reconstructed = common.reconstruct_iscc_id(body, 1)

    assert reconstructed.startswith("ISCC:")
    # Verify realm extracted correctly
    assert common.extract_realm_id(reconstructed) == 1
    # Verify body matches
    assert common.extract_iscc_id_body(reconstructed) == body


def test_reconstruct_iscc_id_invalid_realm():
    """Test reconstruct with invalid realm_id."""
    body = b"\x01\x02\x03\x04\x05\x06\x07\x08"

    with pytest.raises(ValueError, match="Invalid realm_id"):
        common.reconstruct_iscc_id(body, 2)

    with pytest.raises(ValueError, match="Invalid realm_id"):
        common.reconstruct_iscc_id(body, -1)


def test_reconstruct_iscc_id_invalid_body_length():
    """Test reconstruct with wrong body length."""
    with pytest.raises(ValueError, match="must be 8 bytes"):
        common.reconstruct_iscc_id(b"\x01\x02\x03", 0)

    with pytest.raises(ValueError, match="must be 8 bytes"):
        common.reconstruct_iscc_id(b"\x01" * 10, 0)


def test_extract_unit_body(sample_content_units):
    """Test extracting body from ISCC-UNIT."""
    unit = sample_content_units[0]
    body = common.extract_unit_body(unit)

    assert isinstance(body, bytes)
    assert len(body) > 0


def test_extract_unit_body_variable_lengths(sample_meta_units):
    """Test extracting bodies from units of different lengths."""
    # sample_meta_units has units of 64, 128, 192, 256 bits
    for unit in sample_meta_units:
        body = common.extract_unit_body(unit)
        assert isinstance(body, bytes)
        assert len(body) in (8, 16, 24, 32)  # Valid ISCC-UNIT body sizes


def test_extract_unit_body_invalid():
    """Test extracting body from invalid UNIT."""
    with pytest.raises(Exception):  # IsccUnit will raise on invalid format
        common.extract_unit_body("NOT_A_UNIT")


def test_get_unit_type(sample_content_units):
    """Test extracting unit type from ISCC-UNIT."""
    unit = sample_content_units[0]
    unit_type = common.get_unit_type(unit)

    assert isinstance(unit_type, str)
    assert "CONTENT" in unit_type


def test_get_unit_type_different_types(sample_content_units, sample_data_units):
    """Test unit type extraction for different ISCC-UNIT types."""
    # Content unit
    content_type = common.get_unit_type(sample_content_units[0])
    assert "CONTENT" in content_type

    # Data unit
    data_type = common.get_unit_type(sample_data_units[0])
    assert "DATA" in data_type


def test_get_unit_type_invalid():
    """Test unit type extraction with invalid unit."""
    with pytest.raises(Exception):
        common.get_unit_type("INVALID")


def test_validate_index_name_valid():
    """Test index name validation with valid names."""
    valid_names = [
        "abc",
        "test123",
        "myindex",
        "a",  # Single letter
        "index2024",
        "production",
    ]

    for name in valid_names:
        # Should not raise
        common.validate_index_name(name)


def test_validate_index_name_invalid():
    """Test index name validation with invalid names."""
    invalid_names = [
        "123abc",  # Starts with digit
        "Test",  # Uppercase
        "my-index",  # Hyphen
        "my_index",  # Underscore
        "my.index",  # Dot
        "",  # Empty
        "My Index",  # Space
        "ÄBC",  # Non-ASCII
    ]

    for name in invalid_names:
        with pytest.raises(ValueError, match="Invalid index name"):
            common.validate_index_name(name)


def test_validate_iscc_id_valid(sample_iscc_ids):
    """Test ISCC-ID validation with valid ID."""
    # Should not raise
    common.validate_iscc_id(sample_iscc_ids[0])


def test_validate_iscc_id_invalid_format():
    """Test ISCC-ID validation with invalid formats."""
    invalid_ids = [
        "NOT_ISCC",  # Wrong prefix
        "ISCC:",  # No code
        "",  # Empty
        "ISCC:INVALID!!!",  # Invalid base32
    ]

    for invalid_id in invalid_ids:
        with pytest.raises(ValueError, match="Invalid ISCC-ID"):
            common.validate_iscc_id(invalid_id)


def test_validate_iscc_id_wrong_length():
    """Test ISCC-ID validation with wrong length."""
    # Create code with wrong length (should be 10 bytes, make it 5)
    header = ic.encode_header(ic.MT.CONTENT, 0, ic.VS.V1, 0)  # Wrong main type
    body = b"\x01\x02\x03"  # Too short
    wrong_code = "ISCC:" + ic.encode_base32(header + body)

    with pytest.raises(ValueError, match="Invalid ISCC-ID length"):
        common.validate_iscc_id(wrong_code)


def test_validate_iscc_id_wrong_maintype():
    """Test ISCC-ID validation with wrong main type."""
    # Create ISCC with CONTENT main type instead of ID
    header = ic.encode_header(ic.MT.CONTENT, 0, ic.VS.V1, 0)
    body = b"\x01\x02\x03\x04\x05\x06\x07\x08"
    wrong_id = "ISCC:" + ic.encode_base32(header + body)

    with pytest.raises(ValueError, match="Invalid ISCC-ID main type"):
        common.validate_iscc_id(wrong_id)


def test_validate_iscc_id_invalid_length_field():
    """Test ISCC-ID validation rejects invalid length field."""
    # Create ISCC-ID with length=1 (invalid for 64-bit ISCC-ID v1)
    # Valid length field for ISCC-ID v1 is 0
    header = ic.encode_header(ic.MT.ID, 0, ic.VS.V1, 1)  # length=1 is invalid
    body = b"\x01\x02\x03\x04\x05\x06\x07\x08"
    malformed_id = "ISCC:" + ic.encode_base32(header + body)

    with pytest.raises(ValueError, match="Invalid ISCC-ID length field"):
        common.validate_iscc_id(malformed_id)


def test_roundtrip_iscc_id_body_reconstruction(sample_iscc_ids):
    """Test full roundtrip: extract body and realm, then reconstruct."""
    original_iscc_id = sample_iscc_ids[0]

    # Extract components
    body = common.extract_iscc_id_body(original_iscc_id)
    realm = common.extract_realm_id(original_iscc_id)

    # Reconstruct
    reconstructed = common.reconstruct_iscc_id(body, realm)

    # Should match original
    assert reconstructed == original_iscc_id


def test_normalize_query_asset_with_iscc_code_only(sample_iscc_codes):
    """Test normalize_query_asset derives units from iscc_code."""
    query = IsccEntry(iscc_code=sample_iscc_codes[0])
    normalized = common.normalize_query_asset(query)

    # Should have both iscc_code and units after normalization
    assert normalized.iscc_code == sample_iscc_codes[0]
    assert normalized.units is not None
    assert len(normalized.units) > 0


def test_normalize_query_asset_with_units_only(sample_iscc_codes):
    """Test normalize_query_asset derives iscc_code from units."""
    from iscc_search.models import IsccCode

    # Get units from a valid ISCC-CODE
    code_obj = IsccCode(sample_iscc_codes[0])
    units = [str(unit) for unit in code_obj.units]

    # Query with units only (no iscc_code)
    query = IsccEntry(units=units)
    normalized = common.normalize_query_asset(query)

    # Should have both units and iscc_code after normalization
    assert normalized.units == units
    assert normalized.iscc_code is not None
    assert normalized.iscc_code.startswith("ISCC:")


def test_normalize_query_asset_with_both(sample_iscc_codes):
    """Test normalize_query_asset keeps both when provided."""
    from iscc_search.models import IsccCode

    # Derive units from code for test data
    code_obj = IsccCode(sample_iscc_codes[0])
    units = [str(u) for u in code_obj.units]

    query = IsccEntry(iscc_code=sample_iscc_codes[0], units=units)
    normalized = common.normalize_query_asset(query)

    # Should preserve both
    assert normalized.iscc_code == sample_iscc_codes[0]
    assert normalized.units == units


def test_normalize_query_asset_with_neither(sample_iscc_ids):
    """Test normalize_query_asset raises error when neither iscc_code nor units provided."""
    # Query with only iscc_id (no iscc_code or units)
    query = IsccEntry(iscc_id=sample_iscc_ids[0])
    with pytest.raises(ValueError, match="must have either 'iscc_code' or 'units'"):
        common.normalize_query_asset(query)
