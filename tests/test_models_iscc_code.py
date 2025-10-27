"""Tests for IsccCode class."""

import iscc_core as ic
from io import BytesIO
from iscc_search.models import IsccCode, IsccUnit


def test_iscc_code_init_with_string():
    # type: () -> None
    """Test IsccCode initialization with string."""
    meta = ic.gen_meta_code("Test Title")
    data = ic.gen_data_code(BytesIO(b"Test Data"))
    instance = ic.gen_instance_code(BytesIO(b"Test Data"))
    code = ic.gen_iscc_code([meta["iscc"], data["iscc"], instance["iscc"]])

    iscc_code = IsccCode(code["iscc"])
    assert isinstance(iscc_code, IsccCode)
    assert str(iscc_code) == code["iscc"]


def test_iscc_code_init_with_bytes():
    # type: () -> None
    """Test IsccCode initialization with bytes."""
    meta = ic.gen_meta_code("Test")
    data = ic.gen_data_code(BytesIO(b"Test"))
    instance = ic.gen_instance_code(BytesIO(b"Test"))
    code = ic.gen_iscc_code([meta["iscc"], data["iscc"], instance["iscc"]])

    digest = ic.decode_base32(code["iscc"].removeprefix("ISCC:"))
    iscc_code = IsccCode(digest)
    assert isinstance(iscc_code, IsccCode)
    assert bytes(iscc_code) == digest


def test_iscc_code_init_with_prefix():
    # type: () -> None
    """Test IsccCode handles ISCC: prefix correctly."""
    meta = ic.gen_meta_code("Test")
    data = ic.gen_data_code(BytesIO(b"Test"))
    instance = ic.gen_instance_code(BytesIO(b"Test"))
    code = ic.gen_iscc_code([meta["iscc"], data["iscc"], instance["iscc"]])

    # With prefix
    iscc_with = IsccCode(code["iscc"])
    # Without prefix
    iscc_without = IsccCode(code["iscc"].removeprefix("ISCC:"))

    assert str(iscc_with) == str(iscc_without)
    assert bytes(iscc_with) == bytes(iscc_without)


def test_iscc_code_inherits_iscc_base():
    # type: () -> None
    """Test IsccCode inherits all IsccBase properties."""
    meta = ic.gen_meta_code("Test")
    data = ic.gen_data_code(BytesIO(b"Test"))
    instance = ic.gen_instance_code(BytesIO(b"Test"))
    code = ic.gen_iscc_code([meta["iscc"], data["iscc"], instance["iscc"]])

    iscc_code = IsccCode(code["iscc"])

    # Test inherited properties
    assert hasattr(iscc_code, "digest")
    assert hasattr(iscc_code, "body")
    assert hasattr(iscc_code, "fields")
    assert hasattr(iscc_code, "iscc_type")

    # Test digest
    expected_digest = ic.decode_base32(code["iscc"].removeprefix("ISCC:"))
    assert iscc_code.digest == expected_digest

    # Test body (digest without header)
    assert iscc_code.body == expected_digest[2:]

    # Test fields
    header = ic.decode_header(expected_digest)
    assert iscc_code.fields == header

    # Test iscc_type
    assert "ISCC" in iscc_code.iscc_type
    assert ic.MT.ISCC.name in iscc_code.iscc_type


def test_iscc_code_str():
    # type: () -> None
    """Test IsccCode string representation."""
    meta = ic.gen_meta_code("Test")
    data = ic.gen_data_code(BytesIO(b"Test"))
    instance = ic.gen_instance_code(BytesIO(b"Test"))
    code = ic.gen_iscc_code([meta["iscc"], data["iscc"], instance["iscc"]])

    iscc_code = IsccCode(code["iscc"])
    assert str(iscc_code) == code["iscc"]
    assert str(iscc_code).startswith("ISCC:")


def test_iscc_code_len():
    # type: () -> None
    """Test IsccCode length returns ISCC-BODY bit-length."""
    meta = ic.gen_meta_code("Test")
    data = ic.gen_data_code(BytesIO(b"Test"))
    instance = ic.gen_instance_code(BytesIO(b"Test"))
    code = ic.gen_iscc_code([meta["iscc"], data["iscc"], instance["iscc"]])

    iscc_code = IsccCode(code["iscc"])
    # Body is digest[2:], each byte is 8 bits
    expected_bits = (len(iscc_code.digest) - 2) * 8
    assert len(iscc_code) == expected_bits


def test_iscc_code_bytes():
    # type: () -> None
    """Test IsccCode bytes representation."""
    meta = ic.gen_meta_code("Test")
    data = ic.gen_data_code(BytesIO(b"Test"))
    instance = ic.gen_instance_code(BytesIO(b"Test"))
    code = ic.gen_iscc_code([meta["iscc"], data["iscc"], instance["iscc"]])

    iscc_code = IsccCode(code["iscc"])
    expected_digest = ic.decode_base32(code["iscc"].removeprefix("ISCC:"))
    assert bytes(iscc_code) == expected_digest


def test_iscc_code_units_returns_list():
    # type: () -> None
    """Test units property returns a list."""
    meta = ic.gen_meta_code("Test")
    data = ic.gen_data_code(BytesIO(b"Test"))
    instance = ic.gen_instance_code(BytesIO(b"Test"))
    code = ic.gen_iscc_code([meta["iscc"], data["iscc"], instance["iscc"]])

    iscc_code = IsccCode(code["iscc"])
    assert isinstance(iscc_code.units, list)


def test_iscc_code_units_returns_iscc_unit_objects():
    # type: () -> None
    """Test units property returns list of IsccUnit objects."""
    meta = ic.gen_meta_code("Test")
    data = ic.gen_data_code(BytesIO(b"Test"))
    instance = ic.gen_instance_code(BytesIO(b"Test"))
    code = ic.gen_iscc_code([meta["iscc"], data["iscc"], instance["iscc"]])

    iscc_code = IsccCode(code["iscc"])
    for unit in iscc_code.units:
        assert isinstance(unit, IsccUnit)


def test_iscc_code_units_cached():
    # type: () -> None
    """Test units property is cached."""
    meta = ic.gen_meta_code("Test")
    data = ic.gen_data_code(BytesIO(b"Test"))
    instance = ic.gen_instance_code(BytesIO(b"Test"))
    code = ic.gen_iscc_code([meta["iscc"], data["iscc"], instance["iscc"]])

    iscc_code = IsccCode(code["iscc"])
    units1 = iscc_code.units
    units2 = iscc_code.units
    # Should return the same cached object
    assert units1 is units2


def test_iscc_code_minimum_valid_meta_data_instance():
    # type: () -> None
    """Test minimum valid ISCC-CODE: META + DATA + INSTANCE."""
    meta = ic.gen_meta_code("Test Title")
    data = ic.gen_data_code(BytesIO(b"Test Data"))
    instance = ic.gen_instance_code(BytesIO(b"Test Data"))
    code = ic.gen_iscc_code([meta["iscc"], data["iscc"], instance["iscc"]])

    iscc_code = IsccCode(code["iscc"])
    units = iscc_code.units
    expected = ic.iscc_decompose(code["iscc"])

    # Should have 3 units
    assert len(units) == 3
    assert len(units) == len(expected)

    # Verify each unit matches expected
    for i, unit in enumerate(units):
        expected_str = f"ISCC:{expected[i]}"
        assert str(unit) == expected_str, f"Unit {i}: {str(unit)} != {expected_str}"


def test_iscc_code_content_data_instance():
    # type: () -> None
    """Test ISCC-CODE: CONTENT (TEXT) + DATA + INSTANCE."""
    text = ic.gen_text_code("This is test text for content code generation.")
    data = ic.gen_data_code(BytesIO(b"Test"))
    instance = ic.gen_instance_code(BytesIO(b"Test"))
    code = ic.gen_iscc_code([text["iscc"], data["iscc"], instance["iscc"]])

    iscc_code = IsccCode(code["iscc"])
    units = iscc_code.units
    expected = ic.iscc_decompose(code["iscc"])

    # Should have 3 units
    assert len(units) == 3
    assert len(units) == len(expected)

    # Verify each unit matches expected
    for i, unit in enumerate(units):
        expected_str = f"ISCC:{expected[i]}"
        assert str(unit) == expected_str, f"Unit {i}: {str(unit)} != {expected_str}"


def test_iscc_code_meta_content_data_instance():
    # type: () -> None
    """Test ISCC-CODE: META + CONTENT + DATA + INSTANCE."""
    meta = ic.gen_meta_code("Test Title")
    text = ic.gen_text_code("Test text content for code generation.")
    data = ic.gen_data_code(BytesIO(b"Test data"))
    instance = ic.gen_instance_code(BytesIO(b"Test data"))
    code = ic.gen_iscc_code([meta["iscc"], text["iscc"], data["iscc"], instance["iscc"]])

    iscc_code = IsccCode(code["iscc"])
    units = iscc_code.units
    expected = ic.iscc_decompose(code["iscc"])

    # Should have 4 units
    assert len(units) == 4
    assert len(units) == len(expected)

    # Verify each unit matches expected
    for i, unit in enumerate(units):
        expected_str = f"ISCC:{expected[i]}"
        assert str(unit) == expected_str, f"Unit {i}: {str(unit)} != {expected_str}"


def test_iscc_code_wide_subtype():
    # type: () -> None
    """Test ISCC-CODE WIDE subtype: DATA-128 + INSTANCE-128."""
    from iscc_core.iscc_code import gen_iscc_code_v0

    data_128 = ic.gen_data_code(BytesIO(b"Test data for wide code"), bits=128)
    instance_128 = ic.gen_instance_code(BytesIO(b"Test data for wide code"), bits=128)
    wide_code = gen_iscc_code_v0([data_128["iscc"], instance_128["iscc"]], wide=True)

    iscc_code = IsccCode(wide_code["iscc"])
    units = iscc_code.units
    expected = ic.iscc_decompose(wide_code["iscc"])

    # Should have 2 units (DATA-128 + INSTANCE-128)
    assert len(units) == 2
    assert len(units) == len(expected)

    # Verify subtype is WIDE
    assert iscc_code.fields[1] == ic.ST_ISCC.WIDE

    # Verify each unit matches expected
    for i, unit in enumerate(units):
        expected_str = f"ISCC:{expected[i]}"
        assert str(unit) == expected_str, f"Unit {i}: {str(unit)} != {expected_str}"

    # Verify unit lengths are 128 bits each
    assert len(units[0]) == 128
    assert len(units[1]) == 128


def test_iscc_code_units_have_correct_main_types():
    # type: () -> None
    """Test parsed units have correct MainTypes."""
    meta = ic.gen_meta_code("Title")
    text = ic.gen_text_code("Text content")
    data = ic.gen_data_code(BytesIO(b"Data"))
    instance = ic.gen_instance_code(BytesIO(b"Data"))
    code = ic.gen_iscc_code([meta["iscc"], text["iscc"], data["iscc"], instance["iscc"]])

    iscc_code = IsccCode(code["iscc"])
    units = iscc_code.units

    # Check MainTypes
    assert units[0].fields[0] == ic.MT.META
    assert units[1].fields[0] == ic.MT.CONTENT
    assert units[2].fields[0] == ic.MT.DATA
    assert units[3].fields[0] == ic.MT.INSTANCE


def test_iscc_code_units_have_correct_subtypes():
    # type: () -> None
    """Test parsed units have correct SubTypes."""
    meta = ic.gen_meta_code("Title")
    text = ic.gen_text_code("Text content")
    data = ic.gen_data_code(BytesIO(b"Data"))
    instance = ic.gen_instance_code(BytesIO(b"Data"))
    code = ic.gen_iscc_code([meta["iscc"], text["iscc"], data["iscc"], instance["iscc"]])

    iscc_code = IsccCode(code["iscc"])
    units = iscc_code.units

    # META has ST.NONE
    assert units[0].fields[1] == ic.ST.NONE
    # CONTENT has TEXT subtype
    assert units[1].fields[1] == ic.ST_CC.TEXT
    # DATA has ST.NONE
    assert units[2].fields[1] == ic.ST.NONE
    # INSTANCE has ST.NONE
    assert units[3].fields[1] == ic.ST.NONE


def test_iscc_code_units_have_correct_body_length():
    # type: () -> None
    """Test parsed units have correct body lengths (64 bits for standard)."""
    meta = ic.gen_meta_code("Title")
    data = ic.gen_data_code(BytesIO(b"Data"))
    instance = ic.gen_instance_code(BytesIO(b"Data"))
    code = ic.gen_iscc_code([meta["iscc"], data["iscc"], instance["iscc"]])

    iscc_code = IsccCode(code["iscc"])
    units = iscc_code.units

    # Standard ISCC-CODE units are 64 bits
    for unit in units:
        assert len(unit) == 64


def test_iscc_code_unit_reconstruction_accuracy():
    # type: () -> None
    """Test that units are reconstructed with exact header + body match."""
    meta = ic.gen_meta_code("Test")
    data = ic.gen_data_code(BytesIO(b"Test"))
    instance = ic.gen_instance_code(BytesIO(b"Test"))
    code = ic.gen_iscc_code([meta["iscc"], data["iscc"], instance["iscc"]])

    iscc_code = IsccCode(code["iscc"])
    expected = ic.iscc_decompose(code["iscc"])

    for i, unit in enumerate(iscc_code.units):
        expected_digest = ic.decode_base32(expected[i])
        actual_digest = bytes(unit)

        # Verify entire digest matches (header + body)
        assert actual_digest == expected_digest

        # Verify header matches
        expected_header = ic.decode_header(expected_digest)
        actual_header = unit.fields
        assert actual_header == expected_header

        # Verify body matches
        expected_body = expected_digest[2:]
        actual_body = unit.body
        assert actual_body == expected_body


def test_iscc_code_data_only():
    # type: () -> None
    """Test ISCC-CODE with only DATA + INSTANCE (no dynamic units)."""
    data = ic.gen_data_code(BytesIO(b"Just data"))
    instance = ic.gen_instance_code(BytesIO(b"Just data"))
    code = ic.gen_iscc_code([data["iscc"], instance["iscc"]])

    iscc_code = IsccCode(code["iscc"])
    units = iscc_code.units
    expected = ic.iscc_decompose(code["iscc"])

    # Should have 2 units (DATA + INSTANCE)
    assert len(units) == 2
    assert len(units) == len(expected)

    # Verify reconstruction
    for i, unit in enumerate(units):
        expected_str = f"ISCC:{expected[i]}"
        assert str(unit) == expected_str


def test_iscc_code_multiple_instances():
    # type: () -> None
    """Test creating multiple IsccCode instances from same code."""
    meta = ic.gen_meta_code("Test")
    data = ic.gen_data_code(BytesIO(b"Test"))
    instance = ic.gen_instance_code(BytesIO(b"Test"))
    code = ic.gen_iscc_code([meta["iscc"], data["iscc"], instance["iscc"]])

    # Create multiple instances
    iscc1 = IsccCode(code["iscc"])
    iscc2 = IsccCode(code["iscc"])

    # They should be independent
    assert iscc1 is not iscc2

    # But have same content
    assert str(iscc1) == str(iscc2)
    assert bytes(iscc1) == bytes(iscc2)
    assert len(iscc1.units) == len(iscc2.units)


def test_iscc_code_units_are_valid_iscc_units():
    # type: () -> None
    """Test that parsed units can be used as standalone ISCC-UNITs."""
    meta = ic.gen_meta_code("Test")
    data = ic.gen_data_code(BytesIO(b"Test"))
    instance = ic.gen_instance_code(BytesIO(b"Test"))
    code = ic.gen_iscc_code([meta["iscc"], data["iscc"], instance["iscc"]])

    iscc_code = IsccCode(code["iscc"])

    for unit in iscc_code.units:
        # Each unit should be valid
        assert isinstance(unit, IsccUnit)
        # Should have valid ISCC string representation
        assert str(unit).startswith("ISCC:")
        # Should be decodable by iscc_core
        decoded = ic.iscc_decode(str(unit))
        assert decoded is not None


def test_iscc_code_chained_units():
    # type: () -> None
    """Test parsing of chained ISCC-UNITs (standard units concatenated)."""
    # Create two separate META units
    meta1 = ic.gen_meta_code("Title 1")
    meta2 = ic.gen_meta_code("Title 2")

    # Concatenate their digests
    meta1_clean = ic.iscc_clean(meta1["iscc"])
    meta2_clean = ic.iscc_clean(meta2["iscc"])
    digest1 = ic.decode_base32(meta1_clean)
    digest2 = ic.decode_base32(meta2_clean)
    combined = digest1 + digest2

    # Parse the chained units
    iscc_chain = IsccCode(combined)
    units = iscc_chain.units

    # Should parse both units
    assert len(units) == 2
    assert str(units[0]) == f"ISCC:{meta1_clean}"
    assert str(units[1]) == f"ISCC:{meta2_clean}"
