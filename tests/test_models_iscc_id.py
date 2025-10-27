"""Tests for IsccID class."""

import iscc_core as ic
import pytest

from iscc_search.models import IsccID


def test_init_with_string():
    # type: () -> None
    """Constructor accepts ISCC-ID string."""
    iscc_id_str = ic.gen_iscc_id(timestamp=1000000, hub_id=5, realm_id=0)["iscc"]
    iscc_id = IsccID(iscc_id_str)

    assert isinstance(iscc_id, IsccID)
    assert str(iscc_id) == iscc_id_str


def test_init_with_string_no_prefix():
    # type: () -> None
    """Constructor accepts ISCC-ID string without 'ISCC:' prefix."""
    iscc_id_str = ic.gen_iscc_id(timestamp=1000000, hub_id=5, realm_id=0)["iscc"]
    iscc_id_no_prefix = iscc_id_str.removeprefix("ISCC:")

    iscc_id = IsccID(iscc_id_no_prefix)

    assert isinstance(iscc_id, IsccID)
    assert str(iscc_id) == iscc_id_str


def test_init_with_bytes():
    # type: () -> None
    """Constructor accepts ISCC-ID digest bytes."""
    iscc_id_str = ic.gen_iscc_id(timestamp=1000000, hub_id=5, realm_id=0)["iscc"]
    digest = ic.decode_base32(iscc_id_str.removeprefix("ISCC:"))

    iscc_id = IsccID(digest)

    assert isinstance(iscc_id, IsccID)
    assert bytes(iscc_id) == digest


def test_init_invalid_type():
    # type: () -> None
    """Constructor raises TypeError for invalid input types."""
    with pytest.raises(TypeError, match="`iscc` must be str, bytes"):
        IsccID(12345)  # type: ignore


def test_str_returns_canonical_iscc(sample_iscc_ids):
    # type: (list[str]) -> None
    """__str__ returns canonical ISCC string with 'ISCC:' prefix."""
    for iscc_id_str in sample_iscc_ids:
        iscc_id = IsccID(iscc_id_str)
        assert str(iscc_id) == iscc_id_str
        assert str(iscc_id).startswith("ISCC:")


def test_bytes_returns_digest(sample_iscc_ids):
    # type: (list[str]) -> None
    """__bytes__ returns ISCC-DIGEST bytes."""
    for iscc_id_str in sample_iscc_ids:
        iscc_id = IsccID(iscc_id_str)
        digest = ic.decode_base32(iscc_id_str.removeprefix("ISCC:"))
        assert bytes(iscc_id) == digest


def test_len_returns_body_bit_length(sample_iscc_ids):
    # type: (list[str]) -> None
    """__len__ returns ISCC-BODY bit length (always 64 bits for ISCC-ID)."""
    for iscc_id_str in sample_iscc_ids:
        iscc_id = IsccID(iscc_id_str)
        # ISCC-ID body is always 8 bytes = 64 bits
        assert len(iscc_id) == 64


def test_body_property(sample_iscc_ids):
    # type: (list[str]) -> None
    """body property returns ISCC-BODY bytes (without header)."""
    for iscc_id_str in sample_iscc_ids:
        iscc_id = IsccID(iscc_id_str)
        # Body is digest without first 2 bytes (header)
        assert len(iscc_id.body) == 8
        assert iscc_id.body == iscc_id.digest[2:]


def test_fields_property(sample_iscc_ids):
    # type: (list[str]) -> None
    """fields property returns decoded ISCC header fields."""
    for iscc_id_str in sample_iscc_ids:
        iscc_id = IsccID(iscc_id_str)
        fields = iscc_id.fields

        # Fields is an IsccTuple: (maintype, subtype, version, length, body)
        assert isinstance(fields, tuple)
        assert len(fields) == 5
        assert fields[0] == ic.MT.ID  # MainType is ID
        assert fields[2] == ic.VS.V1  # Version is V1


def test_iscc_type_property():
    # type: () -> None
    """iscc_type property returns formatted type string."""
    # Test realm_id 0
    iscc_id_r0 = IsccID(ic.gen_iscc_id(timestamp=1000000, hub_id=5, realm_id=0)["iscc"])
    assert iscc_id_r0.iscc_type == "ID_REALM_0_V1"

    # Test realm_id 1
    iscc_id_r1 = IsccID(ic.gen_iscc_id(timestamp=2000000, hub_id=10, realm_id=1)["iscc"])
    assert iscc_id_r1.iscc_type == "ID_REALM_1_V1"


def test_int_returns_body_as_integer():
    # type: () -> None
    """__int__ returns body-only as integer (header excluded)."""
    iscc_id_str = ic.gen_iscc_id(timestamp=1000000, hub_id=5, realm_id=0)["iscc"]
    iscc_id = IsccID(iscc_id_str)

    # __int__ should return only the body as integer
    int_val = int(iscc_id)
    expected_int = int.from_bytes(iscc_id.body, "big", signed=False)

    assert int_val == expected_int
    assert int_val > 0


def test_int_excludes_header():
    # type: () -> None
    """__int__ excludes header from integer representation (body-only)."""
    iscc_id_str = ic.gen_iscc_id(timestamp=1000000, hub_id=5, realm_id=0)["iscc"]
    iscc_id = IsccID(iscc_id_str)

    int_val = int(iscc_id)
    body_int = int.from_bytes(iscc_id.body, "big", signed=False)

    # __int__ should return body-only
    assert int_val == body_int
    # Body is 8 bytes = 64 bits
    assert int_val.bit_length() <= 64


def test_from_int_with_realm_0():
    # type: () -> None
    """from_int class method builds ISCC-ID with realm_id=0."""
    # Create original ISCC-ID
    original_str = ic.gen_iscc_id(timestamp=1000000, hub_id=5, realm_id=0)["iscc"]
    original = IsccID(original_str)

    # Extract body as integer
    body_int = int.from_bytes(original.body, "big", signed=False)

    # Reconstruct using from_int
    reconstructed = IsccID.from_int(body_int, realm_id=0)

    # Should match original
    assert str(reconstructed) == str(original)
    assert reconstructed.fields == original.fields


def test_from_int_with_realm_1():
    # type: () -> None
    """from_int class method builds ISCC-ID with realm_id=1."""
    # Create original ISCC-ID with realm_id=1
    original_str = ic.gen_iscc_id(timestamp=2000000, hub_id=10, realm_id=1)["iscc"]
    original = IsccID(original_str)

    # Extract body as integer
    body_int = int.from_bytes(original.body, "big", signed=False)

    # Reconstruct using from_int
    reconstructed = IsccID.from_int(body_int, realm_id=1)

    # Should match original
    assert str(reconstructed) == str(original)
    assert reconstructed.fields == original.fields


def test_from_int_changes_realm_id():
    # type: () -> None
    """from_int can reconstruct with different realm_id (header is lost)."""
    # Create original ISCC-ID with realm_id=1
    original_str = ic.gen_iscc_id(timestamp=2000000, hub_id=10, realm_id=1)["iscc"]
    original = IsccID(original_str)

    # Extract body as integer
    body_int = int.from_bytes(original.body, "big", signed=False)

    # Reconstruct with realm_id=0 (different from original)
    reconstructed = IsccID.from_int(body_int, realm_id=0)

    # Body should be identical
    assert reconstructed.body == original.body
    # But header should be different
    assert reconstructed.digest != original.digest
    assert reconstructed.iscc_type == "ID_REALM_0_V1"
    assert original.iscc_type == "ID_REALM_1_V1"


def test_roundtrip_body_preserves_iscc_id():
    # type: () -> None
    """Roundtrip: ISCC-ID -> body_int -> ISCC-ID preserves ID with correct realm_id."""
    original_str = ic.gen_iscc_id(timestamp=1234567, hub_id=42, realm_id=0)["iscc"]
    original = IsccID(original_str)

    # Convert to int (body only)
    body_int = int.from_bytes(original.body, "big", signed=False)

    # Reconstruct with same realm_id
    reconstructed = IsccID.from_int(body_int, realm_id=0)

    # Should be identical
    assert str(reconstructed) == str(original)
    assert bytes(reconstructed) == bytes(original)


def test_roundtrip_with_int_succeeds():
    # type: () -> None
    """Roundtrip with __int__ succeeds because it returns body-only."""
    original_str = ic.gen_iscc_id(timestamp=1000000, hub_id=5, realm_id=0)["iscc"]
    original = IsccID(original_str)

    # Get body as int using __int__
    body_int = int(original)

    # Reconstruct with same realm_id should succeed
    reconstructed = IsccID.from_int(body_int, realm_id=0)

    # Should match original
    assert str(reconstructed) == str(original)
    assert reconstructed.digest == original.digest


def test_with_sample_iscc_ids_fixture(sample_iscc_ids):
    # type: (list[str]) -> None
    """Test with sample_iscc_ids fixture."""
    for iscc_id_str in sample_iscc_ids:
        iscc_id = IsccID(iscc_id_str)

        # Basic properties
        assert isinstance(iscc_id, IsccID)
        assert str(iscc_id) == iscc_id_str
        assert len(iscc_id) == 64

        # Roundtrip through body int
        body_int = int.from_bytes(iscc_id.body, "big", signed=False)
        reconstructed = IsccID.from_int(body_int, realm_id=0)
        assert str(reconstructed) == iscc_id_str


def test_with_iscc_id_key_pairs_fixture(iscc_id_key_pairs):
    # type: (list[tuple[str, int]]) -> None
    """Test with iscc_id_key_pairs fixture."""
    for iscc_id_str, key in iscc_id_key_pairs:
        iscc_id = IsccID(iscc_id_str)

        # The key should be the body as integer
        body_int = int.from_bytes(iscc_id.body, "big", signed=False)
        assert key == body_int

        # Reconstruct from key
        reconstructed = IsccID.from_int(key, realm_id=0)
        assert str(reconstructed) == iscc_id_str


def test_edge_case_very_large_timestamp():
    # type: () -> None
    """Test with very large timestamp (near max for 52 bits)."""
    # Max 52-bit value is 2^52 - 1 = 4503599627370495
    # This represents microseconds, so it's year ~2112
    max_timestamp = 4503599627370495 - 100

    iscc_id_str = ic.gen_iscc_id(timestamp=max_timestamp, hub_id=0, realm_id=0)["iscc"]
    iscc_id = IsccID(iscc_id_str)

    # Should work fine
    assert isinstance(iscc_id, IsccID)
    assert len(iscc_id) == 64

    # Roundtrip
    body_int = int.from_bytes(iscc_id.body, "big", signed=False)
    reconstructed = IsccID.from_int(body_int, realm_id=0)
    assert str(reconstructed) == iscc_id_str


def test_edge_case_zero_timestamp():
    # type: () -> None
    """Test with zero timestamp."""
    iscc_id_str = ic.gen_iscc_id(timestamp=0, hub_id=0, realm_id=0)["iscc"]
    iscc_id = IsccID(iscc_id_str)

    assert isinstance(iscc_id, IsccID)
    assert len(iscc_id) == 64

    # Roundtrip
    body_int = int.from_bytes(iscc_id.body, "big", signed=False)
    reconstructed = IsccID.from_int(body_int, realm_id=0)
    assert str(reconstructed) == iscc_id_str


def test_edge_case_max_hub_id():
    # type: () -> None
    """Test with maximum hub_id (12 bits = 4095)."""
    max_hub_id = 4095

    iscc_id_str = ic.gen_iscc_id(timestamp=1000000, hub_id=max_hub_id, realm_id=0)["iscc"]
    iscc_id = IsccID(iscc_id_str)

    assert isinstance(iscc_id, IsccID)
    assert len(iscc_id) == 64

    # Roundtrip
    body_int = int.from_bytes(iscc_id.body, "big", signed=False)
    reconstructed = IsccID.from_int(body_int, realm_id=0)
    assert str(reconstructed) == iscc_id_str


def test_multiple_iscc_ids_different_realm_ids():
    # type: () -> None
    """Test multiple ISCC-IDs with different realm_ids."""
    for realm_id in [0, 1]:
        iscc_id_str = ic.gen_iscc_id(timestamp=3000000, hub_id=100, realm_id=realm_id)["iscc"]
        iscc_id = IsccID(iscc_id_str)

        # Check type
        expected_type = f"ID_REALM_{realm_id}_V1"
        assert iscc_id.iscc_type == expected_type

        # Roundtrip
        body_int = int.from_bytes(iscc_id.body, "big", signed=False)
        reconstructed = IsccID.from_int(body_int, realm_id=realm_id)
        assert str(reconstructed) == iscc_id_str


def test_from_int_with_zero():
    # type: () -> None
    """from_int handles zero body value."""
    # Zero body represents timestamp=0, hub_id=0
    reconstructed = IsccID.from_int(0, realm_id=0)

    assert isinstance(reconstructed, IsccID)
    assert int.from_bytes(reconstructed.body, "big", signed=False) == 0


def test_from_int_with_max_body_value():
    # type: () -> None
    """from_int handles maximum 64-bit body value."""
    # Max 64-bit value
    max_body = 2**64 - 1

    reconstructed = IsccID.from_int(max_body, realm_id=0)

    assert isinstance(reconstructed, IsccID)
    assert int.from_bytes(reconstructed.body, "big", signed=False) == max_body


def test_cached_int_property():
    # type: () -> None
    """__int__ is cached and returns same value on multiple calls."""
    iscc_id_str = ic.gen_iscc_id(timestamp=1000000, hub_id=5, realm_id=0)["iscc"]
    iscc_id = IsccID(iscc_id_str)

    # Call multiple times
    int1 = int(iscc_id)
    int2 = int(iscc_id)
    int3 = int(iscc_id)

    # Should all be identical
    assert int1 == int2 == int3


def test_cached_str_property():
    # type: () -> None
    """__str__ is cached and returns same value on multiple calls."""
    iscc_id_str = ic.gen_iscc_id(timestamp=1000000, hub_id=5, realm_id=0)["iscc"]
    iscc_id = IsccID(iscc_id_str)

    # Call multiple times
    str1 = str(iscc_id)
    str2 = str(iscc_id)
    str3 = str(iscc_id)

    # Should all be identical
    assert str1 == str2 == str3
    assert str1 == iscc_id_str


def test_cached_len_property():
    # type: () -> None
    """__len__ is cached and returns same value on multiple calls."""
    iscc_id_str = ic.gen_iscc_id(timestamp=1000000, hub_id=5, realm_id=0)["iscc"]
    iscc_id = IsccID(iscc_id_str)

    # Call multiple times
    len1 = len(iscc_id)
    len2 = len(iscc_id)
    len3 = len(iscc_id)

    # Should all be identical
    assert len1 == len2 == len3 == 64


def test_random_returns_iscc_id_instance():
    # type: () -> None
    """random() class method returns IsccID instance."""
    iscc_id = IsccID.random()

    assert isinstance(iscc_id, IsccID)


def test_random_generates_different_ids():
    # type: () -> None
    """random() generates different IDs on subsequent calls."""
    import time

    id1 = IsccID.random()
    # Small delay to ensure different timestamp
    time.sleep(0.001)
    id2 = IsccID.random()
    time.sleep(0.001)
    id3 = IsccID.random()

    # All IDs should be different
    assert str(id1) != str(id2)
    assert str(id1) != str(id3)
    assert str(id2) != str(id3)

    # Bytes should also be different
    assert bytes(id1) != bytes(id2)
    assert bytes(id1) != bytes(id3)
    assert bytes(id2) != bytes(id3)


def test_random_has_correct_type():
    # type: () -> None
    """random() generates ISCC-ID with type ID_REALM_0_V1."""
    iscc_id = IsccID.random()

    assert iscc_id.iscc_type == "ID_REALM_0_V1"


def test_random_has_valid_structure():
    # type: () -> None
    """random() generates ISCC-ID with valid structure (10 bytes digest)."""
    iscc_id = IsccID.random()

    # Total digest should be 10 bytes (2-byte header + 8-byte body)
    assert len(iscc_id.digest) == 10
    # Body should be 8 bytes
    assert len(iscc_id.body) == 8
    # Body bit length should be 64 bits
    assert len(iscc_id) == 64


def test_random_has_reasonable_timestamp():
    # type: () -> None
    """random() generates ISCC-ID with timestamp close to current time."""
    import time

    # Capture current timestamp in microseconds
    before_us = time.time_ns() // 1000
    iscc_id = IsccID.random()
    after_us = time.time_ns() // 1000

    # Extract timestamp from ISCC-ID body
    # Body is 8 bytes: 52 bits timestamp + 12 bits hub_id
    body_int = int.from_bytes(iscc_id.body, "big", signed=False)
    timestamp_us = body_int >> 12

    # Timestamp should be within the time window
    assert before_us <= timestamp_us <= after_us

    # Timestamp should be reasonable (not zero, not too far in future)
    assert timestamp_us > 0
    # Should be less than max 52-bit value
    assert timestamp_us < 2**52


def test_random_can_convert_to_string():
    # type: () -> None
    """random() generates ISCC-ID that can be converted to string."""
    iscc_id = IsccID.random()

    # Should convert to valid ISCC string
    iscc_str = str(iscc_id)
    assert iscc_str.startswith("ISCC:")
    assert len(iscc_str) > 5  # "ISCC:" prefix + base32 content

    # Should be valid canonical format
    assert iscc_str == iscc_str.upper()  # base32 should be uppercase


def test_random_can_convert_to_bytes():
    # type: () -> None
    """random() generates ISCC-ID that can be converted to bytes."""
    iscc_id = IsccID.random()

    # Should convert to bytes
    digest = bytes(iscc_id)
    assert isinstance(digest, bytes)
    assert len(digest) == 10

    # Should match the digest property
    assert digest == iscc_id.digest


def test_random_can_roundtrip_through_string():
    # type: () -> None
    """random() ISCC-ID can roundtrip through string representation."""
    original = IsccID.random()

    # Convert to string and back
    iscc_str = str(original)
    reconstructed = IsccID(iscc_str)

    # Should be identical
    assert str(reconstructed) == str(original)
    assert bytes(reconstructed) == bytes(original)


def test_random_can_roundtrip_through_bytes():
    # type: () -> None
    """random() ISCC-ID can roundtrip through bytes representation."""
    original = IsccID.random()

    # Convert to bytes and back
    digest = bytes(original)
    reconstructed = IsccID(digest)

    # Should be identical
    assert str(reconstructed) == str(original)
    assert bytes(reconstructed) == bytes(original)
