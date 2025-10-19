"""Tests for types module utility functions."""

import time

import iscc_core as ic
import pytest

from iscc_vdb.types import new_iscc_id, split_iscc_sequence


def test_new_iscc_id_returns_bytes():
    # type: () -> None
    """new_iscc_id returns bytes."""
    result = new_iscc_id()
    assert isinstance(result, bytes)


def test_new_iscc_id_correct_length():
    # type: () -> None
    """new_iscc_id returns 10 bytes (2-byte header + 8-byte body)."""
    result = new_iscc_id()
    assert len(result) == 10


def test_new_iscc_id_valid_structure():
    # type: () -> None
    """new_iscc_id produces valid ISCC-ID that can be decoded."""
    result = new_iscc_id()

    # Should be decodable by iscc_core
    mt, st, vs, ln, body = ic.decode_header(result)

    # Verify ISCC-ID structure
    assert mt == ic.MT.ID
    assert vs == ic.VS.V1
    assert len(body) == 8


def test_new_iscc_id_has_realm_0():
    # type: () -> None
    """new_iscc_id uses REALM_0 subtype."""
    result = new_iscc_id()

    mt, st, vs, ln, body = ic.decode_header(result)

    # Verify REALM_0 subtype
    assert st == ic.ST_ID_REALM.REALM_0


def test_new_iscc_id_timestamp_reasonable():
    # type: () -> None
    """new_iscc_id timestamp is close to current time."""
    # Get current time in microseconds before and after generating ID
    before_us = time.time_ns() // 1000
    result = new_iscc_id()
    after_us = time.time_ns() // 1000

    # Extract timestamp from ISCC-ID body
    body = result[2:]
    identifier = int.from_bytes(body, "big")
    timestamp = identifier >> 12

    # Timestamp should be between before and after
    assert before_us <= timestamp <= after_us


def test_new_iscc_id_randomness():
    # type: () -> None
    """new_iscc_id returns different values on subsequent calls."""
    # Generate multiple ISCC-IDs
    ids = [new_iscc_id() for _ in range(10)]

    # All should be different (due to random hub_id)
    assert len(ids) == len(set(ids))


def test_new_iscc_id_hub_id_range():
    # type: () -> None
    """new_iscc_id generates hub_id within valid 12-bit range (0-4095)."""
    # Generate multiple IDs and check hub_id values
    for _ in range(20):
        result = new_iscc_id()
        body = result[2:]
        identifier = int.from_bytes(body, "big")
        hub_id = identifier & 0xFFF  # Extract 12-bit hub_id

        # Hub ID should be in valid range
        assert 0 <= hub_id <= 4095


def test_new_iscc_id_can_encode_to_string():
    # type: () -> None
    """new_iscc_id result can be encoded to canonical ISCC string."""
    result = new_iscc_id()

    # Should be encodable to base32 string
    iscc_str = f"ISCC:{ic.encode_base32(result)}"

    assert iscc_str.startswith("ISCC:")
    assert len(iscc_str) > 5  # Has content after prefix


def test_split_iscc_sequence_single_unit():
    # type: () -> None
    """split_iscc_sequence correctly splits single ISCC-UNIT."""
    # Generate single 64-bit META unit
    unit_str = str(ic.Code.rnd(ic.MT.META, bits=64))
    unit_digest = ic.decode_base32(unit_str)

    result = split_iscc_sequence(unit_digest)

    assert len(result) == 1
    assert result[0] == unit_digest


def test_split_iscc_sequence_multiple_same_length():
    # type: () -> None
    """split_iscc_sequence splits multiple ISCC-UNITs of same length."""
    # Generate three 64-bit META units
    units = [str(ic.Code.rnd(ic.MT.META, bits=64)) for _ in range(3)]
    unit_digests = [ic.decode_base32(u) for u in units]
    concatenated = b"".join(unit_digests)

    result = split_iscc_sequence(concatenated)

    assert len(result) == 3
    for i, expected in enumerate(unit_digests):
        assert result[i] == expected


def test_split_iscc_sequence_variable_lengths():
    # type: () -> None
    """split_iscc_sequence splits ISCC-UNITs of variable lengths (64, 128, 192, 256 bits)."""
    # Generate units with different bit lengths
    units = [
        str(ic.Code.rnd(ic.MT.META, bits=64)),  # 10 bytes total
        str(ic.Code.rnd(ic.MT.META, bits=128)),  # 18 bytes total
        str(ic.Code.rnd(ic.MT.META, bits=192)),  # 26 bytes total
        str(ic.Code.rnd(ic.MT.META, bits=256)),  # 34 bytes total
    ]
    unit_digests = [ic.decode_base32(u) for u in units]
    concatenated = b"".join(unit_digests)

    result = split_iscc_sequence(concatenated)

    assert len(result) == 4
    # Verify lengths
    assert len(result[0]) == 10  # 2 + 8 bytes
    assert len(result[1]) == 18  # 2 + 16 bytes
    assert len(result[2]) == 26  # 2 + 24 bytes
    assert len(result[3]) == 34  # 2 + 32 bytes

    # Verify content matches
    for i, expected in enumerate(unit_digests):
        assert result[i] == expected


def test_split_iscc_sequence_empty_input():
    # type: () -> None
    """split_iscc_sequence handles empty input."""
    result = split_iscc_sequence(b"")
    assert result == []


def test_split_iscc_sequence_roundtrip():
    # type: () -> None
    """Concatenate -> split -> concatenate produces same result."""
    # Generate mixed-length units
    units = [
        str(ic.Code.rnd(ic.MT.META, bits=64)),
        str(ic.Code.rnd(ic.MT.SEMANTIC, ic.ST_CC.TEXT, bits=128)),
        str(ic.Code.rnd(ic.MT.CONTENT, ic.ST_CC.IMAGE, bits=256)),
        str(ic.Code.rnd(ic.MT.DATA, bits=192)),
    ]
    unit_digests = [ic.decode_base32(u) for u in units]
    original = b"".join(unit_digests)

    # Split and recombine
    split_result = split_iscc_sequence(original)
    recombined = b"".join(split_result)

    assert recombined == original


def test_split_iscc_sequence_different_main_types():
    # type: () -> None
    """split_iscc_sequence works with different MainTypes."""
    # Generate units with different MainTypes
    units = [
        str(ic.Code.rnd(ic.MT.META, bits=128)),
        str(ic.Code.rnd(ic.MT.SEMANTIC, ic.ST_CC.TEXT, bits=128)),
        str(ic.Code.rnd(ic.MT.CONTENT, ic.ST_CC.VIDEO, bits=128)),
        str(ic.Code.rnd(ic.MT.DATA, bits=128)),
    ]
    unit_digests = [ic.decode_base32(u) for u in units]
    concatenated = b"".join(unit_digests)

    result = split_iscc_sequence(concatenated)

    assert len(result) == 4
    for i, expected in enumerate(unit_digests):
        assert result[i] == expected


def test_split_iscc_sequence_with_iscc_id():
    # type: () -> None
    """split_iscc_sequence correctly splits ISCC-ID."""
    # Generate ISCC-ID
    iscc_id = new_iscc_id()

    result = split_iscc_sequence(iscc_id)

    assert len(result) == 1
    assert result[0] == iscc_id


def test_split_iscc_sequence_mixed_ids_and_units():
    # type: () -> None
    """split_iscc_sequence handles mix of ISCC-IDs and ISCC-UNITs."""
    # Generate mix of IDs and units
    id1 = new_iscc_id()
    unit1 = ic.decode_base32(str(ic.Code.rnd(ic.MT.META, bits=128)))
    id2 = new_iscc_id()
    unit2 = ic.decode_base32(str(ic.Code.rnd(ic.MT.DATA, bits=64)))

    concatenated = id1 + unit1 + id2 + unit2

    result = split_iscc_sequence(concatenated)

    assert len(result) == 4
    assert result[0] == id1
    assert result[1] == unit1
    assert result[2] == id2
    assert result[3] == unit2


def test_split_iscc_sequence_preserves_header():
    # type: () -> None
    """split_iscc_sequence preserves ISCC header information."""
    # Generate unit with specific subtype
    unit_str = str(ic.Code.rnd(ic.MT.SEMANTIC, ic.ST_CC.TEXT, bits=128))
    unit_digest = ic.decode_base32(unit_str)

    # Split and verify header is preserved
    result = split_iscc_sequence(unit_digest)

    mt, st, vs, ln, body = ic.decode_header(result[0])
    assert mt == ic.MT.SEMANTIC
    assert st == ic.ST_CC.TEXT
    assert vs == ic.VS.V0


def test_split_iscc_sequence_large_sequence():
    # type: () -> None
    """split_iscc_sequence handles large sequences efficiently."""
    # Generate 50 units
    units = [str(ic.Code.rnd(ic.MT.META, bits=64)) for _ in range(50)]
    unit_digests = [ic.decode_base32(u) for u in units]
    concatenated = b"".join(unit_digests)

    result = split_iscc_sequence(concatenated)

    assert len(result) == 50
    for i, expected in enumerate(unit_digests):
        assert result[i] == expected


def test_split_iscc_sequence_all_zeros():
    # type: () -> None
    """split_iscc_sequence handles all-zeros body."""
    # Generate unit with all-zeros body using encode_component
    zeros_bytes = bytes([0] * 8)
    unit_str = ic.encode_component(ic.MT.META, ic.ST.NONE, ic.VS.V0, 64, zeros_bytes)
    unit_digest = ic.decode_base32(unit_str)

    result = split_iscc_sequence(unit_digest)

    assert len(result) == 1
    assert result[0] == unit_digest


def test_split_iscc_sequence_all_ones():
    # type: () -> None
    """split_iscc_sequence handles all-ones body."""
    # Generate unit with all-ones body using encode_component
    ones_bytes = bytes([255] * 8)
    unit_str = ic.encode_component(ic.MT.META, ic.ST.NONE, ic.VS.V0, 64, ones_bytes)
    unit_digest = ic.decode_base32(unit_str)

    result = split_iscc_sequence(unit_digest)

    assert len(result) == 1
    assert result[0] == unit_digest


def test_split_iscc_sequence_min_length_unit():
    # type: () -> None
    """split_iscc_sequence handles minimum length unit (64 bits)."""
    unit_str = str(ic.Code.rnd(ic.MT.META, bits=64))
    unit_digest = ic.decode_base32(unit_str)

    result = split_iscc_sequence(unit_digest)

    assert len(result) == 1
    assert len(result[0]) == 10  # 2-byte header + 8-byte body


def test_split_iscc_sequence_max_length_unit():
    # type: () -> None
    """split_iscc_sequence handles maximum length unit (256 bits)."""
    unit_str = str(ic.Code.rnd(ic.MT.META, bits=256))
    unit_digest = ic.decode_base32(unit_str)

    result = split_iscc_sequence(unit_digest)

    assert len(result) == 1
    assert len(result[0]) == 34  # 2-byte header + 32-byte body


def test_new_iscc_id_integration_with_split():
    # type: () -> None
    """new_iscc_id output can be processed by split_iscc_sequence."""
    # Generate multiple ISCC-IDs
    ids = [new_iscc_id() for _ in range(5)]
    concatenated = b"".join(ids)

    # Split should correctly identify all IDs
    result = split_iscc_sequence(concatenated)

    assert len(result) == 5
    for i, expected_id in enumerate(ids):
        assert result[i] == expected_id


def test_new_iscc_id_multiple_calls_unique_timestamps():
    # type: () -> None
    """Multiple new_iscc_id calls may have same timestamp but different hub_id."""
    # Generate IDs in quick succession
    ids = [new_iscc_id() for _ in range(100)]

    # Extract timestamps and hub_ids
    timestamps = []
    hub_ids = []
    for id_bytes in ids:
        body = id_bytes[2:]
        identifier = int.from_bytes(body, "big")
        timestamps.append(identifier >> 12)
        hub_ids.append(identifier & 0xFFF)

    # Most IDs should be unique (allowing for rare random collisions)
    # With 100 selections from 4096 values, probability of collision is low but non-zero
    unique_ids = len(set(ids))
    assert unique_ids >= 95  # Allow up to 5 collisions due to randomness

    # Hub IDs should have variety
    unique_hub_ids = set(hub_ids)
    # With random selection from 4096 values, we expect many unique values
    assert len(unique_hub_ids) > 50  # Probabilistically very likely
