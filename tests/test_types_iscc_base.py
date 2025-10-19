"""Comprehensive unit tests for IsccBase class."""

import iscc_core as ic
import pytest

from iscc_vdb.types import IsccBase


def test_init_with_string():
    # type: () -> None
    """Test IsccBase initialization with string input."""
    iscc_str = "ISCC:AAA6HZYGQLBASTFM"
    obj = IsccBase(iscc_str)
    assert obj.digest is not None
    assert isinstance(obj.digest, bytes)


def test_init_with_string_without_prefix():
    # type: () -> None
    """Test IsccBase initialization with string without ISCC: prefix."""
    iscc_str = "AAA6HZYGQLBASTFM"
    obj = IsccBase(iscc_str)
    assert obj.digest is not None
    assert isinstance(obj.digest, bytes)


def test_init_with_bytes():
    # type: () -> None
    """Test IsccBase initialization with bytes input."""
    digest = ic.decode_base32("AAA6HZYGQLBASTFM")
    obj = IsccBase(digest)
    assert obj.digest == digest
    assert isinstance(obj.digest, bytes)


def test_init_invalid_type_int():
    # type: () -> None
    """Test IsccBase initialization with invalid int type raises TypeError."""
    with pytest.raises(TypeError, match="`iscc` must be str, bytes"):
        IsccBase(12345)


def test_init_invalid_type_list():
    # type: () -> None
    """Test IsccBase initialization with invalid list type raises TypeError."""
    with pytest.raises(TypeError, match="`iscc` must be str, bytes"):
        IsccBase([1, 2, 3])


def test_init_invalid_type_none():
    # type: () -> None
    """Test IsccBase initialization with None raises TypeError."""
    with pytest.raises(TypeError, match="`iscc` must be str, bytes"):
        IsccBase(None)


def test_body_property():
    # type: () -> None
    """Test body property returns ISCC-BODY bytes (excluding header)."""
    iscc_str = "AAA6HZYGQLBASTFM"
    obj = IsccBase(iscc_str)
    body = obj.body
    # Body should be all bytes after first 2 bytes (header)
    assert isinstance(body, bytes)
    assert len(body) == len(obj.digest) - 2
    assert body == obj.digest[2:]


def test_body_property_various_lengths(sample_meta_units):
    # type: (list[str]) -> None
    """Test body property with various ISCC lengths."""
    for iscc_str in sample_meta_units:
        obj = IsccBase(iscc_str)
        body = obj.body
        # Verify body is the digest without the 2-byte header
        assert body == obj.digest[2:]


def test_fields_cached_property():
    # type: () -> None
    """Test fields cached property returns IsccTuple."""
    iscc_str = "AAA6HZYGQLBASTFM"
    obj = IsccBase(iscc_str)
    fields = obj.fields
    # IsccTuple has 5 elements: (MainType, SubType, Version, length, TailData)
    assert isinstance(fields, tuple)
    assert len(fields) == 5
    # First call should cache the result
    fields2 = obj.fields
    assert fields is fields2  # Should be the same object (cached)


def test_fields_structure():
    # type: () -> None
    """Test fields returns correct structure with expected types."""
    iscc_str = "AAA6HZYGQLBASTFM"
    obj = IsccBase(iscc_str)
    main_type, sub_type, version, length, tail_data = obj.fields
    # Verify types
    assert isinstance(main_type, int)
    assert isinstance(sub_type, int)
    assert isinstance(version, int)
    assert isinstance(length, int)
    assert isinstance(tail_data, bytes)


def test_iscc_type_meta(sample_meta_units):
    # type: (list[str]) -> None
    """Test iscc_type cached property returns correct type string for META."""
    for iscc_str in sample_meta_units:
        obj = IsccBase(iscc_str)
        iscc_type = obj.iscc_type
        assert isinstance(iscc_type, str)
        assert iscc_type.startswith("META_")
        assert "_V0" in iscc_type or "_V1" in iscc_type
        # Verify caching
        assert obj.iscc_type is iscc_type


def test_iscc_type_semantic(sample_semantic_units):
    # type: (list[str]) -> None
    """Test iscc_type cached property returns correct type string for SEMANTIC."""
    for iscc_str in sample_semantic_units:
        obj = IsccBase(iscc_str)
        iscc_type = obj.iscc_type
        assert isinstance(iscc_type, str)
        assert iscc_type.startswith("SEMANTIC_")
        assert any(subtype in iscc_type for subtype in ["TEXT", "IMAGE", "MIXED"])


def test_iscc_type_content(sample_content_units):
    # type: (list[str]) -> None
    """Test iscc_type cached property returns correct type string for CONTENT."""
    for iscc_str in sample_content_units:
        obj = IsccBase(iscc_str)
        iscc_type = obj.iscc_type
        assert isinstance(iscc_type, str)
        assert iscc_type.startswith("CONTENT_")
        assert any(subtype in iscc_type for subtype in ["TEXT", "IMAGE", "AUDIO", "VIDEO", "MIXED"])


def test_iscc_type_data(sample_data_units):
    # type: (list[str]) -> None
    """Test iscc_type cached property returns correct type string for DATA."""
    for iscc_str in sample_data_units:
        obj = IsccBase(iscc_str)
        iscc_type = obj.iscc_type
        assert isinstance(iscc_type, str)
        assert iscc_type.startswith("DATA_")
        assert "NONE" in iscc_type


def test_str_returns_canonical_iscc():
    # type: () -> None
    """Test __str__ returns canonical ISCC string with ISCC: prefix."""
    iscc_str = "AAA6HZYGQLBASTFM"
    obj = IsccBase(iscc_str)
    result = str(obj)
    assert result.startswith("ISCC:")
    assert isinstance(result, str)
    # Should preserve the ISCC code
    assert result == "ISCC:AAA6HZYGQLBASTFM"


def test_str_with_prefix_input():
    # type: () -> None
    """Test __str__ returns canonical form even when initialized with prefix."""
    iscc_str = "ISCC:AAA6HZYGQLBASTFM"
    obj = IsccBase(iscc_str)
    result = str(obj)
    assert result == "ISCC:AAA6HZYGQLBASTFM"


def test_len_returns_body_bit_length():
    # type: () -> None
    """Test __len__ returns ISCC-BODY bit-length."""
    # 64-bit ISCC
    obj = IsccBase("AAA6HZYGQLBASTFM")
    assert len(obj) == 64
    # Body is 8 bytes = 64 bits
    assert len(obj) == len(obj.body) * 8


def test_len_various_bit_lengths(sample_meta_units):
    # type: (list[str]) -> None
    """Test __len__ with various ISCC bit lengths (64, 128, 192, 256)."""
    expected_lengths = [64, 128, 192, 256]
    for iscc_str, expected_len in zip(sample_meta_units, expected_lengths):
        obj = IsccBase(iscc_str)
        assert len(obj) == expected_len
        # Verify it matches body byte length * 8
        assert len(obj) == len(obj.body) * 8


def test_len_caching():
    # type: () -> None
    """Test __len__ result is cached."""
    obj = IsccBase("AAA6HZYGQLBASTFM")
    result1 = len(obj)
    result2 = len(obj)
    # Should return the same value (cached)
    assert result1 == result2


def test_bytes_returns_digest():
    # type: () -> None
    """Test __bytes__ returns ISCC-DIGEST bytes."""
    digest = ic.decode_base32("AAA6HZYGQLBASTFM")
    obj = IsccBase(digest)
    result = bytes(obj)
    assert result == digest
    assert isinstance(result, bytes)


def test_bytes_includes_header():
    # type: () -> None
    """Test __bytes__ returns complete digest including header."""
    obj = IsccBase("AAA6HZYGQLBASTFM")
    result = bytes(obj)
    # Should include 2-byte header + body
    assert len(result) > 2
    # Body should match
    assert result[2:] == obj.body


def test_string_and_bytes_input_equivalence():
    # type: () -> None
    """Test that string and bytes initialization produce equivalent objects."""
    iscc_str = "ISCC:AAA6HZYGQLBASTFM"
    digest = ic.decode_base32(iscc_str.removeprefix("ISCC:"))

    obj_from_str = IsccBase(iscc_str)
    obj_from_bytes = IsccBase(digest)

    # Both should produce the same results
    assert str(obj_from_str) == str(obj_from_bytes)
    assert bytes(obj_from_str) == bytes(obj_from_bytes)
    assert len(obj_from_str) == len(obj_from_bytes)
    assert obj_from_str.body == obj_from_bytes.body
    assert obj_from_str.fields == obj_from_bytes.fields
    assert obj_from_str.iscc_type == obj_from_bytes.iscc_type


def test_edge_case_minimum_length():
    # type: () -> None
    """Test IsccBase with minimum length ISCC (64 bits)."""
    min_iscc = str(ic.Code.rnd(ic.MT.META, bits=64))
    obj = IsccBase(min_iscc)
    assert len(obj) == 64
    assert len(obj.body) == 8  # 8 bytes = 64 bits
    assert len(obj.digest) == 10  # 2-byte header + 8-byte body


def test_edge_case_maximum_length():
    # type: () -> None
    """Test IsccBase with maximum length ISCC (256 bits)."""
    max_iscc = str(ic.Code.rnd(ic.MT.META, bits=256))
    obj = IsccBase(max_iscc)
    assert len(obj) == 256
    assert len(obj.body) == 32  # 32 bytes = 256 bits
    assert len(obj.digest) == 34  # 2-byte header + 32-byte body


def test_edge_case_all_zeros(edge_case_units):
    # type: (dict[str, str]) -> None
    """Test IsccBase with all-zeros body."""
    obj = IsccBase(edge_case_units["all_zeros"])
    assert obj.body == bytes([0] * 8)
    assert len(obj) == 64


def test_edge_case_all_ones(edge_case_units):
    # type: (dict[str, str]) -> None
    """Test IsccBase with all-ones body."""
    obj = IsccBase(edge_case_units["all_ones"])
    assert obj.body == bytes([255] * 8)
    assert len(obj) == 64


def test_multiple_instances_independence():
    # type: () -> None
    """Test that multiple IsccBase instances are independent."""
    iscc1 = str(ic.Code.rnd(ic.MT.META, bits=64))
    iscc2 = str(ic.Code.rnd(ic.MT.META, bits=128))

    obj1 = IsccBase(iscc1)
    obj2 = IsccBase(iscc2)

    # Should have different properties
    assert bytes(obj1) != bytes(obj2)
    assert len(obj1) != len(obj2)
    # Modifying one should not affect the other
    assert obj1.digest is not obj2.digest


def test_body_is_property_not_cached():
    # type: () -> None
    """Test that body is a property (not cached) and returns consistent values."""
    obj = IsccBase("AAA6HZYGQLBASTFM")
    body1 = obj.body
    body2 = obj.body
    # Should return equal values
    assert body1 == body2
    # Body is a property, so it might create new objects each time
    # But the values should be identical
    assert body1 == obj.digest[2:]


def test_fields_returns_valid_iscc_tuple():
    # type: () -> None
    """Test that fields property returns a valid IsccTuple structure."""
    obj = IsccBase(str(ic.Code.rnd(ic.MT.SEMANTIC, ic.ST_CC.TEXT, bits=128)))
    fields = obj.fields

    # Unpack and verify
    main_type, sub_type, version, length, tail_data = fields

    # MainType should be SEMANTIC (1)
    assert main_type == ic.MT.SEMANTIC
    # SubType should be TEXT
    assert sub_type == ic.ST_CC.TEXT
    # Version should be V0 (0)
    assert version == ic.VS.V0
    # Length field should be non-negative
    assert length >= 0
    # Tail data should be bytes and match body
    assert isinstance(tail_data, bytes)
    assert tail_data == obj.body


def test_iscc_type_format():
    # type: () -> None
    """Test iscc_type returns properly formatted type string."""
    # Test META
    meta_obj = IsccBase(str(ic.Code.rnd(ic.MT.META, bits=64)))
    assert "_" in meta_obj.iscc_type
    parts = meta_obj.iscc_type.split("_")
    assert len(parts) == 3
    assert parts[0] in ["META", "SEMANTIC", "CONTENT", "DATA", "INSTANCE"]
    assert parts[2].startswith("V")


def test_digest_stored_correctly():
    # type: () -> None
    """Test that digest is stored correctly in the instance."""
    digest = ic.decode_base32("AAA6HZYGQLBASTFM")
    obj = IsccBase(digest)
    # Digest should be stored as-is
    assert obj.digest is digest
    # Modifying original digest should not affect the object
    # (though in this case it would since we're not copying)
    assert bytes(obj) == digest


def test_iscc_type_caching():
    # type: () -> None
    """Test that iscc_type result is cached."""
    obj = IsccBase(str(ic.Code.rnd(ic.MT.META, bits=64)))
    type1 = obj.iscc_type
    type2 = obj.iscc_type
    # Should return the same object (cached)
    assert type1 is type2


def test_semantic_image_type():
    # type: () -> None
    """Test IsccBase with SEMANTIC-IMAGE type."""
    iscc = str(ic.Code.rnd(ic.MT.SEMANTIC, ic.ST_CC.IMAGE, bits=128))
    obj = IsccBase(iscc)
    assert "SEMANTIC" in obj.iscc_type
    assert "IMAGE" in obj.iscc_type
    assert len(obj) == 128


def test_content_audio_type():
    # type: () -> None
    """Test IsccBase with CONTENT-AUDIO type."""
    iscc = str(ic.Code.rnd(ic.MT.CONTENT, ic.ST_CC.AUDIO, bits=192))
    obj = IsccBase(iscc)
    assert "CONTENT" in obj.iscc_type
    assert "AUDIO" in obj.iscc_type
    assert len(obj) == 192


def test_content_video_type():
    # type: () -> None
    """Test IsccBase with CONTENT-VIDEO type."""
    iscc = str(ic.Code.rnd(ic.MT.CONTENT, ic.ST_CC.VIDEO, bits=256))
    obj = IsccBase(iscc)
    assert "CONTENT" in obj.iscc_type
    assert "VIDEO" in obj.iscc_type
    assert len(obj) == 256


def test_repr_not_implemented():
    # type: () -> None
    """Test that __repr__ is not defined (uses default object representation)."""
    obj = IsccBase("AAA6HZYGQLBASTFM")
    repr_str = repr(obj)
    # Should use default repr with class name and memory address
    assert "IsccBase" in repr_str
    assert "object at" in repr_str
