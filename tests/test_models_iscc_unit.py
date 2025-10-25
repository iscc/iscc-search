"""Unit tests for IsccUnit class."""

import iscc_core as ic
import numpy as np
import pytest

from iscc_vdb.models import IsccUnit


def test_iscc_unit_init_from_string(sample_meta_units):
    # type: (list[str]) -> None
    """Test IsccUnit initialization from ISCC string."""
    unit = IsccUnit(sample_meta_units[0])
    assert isinstance(unit, IsccUnit)
    assert isinstance(unit.digest, bytes)


def test_iscc_unit_init_from_bytes(sample_data_units):
    # type: (list[str]) -> None
    """Test IsccUnit initialization from bytes."""
    unit_str = sample_data_units[0]
    digest = ic.decode_base32(unit_str.removeprefix("ISCC:"))
    unit = IsccUnit(digest)
    assert isinstance(unit, IsccUnit)
    assert unit.digest == digest


def test_iscc_unit_init_invalid_type():
    # type: () -> None
    """Test IsccUnit initialization with invalid type raises TypeError."""
    with pytest.raises(TypeError, match="`iscc` must be str, bytes"):
        IsccUnit(12345)  # type: ignore


def test_iscc_unit_body_property(sample_meta_units):
    # type: (list[str]) -> None
    """Test body property returns digest without header."""
    unit = IsccUnit(sample_meta_units[0])
    assert isinstance(unit.body, bytes)
    assert len(unit.digest) == len(unit.body) + 2
    assert unit.body == unit.digest[2:]


def test_iscc_unit_fields_property(sample_semantic_units):
    # type: (list[str]) -> None
    """Test fields property returns IsccTuple."""
    unit = IsccUnit(sample_semantic_units[0])
    fields = unit.fields
    assert isinstance(fields, tuple)
    assert len(fields) == 5
    # Fields: (maintype, subtype, version, length, body)
    assert isinstance(fields[0], int)  # maintype
    assert isinstance(fields[1], int)  # subtype
    assert isinstance(fields[2], int)  # version
    assert isinstance(fields[3], int)  # length
    assert isinstance(fields[4], bytes)  # body


def test_iscc_unit_iscc_type_property(sample_content_units):
    # type: (list[str]) -> None
    """Test iscc_type property returns formatted type string."""
    unit = IsccUnit(sample_content_units[0])
    iscc_type = unit.iscc_type
    assert isinstance(iscc_type, str)
    assert "_" in iscc_type
    parts = iscc_type.split("_")
    assert len(parts) >= 3  # MAINTYPE_SUBTYPE_VERSION (or more with multi-word types)


def test_iscc_unit_unit_type_property(sample_meta_units):
    # type: (list[str]) -> None
    """Test unit_type property returns same as iscc_type."""
    unit = IsccUnit(sample_meta_units[0])
    assert unit.unit_type == unit.iscc_type


def test_iscc_unit_str_canonical_format(sample_data_units):
    # type: (list[str]) -> None
    """Test __str__ returns canonical ISCC format."""
    unit_str = sample_data_units[0]
    unit = IsccUnit(unit_str)
    result = str(unit)
    assert result.startswith("ISCC:")
    assert result == unit_str


def test_iscc_unit_str_preserves_prefix(sample_semantic_units):
    # type: (list[str]) -> None
    """Test __str__ adds ISCC: prefix even when initialized without it."""
    unit_str = sample_semantic_units[0]
    # Initialize without prefix
    digest = ic.decode_base32(unit_str.removeprefix("ISCC:"))
    unit = IsccUnit(digest)
    result = str(unit)
    assert result.startswith("ISCC:")


def test_iscc_unit_len_returns_body_bit_length(sample_meta_units):
    # type: (list[str]) -> None
    """Test __len__ returns ISCC-BODY bit-length."""
    for unit_str in sample_meta_units:
        unit = IsccUnit(unit_str)
        body_bytes = len(unit.body)
        expected_bits = body_bytes * 8
        assert len(unit) == expected_bits


def test_iscc_unit_len_64_bits():
    # type: () -> None
    """Test __len__ with 64-bit ISCC-UNIT."""
    unit_str = f"ISCC:{ic.Code.rnd(ic.MT.META, bits=64)}"
    unit = IsccUnit(unit_str)
    assert len(unit) == 64


def test_iscc_unit_len_128_bits():
    # type: () -> None
    """Test __len__ with 128-bit ISCC-UNIT."""
    unit_str = f"ISCC:{ic.Code.rnd(ic.MT.SEMANTIC, ic.ST_CC.TEXT, bits=128)}"
    unit = IsccUnit(unit_str)
    assert len(unit) == 128


def test_iscc_unit_len_192_bits():
    # type: () -> None
    """Test __len__ with 192-bit ISCC-UNIT."""
    unit_str = f"ISCC:{ic.Code.rnd(ic.MT.CONTENT, ic.ST_CC.IMAGE, bits=192)}"
    unit = IsccUnit(unit_str)
    assert len(unit) == 192


def test_iscc_unit_len_256_bits():
    # type: () -> None
    """Test __len__ with 256-bit ISCC-UNIT."""
    unit_str = f"ISCC:{ic.Code.rnd(ic.MT.DATA, bits=256)}"
    unit = IsccUnit(unit_str)
    assert len(unit) == 256


def test_iscc_unit_bytes_returns_digest(sample_content_units):
    # type: (list[str]) -> None
    """Test __bytes__ returns ISCC-DIGEST bytes."""
    unit_str = sample_content_units[0]
    unit = IsccUnit(unit_str)
    result = bytes(unit)
    assert result == unit.digest


def test_iscc_unit_array_default_dtype(sample_meta_units):
    # type: (list[str]) -> None
    """Test __array__ returns numpy array with default uint8 dtype."""
    unit = IsccUnit(sample_meta_units[0])
    arr = np.array(unit)
    assert isinstance(arr, np.ndarray)
    assert arr.dtype == np.uint8


def test_iscc_unit_array_explicit_uint8(sample_semantic_units):
    # type: (list[str]) -> None
    """Test __array__ with explicit uint8 dtype."""
    unit = IsccUnit(sample_semantic_units[0])
    arr = np.array(unit, dtype=np.uint8)
    assert isinstance(arr, np.ndarray)
    assert arr.dtype == np.uint8


def test_iscc_unit_array_int16_dtype(sample_data_units):
    # type: (list[str]) -> None
    """Test __array__ with int16 dtype."""
    unit = IsccUnit(sample_data_units[0])
    arr = np.array(unit, dtype=np.int16)
    assert isinstance(arr, np.ndarray)
    assert arr.dtype == np.int16


def test_iscc_unit_array_int32_dtype(sample_content_units):
    # type: (list[str]) -> None
    """Test __array__ with int32 dtype."""
    unit = IsccUnit(sample_content_units[0])
    arr = np.array(unit, dtype=np.int32)
    assert isinstance(arr, np.ndarray)
    assert arr.dtype == np.int32


def test_iscc_unit_array_float32_dtype(sample_meta_units):
    # type: (list[str]) -> None
    """Test __array__ with float32 dtype."""
    unit = IsccUnit(sample_meta_units[1])
    arr = np.array(unit, dtype=np.float32)
    assert isinstance(arr, np.ndarray)
    assert arr.dtype == np.float32


def test_iscc_unit_array_shape_matches_body_length(sample_data_units):
    # type: (list[str]) -> None
    """Test array shape matches body length in bytes."""
    for unit_str in sample_data_units:
        unit = IsccUnit(unit_str)
        arr = np.array(unit)
        assert arr.shape == (len(unit.body),)


def test_iscc_unit_array_values_match_body_bytes(sample_meta_units):
    # type: (list[str]) -> None
    """Test array values match body bytes."""
    unit = IsccUnit(sample_meta_units[0])
    arr = np.array(unit)
    body_bytes = list(unit.body)
    arr_values = arr.tolist()
    assert arr_values == body_bytes


def test_iscc_unit_array_64bit_shape():
    # type: () -> None
    """Test array shape for 64-bit ISCC-UNIT."""
    unit_str = f"ISCC:{ic.Code.rnd(ic.MT.META, bits=64)}"
    unit = IsccUnit(unit_str)
    arr = np.array(unit)
    assert arr.shape == (8,)  # 64 bits = 8 bytes


def test_iscc_unit_array_128bit_shape():
    # type: () -> None
    """Test array shape for 128-bit ISCC-UNIT."""
    unit_str = f"ISCC:{ic.Code.rnd(ic.MT.SEMANTIC, ic.ST_CC.TEXT, bits=128)}"
    unit = IsccUnit(unit_str)
    arr = np.array(unit)
    assert arr.shape == (16,)  # 128 bits = 16 bytes


def test_iscc_unit_array_192bit_shape():
    # type: () -> None
    """Test array shape for 192-bit ISCC-UNIT."""
    unit_str = f"ISCC:{ic.Code.rnd(ic.MT.CONTENT, ic.ST_CC.IMAGE, bits=192)}"
    unit = IsccUnit(unit_str)
    arr = np.array(unit)
    assert arr.shape == (24,)  # 192 bits = 24 bytes


def test_iscc_unit_array_256bit_shape():
    # type: () -> None
    """Test array shape for 256-bit ISCC-UNIT."""
    unit_str = f"ISCC:{ic.Code.rnd(ic.MT.DATA, bits=256)}"
    unit = IsccUnit(unit_str)
    arr = np.array(unit)
    assert arr.shape == (32,)  # 256 bits = 32 bytes


def test_iscc_unit_meta_units(sample_meta_units):
    # type: (list[str]) -> None
    """Test IsccUnit with all META-type units from fixture."""
    for unit_str in sample_meta_units:
        unit = IsccUnit(unit_str)
        assert "META" in unit.iscc_type
        assert unit.unit_type == unit.iscc_type
        arr = np.array(unit)
        assert len(arr) == len(unit.body)


def test_iscc_unit_semantic_units(sample_semantic_units):
    # type: (list[str]) -> None
    """Test IsccUnit with all SEMANTIC-type units from fixture."""
    for unit_str in sample_semantic_units:
        unit = IsccUnit(unit_str)
        assert "SEMANTIC" in unit.iscc_type
        assert unit.unit_type == unit.iscc_type
        arr = np.array(unit)
        assert len(arr) == len(unit.body)


def test_iscc_unit_content_units(sample_content_units):
    # type: (list[str]) -> None
    """Test IsccUnit with all CONTENT-type units from fixture."""
    for unit_str in sample_content_units:
        unit = IsccUnit(unit_str)
        assert "CONTENT" in unit.iscc_type
        assert unit.unit_type == unit.iscc_type
        arr = np.array(unit)
        assert len(arr) == len(unit.body)


def test_iscc_unit_data_units(sample_data_units):
    # type: (list[str]) -> None
    """Test IsccUnit with all DATA-type units from fixture."""
    for unit_str in sample_data_units:
        unit = IsccUnit(unit_str)
        assert "DATA" in unit.iscc_type
        assert unit.unit_type == unit.iscc_type
        arr = np.array(unit)
        assert len(arr) == len(unit.body)


def test_iscc_unit_edge_case_min_length(edge_case_units):
    # type: (dict[str, str]) -> None
    """Test IsccUnit with minimum length (64 bits)."""
    unit = IsccUnit(edge_case_units["min_length"])
    assert len(unit) == 64
    arr = np.array(unit)
    assert arr.shape == (8,)


def test_iscc_unit_edge_case_max_length(edge_case_units):
    # type: (dict[str, str]) -> None
    """Test IsccUnit with maximum length (256 bits)."""
    unit = IsccUnit(edge_case_units["max_length"])
    assert len(unit) == 256
    arr = np.array(unit)
    assert arr.shape == (32,)


def test_iscc_unit_edge_case_all_zeros(edge_case_units):
    # type: (dict[str, str]) -> None
    """Test IsccUnit with all-zeros body."""
    unit = IsccUnit(edge_case_units["all_zeros"])
    arr = np.array(unit)
    assert np.all(arr == 0)


def test_iscc_unit_edge_case_all_ones(edge_case_units):
    # type: (dict[str, str]) -> None
    """Test IsccUnit with all-ones body."""
    unit = IsccUnit(edge_case_units["all_ones"])
    arr = np.array(unit)
    assert np.all(arr == 255)


def test_iscc_unit_array_byte_order_preserved():
    # type: () -> None
    """Test that array conversion preserves byte order."""
    # Create unit with known byte pattern
    body_bytes = bytes([1, 2, 3, 4, 5, 6, 7, 8])
    unit_str = f"ISCC:{ic.encode_component(ic.MT.META, ic.ST.NONE, ic.VS.V0, 64, body_bytes)}"
    unit = IsccUnit(unit_str)
    arr = np.array(unit)
    assert arr.tolist() == [1, 2, 3, 4, 5, 6, 7, 8]


def test_iscc_unit_array_readonly():
    # type: () -> None
    """Test that modifying array doesn't affect original unit."""
    unit_str = f"ISCC:{ic.Code.rnd(ic.MT.META, bits=64)}"
    unit = IsccUnit(unit_str)
    arr1 = np.array(unit)
    original_values = arr1.copy()

    # Modify the array
    arr1[0] = 99

    # Get a fresh array from the unit
    arr2 = np.array(unit)

    # Original unit should be unchanged
    assert np.array_equal(arr2, original_values)


def test_iscc_unit_multiple_array_calls():
    # type: () -> None
    """Test that multiple __array__ calls return consistent results."""
    unit_str = f"ISCC:{ic.Code.rnd(ic.MT.DATA, bits=128)}"
    unit = IsccUnit(unit_str)

    arr1 = np.array(unit)
    arr2 = np.array(unit)
    arr3 = np.array(unit, dtype=np.uint8)

    assert np.array_equal(arr1, arr2)
    assert np.array_equal(arr1, arr3)


def test_iscc_unit_array_with_different_subtypes():
    # type: () -> None
    """Test array conversion with different CONTENT subtypes."""
    subtypes = [ic.ST_CC.TEXT, ic.ST_CC.IMAGE, ic.ST_CC.AUDIO, ic.ST_CC.VIDEO, ic.ST_CC.MIXED]

    for st in subtypes:
        unit_str = f"ISCC:{ic.Code.rnd(ic.MT.CONTENT, st, bits=128)}"
        unit = IsccUnit(unit_str)
        arr = np.array(unit)
        assert arr.dtype == np.uint8
        assert arr.shape == (16,)


def test_iscc_unit_semantic_text_subtype():
    # type: () -> None
    """Test IsccUnit with SEMANTIC-TEXT subtype."""
    unit_str = f"ISCC:{ic.Code.rnd(ic.MT.SEMANTIC, ic.ST_CC.TEXT, bits=128)}"
    unit = IsccUnit(unit_str)
    assert "SEMANTIC" in unit.iscc_type
    assert "TEXT" in unit.iscc_type


def test_iscc_unit_semantic_image_subtype():
    # type: () -> None
    """Test IsccUnit with SEMANTIC-IMAGE subtype."""
    unit_str = f"ISCC:{ic.Code.rnd(ic.MT.SEMANTIC, ic.ST_CC.IMAGE, bits=128)}"
    unit = IsccUnit(unit_str)
    assert "SEMANTIC" in unit.iscc_type
    assert "IMAGE" in unit.iscc_type


def test_iscc_unit_semantic_mixed_subtype():
    # type: () -> None
    """Test IsccUnit with SEMANTIC-MIXED subtype."""
    unit_str = f"ISCC:{ic.Code.rnd(ic.MT.SEMANTIC, ic.ST_CC.MIXED, bits=128)}"
    unit = IsccUnit(unit_str)
    assert "SEMANTIC" in unit.iscc_type
    assert "MIXED" in unit.iscc_type


def test_iscc_unit_content_text_subtype():
    # type: () -> None
    """Test IsccUnit with CONTENT-TEXT subtype."""
    unit_str = f"ISCC:{ic.Code.rnd(ic.MT.CONTENT, ic.ST_CC.TEXT, bits=128)}"
    unit = IsccUnit(unit_str)
    assert "CONTENT" in unit.iscc_type
    assert "TEXT" in unit.iscc_type


def test_iscc_unit_content_image_subtype():
    # type: () -> None
    """Test IsccUnit with CONTENT-IMAGE subtype."""
    unit_str = f"ISCC:{ic.Code.rnd(ic.MT.CONTENT, ic.ST_CC.IMAGE, bits=128)}"
    unit = IsccUnit(unit_str)
    assert "CONTENT" in unit.iscc_type
    assert "IMAGE" in unit.iscc_type


def test_iscc_unit_content_audio_subtype():
    # type: () -> None
    """Test IsccUnit with CONTENT-AUDIO subtype."""
    unit_str = f"ISCC:{ic.Code.rnd(ic.MT.CONTENT, ic.ST_CC.AUDIO, bits=128)}"
    unit = IsccUnit(unit_str)
    assert "CONTENT" in unit.iscc_type
    assert "AUDIO" in unit.iscc_type


def test_iscc_unit_content_video_subtype():
    # type: () -> None
    """Test IsccUnit with CONTENT-VIDEO subtype."""
    unit_str = f"ISCC:{ic.Code.rnd(ic.MT.CONTENT, ic.ST_CC.VIDEO, bits=128)}"
    unit = IsccUnit(unit_str)
    assert "CONTENT" in unit.iscc_type
    assert "VIDEO" in unit.iscc_type


def test_iscc_unit_content_mixed_subtype():
    # type: () -> None
    """Test IsccUnit with CONTENT-MIXED subtype."""
    unit_str = f"ISCC:{ic.Code.rnd(ic.MT.CONTENT, ic.ST_CC.MIXED, bits=128)}"
    unit = IsccUnit(unit_str)
    assert "CONTENT" in unit.iscc_type
    assert "MIXED" in unit.iscc_type


def test_iscc_unit_roundtrip_string_to_array():
    # type: () -> None
    """Test roundtrip: string -> IsccUnit -> array -> bytes matches original."""
    unit_str = f"ISCC:{ic.Code.rnd(ic.MT.DATA, bits=256)}"
    unit = IsccUnit(unit_str)
    arr = np.array(unit)
    reconstructed_body = arr.tobytes()
    assert reconstructed_body == unit.body


def test_iscc_unit_comparison_via_arrays():
    # type: () -> None
    """Test comparing two units via their numpy arrays."""
    # Create two identical units (same body)
    body_bytes = bytes([42] * 8)
    unit_str1 = f"ISCC:{ic.encode_component(ic.MT.META, ic.ST.NONE, ic.VS.V0, 64, body_bytes)}"
    unit_str2 = f"ISCC:{ic.encode_component(ic.MT.META, ic.ST.NONE, ic.VS.V0, 64, body_bytes)}"

    unit1 = IsccUnit(unit_str1)
    unit2 = IsccUnit(unit_str2)

    arr1 = np.array(unit1)
    arr2 = np.array(unit2)

    assert np.array_equal(arr1, arr2)


def test_iscc_unit_array_copy_parameter(sample_meta_units):
    # type: (list[str]) -> None
    """Test __array__ method respects copy parameter (NumPy 2.0 compatibility)."""
    unit = IsccUnit(sample_meta_units[0])

    # Test copy=None (default) - should return view
    arr_default = np.array(unit)
    arr_none = np.array(unit, copy=None)
    assert np.array_equal(arr_default, arr_none)

    # Test copy=False - should return view (no copy)
    arr_no_copy = np.array(unit, copy=False)
    assert np.array_equal(arr_default, arr_no_copy)

    # Test copy=True - should return a copy
    arr_copy = np.array(unit, copy=True)
    assert np.array_equal(arr_default, arr_copy)
    # Verify it's actually a copy by modifying and checking independence
    # Note: We can't modify the view since it's read-only from frombuffer,
    # but we can verify that both exist as separate objects
    assert arr_copy is not arr_default
