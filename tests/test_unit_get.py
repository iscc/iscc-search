"""Tests for UnitIndex.get() method."""

import iscc_core as ic

from iscc_vdb.unit import UnitIndex


def test_get_single_key_exists_returns_iscc_unit():
    """Single key that exists returns ISCC-UNIT string."""
    idx = UnitIndex(unit_type="META-NONE-V0", max_dim=256, realm_id=0)

    # Add a vector
    unit = ic.Code.rnd(ic.MT.META, bits=128)
    iscc_unit = f"ISCC:{unit}"
    iscc_id = ic.gen_iscc_id(timestamp=1000000, hub_id=1, realm_id=0)["iscc"]

    idx.add(iscc_id, iscc_unit)

    # Get the vector
    result = idx.get(iscc_id)

    assert isinstance(result, str)
    assert result == iscc_unit


def test_get_single_key_missing_returns_none():
    """Single key that doesn't exist returns None."""
    idx = UnitIndex(unit_type="META-NONE-V0", max_dim=256, realm_id=0)

    # Try to get a non-existent key
    iscc_id = ic.gen_iscc_id(timestamp=1000000, hub_id=1, realm_id=0)["iscc"]
    result = idx.get(iscc_id)

    assert result is None


def test_get_multiple_keys_all_exist_returns_list():
    """Multiple existing keys return list of ISCC-UNIT strings."""
    idx = UnitIndex(unit_type="META-NONE-V0", max_dim=256, realm_id=0)

    # Add multiple vectors
    units = [ic.Code.rnd(ic.MT.META, bits=128) for _ in range(3)]
    iscc_units = [f"ISCC:{unit}" for unit in units]
    ids = [ic.gen_iscc_id(timestamp=1000000 + i, hub_id=i, realm_id=0)["iscc"] for i in range(3)]

    idx.add(ids, iscc_units)

    # Get all vectors
    results = idx.get(ids)

    assert isinstance(results, list)
    assert len(results) == 3
    assert all(isinstance(r, str) for r in results)
    assert results == iscc_units


def test_get_multiple_keys_mixed_returns_list_with_none():
    """Multiple keys with some missing return list with None values."""
    idx = UnitIndex(unit_type="META-NONE-V0", max_dim=256, realm_id=0)

    # Add only some vectors
    units = [ic.Code.rnd(ic.MT.META, bits=128) for _ in range(2)]
    iscc_units = [f"ISCC:{unit}" for unit in units]
    ids = [ic.gen_iscc_id(timestamp=1000000 + i, hub_id=i, realm_id=0)["iscc"] for i in range(3)]

    idx.add([ids[0], ids[2]], [iscc_units[0], iscc_units[1]])

    # Get all vectors (including missing one)
    results = idx.get(ids)

    assert isinstance(results, list)
    assert len(results) == 3
    assert results[0] == iscc_units[0]
    assert results[1] is None
    assert results[2] == iscc_units[1]


def test_get_multiple_keys_all_missing_returns_list_of_none():
    """Multiple non-existing keys return list of None values."""
    idx = UnitIndex(unit_type="META-NONE-V0", max_dim=256, realm_id=0)

    # Try to get non-existent keys
    ids = [ic.gen_iscc_id(timestamp=1000000 + i, hub_id=i, realm_id=0)["iscc"] for i in range(3)]
    results = idx.get(ids)

    assert isinstance(results, list)
    assert len(results) == 3
    assert all(r is None for r in results)


def test_get_semantic_units():
    """Get SEMANTIC type ISCC-UNITs."""
    idx = UnitIndex(unit_type="SEMANTIC-TEXT-V0", max_dim=256, realm_id=0)

    # Add a semantic unit
    unit = ic.Code.rnd(ic.MT.SEMANTIC, ic.ST_CC.TEXT, bits=128)
    iscc_unit = f"ISCC:{unit}"
    iscc_id = ic.gen_iscc_id(timestamp=1000000, hub_id=1, realm_id=0)["iscc"]

    idx.add(iscc_id, iscc_unit)

    # Get the vector
    result = idx.get(iscc_id)

    assert result == iscc_unit


def test_get_content_units():
    """Get CONTENT type ISCC-UNITs."""
    idx = UnitIndex(unit_type="CONTENT-IMAGE-V0", max_dim=256, realm_id=0)

    # Add a content unit
    unit = ic.Code.rnd(ic.MT.CONTENT, ic.ST_CC.IMAGE, bits=128)
    iscc_unit = f"ISCC:{unit}"
    iscc_id = ic.gen_iscc_id(timestamp=1000000, hub_id=1, realm_id=0)["iscc"]

    idx.add(iscc_id, iscc_unit)

    # Get the vector
    result = idx.get(iscc_id)

    assert result == iscc_unit


def test_get_data_units():
    """Get DATA type ISCC-UNITs."""
    idx = UnitIndex(unit_type="DATA-NONE-V0", max_dim=256, realm_id=0)

    # Add a data unit
    unit = ic.Code.rnd(ic.MT.DATA, bits=128)
    iscc_unit = f"ISCC:{unit}"
    iscc_id = ic.gen_iscc_id(timestamp=1000000, hub_id=1, realm_id=0)["iscc"]

    idx.add(iscc_id, iscc_unit)

    # Get the vector
    result = idx.get(iscc_id)

    assert result == iscc_unit


def test_get_variable_length_units():
    """Get ISCC-UNITs with different bit lengths."""
    idx = UnitIndex(unit_type="META-NONE-V0", max_dim=256, realm_id=0)

    # Add units with different lengths
    bit_lengths = [64, 128, 192, 256]
    units = [ic.Code.rnd(ic.MT.META, bits=bl) for bl in bit_lengths]
    iscc_units = [f"ISCC:{unit}" for unit in units]
    ids = [ic.gen_iscc_id(timestamp=1000000 + i, hub_id=i, realm_id=0)["iscc"] for i in range(len(bit_lengths))]

    idx.add(ids, iscc_units)

    # Get all vectors
    results = idx.get(ids)

    assert results == iscc_units


def test_get_reconstructs_iscc_unit_correctly():
    """Verify that get() correctly reconstructs the original ISCC-UNIT."""
    idx = UnitIndex(unit_type="META-NONE-V0", max_dim=256, realm_id=0)

    # Create a known ISCC-UNIT
    vector_bytes = bytes([0xFF, 0xAA, 0x55, 0x00] * 4)  # 128 bits
    original_unit = f"ISCC:{ic.encode_component(ic.MT.META, ic.ST.NONE, ic.VS.V0, 128, vector_bytes)}"
    iscc_id = ic.gen_iscc_id(timestamp=1000000, hub_id=1, realm_id=0)["iscc"]

    idx.add(iscc_id, original_unit)

    # Get the vector and verify it matches
    result = idx.get(iscc_id)

    assert result == original_unit

    # Decode and verify the bytes match
    result_bytes = ic.decode_base32(result.removeprefix("ISCC:"))[2:]
    assert result_bytes == vector_bytes


def test_get_with_sample_meta_units(sample_meta_units, sample_iscc_ids):
    """Test with sample META units from fixtures."""
    idx = UnitIndex(max_dim=256, realm_id=0)

    # Add units (unit_type will be auto-detected from first unit)
    idx.add(sample_iscc_ids[:4], sample_meta_units)

    # Get all units
    results = idx.get(sample_iscc_ids[:4])

    assert results == sample_meta_units


def test_get_with_autodetect_unit_type():
    """Test get() with auto-detected unit_type."""
    idx = UnitIndex(max_dim=256, realm_id=0)

    # Add units without specifying unit_type
    units = [ic.Code.rnd(ic.MT.META, bits=128) for _ in range(3)]
    iscc_units = [f"ISCC:{unit}" for unit in units]
    ids = [ic.gen_iscc_id(timestamp=1000000 + i, hub_id=i, realm_id=0)["iscc"] for i in range(3)]

    idx.add(ids, iscc_units)

    # Verify unit_type was set
    assert idx.unit_type == "META-NONE-V0"

    # Get all units
    results = idx.get(ids)

    assert results == iscc_units


def test_get_empty_list_returns_empty_list():
    """Get with empty list returns empty list."""
    idx = UnitIndex(unit_type="META-NONE-V0", max_dim=256, realm_id=0)

    results = idx.get([])

    assert results == []


def test_get_preserves_order():
    """Get preserves the order of requested keys."""
    idx = UnitIndex(unit_type="META-NONE-V0", max_dim=256, realm_id=0)

    # Add units
    units = [ic.Code.rnd(ic.MT.META, bits=128) for _ in range(5)]
    iscc_units = [f"ISCC:{unit}" for unit in units]
    ids = [ic.gen_iscc_id(timestamp=1000000 + i, hub_id=i, realm_id=0)["iscc"] for i in range(5)]

    idx.add(ids, iscc_units)

    # Get in different order
    requested_order = [ids[2], ids[0], ids[4], ids[1], ids[3]]
    expected_order = [iscc_units[2], iscc_units[0], iscc_units[4], iscc_units[1], iscc_units[3]]

    results = idx.get(requested_order)

    assert results == expected_order
