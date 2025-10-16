"""Tests for UnitIndex.add() method."""

import iscc_core as ic
import pytest

from iscc_vdb import UnitIndex


def test_add_single_unit_with_auto_key(sample_meta_units):
    """Test adding a single ISCC-UNIT with auto-generated key."""
    idx = UnitIndex()
    unit = sample_meta_units[0]

    result = idx.add(None, unit)

    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0].startswith("ISCC:")
    assert idx.realm_id == 0  # Auto-set to 0 for auto-generated keys
    assert idx.unit_type is not None


def test_add_single_unit_with_explicit_key(sample_iscc_ids, sample_meta_units):
    """Test adding a single ISCC-UNIT with explicit ISCC-ID key."""
    idx = UnitIndex()
    iscc_id = sample_iscc_ids[0]
    unit = sample_meta_units[0]

    result = idx.add(iscc_id, unit)

    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0] == iscc_id
    assert idx.realm_id == 0  # Extracted from ISCC-ID
    assert idx.unit_type is not None


def test_add_multiple_units_with_auto_keys(sample_meta_units):
    """Test adding multiple ISCC-UNITs with auto-generated keys."""
    idx = UnitIndex()
    units = sample_meta_units[:3]

    result = idx.add(None, units)

    assert isinstance(result, list)
    assert len(result) == 3
    for iscc_id in result:
        assert iscc_id.startswith("ISCC:")
    assert idx.realm_id == 0


def test_add_multiple_units_with_explicit_keys(sample_iscc_ids, sample_meta_units):
    """Test adding multiple ISCC-UNITs with explicit ISCC-ID keys."""
    idx = UnitIndex()
    iscc_ids = sample_iscc_ids[:3]
    units = sample_meta_units[:3]

    result = idx.add(iscc_ids, units)

    assert isinstance(result, list)
    assert len(result) == 3
    assert result == iscc_ids


def test_unit_type_auto_detection(sample_meta_units):
    """Test that unit_type is auto-detected from first ISCC-UNIT."""
    idx = UnitIndex()
    assert idx.unit_type is None

    idx.add(None, sample_meta_units[0])

    assert idx.unit_type is not None
    assert idx.unit_type == "META-NONE-V0"


def test_realm_id_auto_detection_from_key(sample_iscc_ids, sample_meta_units):
    """Test that realm_id is auto-detected from first ISCC-ID."""
    idx = UnitIndex()
    assert idx.realm_id is None

    idx.add(sample_iscc_ids[0], sample_meta_units[0])

    assert idx.realm_id == 0


def test_realm_id_defaults_to_zero_for_auto_keys(sample_meta_units):
    """Test that realm_id defaults to 0 when keys are auto-generated."""
    idx = UnitIndex()
    assert idx.realm_id is None

    idx.add(None, sample_meta_units[0])

    assert idx.realm_id == 0


def test_explicit_unit_type_preserved(sample_meta_units):
    """Test that explicitly set unit_type is not overridden."""
    idx = UnitIndex(unit_type="META-NONE-V0")

    idx.add(None, sample_meta_units[0])

    assert idx.unit_type == "META-NONE-V0"


def test_explicit_realm_id_preserved(sample_iscc_ids, sample_meta_units):
    """Test that explicitly set realm_id is not overridden."""
    idx = UnitIndex(realm_id=5)

    idx.add(sample_iscc_ids[0], sample_meta_units[0])

    assert idx.realm_id == 5


def test_returned_keys_are_valid_iscc_ids(sample_meta_units):
    """Test that returned keys are valid ISCC-IDs."""
    idx = UnitIndex()
    units = sample_meta_units[:3]

    result = idx.add(None, units)

    for iscc_id in result:
        # Verify it can be decoded as ISCC-ID
        mt, st, vs, ln, body = ic.iscc_decode(iscc_id)
        assert mt == ic.MT.ID
        assert vs == ic.VS.V1


def test_add_semantic_units(sample_semantic_units):
    """Test adding SEMANTIC type ISCC-UNITs."""
    idx = UnitIndex()

    result = idx.add(None, sample_semantic_units[:3])

    assert len(result) == 3
    assert idx.unit_type.startswith("SEMANTIC-")


def test_add_content_units(sample_content_units):
    """Test adding CONTENT type ISCC-UNITs."""
    idx = UnitIndex()

    result = idx.add(None, sample_content_units[:3])

    assert len(result) == 3
    assert idx.unit_type.startswith("CONTENT-")


def test_add_data_units(sample_data_units):
    """Test adding DATA type ISCC-UNITs."""
    idx = UnitIndex()

    result = idx.add(None, sample_data_units[:3])

    assert len(result) == 3
    assert idx.unit_type == "DATA-NONE-V0"


def test_add_units_of_different_lengths(sample_meta_units):
    """Test adding ISCC-UNITs with different bit lengths."""
    idx = UnitIndex()
    # sample_meta_units has units with lengths: 64, 128, 192, 256
    all_units = sample_meta_units

    result = idx.add(None, all_units)

    assert len(result) == len(all_units)
    # All should have same unit_type
    assert idx.unit_type == "META-NONE-V0"


def test_auto_generated_keys_are_sequential(sample_meta_units):
    """Test that auto-generated ISCC-ID keys use sequential integer keys internally."""
    idx = UnitIndex()
    units = sample_meta_units[:3]

    result = idx.add(None, units)

    # Verify that internally the keys are 0, 1, 2
    assert len(idx) == 3
    # Get the internal keys by decoding ISCC-IDs
    for i, iscc_id in enumerate(result):
        body = ic.decode_base32(iscc_id.removeprefix("ISCC:"))[2:]
        key = int.from_bytes(body, "big", signed=False)
        assert key == i


def test_explicit_keys_preserved_in_result(sample_iscc_ids, sample_meta_units):
    """Test that explicit ISCC-ID keys are preserved in result."""
    idx = UnitIndex()
    iscc_ids = sample_iscc_ids[:5]
    units = sample_meta_units[:4] + [sample_meta_units[0]]  # 5 units

    result = idx.add(iscc_ids, units)

    assert result == iscc_ids


def test_mixed_add_operations(sample_iscc_ids, sample_meta_units):
    """Test mixed auto-generated and explicit keys across multiple add operations."""
    idx = UnitIndex()

    # First add with auto-generated keys
    result1 = idx.add(None, sample_meta_units[0])
    assert len(result1) == 1

    # Second add with explicit key
    result2 = idx.add(sample_iscc_ids[0], sample_meta_units[1])
    assert len(result2) == 1
    assert result2[0] == sample_iscc_ids[0]

    # Third add with auto-generated keys again
    result3 = idx.add(None, sample_meta_units[2])
    assert len(result3) == 1

    # Total of 3 vectors in index
    assert len(idx) == 3


def test_iscc_id_header_requires_realm_id():
    """Test that accessing _iscc_id_header without realm_id raises ValueError."""
    idx = UnitIndex()

    with pytest.raises(ValueError, match="realm_id must be set"):
        _ = idx._iscc_id_header
