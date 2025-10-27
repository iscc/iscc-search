"""Unit tests for UnitIndex class."""

import iscc_core as ic

from iscc_search import UnitIndex


def test_unit_index_init_default():
    """Test UnitIndex initialization with default parameters."""
    index = UnitIndex()

    assert index.max_dim == 256
    assert index.max_bytes == 32
    assert index.ndim == 264  # 256 + 8 bits for length signal
    assert index.unit_type is None
    assert index.realm_id is None


def test_unit_index_init_with_unit_type():
    """Test UnitIndex initialization with explicit unit_type string."""
    index = UnitIndex(unit_type="META_NONE_V0")

    assert index.unit_type == "META_NONE_V0"
    assert index.realm_id is None
    assert index.max_dim == 256


def test_unit_index_init_with_realm_id():
    """Test UnitIndex initialization with explicit realm_id."""
    index = UnitIndex(realm_id=1)

    assert index.unit_type is None
    assert index.realm_id == 1
    assert index.max_dim == 256


def test_unit_index_init_with_both_parameters():
    """Test UnitIndex initialization with both unit_type and realm_id."""
    index = UnitIndex(unit_type="SEMANTIC_TEXT_V0", realm_id=1)

    assert index.unit_type == "SEMANTIC_TEXT_V0"
    assert index.realm_id == 1
    assert index.max_dim == 256


def test_unit_index_init_custom_max_dim():
    """Test UnitIndex initialization with custom max_dim."""
    index = UnitIndex(max_dim=128)

    assert index.max_dim == 128
    assert index.max_bytes == 16
    assert index.ndim == 136  # 128 + 8 bits
    assert index.unit_type is None
    assert index.realm_id is None


def test_unit_index_init_all_parameters():
    """Test UnitIndex initialization with all parameters."""
    index = UnitIndex(unit_type="CONTENT_IMAGE_V0", max_dim=192, realm_id=1)

    assert index.unit_type == "CONTENT_IMAGE_V0"
    assert index.max_dim == 192
    assert index.max_bytes == 24
    assert index.realm_id == 1
    assert index.ndim == 200  # 192 + 8 bits


def test_unit_index_init_realm_id_zero():
    """Test UnitIndex initialization with realm_id=0."""
    index = UnitIndex(realm_id=0)

    assert index.realm_id == 0
    assert index.unit_type is None


def test_unit_index_init_realm_id_max():
    """Test UnitIndex initialization with maximum realm_id (15)."""
    index = UnitIndex(realm_id=15)

    assert index.realm_id == 15
    assert index.unit_type is None


def test_unit_index_init_various_unit_types():
    """Test UnitIndex initialization with various unit type strings."""
    # META
    index_meta = UnitIndex(unit_type="META_NONE_V0")
    assert index_meta.unit_type == "META_NONE_V0"

    # SEMANTIC TEXT
    index_sem = UnitIndex(unit_type="SEMANTIC_TEXT_V0")
    assert index_sem.unit_type == "SEMANTIC_TEXT_V0"

    # CONTENT IMAGE
    index_cont = UnitIndex(unit_type="CONTENT_IMAGE_V0")
    assert index_cont.unit_type == "CONTENT_IMAGE_V0"

    # DATA
    index_data = UnitIndex(unit_type="DATA_NONE_V0")
    assert index_data.unit_type == "DATA_NONE_V0"


def test_unit_index_inherits_from_nphd():
    """Test that UnitIndex properly inherits from NphdIndex."""
    index = UnitIndex()

    # Check inherited attributes
    assert hasattr(index, "max_dim")
    assert hasattr(index, "max_bytes")
    assert hasattr(index, "ndim")
    assert hasattr(index, "add")
    assert hasattr(index, "search")
    assert hasattr(index, "get")


def test_unit_index_init_with_kwargs():
    """Test UnitIndex initialization with additional kwargs passed to NphdIndex."""
    index = UnitIndex(connectivity=32, expansion_add=64, expansion_search=128)

    assert index.connectivity == 32
    assert index.expansion_add == 64
    assert index.expansion_search == 128
    assert index.unit_type is None
    assert index.realm_id is None


def test_unit_index_unit_type_extraction():
    """Test extracting unit_type string from ISCC-UNIT."""
    # Create sample ISCC-UNIT
    unit = ic.Code.rnd(ic.MT.META, bits=128)
    iscc_unit = f"ISCC:{unit}"

    # Extract unit_type using the same method as will be used in add()
    # ic.iscc_type_id returns "META_NONE_V0-128", we need "META_NONE_V0"
    unit_type = "_".join(ic.iscc_type_id(iscc_unit).split("-")[:-1])

    assert unit_type == "META_NONE_V0"

    # Test with SEMANTIC TEXT
    unit_sem = ic.Code.rnd(ic.MT.SEMANTIC, ic.ST_CC.TEXT, bits=128)
    iscc_unit_sem = f"ISCC:{unit_sem}"
    unit_type_sem = "_".join(ic.iscc_type_id(iscc_unit_sem).split("-")[:-1])

    assert unit_type_sem == "SEMANTIC_TEXT_V0"
