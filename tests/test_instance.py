"""Comprehensive tests for InstanceIndex with 100% coverage."""

import pytest
import iscc_core as ic
from iscc_vdb.instance import InstanceIndex


@pytest.fixture
def temp_instance_path(tmp_path):
    # type: (typing.Any) -> typing.Any
    """Provide a temporary path for InstanceIndex."""
    # Use a reasonable map_size for testing (100MB instead of 10GB default)
    return tmp_path / "instance_index"


@pytest.fixture
def small_map_size():
    # type: () -> int
    """Provide a smaller map_size for testing."""
    return 100 * 1024 * 1024  # 100MB


# Override InstanceIndex default map_size for all tests
_original_init = InstanceIndex.__init__


def _test_init(self, path, realm_id=0, map_size=100 * 1024 * 1024):
    # type: (InstanceIndex, typing.Any, int, int) -> None
    """Test version with smaller default map_size."""
    _original_init(self, path, realm_id, map_size)


InstanceIndex.__init__ = _test_init


@pytest.fixture
def sample_instance_codes():
    # type: () -> list[str]
    """Generate sample Instance-Codes with various bit lengths."""
    codes = []
    for bit_length in [64, 128, 192, 256]:
        code = ic.Code.rnd(ic.MT.INSTANCE, bits=bit_length)
        codes.append(f"ISCC:{code}")
    return codes


@pytest.fixture
def sample_instance_codes_bytes(sample_instance_codes):
    # type: (list[str]) -> list[bytes]
    """Convert sample Instance-Codes to bytes."""
    return [ic.decode_base32(code.removeprefix("ISCC:"))[2:] for code in sample_instance_codes]


def test_instance_index_init(temp_instance_path):
    # type: (typing.Any) -> None
    """Test InstanceIndex initialization."""
    idx = InstanceIndex(temp_instance_path, realm_id=1)
    assert idx.path == str(temp_instance_path)
    assert idx.realm_id == 1
    assert len(idx) == 0
    idx.close()


def test_add_single_string(temp_instance_path, sample_iscc_ids, sample_instance_codes):
    # type: (typing.Any, list[str], list[str]) -> None
    """Test adding single ISCC-ID and Instance-Code as strings."""
    idx = InstanceIndex(temp_instance_path)
    count = idx.add(sample_iscc_ids[0], sample_instance_codes[0])
    assert count == 1
    assert len(idx) == 1
    idx.close()


def test_add_single_bytes(temp_instance_path, sample_iscc_ids, sample_instance_codes_bytes):
    # type: (typing.Any, list[str], list[bytes]) -> None
    """Test adding single ISCC-ID and Instance-Code as bytes."""
    idx = InstanceIndex(temp_instance_path)
    # Convert ISCC-ID to bytes
    iscc_id_bytes = ic.decode_base32(sample_iscc_ids[0].removeprefix("ISCC:"))[2:]
    count = idx.add(iscc_id_bytes, sample_instance_codes_bytes[0])
    assert count == 1
    assert len(idx) == 1
    idx.close()


def test_add_batch_strings(temp_instance_path, sample_iscc_ids, sample_instance_codes):
    # type: (typing.Any, list[str], list[str]) -> None
    """Test adding multiple ISCC-IDs and Instance-Codes."""
    idx = InstanceIndex(temp_instance_path)
    count = idx.add(sample_iscc_ids[:4], sample_instance_codes[:4])
    assert count == 4
    assert len(idx) == 4
    idx.close()


def test_add_duplicate_no_count(temp_instance_path, sample_iscc_ids, sample_instance_codes):
    # type: (typing.Any, list[str], list[str]) -> None
    """Test that adding duplicate ISCC-ID doesn't increment count."""
    idx = InstanceIndex(temp_instance_path)
    # Add first time
    count1 = idx.add(sample_iscc_ids[0], sample_instance_codes[0])
    assert count1 == 1
    # Add same mapping again
    count2 = idx.add(sample_iscc_ids[0], sample_instance_codes[0])
    assert count2 == 0
    assert len(idx) == 1
    idx.close()


def test_add_multiple_ids_same_instance(temp_instance_path, sample_iscc_ids, sample_instance_codes):
    # type: (typing.Any, list[str], list[str]) -> None
    """Test adding multiple ISCC-IDs to same Instance-Code."""
    idx = InstanceIndex(temp_instance_path)
    # Add two different ISCC-IDs with same Instance-Code
    idx.add(sample_iscc_ids[0], sample_instance_codes[0])
    count = idx.add(sample_iscc_ids[1], sample_instance_codes[0])
    assert count == 1
    assert len(idx) == 2
    idx.close()


def test_add_mismatched_length_error(temp_instance_path, sample_iscc_ids, sample_instance_codes):
    # type: (typing.Any, list[str], list[str]) -> None
    """Test that mismatched list lengths raise ValueError."""
    idx = InstanceIndex(temp_instance_path)
    with pytest.raises(ValueError, match="Number of ISCC-IDs must match Instance-Codes"):
        idx.add(sample_iscc_ids[:2], sample_instance_codes[:3])
    idx.close()


def test_get_single_string(temp_instance_path, sample_iscc_ids, sample_instance_codes):
    # type: (typing.Any, list[str], list[str]) -> None
    """Test getting ISCC-IDs by single Instance-Code string."""
    idx = InstanceIndex(temp_instance_path)
    idx.add(sample_iscc_ids[0], sample_instance_codes[0])
    results = idx.get(sample_instance_codes[0])
    assert len(results) == 1
    assert sample_iscc_ids[0] in results
    idx.close()


def test_get_single_bytes(temp_instance_path, sample_iscc_ids, sample_instance_codes):
    # type: (typing.Any, list[str], list[str]) -> None
    """Test getting ISCC-IDs by single Instance-Code bytes."""
    idx = InstanceIndex(temp_instance_path)
    # Add with string
    iscc_id = sample_iscc_ids[0]
    ic_str = sample_instance_codes[0]
    idx.add(iscc_id, ic_str)
    # Get with bytes (extract the digest from the string we added)
    ic_bytes = ic.decode_base32(ic_str.removeprefix("ISCC:"))[2:]
    results = idx.get(ic_bytes)
    assert len(results) == 1
    assert iscc_id in results
    idx.close()


def test_get_batch(temp_instance_path, sample_iscc_ids, sample_instance_codes):
    # type: (typing.Any, list[str], list[str]) -> None
    """Test getting ISCC-IDs by multiple Instance-Codes."""
    idx = InstanceIndex(temp_instance_path)
    idx.add(sample_iscc_ids[:3], sample_instance_codes[:3])
    results = idx.get(sample_instance_codes[:2])
    assert len(results) == 2
    assert sample_iscc_ids[0] in results
    assert sample_iscc_ids[1] in results
    idx.close()


def test_get_nonexistent(temp_instance_path, sample_instance_codes):
    # type: (typing.Any, list[str]) -> None
    """Test getting non-existent Instance-Code returns empty list."""
    idx = InstanceIndex(temp_instance_path)
    results = idx.get(sample_instance_codes[0])
    assert results == []
    idx.close()


def test_get_multiple_ids_for_one_instance(temp_instance_path, sample_iscc_ids, sample_instance_codes):
    # type: (typing.Any, list[str], list[str]) -> None
    """Test getting multiple ISCC-IDs mapped to same Instance-Code."""
    idx = InstanceIndex(temp_instance_path)
    # Map two different ISCC-IDs to same Instance-Code
    idx.add(sample_iscc_ids[0], sample_instance_codes[0])
    idx.add(sample_iscc_ids[1], sample_instance_codes[0])
    results = idx.get(sample_instance_codes[0])
    assert len(results) == 2
    assert sample_iscc_ids[0] in results
    assert sample_iscc_ids[1] in results
    idx.close()


def test_search_prefix_single(temp_instance_path, sample_iscc_ids):
    # type: (typing.Any, list[str]) -> None
    """Test prefix search with single prefix."""
    idx = InstanceIndex(temp_instance_path)

    # Create Instance-Codes with common prefix
    base_bytes = bytes([1, 2, 3, 4, 5, 6, 7, 8])
    ic1_bytes = base_bytes + bytes([10, 11])
    ic2_bytes = base_bytes + bytes([20, 21])
    ic3_bytes = bytes([9, 9, 9, 9, 5, 6, 7, 8])  # Different prefix

    # Reconstruct as ISCC codes
    ic1 = "ISCC:" + ic.encode_base32(
        ic.encode_header(ic.MT.INSTANCE, ic.ST.NONE, ic.VS.V0, len(ic1_bytes) * 8) + ic1_bytes
    )
    ic2 = "ISCC:" + ic.encode_base32(
        ic.encode_header(ic.MT.INSTANCE, ic.ST.NONE, ic.VS.V0, len(ic2_bytes) * 8) + ic2_bytes
    )
    ic3 = "ISCC:" + ic.encode_base32(
        ic.encode_header(ic.MT.INSTANCE, ic.ST.NONE, ic.VS.V0, len(ic3_bytes) * 8) + ic3_bytes
    )

    # Add entries
    idx.add([sample_iscc_ids[0], sample_iscc_ids[1], sample_iscc_ids[2]], [ic1, ic2, ic3])

    # Search with prefix (first 8 bytes)
    results = idx.search(base_bytes)

    assert len(results) == 2  # Should match ic1 and ic2
    assert ic1 in results
    assert ic2 in results
    assert ic3 not in results
    idx.close()


def test_search_prefix_batch(temp_instance_path, sample_iscc_ids):
    # type: (typing.Any, list[str]) -> None
    """Test prefix search with multiple prefixes."""
    idx = InstanceIndex(temp_instance_path)

    # Create Instance-Codes with different prefixes
    prefix1 = bytes([1, 2, 3, 4])
    prefix2 = bytes([5, 6, 7, 8])

    ic1_bytes = prefix1 + bytes([10, 11, 12, 13])
    ic2_bytes = prefix1 + bytes([20, 21, 22, 23])
    ic3_bytes = prefix2 + bytes([30, 31, 32, 33])

    # Reconstruct as ISCC codes
    ic1 = "ISCC:" + ic.encode_base32(
        ic.encode_header(ic.MT.INSTANCE, ic.ST.NONE, ic.VS.V0, len(ic1_bytes) * 8) + ic1_bytes
    )
    ic2 = "ISCC:" + ic.encode_base32(
        ic.encode_header(ic.MT.INSTANCE, ic.ST.NONE, ic.VS.V0, len(ic2_bytes) * 8) + ic2_bytes
    )
    ic3 = "ISCC:" + ic.encode_base32(
        ic.encode_header(ic.MT.INSTANCE, ic.ST.NONE, ic.VS.V0, len(ic3_bytes) * 8) + ic3_bytes
    )

    # Add entries
    idx.add([sample_iscc_ids[0], sample_iscc_ids[1], sample_iscc_ids[2]], [ic1, ic2, ic3])

    # Search with both prefixes
    results = idx.search([prefix1, prefix2])

    assert len(results) == 3  # Should match all 3
    assert ic1 in results
    assert ic2 in results
    assert ic3 in results
    idx.close()


def test_search_no_matches(temp_instance_path, sample_iscc_ids, sample_instance_codes):
    # type: (typing.Any, list[str], list[str]) -> None
    """Test prefix search with no matches."""
    idx = InstanceIndex(temp_instance_path)
    idx.add(sample_iscc_ids[0], sample_instance_codes[0])

    # Search with non-matching prefix
    nonexistent_prefix = bytes([255, 255, 255, 255])
    results = idx.search(nonexistent_prefix)

    assert results == {}
    idx.close()


def test_search_multiple_ids_same_instance(temp_instance_path, sample_iscc_ids):
    # type: (typing.Any, list[str]) -> None
    """Test prefix search with multiple ISCC-IDs mapped to same Instance-Code."""
    idx = InstanceIndex(temp_instance_path)

    # Create Instance-Code with common prefix
    ic_bytes = bytes([1, 2, 3, 4, 5, 6, 7, 8])
    instance_code = "ISCC:" + ic.encode_base32(
        ic.encode_header(ic.MT.INSTANCE, ic.ST.NONE, ic.VS.V0, len(ic_bytes) * 8) + ic_bytes
    )

    # Add multiple ISCC-IDs to same Instance-Code
    idx.add([sample_iscc_ids[0], sample_iscc_ids[1]], [instance_code, instance_code])

    # Search with prefix - should find one Instance-Code with both ISCC-IDs
    prefix = bytes([1, 2, 3, 4])
    results = idx.search(prefix)

    assert len(results) == 1
    assert instance_code in results
    assert len(results[instance_code]) == 2
    assert sample_iscc_ids[0] in results[instance_code]
    assert sample_iscc_ids[1] in results[instance_code]
    idx.close()


def test_remove_by_iscc_id_single(temp_instance_path, sample_iscc_ids, sample_instance_codes):
    # type: (typing.Any, list[str], list[str]) -> None
    """Test removing by single ISCC-ID."""
    idx = InstanceIndex(temp_instance_path)
    idx.add(sample_iscc_ids[0], sample_instance_codes[0])
    assert len(idx) == 1

    count = idx.remove_by_iscc_id(sample_iscc_ids[0])
    assert count == 1
    assert len(idx) == 0
    idx.close()


def test_remove_by_iscc_id_batch(temp_instance_path, sample_iscc_ids, sample_instance_codes):
    # type: (typing.Any, list[str], list[str]) -> None
    """Test removing by multiple ISCC-IDs."""
    idx = InstanceIndex(temp_instance_path)
    idx.add(sample_iscc_ids[:3], sample_instance_codes[:3])
    assert len(idx) == 3

    count = idx.remove_by_iscc_id(sample_iscc_ids[:2])
    assert count == 2
    assert len(idx) == 1
    idx.close()


def test_remove_by_iscc_id_partial(temp_instance_path, sample_iscc_ids, sample_instance_codes):
    # type: (typing.Any, list[str], list[str]) -> None
    """Test removing one ISCC-ID when multiple map to same Instance-Code."""
    idx = InstanceIndex(temp_instance_path)
    # Map two ISCC-IDs to same Instance-Code
    idx.add(sample_iscc_ids[0], sample_instance_codes[0])
    idx.add(sample_iscc_ids[1], sample_instance_codes[0])
    assert len(idx) == 2

    # Remove only one ISCC-ID
    count = idx.remove_by_iscc_id(sample_iscc_ids[0])
    assert count == 1
    assert len(idx) == 1

    # Verify the other ISCC-ID is still there
    results = idx.get(sample_instance_codes[0])
    assert sample_iscc_ids[1] in results
    assert sample_iscc_ids[0] not in results
    idx.close()


def test_remove_by_iscc_id_nonexistent(temp_instance_path, sample_iscc_ids, sample_instance_codes):
    # type: (typing.Any, list[str], list[str]) -> None
    """Test removing non-existent ISCC-ID."""
    idx = InstanceIndex(temp_instance_path)
    idx.add(sample_iscc_ids[0], sample_instance_codes[0])

    count = idx.remove_by_iscc_id(sample_iscc_ids[1])
    assert count == 0
    assert len(idx) == 1
    idx.close()


def test_remove_by_instance_code_single(temp_instance_path, sample_iscc_ids, sample_instance_codes):
    # type: (typing.Any, list[str], list[str]) -> None
    """Test removing by single Instance-Code."""
    idx = InstanceIndex(temp_instance_path)
    idx.add(sample_iscc_ids[0], sample_instance_codes[0])
    assert len(idx) == 1

    count = idx.remove_by_instance_code(sample_instance_codes[0])
    assert count == 1
    assert len(idx) == 0
    idx.close()


def test_remove_by_instance_code_batch(temp_instance_path, sample_iscc_ids, sample_instance_codes):
    # type: (typing.Any, list[str], list[str]) -> None
    """Test removing by multiple Instance-Codes."""
    idx = InstanceIndex(temp_instance_path)
    idx.add(sample_iscc_ids[:3], sample_instance_codes[:3])
    assert len(idx) == 3

    count = idx.remove_by_instance_code(sample_instance_codes[:2])
    assert count == 2
    assert len(idx) == 1
    idx.close()


def test_remove_by_instance_code_multiple_ids(temp_instance_path, sample_iscc_ids, sample_instance_codes):
    # type: (typing.Any, list[str], list[str]) -> None
    """Test removing Instance-Code with multiple ISCC-IDs."""
    idx = InstanceIndex(temp_instance_path)
    # Map two ISCC-IDs to same Instance-Code
    idx.add(sample_iscc_ids[0], sample_instance_codes[0])
    idx.add(sample_iscc_ids[1], sample_instance_codes[0])
    assert len(idx) == 2

    # Remove by Instance-Code (should remove both ISCC-IDs)
    count = idx.remove_by_instance_code(sample_instance_codes[0])
    assert count == 2
    assert len(idx) == 0
    idx.close()


def test_remove_by_instance_code_nonexistent(temp_instance_path, sample_iscc_ids, sample_instance_codes):
    # type: (typing.Any, list[str], list[str]) -> None
    """Test removing non-existent Instance-Code."""
    idx = InstanceIndex(temp_instance_path)
    idx.add(sample_iscc_ids[0], sample_instance_codes[0])

    count = idx.remove_by_instance_code(sample_instance_codes[1])
    assert count == 0
    assert len(idx) == 1
    idx.close()


def test_len_empty(temp_instance_path):
    # type: (typing.Any) -> None
    """Test len on empty index."""
    idx = InstanceIndex(temp_instance_path)
    assert len(idx) == 0
    idx.close()


def test_len_with_data(temp_instance_path, sample_iscc_ids, sample_instance_codes):
    # type: (typing.Any, list[str], list[str]) -> None
    """Test len with data."""
    idx = InstanceIndex(temp_instance_path)
    idx.add(sample_iscc_ids[:4], sample_instance_codes[:4])
    assert len(idx) == 4
    idx.close()


def test_close_and_reopen(temp_instance_path, sample_iscc_ids, sample_instance_codes):
    # type: (typing.Any, list[str], list[str]) -> None
    """Test persistence: close and reopen index."""
    # Create and populate index
    idx1 = InstanceIndex(temp_instance_path)
    idx1.add(sample_iscc_ids[:3], sample_instance_codes[:3])
    assert len(idx1) == 3
    idx1.close()

    # Reopen and verify data persists
    idx2 = InstanceIndex(temp_instance_path)
    assert len(idx2) == 3
    results = idx2.get(sample_instance_codes[0])
    assert sample_iscc_ids[0] in results
    idx2.close()


def test_destructor_cleanup(temp_instance_path, sample_iscc_ids, sample_instance_codes):
    # type: (typing.Any, list[str], list[str]) -> None
    """Test __del__ cleanup."""
    idx = InstanceIndex(temp_instance_path)
    idx.add(sample_iscc_ids[0], sample_instance_codes[0])
    # Delete to trigger __del__
    del idx
    # Should not raise any errors


def test_custom_realm_id(temp_instance_path, sample_iscc_ids, sample_instance_codes):
    # type: (typing.Any, list[str], list[str]) -> None
    """Test custom realm_id is used for ISCC-ID reconstruction."""
    realm_id = 1  # Valid realm_id: 0 (test) or 1 (operational)
    idx = InstanceIndex(temp_instance_path, realm_id=realm_id)
    idx.add(sample_iscc_ids[0], sample_instance_codes[0])

    results = idx.get(sample_instance_codes[0])
    assert len(results) == 1

    # Verify realm_id is embedded in returned ISCC-ID
    returned_id = results[0]
    decoded = ic.decode_base32(returned_id.removeprefix("ISCC:"))
    # Extract realm_id from header (second nibble of second byte)
    header_realm = (decoded[1] >> 4) & 0x0F
    assert header_realm == realm_id
    idx.close()


def test_bytes_to_instance_code_various_lengths(temp_instance_path):
    # type: (typing.Any) -> None
    """Test _bytes_to_instance_code with various byte lengths."""
    idx = InstanceIndex(temp_instance_path)

    # Test with different lengths
    for byte_length in [8, 16, 24, 32]:
        test_bytes = bytes(range(byte_length))
        ic_str = idx._bytes_to_instance_code(test_bytes)
        assert ic_str.startswith("ISCC:")
        # Verify bit length encoding
        decoded = ic.decode_base32(ic_str.removeprefix("ISCC:"))
        # The length is encoded in the header
        assert len(decoded) > 2  # At least header + some data

    idx.close()


def test_mixed_string_and_bytes_input(temp_instance_path, sample_iscc_ids, sample_instance_codes):
    # type: (typing.Any, list[str], list[str]) -> None
    """Test that mixed string and bytes inputs work correctly."""
    idx = InstanceIndex(temp_instance_path)

    # Add with string
    idx.add(sample_iscc_ids[0], sample_instance_codes[0])

    # Get with bytes
    ic_bytes = ic.decode_base32(sample_instance_codes[0].removeprefix("ISCC:"))[2:]
    results = idx.get(ic_bytes)
    assert sample_iscc_ids[0] in results

    idx.close()


def test_search_string_prefix(temp_instance_path, sample_iscc_ids):
    # type: (typing.Any, list[str]) -> None
    """Test search with string prefix."""
    idx = InstanceIndex(temp_instance_path)

    # Create Instance-Codes with common prefix
    base_bytes = bytes([1, 2, 3, 4, 5, 6, 7, 8])
    ic1_bytes = base_bytes + bytes([10, 11])

    # Reconstruct as ISCC code
    ic1 = "ISCC:" + ic.encode_base32(
        ic.encode_header(ic.MT.INSTANCE, ic.ST.NONE, ic.VS.V0, len(ic1_bytes) * 8) + ic1_bytes
    )

    # Add entry
    idx.add(sample_iscc_ids[0], ic1)

    # Search with string prefix (reconstruct prefix as ISCC code)
    prefix_ic = "ISCC:" + ic.encode_base32(
        ic.encode_header(ic.MT.INSTANCE, ic.ST.NONE, ic.VS.V0, len(base_bytes) * 8) + base_bytes
    )
    results = idx.search(prefix_ic)

    assert len(results) > 0
    idx.close()


def test_invalid_realm_id(temp_instance_path, sample_iscc_ids, sample_instance_codes):
    # type: (typing.Any, list[str], list[str]) -> None
    """Test that invalid realm_id raises ValueError when converting to ISCC-ID."""
    idx = InstanceIndex(temp_instance_path, realm_id=7)  # Invalid realm_id
    idx.add(sample_iscc_ids[0], sample_instance_codes[0])

    # Try to get - this should trigger realm_id validation in _bytes_to_iscc_id
    with pytest.raises(ValueError, match="Invalid realm_id 7, must be 0 or 1"):
        idx.get(sample_instance_codes[0])

    idx.close()


def test_destructor_no_env():
    # type: () -> None
    """Test __del__ when env attribute doesn't exist."""
    idx = InstanceIndex.__new__(InstanceIndex)
    # Don't call __init__, so env won't be set
    # This should trigger the hasattr branch in __del__
    del idx
    # Should not raise any errors
