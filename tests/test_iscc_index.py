"""Test the IsccIndex multi-index class implementation."""

import io
import json
import os
import tempfile
import typing
from datetime import datetime
from pathlib import Path

import iscc_core as ic
import platformdirs
import pytest

from iscc_vdb.iscc_index import IsccIndex


def test_init_with_custom_path(tmp_path):
    # type: (typing.Any) -> None
    """Test initialization with a custom path."""
    custom_path = tmp_path / "custom_index"
    index = IsccIndex(custom_path)

    assert index.path == custom_path
    assert index.max_bits == 256
    assert custom_path.exists()
    assert (custom_path / "index.json").exists()

    # Check metadata file
    with open(custom_path / "index.json") as f:
        metadata = json.load(f)

    assert metadata["max_bits"] == 256
    assert metadata["version"] == "0.0.1"
    assert "created" in metadata
    # Verify timestamp is recent (within last minute)
    created_time = datetime.fromisoformat(metadata["created"].replace("Z", "+00:00"))
    assert (datetime.now(created_time.tzinfo) - created_time).total_seconds() < 60


def test_init_with_default_path():
    # type: () -> None
    """Test initialization without providing a path (uses default)."""
    index = IsccIndex()

    # Should use platformdirs default location
    expected_base = Path(platformdirs.user_data_dir("iscc-vdb", "iscc"))
    expected_path = expected_base / "default"

    assert index.path == expected_path
    assert index.max_bits == 256

    # Clean up default directory
    if expected_path.exists():
        import shutil

        shutil.rmtree(expected_path)
        # Also try to clean up parent if empty
        if expected_base.exists() and not any(expected_base.iterdir()):
            expected_base.rmdir()


def test_init_with_string_path(tmp_path):
    # type: (typing.Any) -> None
    """Test initialization with string path instead of Path object."""
    custom_path = str(tmp_path / "string_path_index")
    index = IsccIndex(custom_path)

    assert index.path == Path(custom_path)
    assert Path(custom_path).exists()


def test_init_with_custom_max_bits(tmp_path):
    # type: (typing.Any) -> None
    """Test initialization with custom max_bits."""
    custom_path = tmp_path / "custom_bits_index"
    index = IsccIndex(custom_path, max_bits=128)

    assert index.max_bits == 128

    # Check metadata file
    with open(custom_path / "index.json") as f:
        metadata = json.load(f)

    assert metadata["max_bits"] == 128


def test_load_existing_metadata(tmp_path):
    # type: (typing.Any) -> None
    """Test loading from existing metadata file."""
    # Create index first
    custom_path = tmp_path / "existing_index"
    IsccIndex(custom_path, max_bits=192)

    # Create another instance that should load existing metadata
    index2 = IsccIndex(custom_path, max_bits=256)  # Different max_bits

    # Should use max_bits from metadata, not constructor argument
    assert index2.max_bits == 192


def test_metadata_validation_error(tmp_path):
    # type: (typing.Any) -> None
    """Test error handling for invalid metadata."""
    custom_path = tmp_path / "invalid_metadata_index"
    custom_path.mkdir(parents=True)

    # Create invalid metadata (missing max_bits)
    metadata_path = custom_path / "index.json"
    with open(metadata_path, "w") as f:
        json.dump({"version": "0.0.1"}, f)

    with pytest.raises(ValueError, match="Invalid metadata.*missing max_bits"):
        IsccIndex(custom_path)


def test_metadata_json_decode_error(tmp_path):
    # type: (typing.Any) -> None
    """Test error handling for corrupted metadata JSON."""
    custom_path = tmp_path / "corrupted_metadata_index"
    custom_path.mkdir(parents=True)

    # Create corrupted JSON
    metadata_path = custom_path / "index.json"
    with open(metadata_path, "w") as f:
        f.write("{ invalid json")

    with pytest.raises(RuntimeError, match="Failed to load metadata"):
        IsccIndex(custom_path)


def test_metadata_read_permission_error(tmp_path):
    # type: (typing.Any) -> None
    """Test error handling for permission errors reading metadata."""
    custom_path = tmp_path / "permission_error_index"
    custom_path.mkdir(parents=True)

    # Create metadata file
    metadata_path = custom_path / "index.json"
    with open(metadata_path, "w") as f:
        json.dump({"max_bits": 256, "version": "0.0.1"}, f)

    # Make file unreadable (skip on Windows as it handles permissions differently)
    if os.name != "nt":
        os.chmod(metadata_path, 0o000)

        try:
            with pytest.raises(RuntimeError, match="Failed to load metadata"):
                IsccIndex(custom_path)
        finally:
            # Restore permissions for cleanup
            os.chmod(metadata_path, 0o644)


def test_directory_creation_error():
    # type: () -> None
    """Test error handling when directory creation fails."""
    # Use a path that cannot be created
    if os.name == "nt":
        # Windows: use invalid path characters
        invalid_path = "C:\\<>:|?*invalid"
    else:
        # Unix: use a file as parent directory
        with tempfile.NamedTemporaryFile(delete=False) as f:
            invalid_path = f.name + "/subdir"

    try:
        with pytest.raises(RuntimeError, match="Failed to create index directory"):
            IsccIndex(invalid_path)
    finally:
        # Clean up temp file on Unix
        if os.name != "nt" and os.path.exists(f.name):
            os.unlink(f.name)


def test_metadata_write_permission_error(tmp_path):
    # type: (typing.Any) -> None
    """Test error handling for permission errors writing metadata."""
    # Skip on Windows as it handles permissions differently
    if os.name == "nt":
        return

    custom_path = tmp_path / "write_permission_error"
    custom_path.mkdir(parents=True)

    # Make directory read-only
    os.chmod(custom_path, 0o555)  # noqa: S103

    try:
        with pytest.raises(RuntimeError, match="Failed to create metadata file"):
            IsccIndex(custom_path)
    finally:
        # Restore permissions for cleanup
        os.chmod(custom_path, 0o755)  # noqa: S103


def test_indices_initialization(tmp_path):
    # type: (typing.Any) -> None
    """Test that indices dictionary is properly initialized."""
    custom_path = tmp_path / "indices_test"
    index = IsccIndex(custom_path)

    assert isinstance(index.indices, dict)
    assert len(index.indices) == 0


def test_multiple_instances_same_path(tmp_path):
    # type: (typing.Any) -> None
    """Test multiple instances can access the same index path."""
    custom_path = tmp_path / "shared_index"

    # Create first instance
    index1 = IsccIndex(custom_path, max_bits=128)

    # Create second instance - should load existing metadata
    index2 = IsccIndex(custom_path, max_bits=256)

    # Both should have same configuration from metadata
    assert index1.max_bits == 128
    assert index2.max_bits == 128


def test_parent_directory_creation(tmp_path):
    # type: (typing.Any) -> None
    """Test that parent directories are created if they don't exist."""
    custom_path = tmp_path / "deep" / "nested" / "path" / "index"

    # Parent directories don't exist yet
    assert not custom_path.parent.exists()

    IsccIndex(custom_path)

    # All parent directories should be created
    assert custom_path.exists()
    assert (custom_path / "index.json").exists()


def test_metadata_same_max_bits(tmp_path):
    # type: (typing.Any) -> None
    """Test loading metadata when max_bits matches constructor argument."""
    custom_path = tmp_path / "same_max_bits_index"

    # Create first instance with max_bits=256
    index1 = IsccIndex(custom_path, max_bits=256)

    # Create second instance with same max_bits
    # This tests the branch where metadata["max_bits"] == self.max_bits
    index2 = IsccIndex(custom_path, max_bits=256)

    # Both should have the same max_bits
    assert index1.max_bits == 256
    assert index2.max_bits == 256


# Tests for Checkpoint 2: ISCC Processing and Validation


def test_validate_iscc_valid(tmp_path):
    # type: (typing.Any) -> None
    """Test validation of valid ISCC strings."""
    index = IsccIndex(tmp_path / "validate_test")

    # Valid single component ISCCs
    assert index._validate_iscc("ISCC:KACT4EBWK27737D2AYCJRAL5Z36G76RFRMO4554RU26HZ4ORJGIVHDI") is True
    assert index._validate_iscc("ISCC:AAAT4EBWK27737D2") is True
    assert index._validate_iscc("ISCC:GAAT4EBWK27737D2") is True
    assert index._validate_iscc("ISCC:EAAT4EBWK27737D2") is True
    assert index._validate_iscc("ISCC:IAAT4EBWK27737D2") is True

    # Valid ISCC-ID
    assert index._validate_iscc("ISCC:MAAGZTFQTTVIZPJR") is True

    # Valid composite ISCC
    assert index._validate_iscc("ISCC:KECWJHBQRJCQG7HGLC7M3NRHWFR6BMBHOJ2T2PC6GG7F5QI4TJEJV5I") is True


def test_validate_iscc_invalid(tmp_path):
    # type: (typing.Any) -> None
    """Test validation of invalid ISCC strings."""
    index = IsccIndex(tmp_path / "validate_invalid_test")

    # Invalid ISCCs
    assert index._validate_iscc("not_an_iscc") is False
    assert index._validate_iscc("ISCC:INVALID") is False
    assert index._validate_iscc("") is False
    assert index._validate_iscc("ISCC:") is False
    assert index._validate_iscc("ISCC:12345") is False  # Invalid base32

    # Edge cases
    assert index._validate_iscc(None) is False
    assert index._validate_iscc(123) is False


def test_decompose_iscc_single_component(tmp_path):
    # type: (typing.Any) -> None
    """Test decomposition of single-component ISCCs."""
    index = IsccIndex(tmp_path / "decompose_single_test")

    # Single Meta-Code
    components = index._decompose_iscc("ISCC:AAAT4EBWK27737D2")
    assert len(components) == 1
    # Components are returned without ISCC: prefix
    assert components[0][0] == "AAAT4EBWK27737D2"
    assert "meta" in components[0][1].lower()

    # Single Content-Code
    components = index._decompose_iscc("ISCC:EAAT4EBWK27737D2")
    assert len(components) == 1
    assert components[0][0] == "EAAT4EBWK27737D2"
    assert "content" in components[0][1].lower()


def test_decompose_iscc_multi_component(tmp_path):
    # type: (typing.Any) -> None
    """Test decomposition of multi-component ISCCs."""
    index = IsccIndex(tmp_path / "decompose_multi_test")

    # Create a composite ISCC from multiple components
    # ISCC-CODE requires at least DATA and INSTANCE components
    test_data = io.BytesIO(b"Test content for ISCC")
    meta = ic.gen_meta_code("Test Title")
    content = ic.gen_text_code("Test content for ISCC")
    data = ic.gen_data_code(test_data)
    test_data.seek(0)  # Reset for instance code
    instance = ic.gen_instance_code(test_data)

    # Create composite with all components
    composite = ic.gen_iscc_code([meta["iscc"], content["iscc"], data["iscc"], instance["iscc"]])

    components = index._decompose_iscc(composite["iscc"])
    assert len(components) == 4

    # Check that we get all components back
    type_ids = [comp[1] for comp in components]
    assert any("meta" in tid.lower() for tid in type_ids)
    assert any("content" in tid.lower() for tid in type_ids)
    assert any("data" in tid.lower() for tid in type_ids)
    assert any("instance" in tid.lower() for tid in type_ids)


def test_iscc_id_conversions(tmp_path):
    # type: (typing.Any) -> None
    """Test ISCC-ID to uint64 conversions and back."""
    index = IsccIndex(tmp_path / "conversion_test")

    # Test with a known ISCC-ID
    iscc_id = "ISCC:MAAGZTFQTTVIZPJR"

    # Convert to uint64
    uint64_val = index._iscc_id_to_uint64(iscc_id)
    assert isinstance(uint64_val, int)
    assert 0 <= uint64_val < 2**64

    # Convert back to ISCC-ID
    converted_back = index._uint64_to_iscc_id(uint64_val)
    assert converted_back == iscc_id or converted_back == "MAAGZTFQTTVIZPJR"  # May or may not have prefix

    # Normalize both for comparison
    assert ic.iscc_normalize(converted_back) == ic.iscc_normalize(iscc_id)


def test_iscc_id_conversions_without_prefix(tmp_path):
    # type: (typing.Any) -> None
    """Test ISCC-ID conversions without ISCC: prefix."""
    index = IsccIndex(tmp_path / "conversion_no_prefix_test")

    # Test without prefix
    iscc_id = "MAAGZTFQTTVIZPJR"

    uint64_val = index._iscc_id_to_uint64(iscc_id)
    converted_back = index._uint64_to_iscc_id(uint64_val)

    # Should work the same way
    assert ic.iscc_normalize(converted_back) == ic.iscc_normalize(iscc_id)


def test_iscc_id_conversion_roundtrip(tmp_path):
    # type: (typing.Any) -> None
    """Test multiple ISCC-ID conversion roundtrips."""
    index = IsccIndex(tmp_path / "roundtrip_test")

    # Test with v0 ISCC-IDs (these should roundtrip perfectly)
    test_ids_v0 = []
    for i in range(3):
        # Create v0 ISCC-ID by encoding directly
        from iscc_core.codec import encode_component
        from iscc_core.constants import MT, ST_ID

        test_bytes = (1000000 + i).to_bytes(8, byteorder="big")
        iscc_id = encode_component(MT.ID, ST_ID.PRIVATE, 0, 64, test_bytes)
        test_ids_v0.append(iscc_id)

    for iscc_id in test_ids_v0:
        # Convert to uint64 and back
        uint64_val = index._iscc_id_to_uint64(iscc_id)
        converted_back = index._uint64_to_iscc_id(uint64_val)

        # v0 IDs should roundtrip exactly
        assert ic.iscc_normalize(converted_back) == ic.iscc_normalize(iscc_id)

        # Double roundtrip
        uint64_val2 = index._iscc_id_to_uint64(converted_back)
        assert uint64_val == uint64_val2

    # Test with v1 ISCC-IDs (these convert to v0 format)
    test_ids_v1 = []
    for i in range(3):
        timestamp = 1714503123456789 + i * 1000000
        iscc_id_dict = ic.gen_iscc_id_v1(timestamp=timestamp, server_id=i)
        test_ids_v1.append(iscc_id_dict["iscc"])

    for iscc_id in test_ids_v1:
        # Convert to uint64
        uint64_val = index._iscc_id_to_uint64(iscc_id)
        converted_back = index._uint64_to_iscc_id(uint64_val)

        # v1 IDs get converted to v0 format, so they won't match exactly
        # But the uint64 value should be preserved
        uint64_val2 = index._iscc_id_to_uint64(converted_back)
        assert uint64_val == uint64_val2


def test_iscc_id_invalid_length(tmp_path):
    # type: (typing.Any) -> None
    """Test error handling for invalid ISCC-ID length."""
    index = IsccIndex(tmp_path / "invalid_length_test")

    # Too short ISCC-ID - this will raise a padding error from base32 decode
    import binascii

    with pytest.raises((ValueError, binascii.Error)):
        index._iscc_id_to_uint64("ISCC:MAA")  # Too short


def test_type_id_extraction(tmp_path):
    # type: (typing.Any) -> None
    """Test type_id extraction from different ISCC components."""
    index = IsccIndex(tmp_path / "type_id_test")

    # Generate various ISCC components
    meta = ic.gen_meta_code("Test")
    content = ic.gen_text_code("Test content")
    data = ic.gen_data_code(io.BytesIO(b"test data"))
    instance = ic.gen_instance_code(io.BytesIO(b"test instance"))

    # Test decomposition extracts correct type_ids
    for iscc_dict in [meta, content, data, instance]:
        components = index._decompose_iscc(iscc_dict["iscc"])
        assert len(components) == 1
        component_code, type_id = components[0]
        # Component codes are returned without ISCC: prefix
        assert ic.iscc_normalize(component_code) == ic.iscc_normalize(iscc_dict["iscc"])
        assert isinstance(type_id, str)
        assert len(type_id) > 0
