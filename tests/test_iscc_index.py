"""Test the IsccIndex multi-index class implementation."""

import json
import os
import tempfile
import typing
from datetime import datetime
from pathlib import Path

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
