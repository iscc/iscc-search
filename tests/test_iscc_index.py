"""Test the IsccIndex multi-index class implementation."""

import json
import tempfile
import typing
from pathlib import Path

import pytest

from iscc_vdb.iscc_index import IsccIndex


def test_init_with_valid_path():
    # type: () -> None
    """Test IsccIndex initialization with valid path."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        index_path = Path(tmp_dir) / "test_index"

        # Create index
        index = IsccIndex(index_path)

        # Check initialization
        assert index.path == index_path
        assert index.max_bits == 256  # default
        assert isinstance(index.indices, dict)
        assert len(index.indices) == 0

        # Check directory was created
        assert index_path.exists()
        assert index_path.is_dir()


def test_init_with_string_path():
    # type: () -> None
    """Test IsccIndex initialization with string path."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        index_path = str(Path(tmp_dir) / "test_index")

        # Create index
        index = IsccIndex(index_path)

        # Check path conversion
        assert index.path == Path(index_path)
        assert index.path.exists()


def test_init_with_custom_max_bits():
    # type: () -> None
    """Test IsccIndex initialization with custom max_bits."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        index_path = Path(tmp_dir) / "test_index"

        # Create index with custom max_bits
        index = IsccIndex(index_path, max_bits=128)

        # Check max_bits
        assert index.max_bits == 128


def test_directory_creation():
    # type: () -> None
    """Test directory creation including nested paths."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create nested path
        index_path = Path(tmp_dir) / "nested" / "path" / "test_index"

        # Create index
        IsccIndex(index_path)

        # Check all directories were created
        assert index_path.exists()
        assert index_path.is_dir()
        assert index_path.parent.exists()
        assert index_path.parent.parent.exists()


def test_metadata_file_creation():
    # type: () -> None
    """Test metadata file creation with correct content."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        index_path = Path(tmp_dir) / "test_index"

        # Create index
        IsccIndex(index_path, max_bits=128)

        # Check metadata file exists
        metadata_path = index_path / "index.json"
        assert metadata_path.exists()

        # Check metadata content
        with metadata_path.open("r") as f:
            metadata = json.load(f)

        assert "max_bits" in metadata
        assert metadata["max_bits"] == 128
        assert "version" in metadata
        assert metadata["version"] == "0.0.1"
        assert "created" in metadata


def test_reinit_with_existing_directory():
    # type: () -> None
    """Test re-initialization with existing directory and metadata."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        index_path = Path(tmp_dir) / "test_index"

        # Create first index
        index1 = IsccIndex(index_path, max_bits=128)

        # Create second index with same path
        index2 = IsccIndex(index_path, max_bits=256)

        # Check that max_bits was loaded from metadata (should be 128)
        assert index2.max_bits == 128

        # Check both indices point to same directory
        assert index1.path == index2.path


def test_init_with_existing_metadata():
    # type: () -> None
    """Test initialization with existing metadata file."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        index_path = Path(tmp_dir) / "test_index"
        index_path.mkdir(parents=True)

        # Create metadata file manually
        metadata_path = index_path / "index.json"
        metadata = {"max_bits": 192, "version": "0.0.1", "created": "2024-01-01T00:00:00Z"}
        with metadata_path.open("w") as f:
            json.dump(metadata, f)

        # Create index (should load existing metadata)
        index = IsccIndex(index_path, max_bits=256)

        # Check max_bits was loaded from metadata
        assert index.max_bits == 192


def test_init_with_invalid_metadata():
    # type: () -> None
    """Test initialization with invalid metadata file."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        index_path = Path(tmp_dir) / "test_index"
        index_path.mkdir(parents=True)

        # Create invalid metadata file
        metadata_path = index_path / "index.json"
        with metadata_path.open("w") as f:
            f.write("invalid json")

        # Should raise RuntimeError
        with pytest.raises(RuntimeError, match="Failed to load metadata"):
            IsccIndex(index_path)


def test_init_with_missing_max_bits_in_metadata():
    # type: () -> None
    """Test initialization with metadata missing max_bits."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        index_path = Path(tmp_dir) / "test_index"
        index_path.mkdir(parents=True)

        # Create metadata without max_bits
        metadata_path = index_path / "index.json"
        metadata = {"version": "0.0.1"}
        with metadata_path.open("w") as f:
            json.dump(metadata, f)

        # Should raise ValueError
        with pytest.raises(ValueError, match="Invalid metadata"):
            IsccIndex(index_path)


def test_init_with_readonly_directory():
    # type: () -> None
    """Test initialization with read-only directory."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create directory and make it read-only
        index_path = Path(tmp_dir) / "readonly"
        index_path.mkdir()
        index_path.chmod(0o444)

        try:
            # Should raise RuntimeError for directory creation
            with pytest.raises(RuntimeError, match="Failed to create index directory"):
                IsccIndex(index_path / "test_index")
        finally:
            # Restore permissions for cleanup
            index_path.chmod(0o755)
