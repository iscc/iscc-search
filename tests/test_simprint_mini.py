"""Tests for SimprintMiniIndexRaw"""

import struct
import tempfile
from pathlib import Path
from iscc_search.simprint.mini import SimprintMiniIndexRaw


def test_simprint_mini_index_basic():
    # type: () -> None
    """Test basic add and search operations."""
    with tempfile.TemporaryDirectory() as tmpdir:
        index_path = Path(tmpdir) / "test_index"

        # Create index
        with SimprintMiniIndexRaw(index_path, "TEST_V0") as index:
            # Create test data
            iscc_id_1 = struct.pack("<Q", 1)
            iscc_id_2 = struct.pack("<Q", 2)

            simprint_a = struct.pack("<Q", 100)
            simprint_b = struct.pack("<Q", 200)
            simprint_c = struct.pack("<Q", 300)

            # Add pairs: (simprint, iscc_id)
            pairs = [
                (simprint_a, iscc_id_1),
                (simprint_b, iscc_id_1),
                (simprint_b, iscc_id_2),
                (simprint_c, iscc_id_2),
            ]
            index.add(pairs)

            # Search for simprint_b (should match both ISCC-IDs)
            result = index.search([simprint_b])
            assert isinstance(result, list)
            assert len(result) == 2

            # Both should have count of 1
            assert result[0][1] == 1
            assert result[1][1] == 1


def test_simprint_mini_index_ranking():
    # type: () -> None
    """Test that results are ranked by match count."""
    with tempfile.TemporaryDirectory() as tmpdir:
        index_path = Path(tmpdir) / "test_index"

        with SimprintMiniIndexRaw(index_path, "TEST_V0") as index:
            iscc_id_1 = struct.pack("<Q", 1)
            iscc_id_2 = struct.pack("<Q", 2)
            iscc_id_3 = struct.pack("<Q", 3)

            simprint_a = struct.pack("<Q", 100)
            simprint_b = struct.pack("<Q", 200)
            simprint_c = struct.pack("<Q", 300)

            # ISCC-ID 1: has simprints a, b, c (3 matches)
            # ISCC-ID 2: has simprints a, b (2 matches)
            # ISCC-ID 3: has simprint a (1 match)
            pairs = [
                (simprint_a, iscc_id_1),
                (simprint_b, iscc_id_1),
                (simprint_c, iscc_id_1),
                (simprint_a, iscc_id_2),
                (simprint_b, iscc_id_2),
                (simprint_a, iscc_id_3),
            ]
            index.add(pairs)

            # Search for all three simprints
            result = index.search([simprint_a, simprint_b, simprint_c])

            # Should return 3 matches
            assert len(result) == 3

            # Check ranking: ISCC-ID 1 should be first (3 matches)
            assert result[0] == (iscc_id_1, 3)
            assert result[1] == (iscc_id_2, 2)
            assert result[2] == (iscc_id_3, 1)


def test_simprint_mini_index_no_matches():
    # type: () -> None
    """Test search with no matching simprints."""
    with tempfile.TemporaryDirectory() as tmpdir:
        index_path = Path(tmpdir) / "test_index"

        with SimprintMiniIndexRaw(index_path, "TEST_V0") as index:
            iscc_id_1 = struct.pack("<Q", 1)
            simprint_a = struct.pack("<Q", 100)
            simprint_b = struct.pack("<Q", 200)

            pairs = [(simprint_a, iscc_id_1)]
            index.add(pairs)

            # Search for non-existent simprint
            result = index.search([simprint_b])
            assert len(result) == 0


def test_simprint_mini_index_persistence():
    # type: () -> None
    """Test that index persists data across sessions."""
    with tempfile.TemporaryDirectory() as tmpdir:
        index_path = Path(tmpdir) / "test_index"

        iscc_id_1 = struct.pack("<Q", 1)
        simprint_a = struct.pack("<Q", 100)

        # Add data in first session
        with SimprintMiniIndexRaw(index_path, "TEST_V0") as index:
            pairs = [(simprint_a, iscc_id_1)]
            index.add(pairs)

        # Read data in second session
        with SimprintMiniIndexRaw(index_path, "TEST_V0") as index:
            result = index.search([simprint_a])
            assert len(result) == 1
            assert result[0] == (iscc_id_1, 1)


def test_simprint_mini_index_map_size_property():
    # type: () -> None
    """Test map_size property returns current LMDB map size."""
    with tempfile.TemporaryDirectory() as tmpdir:
        index_path = Path(tmpdir) / "test_index"

        with SimprintMiniIndexRaw(index_path, "TEST_V0") as index:
            # Check that map_size is accessible
            map_size = index.map_size
            assert isinstance(map_size, int)
            assert map_size > 0
            # Default LMDB map_size is 10MB (10,485,760 bytes)
            assert map_size == 10485760
