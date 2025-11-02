"""Tests for SimprintMiniIndex"""

import struct
import tempfile
from pathlib import Path
from dataclasses import dataclass
from iscc_search.simprint.mini import SimprintMiniIndex, SimprintMiniResult


@dataclass
class SampleItem:
    """Sample item implementing SimprintMiniItem protocol."""

    iscc_id: bytes
    simprints: list[bytes]


def test_simprint_mini_index_basic():
    # type: () -> None
    """Test basic add and search operations."""
    with tempfile.TemporaryDirectory() as tmpdir:
        index_path = Path(tmpdir) / "test_index"

        # Create index
        with SimprintMiniIndex(index_path, "TEST_V0") as index:
            # Create test data
            iscc_id_1 = struct.pack("<Q", 1)
            iscc_id_2 = struct.pack("<Q", 2)

            simprint_a = struct.pack("<Q", 100)
            simprint_b = struct.pack("<Q", 200)
            simprint_c = struct.pack("<Q", 300)

            # Add items
            items = [
                SampleItem(iscc_id_1, [simprint_a, simprint_b]),
                SampleItem(iscc_id_2, [simprint_b, simprint_c]),
            ]
            index.add(items)

            # Search for simprint_b (should match both ISCC-IDs)
            result = index.search([simprint_b])
            assert isinstance(result, SimprintMiniResult)
            assert len(result.matches) == 2

            # Both should have count of 1
            assert result.matches[0][1] == 1
            assert result.matches[1][1] == 1


def test_simprint_mini_index_ranking():
    # type: () -> None
    """Test that results are ranked by match count."""
    with tempfile.TemporaryDirectory() as tmpdir:
        index_path = Path(tmpdir) / "test_index"

        with SimprintMiniIndex(index_path, "TEST_V0") as index:
            iscc_id_1 = struct.pack("<Q", 1)
            iscc_id_2 = struct.pack("<Q", 2)
            iscc_id_3 = struct.pack("<Q", 3)

            simprint_a = struct.pack("<Q", 100)
            simprint_b = struct.pack("<Q", 200)
            simprint_c = struct.pack("<Q", 300)

            # ISCC-ID 1: has simprints a, b, c (3 matches)
            # ISCC-ID 2: has simprints a, b (2 matches)
            # ISCC-ID 3: has simprint a (1 match)
            items = [
                SampleItem(iscc_id_1, [simprint_a, simprint_b, simprint_c]),
                SampleItem(iscc_id_2, [simprint_a, simprint_b]),
                SampleItem(iscc_id_3, [simprint_a]),
            ]
            index.add(items)

            # Search for all three simprints
            result = index.search([simprint_a, simprint_b, simprint_c])

            # Should return 3 matches
            assert len(result.matches) == 3

            # Check ranking: ISCC-ID 1 should be first (3 matches)
            assert result.matches[0] == (iscc_id_1, 3)
            assert result.matches[1] == (iscc_id_2, 2)
            assert result.matches[2] == (iscc_id_3, 1)


def test_simprint_mini_index_no_matches():
    # type: () -> None
    """Test search with no matching simprints."""
    with tempfile.TemporaryDirectory() as tmpdir:
        index_path = Path(tmpdir) / "test_index"

        with SimprintMiniIndex(index_path, "TEST_V0") as index:
            iscc_id_1 = struct.pack("<Q", 1)
            simprint_a = struct.pack("<Q", 100)
            simprint_b = struct.pack("<Q", 200)

            items = [SampleItem(iscc_id_1, [simprint_a])]
            index.add(items)

            # Search for non-existent simprint
            result = index.search([simprint_b])
            assert len(result.matches) == 0


def test_simprint_mini_index_persistence():
    # type: () -> None
    """Test that index persists data across sessions."""
    with tempfile.TemporaryDirectory() as tmpdir:
        index_path = Path(tmpdir) / "test_index"

        iscc_id_1 = struct.pack("<Q", 1)
        simprint_a = struct.pack("<Q", 100)

        # Add data in first session
        with SimprintMiniIndex(index_path, "TEST_V0") as index:
            items = [SampleItem(iscc_id_1, [simprint_a])]
            index.add(items)

        # Read data in second session
        with SimprintMiniIndex(index_path, "TEST_V0") as index:
            result = index.search([simprint_a])
            assert len(result.matches) == 1
            assert result.matches[0] == (iscc_id_1, 1)


def test_simprint_mini_result_repr():
    # type: () -> None
    """Test SimprintMiniResult string representation."""
    matches = [(struct.pack("<Q", 1), 5), (struct.pack("<Q", 2), 3)]
    result = SimprintMiniResult(matches)
    assert repr(result) == "SimprintMiniResult(matches=2)"
