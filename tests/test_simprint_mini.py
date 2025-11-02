"""Tests for SimprintMiniIndexRaw"""

import struct
import tempfile
from pathlib import Path
from iscc_search.simprint.mini import SimprintMiniIndexRaw, SimprintMiniIndex


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


def test_simprint_index_mini_add_and_search():
    # type: () -> None
    """Test SimprintMiniIndex add and search with ISCC-ID strings and base64 simprints."""
    with tempfile.TemporaryDirectory() as tmpdir:
        index_path = Path(tmpdir) / "test_index"

        with SimprintMiniIndex(index_path, "SEMANTIC_TEXT_V0") as index:
            # Test data with base64-encoded simprints
            iscc_id_1 = "ISCC:MAIWIDONMPAVUUAA"
            iscc_id_2 = "MAIWIDONMPAVUUAB"  # Without ISCC: prefix

            features_1 = {
                "simprints": [
                    "8IAnFvInk24iEkDGoxfPid4DLgKjoHcf9U4-_3zPEVk",
                    "GH7W703iOzPEyhD295s0nrKPNujISF5YBbWDpGwiK1Q",
                ]
            }

            features_2 = {
                "simprints": [
                    "8IAnFvInk24iEkDGoxfPid4DLgKjoHcf9U4-_3zPEVk",  # Same as first simprint of id_1
                    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
                ]
            }

            # Add both
            index.add(iscc_id_1, features_1)
            index.add(iscc_id_2, features_2)

            # Search for the shared simprint
            results = index.search(["8IAnFvInk24iEkDGoxfPid4DLgKjoHcf9U4-_3zPEVk"])

            # Should find both ISCC-IDs
            assert len(results) == 2
            assert all("iscc_id" in r and "match_count" in r for r in results)
            assert all(r["match_count"] == 1 for r in results)


def test_simprint_index_mini_search_with_limit():
    # type: () -> None
    """Test SimprintMiniIndex search with limit parameter."""
    with tempfile.TemporaryDirectory() as tmpdir:
        index_path = Path(tmpdir) / "test_index"

        with SimprintMiniIndex(index_path, "SEMANTIC_TEXT_V0") as index:
            # Add multiple ISCC-IDs (using valid base32 characters)
            base_ids = [
                "ISCC:MAIWIDONMPAVUUAA",
                "ISCC:MAIWIDONMPAVUUAB",
                "ISCC:MAIWIDONMPAVUUAC",
                "ISCC:MAIWIDONMPAVUUAD",
                "ISCC:MAIWIDONMPAVUUAE",
            ]
            for iscc_id in base_ids:
                features = {"simprints": ["8IAnFvInk24iEkDGoxfPid4DLgKjoHcf9U4-_3zPEVk"]}
                index.add(iscc_id, features)

            # Search with limit
            results = index.search(["8IAnFvInk24iEkDGoxfPid4DLgKjoHcf9U4-_3zPEVk"], limit=3)

            # Should only return 3 results
            assert len(results) == 3


def test_simprint_index_mini_no_simprints():
    # type: () -> None
    """Test SimprintMiniIndex add with empty simprints list (warning case)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        index_path = Path(tmpdir) / "test_index"

        with SimprintMiniIndex(index_path, "SEMANTIC_TEXT_V0") as index:
            # Add with empty simprints
            iscc_id = "ISCC:MAIWIDONMPAVUUAA"
            features = {"simprints": []}
            index.add(iscc_id, features)

            # Search should return no results
            results = index.search(["8IAnFvInk24iEkDGoxfPid4DLgKjoHcf9U4-_3zPEVk"])
            assert len(results) == 0


def test_simprint_index_mini_realm_id_tracking():
    # type: () -> None
    """Test that realm_id is tracked from first ISCC-ID."""
    with tempfile.TemporaryDirectory() as tmpdir:
        index_path = Path(tmpdir) / "test_index"

        with SimprintMiniIndex(index_path, "SEMANTIC_TEXT_V0") as index:
            # Initially realm_id should be None
            assert index.realm_id is None

            # Add first ISCC-ID (REALM_0)
            iscc_id = "ISCC:MAIWIDONMPAVUUAA"  # REALM_0
            features = {"simprints": ["8IAnFvInk24iEkDGoxfPid4DLgKjoHcf9U4-_3zPEVk"]}
            index.add(iscc_id, features)

            # realm_id should now be set to 0
            assert index.realm_id == 0


def test_simprint_index_mini_realm_id_persistence():
    # type: () -> None
    """Test that realm_id persists across sessions."""
    with tempfile.TemporaryDirectory() as tmpdir:
        index_path = Path(tmpdir) / "test_index"

        # First session: add data
        with SimprintMiniIndex(index_path, "SEMANTIC_TEXT_V0") as index:
            iscc_id = "ISCC:MAIWIDONMPAVUUAA"  # REALM_0
            features = {"simprints": ["8IAnFvInk24iEkDGoxfPid4DLgKjoHcf9U4-_3zPEVk"]}
            index.add(iscc_id, features)
            assert index.realm_id == 0

        # Second session: realm_id should be loaded
        with SimprintMiniIndex(index_path, "SEMANTIC_TEXT_V0") as index:
            assert index.realm_id == 0


def test_simprint_index_mini_search_empty_index_error():
    # type: () -> None
    """Test that search raises ValueError on empty index."""
    with tempfile.TemporaryDirectory() as tmpdir:
        index_path = Path(tmpdir) / "test_index"

        with SimprintMiniIndex(index_path, "SEMANTIC_TEXT_V0") as index:
            # Search on empty index should raise ValueError
            try:
                index.search(["8IAnFvInk24iEkDGoxfPid4DLgKjoHcf9U4-_3zPEVk"])
                assert False, "Expected ValueError"
            except ValueError as e:
                assert "Cannot search empty index" in str(e)


def test_simprint_index_mini_realm_id_reconstruction():
    # type: () -> None
    """Test that ISCC-IDs are correctly reconstructed with tracked realm_id."""
    from iscc_search.models import IsccID

    with tempfile.TemporaryDirectory() as tmpdir:
        index_path = Path(tmpdir) / "test_index"

        # Create ISCC-ID with REALM_1
        iscc_id_realm1 = IsccID.from_int(123456789, realm_id=1)
        iscc_id_str = str(iscc_id_realm1)

        with SimprintMiniIndex(index_path, "SEMANTIC_TEXT_V0") as index:
            features = {"simprints": ["8IAnFvInk24iEkDGoxfPid4DLgKjoHcf9U4-_3zPEVk"]}
            index.add(iscc_id_str, features)

            # realm_id should be set to 1
            assert index.realm_id == 1

            # Search should reconstruct ISCC-ID with correct realm_id
            results = index.search(["8IAnFvInk24iEkDGoxfPid4DLgKjoHcf9U4-_3zPEVk"])
            assert len(results) == 1
            assert results[0]["iscc_id"] == iscc_id_str
