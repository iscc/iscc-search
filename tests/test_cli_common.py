"""Tests for CLI common functions."""

import iscc_core as ic
from iscc_search.cli.common import parse_simprints_from_features


def test_parse_simprints_from_features_basic():
    """Test basic simprint parsing without truncation."""
    features = [
        {
            "maintype": "semantic",
            "subtype": "text",
            "version": 0,
            "simprints": ["abc123", "def456"],
            "offsets": [0, 100],
            "sizes": [50, 60],
        }
    ]

    result = parse_simprints_from_features(features)

    assert result is not None
    assert "SEMANTIC_TEXT_V0" in result
    assert len(result["SEMANTIC_TEXT_V0"]) == 2
    assert result["SEMANTIC_TEXT_V0"][0]["simprint"] == "abc123"
    assert result["SEMANTIC_TEXT_V0"][0]["offset"] == 0
    assert result["SEMANTIC_TEXT_V0"][0]["size"] == 50


def test_parse_simprints_from_features_with_truncation():
    """Test simprint parsing with truncation from 256-bit to 64-bit."""
    # Create a 256-bit (32 byte) simprint
    simprint_256bit = ic.encode_base64(b"x" * 32)

    features = [
        {
            "maintype": "semantic",
            "subtype": "text",
            "version": 0,
            "simprints": [simprint_256bit],
            "offsets": [0],
            "sizes": [100],
        }
    ]

    # Parse without truncation
    result_256 = parse_simprints_from_features(features, simprint_bits=None)
    original = result_256["SEMANTIC_TEXT_V0"][0]["simprint"]
    assert len(ic.decode_base64(original)) == 32  # 256 bits = 32 bytes

    # Parse with 64-bit truncation
    result_64 = parse_simprints_from_features(features, simprint_bits=64)
    truncated = result_64["SEMANTIC_TEXT_V0"][0]["simprint"]
    assert len(ic.decode_base64(truncated)) == 8  # 64 bits = 8 bytes

    # Verify truncated matches first 8 bytes of original
    assert ic.decode_base64(truncated) == ic.decode_base64(original)[:8]


def test_parse_simprints_from_features_truncation_128bit():
    """Test simprint truncation to 128 bits."""
    simprint_256bit = ic.encode_base64(b"y" * 32)

    features = [
        {
            "maintype": "content",
            "subtype": "text",
            "version": 0,
            "simprints": [simprint_256bit],
            "offsets": [0],
            "sizes": [200],
        }
    ]

    result = parse_simprints_from_features(features, simprint_bits=128)
    truncated = result["CONTENT_TEXT_V0"][0]["simprint"]
    assert len(ic.decode_base64(truncated)) == 16  # 128 bits = 16 bytes


def test_parse_simprints_from_features_truncation_too_small():
    """Test that truncation fails when simprint is too small."""
    # Create a 64-bit (8 byte) simprint
    simprint_64bit = ic.encode_base64(b"z" * 8)

    features = [
        {
            "maintype": "semantic",
            "subtype": "text",
            "version": 0,
            "simprints": [simprint_64bit, ic.encode_base64(b"a" * 32)],  # One too small, one ok
            "offsets": [0, 100],
            "sizes": [100, 200],
        }
    ]

    # Try to truncate to 128 bits - should skip the first (too small) simprint
    result = parse_simprints_from_features(features, simprint_bits=128)

    # Should only have 1 simprint (the second one that was large enough)
    assert len(result["SEMANTIC_TEXT_V0"]) == 1
    assert result["SEMANTIC_TEXT_V0"][0]["offset"] == 100  # The second one


def test_parse_simprints_from_features_multiple_types():
    """Test parsing multiple simprint types with truncation."""
    simprint_256bit = ic.encode_base64(b"m" * 32)

    features = [
        {
            "maintype": "semantic",
            "subtype": "text",
            "version": 0,
            "simprints": [simprint_256bit],
            "offsets": [0],
            "sizes": [100],
        },
        {
            "maintype": "content",
            "subtype": "text",
            "version": 0,
            "simprints": [simprint_256bit],
            "offsets": [200],
            "sizes": [150],
        },
    ]

    result = parse_simprints_from_features(features, simprint_bits=64)

    assert "SEMANTIC_TEXT_V0" in result
    assert "CONTENT_TEXT_V0" in result
    assert len(ic.decode_base64(result["SEMANTIC_TEXT_V0"][0]["simprint"])) == 8
    assert len(ic.decode_base64(result["CONTENT_TEXT_V0"][0]["simprint"])) == 8


def test_parse_simprints_from_features_empty():
    """Test parsing empty features."""
    result = parse_simprints_from_features([])
    assert result is None


def test_parse_simprints_from_features_missing_fields():
    """Test parsing with missing required fields."""
    features = [
        {
            "maintype": "semantic",
            # Missing subtype
            "version": 0,
            "simprints": ["abc"],
            "offsets": [0],
            "sizes": [100],
        }
    ]

    result = parse_simprints_from_features(features)
    # Should return None or empty dict when essential fields are missing
    assert result is None or len(result) == 0
