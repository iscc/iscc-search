"""Tests for content processing functions."""

from iscc_search.processing import text_chunks, text_simprints


def test_text_chunks_basic():
    """Test basic text chunking with short text."""
    text = "Hello world! " * 100  # Repeat to create content
    chunks = list(text_chunks(text, avg_size=50))

    # Should produce multiple chunks
    assert len(chunks) > 1
    # Chunks should not be empty
    assert all(len(chunk) > 0 for chunk in chunks)
    # All chunks together should reconstruct the original text
    assert "".join(chunks) == text


def test_text_chunks_empty():
    """Test chunking with empty text."""
    chunks = list(text_chunks(""))
    # CDC algorithm produces one empty chunk for empty input
    assert len(chunks) == 1
    assert chunks[0] == ""


def test_text_chunks_short():
    """Test chunking with text shorter than chunk size."""
    text = "Short text"
    chunks = list(text_chunks(text, avg_size=512))
    # Short text should produce single chunk
    assert len(chunks) == 1
    assert chunks[0] == text


def test_text_chunks_unicode():
    """Test chunking with unicode characters."""
    text = "Hello 世界! " * 100  # Mix ASCII and Chinese
    chunks = list(text_chunks(text, avg_size=50))

    # Should handle unicode properly
    assert len(chunks) > 1
    assert "".join(chunks) == text


def test_text_simprints_basic():
    """Test basic simprint generation."""
    text = "This is a test document. " * 50  # Enough text for chunking
    simprints = text_simprints(text)

    # Should produce simprints
    assert len(simprints) > 0
    # All simprints should be non-empty strings
    assert all(isinstance(s, str) and len(s) > 0 for s in simprints)
    # Simprints should be base64-encoded (alphanumeric + special chars)
    assert all(
        all(c in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/_=-" for c in s) for s in simprints
    )


def test_text_simprints_short():
    """Test simprint generation with short text."""
    text = "Short text"
    simprints = text_simprints(text)

    # Even short text should produce at least one simprint
    assert len(simprints) >= 1


def test_text_simprints_long():
    """Test simprint generation with longer text produces multiple simprints."""
    # Create text that will definitely span multiple chunks
    # Use smaller chunk size to ensure multiple chunks with reasonable text length
    text = "Lorem ipsum dolor sit amet. " * 100  # ~2800 chars
    simprints = text_simprints(text, avg_chunk_size=200)

    # Should produce multiple simprints with smaller chunk size
    assert len(simprints) > 1


def test_text_simprints_cleaned():
    """Test that text is cleaned before processing."""
    # Text with extra whitespace and special chars
    text = "This   has    extra  spaces\n\nand newlines.\t\tTabs too!"
    simprints = text_simprints(text)

    # Should produce valid simprints despite messy input
    assert len(simprints) > 0
    assert all(isinstance(s, str) for s in simprints)


def test_text_simprints_unicode():
    """Test simprint generation with unicode text."""
    text = "Hello 世界! This is a test with unicode characters. " * 50
    simprints = text_simprints(text)

    # Should handle unicode properly
    assert len(simprints) > 0
    assert all(isinstance(s, str) for s in simprints)


def test_text_simprints_deterministic():
    """Test that same text produces same simprints."""
    text = "This is a test document. " * 50
    simprints1 = text_simprints(text)
    simprints2 = text_simprints(text)

    # Same input should produce same output
    assert simprints1 == simprints2


def test_text_simprints_different():
    """Test that different text produces different simprints."""
    text1 = "This is test document A. " * 50
    text2 = "This is test document B. " * 50
    simprints1 = text_simprints(text1)
    simprints2 = text_simprints(text2)

    # Different text should produce different simprints
    assert simprints1 != simprints2


def test_text_simprints_custom_params():
    """Test simprint generation with custom parameters."""
    text = "This is a test document. " * 100
    # Smaller chunks should produce more simprints
    simprints_small = text_simprints(text, avg_chunk_size=256)
    simprints_large = text_simprints(text, avg_chunk_size=1024)

    # Smaller chunks should generally produce more simprints
    # (not guaranteed due to CDC variability, but likely)
    assert len(simprints_small) >= len(simprints_large)
