"""Content processing functions for ISCC-Search."""

import iscc_core as ic
import xxhash


def text_chunks(text, avg_size=512):
    # type: (str, int) -> Generator[str, None, None]
    """
    Generate variable-sized text chunks using content-defined chunking.

    :param text: Input text to chunk
    :param avg_size: Target average chunk size in characters (default: 512)
    :yields: Text chunks
    """
    data = text.encode("utf-32-be")
    avg_size_bytes = avg_size * 4  # 4 bytes per character in utf-32-be
    for chunk_bytes in ic.alg_cdc_chunks(data, utf32=True, avg_chunk_size=avg_size_bytes):
        yield chunk_bytes.decode("utf-32-be")


def text_simprints(text, avg_chunk_size=512, ngram_size=13):
    # type: (str, int, int) -> list[str]
    """
    Generate CONTENT_TEXT_V0 simprints from plain text.

    The text is cleaned, chunked using content-defined chunking, and each chunk
    is processed to produce a minhash-based simprint for similarity search.

    :param text: Plain text input
    :param avg_chunk_size: Target average chunk size in characters (default: 512)
    :param ngram_size: Size of character n-grams for feature extraction (default: 13)
    :return: List of base64-encoded simprint strings
    """
    # Clean text before processing
    cleaned_text = ic.text_clean(text)

    simprints = []
    for chunk in text_chunks(cleaned_text, avg_size=avg_chunk_size):
        # Generate n-grams from collapsed/normalized text
        ngrams = ("".join(chars) for chars in ic.sliding_window(ic.text_collapse(chunk), ngram_size))
        # Hash each n-gram
        features = [xxhash.xxh32_intdigest(s.encode("utf-8")) for s in ngrams]
        # Apply minhash to create similarity-preserving fingerprint
        minimum_hash_digest = ic.alg_minhash_256(features)
        # Encode as base64 simprint
        simprints.append(ic.encode_base64(minimum_hash_digest))

    return simprints
