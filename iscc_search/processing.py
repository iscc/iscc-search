"""Content processing functions for ISCC-Search."""

import iscc_core as ic
import xxhash

# Try to import iscc-sct for semantic simprint generation
try:
    import iscc_sct as sct

    HAS_ISCC_SCT = True  # pragma: no cover
except ImportError:  # pragma: no cover
    sct = None  # type: ignore
    HAS_ISCC_SCT = False


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
    # type: (str, int, int) -> dict[str, list[str]]
    """
    Generate simprints from plain text.

    Generates CONTENT_TEXT_V0 simprints using minhash-based approach.
    If iscc-sct is installed, also generates SEMANTIC_TEXT_V0 simprints.

    :param text: Plain text input
    :param avg_chunk_size: Target average chunk size in characters (default: 512)
    :param ngram_size: Size of character n-grams for feature extraction (default: 13)
    :return: Dictionary mapping simprint types to lists of base64-encoded simprints
    """
    result = {}

    # Generate CONTENT_TEXT_V0 simprints
    cleaned_text = ic.text_clean(text)
    content_simprints = []
    for chunk in text_chunks(cleaned_text, avg_size=avg_chunk_size):
        # Generate n-grams from collapsed/normalized text
        ngrams = ("".join(chars) for chars in ic.sliding_window(ic.text_collapse(chunk), ngram_size))
        # Hash each n-gram
        features = [xxhash.xxh32_intdigest(s.encode("utf-8")) for s in ngrams]
        # Apply minhash to create similarity-preserving fingerprint
        minimum_hash_digest = ic.alg_minhash_256(features)
        # Encode as base64 simprint
        content_simprints.append(ic.encode_base64(minimum_hash_digest))

    result["CONTENT_TEXT_V0"] = content_simprints

    # Generate SEMANTIC_TEXT_V0 simprints if iscc-sct is available
    if HAS_ISCC_SCT:  # pragma: no cover
        semantic_result = sct.gen_text_code_semantic(text, simprints=True, bits_granular=256)
        if semantic_result.get("features") and len(semantic_result["features"]) > 0:
            semantic_simprints = semantic_result["features"][0].get("simprints", [])
            if semantic_simprints:
                result["SEMANTIC_TEXT_V0"] = semantic_simprints

    return result
