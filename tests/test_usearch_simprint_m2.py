"""M2: Integration tests for UsearchIndex simprint search."""

import pytest
from iscc_search.indexes.usearch.index import UsearchIndex
from iscc_search.schema import IsccQuery


def test_usearch_simprint_search(tmp_path, sample_assets_with_simprints, sample_simprints):
    """Test searching with simprints populates chunk_matches."""
    index = UsearchIndex(path=tmp_path / "test_index")
    index.add_assets(sample_assets_with_simprints)

    # Create query with simprints
    query = IsccQuery(simprints={"CONTENT_TEXT_V0": [sample_simprints["CONTENT_TEXT_V0"][0]["simprint"]]})

    # Search
    result = index.search_assets(query, limit=10)

    # Verify chunk_matches populated
    assert len(result.chunk_matches) > 0

    # Verify metadata enriched
    for match in result.chunk_matches:
        assert match.iscc_id.startswith("ISCC:")
        assert match.score >= 0.0 and match.score <= 1.0
        # Metadata may be present (as IsccMetadataModel or dict)
        if match.metadata:
            assert hasattr(match.metadata, "title") or "title" in match.metadata

    # Verify simprints are base64 encoded
    for match in result.chunk_matches:
        for type_stats in match.types.values():
            if type_stats.chunks:
                for chunk in type_stats.chunks:
                    assert isinstance(chunk.query, str)
                    assert isinstance(chunk.match, str)

    index.close()


def test_usearch_simprints_only_query(tmp_path, sample_assets_with_simprints, sample_simprints):
    """Test query with only simprints (no units or iscc_code)."""
    index = UsearchIndex(path=tmp_path / "test_index")
    index.add_assets(sample_assets_with_simprints)

    # Create simprints-only query
    query = IsccQuery(simprints={"CONTENT_TEXT_V0": [sample_simprints["CONTENT_TEXT_V0"][0]["simprint"]]})

    # Search
    result = index.search_assets(query, limit=10)

    # Verify chunk_matches returned, global_matches may be empty
    assert len(result.chunk_matches) > 0
    assert len(result.global_matches) == 0  # No units in query

    index.close()


def test_usearch_mixed_query(tmp_path, sample_assets_with_simprints, sample_content_units, sample_simprints):
    """Test query with both units and simprints."""
    index = UsearchIndex(path=tmp_path / "test_index")
    index.add_assets(sample_assets_with_simprints)

    # Create mixed query
    query = IsccQuery(
        units=[sample_content_units[0]],
        simprints={"CONTENT_TEXT_V0": [sample_simprints["CONTENT_TEXT_V0"][0]["simprint"]]},
    )

    # Search
    result = index.search_assets(query, limit=10)

    # Verify both match types populated
    assert len(result.global_matches) > 0
    assert len(result.chunk_matches) > 0

    index.close()


def test_usearch_simprint_metadata_enrichment(tmp_path, sample_assets_with_simprints, sample_simprints):
    """Test metadata enrichment from LMDB lookup."""
    index = UsearchIndex(path=tmp_path / "test_index")
    index.add_assets(sample_assets_with_simprints)

    # Search
    query = IsccQuery(simprints={"CONTENT_TEXT_V0": [sample_simprints["CONTENT_TEXT_V0"][0]["simprint"]]})
    result = index.search_assets(query, limit=10)

    # Verify at least one match has metadata
    has_metadata = any(match.metadata is not None for match in result.chunk_matches)
    assert has_metadata

    # Verify source field populated when metadata exists
    for match in result.chunk_matches:
        if match.metadata and "source" in match.metadata:
            assert match.source is not None

    index.close()


def test_usearch_simprint_search_errors(tmp_path, sample_assets_with_simprints, sample_simprints, monkeypatch):
    """Test graceful error handling for simprint search failures."""
    index = UsearchIndex(path=tmp_path / "test_index")
    index.add_assets(sample_assets_with_simprints)

    # Mock search_raw_multi to raise exception
    def mock_search(*args, **kwargs):
        raise RuntimeError("Simulated simprint search failure")

    monkeypatch.setattr(index._simprint_index, "search_raw_multi", mock_search)

    # Search should return empty chunk_matches on error
    query = IsccQuery(simprints={"CONTENT_TEXT_V0": [sample_simprints["CONTENT_TEXT_V0"][0]["simprint"]]})
    result = index.search_assets(query, limit=10)

    # Should return empty chunk_matches, not crash
    assert len(result.chunk_matches) == 0

    index.close()


def test_usearch_chunk_details_populated(tmp_path, sample_assets_with_simprints, sample_simprints):
    """Test that chunk details (offset, size, freq) are populated."""
    index = UsearchIndex(path=tmp_path / "test_index")
    index.add_assets(sample_assets_with_simprints)

    query = IsccQuery(simprints={"CONTENT_TEXT_V0": [sample_simprints["CONTENT_TEXT_V0"][0]["simprint"]]})
    result = index.search_assets(query, limit=10)

    # Verify chunk details present
    assert len(result.chunk_matches) > 0
    for match in result.chunk_matches:
        for type_result in match.types.values():
            if type_result.chunks:
                for chunk in type_result.chunks:
                    assert chunk.offset >= 0
                    assert chunk.size > 0
                    assert chunk.freq >= 1
                    assert chunk.score >= 0.0 and chunk.score <= 1.0

    index.close()


def test_normalize_query_simprints_only():
    """Test that normalize_query accepts simprints-only queries."""
    from iscc_search.indexes.common import normalize_query
    from iscc_search.schema import IsccQuery

    # Should not raise
    query = IsccQuery(simprints={"CONTENT_TEXT_V0": ["AXvu3tp2kF8mN9qL4rT1sZ"]})
    normalized = normalize_query(query)
    assert normalized.simprints is not None

    # Should raise for empty query
    with pytest.raises(ValueError, match="must have"):
        empty_query = IsccQuery()
        normalize_query(empty_query)


def test_usearch_threshold_parameter_in_search(tmp_path, sample_assets_with_simprints, sample_simprints):
    """Test that threshold parameter affects search results."""
    # High threshold (0.95) - strict matching
    index_strict = UsearchIndex(path=tmp_path / "test_strict", threshold=0.95)
    index_strict.add_assets(sample_assets_with_simprints)

    query = IsccQuery(simprints={"CONTENT_TEXT_V0": [sample_simprints["CONTENT_TEXT_V0"][0]["simprint"]]})
    result_strict = index_strict.search_assets(query, limit=10)
    strict_count = len(result_strict.chunk_matches)

    index_strict.close()

    # Low threshold (0.5) - permissive matching
    index_permissive = UsearchIndex(path=tmp_path / "test_permissive", threshold=0.5)
    index_permissive.add_assets(sample_assets_with_simprints)

    result_permissive = index_permissive.search_assets(query, limit=10)
    permissive_count = len(result_permissive.chunk_matches)

    index_permissive.close()

    # Permissive should return same or more results (may be same due to test data)
    assert permissive_count >= strict_count


def test_usearch_simprint_metadata_enrichment_error(
    tmp_path, sample_assets_with_simprints, sample_simprints, monkeypatch
):
    """Test that metadata enrichment errors return results without metadata."""
    index = UsearchIndex(path=tmp_path / "test_index")
    index.add_assets(sample_assets_with_simprints)

    # Mock deserialize_asset to raise exception
    from iscc_search.indexes import common

    def mock_deserialize(*args, **kwargs):
        raise ValueError("Simulated asset deserialization error")

    monkeypatch.setattr(common, "deserialize_asset", mock_deserialize)

    # Search should return results without metadata (not empty list)
    query = IsccQuery(simprints={"CONTENT_TEXT_V0": [sample_simprints["CONTENT_TEXT_V0"][0]["simprint"]]})
    result = index.search_assets(query, limit=10)

    # Should return results despite metadata enrichment failure
    assert len(result.chunk_matches) > 0

    # Verify results have no metadata (source should be None)
    for match in result.chunk_matches:
        assert match.iscc_id.startswith("ISCC:")
        assert match.source is None
        assert match.metadata is None

    index.close()


def test_usearch_simprint_asset_not_found(tmp_path):
    """Test _convert_simprint_match when asset is not found in DB."""
    from iscc_search.indexes.simprint.models import MatchedChunkRaw, SimprintMatchMulti, TypeMatchResult

    index = UsearchIndex(path=tmp_path / "test_index")

    # Create a mock match - will search for asset that doesn't exist
    mock_match = SimprintMatchMulti(
        iscc_id=b"\x00\x01" + b"\xff" * 8,  # 10-byte ISCC-ID (non-existent asset)
        score=0.95,
        types={
            "CONTENT_TEXT_V0": TypeMatchResult(
                score=0.95,
                matches=1,
                queried=1,
                chunks=[
                    MatchedChunkRaw(
                        query=b"test_query_data",
                        match=b"test_match_data",
                        score=0.95,
                        freq=1,
                        offset=0,
                        size=10,
                    )
                ],
            )
        },
    )

    # Call with assets_db and txn (but asset won't be found)
    with index.env.begin() as txn:
        assets_db = index.env.open_db(b"__assets__", txn=txn)
        result = index._convert_simprint_match(mock_match, assets_db, txn)

    # Should return result without metadata (asset not found)
    assert result.iscc_id.startswith("ISCC:")
    assert result.source is None
    assert result.metadata is None

    index.close()


def test_usearch_simprint_asset_no_metadata(tmp_path, sample_simprints, sample_iscc_ids):
    """Test simprint search when asset has no metadata field."""
    from iscc_search.schema import IsccEntry, IsccSimprint

    # Create asset with explicit metadata=None to cover the branch
    simprints_obj = {"CONTENT_TEXT_V0": [IsccSimprint(**sp) for sp in sample_simprints["CONTENT_TEXT_V0"][0:2]]}
    assets_no_metadata = [
        IsccEntry(
            iscc_id=sample_iscc_ids[0],
            simprints=simprints_obj,
            metadata=None,  # Explicitly set to None to test the metadata branch
        )
    ]

    index = UsearchIndex(path=tmp_path / "test_index")
    index.add_assets(assets_no_metadata)

    # Search should return results without metadata
    query = IsccQuery(simprints={"CONTENT_TEXT_V0": [sample_simprints["CONTENT_TEXT_V0"][0]["simprint"]]})
    result = index.search_assets(query, limit=10)

    # Should return results (metadata branch not taken)
    assert len(result.chunk_matches) > 0
    for match in result.chunk_matches:
        assert match.source is None
        assert match.metadata is None

    index.close()


def test_usearch_convert_simprint_match_no_chunks(tmp_path):
    """Test _convert_simprint_match with chunks=None (detailed=False path)."""
    from iscc_search.indexes.simprint.models import SimprintMatchMulti, TypeMatchResult

    index = UsearchIndex(path=tmp_path / "test_index")

    # Create a mock match with chunks=None to cover the else branch
    mock_match = SimprintMatchMulti(
        iscc_id=b"\x00\x01" + b"\x00" * 8,  # 10-byte ISCC-ID
        score=0.95,
        types={
            "CONTENT_TEXT_V0": TypeMatchResult(
                score=0.95,
                matches=1,
                queried=1,
                chunks=None,  # This triggers the else branch at line 582
            )
        },
    )

    # Call _convert_simprint_match directly with chunks=None
    result = index._convert_simprint_match(mock_match, None, None)

    # Should successfully convert without chunks
    assert result.iscc_id.startswith("ISCC:")
    assert result.types["CONTENT_TEXT_V0"].chunks is None

    index.close()
