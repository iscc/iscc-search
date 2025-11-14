"""Tests for search endpoints."""

import pytest


def test_search_post_success(test_client, sample_assets):
    """Test POST search with full query asset."""
    # Create index and add assets
    test_client.post("/indexes", json={"name": "testindex"})
    assets_dict = [a.model_dump(mode="json", exclude_none=True) for a in sample_assets]
    test_client.post("/indexes/testindex/assets", json=assets_dict)

    # Search using first asset as query (exclude iscc_id to avoid iscc_id search)
    query_dict = sample_assets[0].model_dump(mode="json", exclude_none=True, exclude={"iscc_id"})
    response = test_client.post("/indexes/testindex/search", json=query_dict)

    assert response.status_code == 200
    data = response.json()

    # Verify response structure
    assert "query" in data
    assert "global_matches" in data

    # Should match at least the exact same asset
    assert len(data["global_matches"]) >= 1

    # Verify match structure
    if len(data["global_matches"]) > 0:
        match = data["global_matches"][0]
        assert "iscc_id" in match
        assert "score" in match
        assert "types" in match


def test_search_post_no_matches(test_client, sample_assets, sample_iscc_codes):
    """Test POST search when no matches exist."""
    # Create index and add one asset
    test_client.post("/indexes", json={"name": "testindex"})
    asset_dict = sample_assets[0].model_dump(mode="json", exclude_none=True)
    test_client.post("/indexes/testindex/assets", json=[asset_dict])

    # Search for different ISCC-CODE
    query_dict = {"iscc_code": sample_iscc_codes[5]}  # Different from added assets
    response = test_client.post("/indexes/testindex/search", json=query_dict)

    assert response.status_code == 200
    data = response.json()
    assert len(data["global_matches"]) == 0


def test_search_post_with_limit(test_client, sample_iscc_ids, sample_iscc_codes):
    """Test POST search respects limit parameter."""
    # Create index and add multiple assets with same ISCC-CODE
    test_client.post("/indexes", json={"name": "testindex"})

    code = sample_iscc_codes[0]
    # Create 10 unique assets with same code but different IDs
    for i in range(10):
        asset_dict = {
            "iscc_id": sample_iscc_ids[i],  # Use unique ISCC-ID for each asset
            "iscc_code": code,
        }
        test_client.post("/indexes/testindex/assets", json=[asset_dict])

    # Search with limit
    query_dict = {"iscc_code": code}
    response = test_client.post("/indexes/testindex/search?limit=5", json=query_dict)

    assert response.status_code == 200
    data = response.json()
    assert len(data["global_matches"]) <= 5


def test_search_post_index_not_found(test_client, sample_assets):
    """Test POST search on non-existent index returns 404."""
    query_dict = sample_assets[0].model_dump(mode="json", exclude_none=True)
    response = test_client.post("/indexes/nonexistent/search", json=query_dict)

    assert response.status_code == 404
    data = response.json()
    assert "detail" in data
    assert "not found" in data["detail"].lower()


def test_search_get_success(test_client, sample_assets):
    """Test GET search with iscc_code query parameter."""
    # Create index and add assets
    test_client.post("/indexes", json={"name": "testindex"})
    assets_dict = [a.model_dump(mode="json", exclude_none=True) for a in sample_assets]
    test_client.post("/indexes/testindex/assets", json=assets_dict)

    # Search using iscc_code from first asset
    iscc_code = sample_assets[0].iscc_code
    response = test_client.get(f"/indexes/testindex/search?iscc_code={iscc_code}")

    assert response.status_code == 200
    data = response.json()

    # Verify response structure
    assert "query" in data
    assert "global_matches" in data

    # Query should contain the iscc_code
    assert data["query"]["iscc_code"] == iscc_code


def test_search_get_with_limit(test_client, sample_assets):
    """Test GET search respects limit parameter."""
    # Create index and add assets
    test_client.post("/indexes", json={"name": "testindex"})
    assets_dict = [a.model_dump(mode="json", exclude_none=True) for a in sample_assets]
    test_client.post("/indexes/testindex/assets", json=assets_dict)

    # Search with limit
    iscc_code = sample_assets[0].iscc_code
    response = test_client.get(f"/indexes/testindex/search?iscc_code={iscc_code}&limit=3")

    assert response.status_code == 200
    data = response.json()
    assert len(data["global_matches"]) <= 3


def test_search_get_index_not_found(test_client, sample_iscc_codes):
    """Test GET search on non-existent index returns 404."""
    response = test_client.get(f"/indexes/nonexistent/search?iscc_code={sample_iscc_codes[0]}")

    assert response.status_code == 404
    data = response.json()
    assert "detail" in data


def test_search_get_missing_iscc_code(test_client):
    """Test GET search without iscc_code parameter returns 422."""
    # Create index
    test_client.post("/indexes", json={"name": "testindex"})

    # Try search without iscc_code parameter
    response = test_client.get("/indexes/testindex/search")

    assert response.status_code == 422  # FastAPI validation error


def test_search_result_structure(test_client, sample_assets):
    """Test that search results have correct structure."""
    # Create index and add assets
    test_client.post("/indexes", json={"name": "testindex"})
    assets_dict = [a.model_dump(mode="json", exclude_none=True) for a in sample_assets]
    test_client.post("/indexes/testindex/assets", json=assets_dict)

    # Search (exclude iscc_id to avoid iscc_id search)
    query_dict = sample_assets[0].model_dump(mode="json", exclude_none=True, exclude={"iscc_id"})
    response = test_client.post("/indexes/testindex/search", json=query_dict)

    assert response.status_code == 200
    data = response.json()

    # Verify top-level structure
    assert "query" in data
    assert "global_matches" in data
    assert isinstance(data["global_matches"], list)

    # Verify match structure if matches exist
    if len(data["global_matches"]) > 0:
        for match in data["global_matches"]:
            assert "iscc_id" in match
            assert "score" in match
            assert "types" in match
            assert isinstance(match["score"], (int, float))
            assert isinstance(match["types"], dict)


def test_search_empty_index(test_client, sample_assets):
    """Test searching an empty index returns no matches."""
    # Create empty index
    test_client.post("/indexes", json={"name": "testindex"})

    # Search empty index (exclude iscc_id to avoid iscc_id lookup failure)
    query_dict = sample_assets[0].model_dump(mode="json", exclude_none=True, exclude={"iscc_id"})
    response = test_client.post("/indexes/testindex/search", json=query_dict)

    assert response.status_code == 200
    data = response.json()
    assert len(data["global_matches"]) == 0


def test_search_text_success(test_client, request):
    """Test text search endpoint with valid text."""
    # Skip for LMDB backend (doesn't support simprint-only queries)
    if "lmdb" in str(request.node.callspec.id):
        pytest.skip("LMDB backend doesn't support simprint-only queries")

    # Create index
    test_client.post("/indexes", json={"name": "testindex"})

    # Search with text
    text_query = {"text": "This is a test document with some content. " * 20}
    response = test_client.post("/indexes/testindex/search/text", json=text_query)

    assert response.status_code == 200
    data = response.json()

    # Verify response structure
    assert "query" in data
    assert "global_matches" in data
    assert isinstance(data["global_matches"], list)


def test_search_text_index_not_found(test_client):
    """Test text search on non-existent index returns 404."""
    text_query = {"text": "Some text content"}
    response = test_client.post("/indexes/nonexistent/search/text", json=text_query)

    assert response.status_code == 404
    data = response.json()
    assert "detail" in data
    assert "not found" in data["detail"].lower()


def test_search_text_empty_text(test_client):
    """Test text search with empty text returns 422 validation error."""
    # Create index
    test_client.post("/indexes", json={"name": "testindex"})

    # Try search with empty text
    text_query = {"text": ""}
    response = test_client.post("/indexes/testindex/search/text", json=text_query)

    # Should fail validation (min_length=1)
    assert response.status_code == 422


def test_search_text_missing_text_field(test_client):
    """Test text search without text field returns 422 validation error."""
    # Create index
    test_client.post("/indexes", json={"name": "testindex"})

    # Try search without text field
    response = test_client.post("/indexes/testindex/search/text", json={})

    # Should fail validation (required field)
    assert response.status_code == 422


def test_search_text_with_limit(test_client, request):
    """Test text search respects limit parameter."""
    # Skip for LMDB backend (doesn't support simprint-only queries)
    if "lmdb" in str(request.node.callspec.id):
        pytest.skip("LMDB backend doesn't support simprint-only queries")

    # Create index
    test_client.post("/indexes", json={"name": "testindex"})

    # Search with limit
    text_query = {"text": "Test content " * 50}
    response = test_client.post("/indexes/testindex/search/text?limit=5", json=text_query)

    assert response.status_code == 200
    data = response.json()
    # Even with no matches, should not exceed limit
    assert len(data["global_matches"]) <= 5


def test_response_excludes_unset_fields(test_client, request):
    """Test that unset fields are excluded from API responses."""
    # Skip for LMDB backend (doesn't support simprint-only queries)
    if "lmdb" in str(request.node.callspec.id):
        pytest.skip("LMDB backend doesn't support simprint-only queries")

    # Create index
    test_client.post("/indexes", json={"name": "testindex"})

    # Search with text (generates simprints, but no iscc_id/iscc_code/units)
    text_query = {"text": "Test content " * 50}
    response = test_client.post("/indexes/testindex/search/text", json=text_query)

    assert response.status_code == 200
    data = response.json()

    # Query object should have simprints but NOT iscc_id/iscc_code/units
    query = data["query"]
    assert "simprints" in query

    # These fields should be OMITTED entirely (not present as null)
    assert "iscc_id" not in query
    assert "iscc_code" not in query
    assert "units" not in query


def test_search_by_iscc_id_success(test_client, sample_assets):
    """Test POST search with iscc_id parameter finds similar assets."""
    # Create index and add multiple assets
    test_client.post("/indexes", json={"name": "testindex"})
    assets_dict = [a.model_dump(mode="json", exclude_none=True) for a in sample_assets]
    test_client.post("/indexes/testindex/assets", json=assets_dict)

    # Search using iscc_id from first asset
    query_dict = {"iscc_id": sample_assets[0].iscc_id}
    response = test_client.post("/indexes/testindex/search", json=query_dict)

    assert response.status_code == 200
    data = response.json()

    # Verify response structure
    assert "query" in data
    assert "global_matches" in data

    # Should find similar assets (excluding the query asset itself)
    assert isinstance(data["global_matches"], list)


def test_search_by_iscc_id_self_exclusion(test_client, sample_assets):
    """Test that query asset is excluded from results when searching by iscc_id."""
    # Create index and add multiple assets
    test_client.post("/indexes", json={"name": "testindex"})
    assets_dict = [a.model_dump(mode="json", exclude_none=True) for a in sample_assets]
    test_client.post("/indexes/testindex/assets", json=assets_dict)

    # Search using iscc_id from first asset
    query_iscc_id = sample_assets[0].iscc_id
    query_dict = {"iscc_id": query_iscc_id}
    response = test_client.post("/indexes/testindex/search", json=query_dict)

    assert response.status_code == 200
    data = response.json()

    # Verify query asset is NOT in results
    result_iscc_ids = [match["iscc_id"] for match in data["global_matches"]]
    assert query_iscc_id not in result_iscc_ids


def test_search_by_iscc_id_not_found(test_client, sample_iscc_ids):
    """Test POST search with iscc_id that doesn't exist returns 404."""
    # Create empty index
    test_client.post("/indexes", json={"name": "testindex"})

    # Search for non-existent iscc_id
    query_dict = {"iscc_id": sample_iscc_ids[0]}
    response = test_client.post("/indexes/testindex/search", json=query_dict)

    # Should return 404 as documented
    assert response.status_code == 404
    data = response.json()
    assert "detail" in data
    assert sample_iscc_ids[0] in data["detail"]


def test_search_by_iscc_id_precedence(test_client, sample_assets, sample_iscc_codes):
    """Test that iscc_id takes precedence over other query fields."""
    # Create index and add assets
    test_client.post("/indexes", json={"name": "testindex"})
    assets_dict = [a.model_dump(mode="json", exclude_none=True) for a in sample_assets]
    test_client.post("/indexes/testindex/assets", json=assets_dict)

    # Search with iscc_id AND other fields (iscc_id should take precedence)
    query_dict = {
        "iscc_id": sample_assets[0].iscc_id,
        "iscc_code": sample_iscc_codes[5],  # Different code - should be ignored
        "units": ["ISCC:AAAUHBUDQUT3LPWR"],  # Different units - should be ignored
    }
    response = test_client.post("/indexes/testindex/search", json=query_dict)

    assert response.status_code == 200
    data = response.json()

    # Verify query asset is excluded (self-exclusion for iscc_id)
    query_iscc_id = sample_assets[0].iscc_id
    result_iscc_ids = [match["iscc_id"] for match in data["global_matches"]]
    assert query_iscc_id not in result_iscc_ids
