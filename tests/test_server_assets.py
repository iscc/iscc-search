"""Tests for asset management endpoints."""


def test_add_assets_single(test_client, sample_assets):
    """Test adding a single asset to an index."""
    # Create index
    test_client.post("/indexes", json={"name": "testindex"})

    # Add one asset
    asset_dict = sample_assets[0].model_dump(mode="json", exclude_none=True)
    response = test_client.post("/indexes/testindex/assets", json=[asset_dict])

    assert response.status_code == 201
    data = response.json()
    assert len(data) == 1
    assert data[0]["iscc_id"] == sample_assets[0].iscc_id
    assert data[0]["status"] == "created"


def test_add_assets_multiple(test_client, sample_assets):
    """Test adding multiple assets in batch."""
    # Create index
    test_client.post("/indexes", json={"name": "testindex"})

    # Add multiple assets
    assets_dict = [a.model_dump(mode="json", exclude_none=True) for a in sample_assets]
    response = test_client.post("/indexes/testindex/assets", json=assets_dict)

    assert response.status_code == 201
    data = response.json()
    assert len(data) == len(sample_assets)
    assert all(r["status"] == "created" for r in data)

    # Verify all ISCC-IDs returned
    returned_ids = {r["iscc_id"] for r in data}
    expected_ids = {a.iscc_id for a in sample_assets}
    assert returned_ids == expected_ids


def test_add_assets_duplicate(test_client, sample_assets):
    """Test adding duplicate asset returns 'updated' status."""
    # Create index
    test_client.post("/indexes", json={"name": "testindex"})

    # Add asset first time
    asset_dict = sample_assets[0].model_dump(mode="json", exclude_none=True)
    response1 = test_client.post("/indexes/testindex/assets", json=[asset_dict])
    assert response1.status_code == 201
    assert response1.json()[0]["status"] == "created"

    # Add same asset again
    response2 = test_client.post("/indexes/testindex/assets", json=[asset_dict])
    assert response2.status_code == 201
    assert response2.json()[0]["status"] == "updated"


def test_add_assets_index_not_found(test_client, sample_assets):
    """Test adding assets to non-existent index returns 404."""
    asset_dict = sample_assets[0].model_dump(mode="json", exclude_none=True)
    response = test_client.post("/indexes/nonexistent/assets", json=[asset_dict])

    assert response.status_code == 404
    data = response.json()
    assert "detail" in data
    assert "not found" in data["detail"].lower()


def test_add_assets_missing_iscc_id(test_client, sample_content_units):
    """Test adding asset without iscc_id raises error."""
    # Create index
    test_client.post("/indexes", json={"name": "testindex"})

    # Asset without iscc_id
    asset_dict = {"units": [sample_content_units[0], sample_content_units[1]]}

    response = test_client.post("/indexes/testindex/assets", json=[asset_dict])

    assert response.status_code == 400
    data = response.json()
    assert "detail" in data


def test_get_asset_success(test_client, sample_assets):
    """Test getting a specific asset by ISCC-ID."""
    # Create index and add assets
    test_client.post("/indexes", json={"name": "testindex"})
    assets_dict = [a.model_dump(mode="json", exclude_none=True) for a in sample_assets]
    test_client.post("/indexes/testindex/assets", json=assets_dict)

    # Get specific asset
    iscc_id = sample_assets[0].iscc_id
    response = test_client.get(f"/indexes/testindex/assets/{iscc_id}")

    assert response.status_code == 200
    data = response.json()
    assert data["iscc_id"] == iscc_id
    assert data["iscc_code"] == sample_assets[0].iscc_code
    assert "metadata" in data
    assert data["metadata"] == sample_assets[0].metadata


def test_get_asset_not_found(test_client, sample_iscc_ids):
    """Test getting non-existent asset returns 404."""
    # Create empty index
    test_client.post("/indexes", json={"name": "testindex"})

    # Try to get non-existent asset
    response = test_client.get(f"/indexes/testindex/assets/{sample_iscc_ids[0]}")

    assert response.status_code == 404
    data = response.json()
    assert "detail" in data
    assert "not found" in data["detail"].lower()


def test_get_asset_index_not_found(test_client, sample_iscc_ids):
    """Test getting asset from non-existent index returns 404."""
    response = test_client.get(f"/indexes/nonexistent/assets/{sample_iscc_ids[0]}")

    assert response.status_code == 404
    data = response.json()
    assert "detail" in data


def test_asset_metadata_preservation(test_client, sample_assets):
    """Test that asset metadata is preserved through add and get."""
    # Create index
    test_client.post("/indexes", json={"name": "testindex"})

    # Add asset with metadata
    asset = sample_assets[0]
    asset_dict = asset.model_dump(mode="json", exclude_none=True)
    test_client.post("/indexes/testindex/assets", json=[asset_dict])

    # Get asset and verify metadata
    response = test_client.get(f"/indexes/testindex/assets/{asset.iscc_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["metadata"] == asset.metadata


def test_add_assets_updates_index_stats(test_client, sample_assets):
    """Test that adding assets updates index asset count."""
    # Create index
    test_client.post("/indexes", json={"name": "testindex"})

    # Verify initial count
    idx_response1 = test_client.get("/indexes/testindex")
    assert idx_response1.json()["assets"] == 0

    # Add assets
    assets_dict = [a.model_dump(mode="json", exclude_none=True) for a in sample_assets]
    test_client.post("/indexes/testindex/assets", json=assets_dict)

    # Verify count updated
    idx_response2 = test_client.get("/indexes/testindex")
    assert idx_response2.json()["assets"] == len(sample_assets)
