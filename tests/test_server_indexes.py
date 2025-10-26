"""Tests for index management endpoints."""


def test_list_indexes_empty(test_client):
    """Test listing indexes when none exist."""
    response = test_client.get("/indexes")
    assert response.status_code == 200
    assert response.json() == []


def test_list_indexes_multiple(test_client):
    """Test listing multiple indexes."""
    # Create indexes
    test_client.post("/indexes", json={"name": "index1"})
    test_client.post("/indexes", json={"name": "index2"})
    test_client.post("/indexes", json={"name": "index3"})

    response = test_client.get("/indexes")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 3

    names = {idx["name"] for idx in data}
    assert names == {"index1", "index2", "index3"}

    # Verify metadata structure
    for idx in data:
        assert "name" in idx
        assert "assets" in idx
        assert "size" in idx
        assert idx["assets"] == 0  # All empty


def test_create_index_success(test_client):
    """Test creating a new index."""
    response = test_client.post("/indexes", json={"name": "myindex"})

    assert response.status_code == 201
    data = response.json()
    assert data["name"] == "myindex"
    assert data["assets"] == 0
    # Size >= 0 (memory returns 0, lmdb returns actual file size in MB)
    assert data["size"] >= 0


def test_create_index_invalid_name(test_client):
    """Test creating index with invalid name."""
    # Invalid names according to pattern ^[a-z][a-z0-9]*$
    invalid_names = [
        "Test",  # Uppercase
        "test-name",  # Hyphen
        "test_name",  # Underscore
        "123test",  # Starts with digit
        "",  # Empty
    ]

    for name in invalid_names:
        response = test_client.post("/indexes", json={"name": name})
        assert response.status_code == 422  # Pydantic validation error


def test_create_index_duplicate(test_client):
    """Test creating duplicate index returns 409 Conflict."""
    # Create first time
    response1 = test_client.post("/indexes", json={"name": "myindex"})
    assert response1.status_code == 201

    # Try to create again
    response2 = test_client.post("/indexes", json={"name": "myindex"})
    assert response2.status_code == 409
    data = response2.json()
    assert "detail" in data
    assert "already exists" in data["detail"]


def test_get_index_success(test_client):
    """Test getting index metadata."""
    # Create index
    test_client.post("/indexes", json={"name": "myindex"})

    # Get it
    response = test_client.get("/indexes/myindex")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "myindex"
    assert data["assets"] == 0
    # Size >= 0 (memory returns 0, lmdb returns actual file size in MB)
    assert data["size"] >= 0


def test_get_index_not_found(test_client):
    """Test getting non-existent index returns 404."""
    response = test_client.get("/indexes/nonexistent")
    assert response.status_code == 404
    data = response.json()
    assert "detail" in data
    assert "not found" in data["detail"].lower()


def test_delete_index_success(test_client):
    """Test deleting an index."""
    # Create index
    test_client.post("/indexes", json={"name": "myindex"})

    # Delete it
    response = test_client.delete("/indexes/myindex")
    assert response.status_code == 204
    assert response.text == ""  # No content

    # Verify it's gone
    get_response = test_client.get("/indexes/myindex")
    assert get_response.status_code == 404


def test_delete_index_not_found(test_client):
    """Test deleting non-existent index returns 404."""
    response = test_client.delete("/indexes/nonexistent")
    assert response.status_code == 404
    data = response.json()
    assert "detail" in data
    assert "not found" in data["detail"].lower()


def test_index_lifecycle(test_client):
    """Test complete index lifecycle: create, use, delete."""
    # Create
    create_response = test_client.post("/indexes", json={"name": "lifecycle"})
    assert create_response.status_code == 201

    # Verify it exists
    get_response = test_client.get("/indexes/lifecycle")
    assert get_response.status_code == 200

    # List includes it
    list_response = test_client.get("/indexes")
    assert list_response.status_code == 200
    names = {idx["name"] for idx in list_response.json()}
    assert "lifecycle" in names

    # Delete
    delete_response = test_client.delete("/indexes/lifecycle")
    assert delete_response.status_code == 204

    # Verify it's gone
    final_get = test_client.get("/indexes/lifecycle")
    assert final_get.status_code == 404
