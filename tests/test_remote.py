"""Tests for remote index client."""

from typing import TYPE_CHECKING

import pytest
from fastapi.testclient import TestClient

if TYPE_CHECKING:
    from iscc_search.remote.client import RemoteIndex  # noqa: F401


@pytest.fixture(scope="function")
def test_server():
    # type: () -> TestClient
    """
    Create test server with in-memory index.

    Note: Memory index persists across tests within same pytest session.
    Tests should use unique index names or clean up after themselves.
    """
    import os

    # Configure for memory index
    os.environ["ISCC_SEARCH_INDEX_LOCATION"] = "memory://"

    from iscc_search.server import app

    # Create test client with lifespan enabled
    with TestClient(app) as client:
        yield client

    # Cleanup environment
    os.environ.pop("ISCC_SEARCH_INDEX_LOCATION", None)


@pytest.fixture
def remote_client(test_server):
    # type: (TestClient) -> RemoteIndex
    """Create RemoteIndex client pointing to test server."""
    from iscc_search.remote import RemoteIndex

    # TestClient uses "http://testserver" as base URL
    client = RemoteIndex(url="http://testserver", index_name="test")

    # Replace httpx.Client with TestClient for synchronous testing
    client._client = test_server

    # Clean up any existing "test" index from previous tests
    try:
        test_server.delete("/indexes/test")
    except Exception:
        pass  # Index doesn't exist, that's fine

    yield client

    # Cleanup after test
    try:
        test_server.delete("/indexes/test")
    except Exception:
        pass

    client.close()


def test_remote_init_imports():
    # type: () -> None
    """Test remote/__init__.py imports work correctly."""
    from iscc_search.remote import RemoteIndex

    assert RemoteIndex is not None
    assert callable(RemoteIndex)


def test_remote_index_initialization():
    # type: () -> None
    """Test RemoteIndex can be initialized."""
    from iscc_search.remote import RemoteIndex

    client = RemoteIndex(url="https://api.example.com", index_name="test")

    assert client.url == "https://api.example.com"
    assert client.index_name == "test"
    assert client.api_key is None
    assert client.chunk_size == 100


def test_remote_index_initialization_with_api_key():
    # type: () -> None
    """Test RemoteIndex initialization with API key."""
    from iscc_search.remote import RemoteIndex

    client = RemoteIndex(url="https://api.example.com", index_name="test", api_key="secret", chunk_size=50)

    assert client.url == "https://api.example.com"
    assert client.index_name == "test"
    assert client.api_key == "secret"
    assert client.chunk_size == 50


def test_remote_index_url_normalization():
    # type: () -> None
    """Test RemoteIndex normalizes URL by removing trailing slash."""
    from iscc_search.remote import RemoteIndex

    client = RemoteIndex(url="https://api.example.com/", index_name="test")

    assert client.url == "https://api.example.com"


def test_remote_index_client_lazy_initialization():
    # type: () -> None
    """Test HTTP client is lazily initialized."""
    from iscc_search.remote import RemoteIndex

    client = RemoteIndex(url="https://api.example.com", index_name="test")

    # Client should not be initialized yet
    assert client._client is None

    # Access client property
    http_client = client.client

    # Now it should be initialized
    assert client._client is not None
    assert http_client is client._client


def test_remote_index_client_with_api_key():
    # type: () -> None
    """Test HTTP client includes API key in headers."""
    from iscc_search.remote import RemoteIndex

    client = RemoteIndex(url="https://api.example.com", index_name="test", api_key="secret")

    http_client = client.client

    assert "X-API-Key" in http_client.headers
    assert http_client.headers["X-API-Key"] == "secret"


def test_remote_index_close():
    # type: () -> None
    """Test RemoteIndex close method."""
    from iscc_search.remote import RemoteIndex

    client = RemoteIndex(url="https://api.example.com", index_name="test")

    # Initialize client
    _ = client.client
    assert client._client is not None

    # Close should cleanup
    client.close()
    assert client._client is None

    # Calling close again should be idempotent
    client.close()
    assert client._client is None


# Integration tests with test server


def test_remote_index_list_indexes(remote_client, test_server):
    # type: (RemoteIndex, TestClient) -> None
    """Test listing indexes from remote server."""
    from iscc_search.schema import IsccIndex

    # Create test index first
    test_server.post("/indexes", json={"name": "test"})

    # List indexes
    indexes = remote_client.list_indexes()

    assert len(indexes) > 0
    assert all(isinstance(idx, IsccIndex) for idx in indexes)
    assert any(idx.name == "test" for idx in indexes)


def test_remote_index_create_index(remote_client, test_server):
    # type: (RemoteIndex, TestClient) -> None
    """Test creating new index on remote server."""
    from iscc_search.schema import IsccIndex

    # Clean up any existing "newindex" from previous test runs
    try:
        test_server.delete("/indexes/newindex")
    except Exception:
        pass

    # Create new index
    new_index = IsccIndex(name="newindex")
    created = remote_client.create_index(new_index)

    assert created.name == "newindex"
    assert created.assets == 0
    assert created.size == 0

    # Cleanup
    try:
        test_server.delete("/indexes/newindex")
    except Exception:
        pass


def test_remote_index_create_duplicate_index(remote_client, test_server):
    # type: (RemoteIndex, TestClient) -> None
    """Test creating duplicate index raises FileExistsError."""
    from iscc_search.schema import IsccIndex

    # Create index
    test_server.post("/indexes", json={"name": "test"})

    # Try to create duplicate
    with pytest.raises(FileExistsError):
        remote_client.create_index(IsccIndex(name="test"))


def test_remote_index_get_index(remote_client, test_server):
    # type: (RemoteIndex, TestClient) -> None
    """Test getting index metadata."""
    from iscc_search.schema import IsccIndex

    # Create index
    test_server.post("/indexes", json={"name": "test"})

    # Get index
    index = remote_client.get_index("test")

    assert isinstance(index, IsccIndex)
    assert index.name == "test"
    assert index.assets == 0


def test_remote_index_get_nonexistent_index(remote_client):
    # type: (RemoteIndex) -> None
    """Test getting nonexistent index raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        remote_client.get_index("nonexistent")


def test_remote_index_delete_index(remote_client, test_server):
    # type: (RemoteIndex, TestClient) -> None
    """Test deleting index."""
    # Create index
    test_server.post("/indexes", json={"name": "test"})

    # Delete index
    remote_client.delete_index("test")

    # Verify deleted
    with pytest.raises(FileNotFoundError):
        remote_client.get_index("test")


def test_remote_index_delete_nonexistent_index(remote_client):
    # type: (RemoteIndex) -> None
    """Test deleting nonexistent index raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        remote_client.delete_index("nonexistent")


def test_remote_index_add_assets_single_batch(remote_client, test_server, sample_content_units, sample_iscc_ids):
    # type: (RemoteIndex, TestClient, list[str], list[str]) -> None
    """Test adding assets in single batch."""
    from iscc_search.schema import IsccEntry, IsccAddResult

    # Create index
    test_server.post("/indexes", json={"name": "test"})

    # Create test assets (need at least 2 units per IsccEntry schema)
    assets = [
        IsccEntry(
            iscc_id=sample_iscc_ids[0],
            units=[sample_content_units[0], sample_content_units[1]],
            metadata={"name": "Asset 1"},
        ),
        IsccEntry(
            iscc_id=sample_iscc_ids[1],
            units=[sample_content_units[0], sample_content_units[1]],
            metadata={"name": "Asset 2"},
        ),
    ]

    # Add assets (small batch, no chunking)
    results = remote_client.add_assets("test", assets)

    assert len(results) == 2
    assert all(isinstance(r, IsccAddResult) for r in results)
    assert all(r.status == "created" for r in results)


def test_remote_index_add_assets_empty_list(remote_client, test_server):
    # type: (RemoteIndex, TestClient) -> None
    """Test adding empty asset list returns empty results."""
    # Create index
    test_server.post("/indexes", json={"name": "test"})

    results = remote_client.add_assets("test", [])

    assert results == []


def test_remote_index_add_assets_chunked_batches(remote_client, test_server, sample_iscc_ids, sample_content_units):
    # type: (RemoteIndex, TestClient, list[str], list[str]) -> None
    """Test adding assets with chunked batches and progress bar."""
    from iscc_search.schema import IsccEntry, IsccAddResult
    import iscc_core as ic

    # Create index
    test_server.post("/indexes", json={"name": "test"})

    # Create more assets than chunk_size to trigger chunking
    # chunk_size is 100 by default, create 150 unique assets
    assets = []
    for i in range(150):
        # Generate unique ISCC-UNIT by using different text each time
        text_unit = ic.gen_text_code_v0(f"Unique text content {i}")["iscc"]
        assets.append(
            IsccEntry(
                iscc_id=sample_iscc_ids[i % len(sample_iscc_ids)],
                units=[text_unit, sample_content_units[1]],  # Need at least 2 units
                metadata={"name": f"Asset {i}"},
            )
        )

    # Add assets (will trigger chunked upload with progress bar)
    results = remote_client.add_assets("test", assets)

    assert len(results) == 150
    assert all(isinstance(r, IsccAddResult) for r in results)
    # Check that all statuses are valid (created, exists, or updated)
    statuses = set(r.status for r in results)
    assert statuses.issubset({"created", "exists", "updated"})


def test_remote_index_get_asset(remote_client, test_server, sample_content_units, sample_iscc_ids):
    # type: (RemoteIndex, TestClient, list[str], list[str]) -> None
    """Test getting specific asset by ISCC-ID."""
    from iscc_search.schema import IsccEntry

    # Create index and add asset
    test_server.post("/indexes", json={"name": "test"})
    asset = IsccEntry(
        iscc_id=sample_iscc_ids[0],
        units=[sample_content_units[0], sample_content_units[1]],
        metadata={"name": "Test Asset"},
    )
    test_server.post("/indexes/test/assets", json=[asset.model_dump(exclude_unset=True)])

    # Get asset
    retrieved = remote_client.get_asset("test", sample_iscc_ids[0])

    assert isinstance(retrieved, IsccEntry)
    assert retrieved.iscc_id == sample_iscc_ids[0]
    assert retrieved.metadata["name"] == "Test Asset"


def test_remote_index_get_nonexistent_asset(remote_client, test_server, sample_iscc_ids):
    # type: (RemoteIndex, TestClient, list[str]) -> None
    """Test getting nonexistent asset raises FileNotFoundError."""
    # Create index
    test_server.post("/indexes", json={"name": "test"})

    with pytest.raises(FileNotFoundError):
        remote_client.get_asset("test", sample_iscc_ids[0])


def test_remote_index_search_assets(remote_client, test_server, sample_content_units, sample_iscc_ids):
    # type: (RemoteIndex, TestClient, list[str], list[str]) -> None
    """Test searching for similar assets."""
    from iscc_search.schema import IsccEntry, IsccQuery, IsccSearchResult

    # Create index and add assets
    test_server.post("/indexes", json={"name": "test"})
    asset = IsccEntry(
        iscc_id=sample_iscc_ids[0],
        units=[sample_content_units[0], sample_content_units[1]],
        metadata={"name": "Test Asset"},
    )
    test_server.post("/indexes/test/assets", json=[asset.model_dump(exclude_unset=True)])

    # Search
    query = IsccQuery(units=[sample_content_units[0], sample_content_units[1]])
    result = remote_client.search_assets("test", query, limit=10)

    assert isinstance(result, IsccSearchResult)
    assert result.query is not None
    assert len(result.global_matches) > 0


def test_remote_index_search_nonexistent_index(remote_client, sample_content_units):
    # type: (RemoteIndex, list[str]) -> None
    """Test searching nonexistent index raises FileNotFoundError."""
    from iscc_search.schema import IsccQuery

    query = IsccQuery(units=[sample_content_units[0], sample_content_units[1]])

    with pytest.raises(FileNotFoundError):
        remote_client.search_assets("nonexistent", query)


def test_remote_index_handle_400_error(remote_client, test_server):
    # type: (RemoteIndex, TestClient) -> None
    """Test ValueError is raised for 400 Bad Request."""
    from iscc_search.schema import IsccIndex

    # Try to create index with invalid name
    with pytest.raises(ValueError):
        remote_client.create_index(IsccIndex(name=""))


def test_remote_index_handle_response_errors_success(remote_client):
    # type: (RemoteIndex) -> None
    """Test _handle_response_errors does nothing for successful responses."""

    # Create a mock successful response
    class MockResponse:
        is_success = True
        status_code = 200

    # Should not raise
    remote_client._handle_response_errors(MockResponse())


def test_remote_index_handle_response_errors_generic(remote_client):
    # type: (RemoteIndex) -> None
    """Test _handle_response_errors raises RuntimeError for other HTTP errors."""

    class MockResponse:
        is_success = False
        status_code = 500
        text = "Internal Server Error"

        def json(self):
            # type: () -> dict
            return {"detail": "Something went wrong"}

    with pytest.raises(RuntimeError, match="HTTP 500"):
        remote_client._handle_response_errors(MockResponse())


def test_remote_index_handle_response_errors_json_exception(remote_client):
    # type: (RemoteIndex) -> None
    """Test _handle_response_errors handles exception when response.json() fails."""

    class MockResponse:
        is_success = False
        status_code = 500
        text = "Plain text error message"

        def json(self):
            # type: () -> dict
            raise Exception("JSON decode error")

    with pytest.raises(RuntimeError, match="Plain text error message"):
        remote_client._handle_response_errors(MockResponse())


def test_remote_index_handle_response_errors_400(remote_client):
    # type: (RemoteIndex) -> None
    """Test _handle_response_errors raises ValueError for 400 Bad Request."""

    class MockResponse:
        is_success = False
        status_code = 400
        text = "Bad request"

        def json(self):
            # type: () -> dict
            return {"detail": "Invalid request"}

    with pytest.raises(ValueError, match="Invalid request"):
        remote_client._handle_response_errors(MockResponse())
