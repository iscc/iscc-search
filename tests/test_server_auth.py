"""Tests for API authentication."""

import typing  # noqa: F401
import pytest
from fastapi.testclient import TestClient
import iscc_search.settings
from iscc_search.server import app


@pytest.fixture
def client_public():
    # type: () -> typing.Generator[TestClient, None, None]
    """Create TestClient with no authentication (public mode)."""
    # Save original settings
    original_uri = iscc_search.settings.search_settings.index_uri
    original_secret = iscc_search.settings.search_settings.api_secret

    try:
        # Configure for public mode
        iscc_search.settings.search_settings.index_uri = "memory://"
        iscc_search.settings.search_settings.api_secret = None

        with TestClient(app) as client:
            yield client
    finally:
        # Restore original settings
        iscc_search.settings.search_settings.index_uri = original_uri
        iscc_search.settings.search_settings.api_secret = original_secret


@pytest.fixture
def client_protected():
    # type: () -> typing.Generator[TestClient, None, None]
    """Create TestClient with authentication enabled (protected mode)."""
    # Save original settings
    original_uri = iscc_search.settings.search_settings.index_uri
    original_secret = iscc_search.settings.search_settings.api_secret

    try:
        # Configure for protected mode
        iscc_search.settings.search_settings.index_uri = "memory://"
        iscc_search.settings.search_settings.api_secret = "test-secret-key-12345"

        with TestClient(app) as client:
            yield client
    finally:
        # Restore original settings
        iscc_search.settings.search_settings.index_uri = original_uri
        iscc_search.settings.search_settings.api_secret = original_secret


def test_public_mode_no_auth_required(client_public):
    # type: (TestClient) -> None
    """Test that all endpoints work without authentication in public mode."""
    # Test indexes endpoints
    response = client_public.get("/indexes")
    assert response.status_code == 200

    response = client_public.post("/indexes", json={"name": "testindex"})
    assert response.status_code == 201

    response = client_public.get("/indexes/testindex")
    assert response.status_code == 200

    # Test assets endpoints
    response = client_public.post(
        "/indexes/testindex/assets",
        json=[
            {
                "iscc_id": "ISCC:MAIGIIFJRDGEQQAA",
                "iscc_code": "ISCC:KECYCMZIOY36XXGZ7S6QJQ2AEEXPOVEHZYPK6GMSFLU3WF54UPZMTPY",
            }
        ],
    )
    assert response.status_code == 201

    response = client_public.get("/indexes/testindex/assets/ISCC:MAIGIIFJRDGEQQAA")
    assert response.status_code == 200

    # Test search endpoints
    response = client_public.post(
        "/indexes/testindex/search",
        json={"iscc_code": "ISCC:KECYCMZIOY36XXGZ7S6QJQ2AEEXPOVEHZYPK6GMSFLU3WF54UPZMTPY"},
    )
    assert response.status_code == 200

    response = client_public.get(
        "/indexes/testindex/search",
        params={"iscc_code": "ISCC:KECYCMZIOY36XXGZ7S6QJQ2AEEXPOVEHZYPK6GMSFLU3WF54UPZMTPY"},
    )
    assert response.status_code == 200

    response = client_public.post(
        "/indexes/testindex/search/text",
        json={"text": "This is a test query"},
    )
    assert response.status_code == 200

    # Cleanup
    response = client_public.delete("/indexes/testindex")
    assert response.status_code == 204


def test_protected_mode_valid_key(client_protected):
    # type: (TestClient) -> None
    """Test that endpoints work with valid API key in protected mode."""
    headers = {"X-API-Key": "test-secret-key-12345"}

    # Test indexes endpoints
    response = client_protected.get("/indexes", headers=headers)
    assert response.status_code == 200

    response = client_protected.post("/indexes", json={"name": "testindex"}, headers=headers)
    assert response.status_code == 201

    response = client_protected.get("/indexes/testindex", headers=headers)
    assert response.status_code == 200

    # Test assets endpoints
    response = client_protected.post(
        "/indexes/testindex/assets",
        json=[
            {
                "iscc_id": "ISCC:MAIGIIFJRDGEQQAA",
                "iscc_code": "ISCC:KECYCMZIOY36XXGZ7S6QJQ2AEEXPOVEHZYPK6GMSFLU3WF54UPZMTPY",
            }
        ],
        headers=headers,
    )
    assert response.status_code == 201

    response = client_protected.get("/indexes/testindex/assets/ISCC:MAIGIIFJRDGEQQAA", headers=headers)
    assert response.status_code == 200

    # Test search endpoints
    response = client_protected.post(
        "/indexes/testindex/search",
        json={"iscc_code": "ISCC:KECYCMZIOY36XXGZ7S6QJQ2AEEXPOVEHZYPK6GMSFLU3WF54UPZMTPY"},
        headers=headers,
    )
    assert response.status_code == 200

    response = client_protected.get(
        "/indexes/testindex/search",
        params={"iscc_code": "ISCC:KECYCMZIOY36XXGZ7S6QJQ2AEEXPOVEHZYPK6GMSFLU3WF54UPZMTPY"},
        headers=headers,
    )
    assert response.status_code == 200

    response = client_protected.post(
        "/indexes/testindex/search/text",
        json={"text": "This is a test query"},
        headers=headers,
    )
    assert response.status_code == 200

    # Cleanup
    response = client_protected.delete("/indexes/testindex", headers=headers)
    assert response.status_code == 204


def test_protected_mode_invalid_key(client_protected):
    # type: (TestClient) -> None
    """Test that endpoints return 401 with invalid API key."""
    headers = {"X-API-Key": "wrong-key"}

    # Test indexes endpoints
    response = client_protected.get("/indexes", headers=headers)
    assert response.status_code == 401
    assert response.json()["detail"] == "Unauthorized"

    response = client_protected.post("/indexes", json={"name": "testindex"}, headers=headers)
    assert response.status_code == 401

    # Test assets endpoints (need to create index first with valid key)
    valid_headers = {"X-API-Key": "test-secret-key-12345"}
    client_protected.post("/indexes", json={"name": "testindex"}, headers=valid_headers)

    response = client_protected.post(
        "/indexes/testindex/assets",
        json=[{"iscc_id": "ISCC:MAIGIIFJRDGEQQAA", "iscc_code": "ISCC:KECT"}],
        headers=headers,
    )
    assert response.status_code == 401

    response = client_protected.get("/indexes/testindex/assets/ISCC:MAIGIIFJRDGEQQAA", headers=headers)
    assert response.status_code == 401

    # Test search endpoints
    response = client_protected.post(
        "/indexes/testindex/search",
        json={"iscc_code": "ISCC:KECYCMZIOY36XXGZ"},
        headers=headers,
    )
    assert response.status_code == 401

    response = client_protected.get(
        "/indexes/testindex/search",
        params={"iscc_code": "ISCC:KECYCMZIOY36XXGZ"},
        headers=headers,
    )
    assert response.status_code == 401

    response = client_protected.post(
        "/indexes/testindex/search/text",
        json={"text": "test"},
        headers=headers,
    )
    assert response.status_code == 401

    # Cleanup
    client_protected.delete("/indexes/testindex", headers=valid_headers)


def test_protected_mode_missing_key(client_protected):
    # type: (TestClient) -> None
    """Test that endpoints return 401 when API key is missing."""
    # Test indexes endpoints
    response = client_protected.get("/indexes")
    assert response.status_code == 401
    assert response.json()["detail"] == "Unauthorized"

    response = client_protected.post("/indexes", json={"name": "testindex"})
    assert response.status_code == 401

    # Test assets endpoints (need to create index first)
    headers = {"X-API-Key": "test-secret-key-12345"}
    client_protected.post("/indexes", json={"name": "testindex"}, headers=headers)

    response = client_protected.post(
        "/indexes/testindex/assets",
        json=[{"iscc_id": "ISCC:MAIGIIFJRDGEQQAA", "iscc_code": "ISCC:KECT"}],
    )
    assert response.status_code == 401

    response = client_protected.get("/indexes/testindex/assets/ISCC:MAIGIIFJRDGEQQAA")
    assert response.status_code == 401

    # Test search endpoints
    response = client_protected.post(
        "/indexes/testindex/search",
        json={"iscc_code": "ISCC:KECYCMZIOY36XXGZ"},
    )
    assert response.status_code == 401

    response = client_protected.get(
        "/indexes/testindex/search",
        params={"iscc_code": "ISCC:KECYCMZIOY36XXGZ"},
    )
    assert response.status_code == 401

    response = client_protected.post(
        "/indexes/testindex/search/text",
        json={"text": "test"},
    )
    assert response.status_code == 401

    # Cleanup
    client_protected.delete("/indexes/testindex", headers=headers)


def test_public_endpoints_always_accessible(client_protected):
    # type: (TestClient) -> None
    """Test that public endpoints are accessible even in protected mode."""
    # Root endpoint
    response = client_protected.get("/")
    assert response.status_code == 200

    # Docs endpoint
    response = client_protected.get("/docs")
    assert response.status_code == 200

    # Playground endpoint
    response = client_protected.get("/playground")
    assert response.status_code == 200

    # OpenAPI static files
    response = client_protected.get("/openapi/openapi.yaml")
    assert response.status_code == 200
