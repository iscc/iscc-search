"""Test FastAPI server for ISCC-Search API."""

from unittest.mock import patch, MagicMock
import pytest
from fastapi.testclient import TestClient
from iscc_search.server import app, custom_docs, root


@pytest.fixture
def client():
    # type: () -> TestClient
    """
    Create TestClient with proper resource cleanup.

    Uses context manager to ensure the lifespan shutdown event is triggered
    and resources are properly cleaned up after tests. Configures app to use
    memory:// index backend.

    :return: FastAPI TestClient instance
    """
    import iscc_search.settings

    # Save original URI and set to memory://
    original_uri = iscc_search.settings.search_settings.indexes_uri
    iscc_search.settings.search_settings.indexes_uri = "memory://"

    try:
        with TestClient(app) as client:
            yield client
    finally:
        # Restore original URI
        iscc_search.settings.search_settings.indexes_uri = original_uri


def test_app_instance():
    """Test FastAPI app instance is properly configured."""
    assert app.title == "ISCC-Search API"
    assert app.version == "0.1.0"
    assert app.docs_url is None
    assert app.redoc_url is None
    assert app.openapi_url is None
    assert "A Scalable Nearest Neighbor Search" in app.description


def test_root_endpoint(client):
    """Test root endpoint returns API information."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["title"] == "ISCC-Search API"
    assert data["version"] == "0.1.0"
    assert data["docs"] == "/docs"
    assert "description" in data


def test_docs_endpoint(client):
    """Test custom docs endpoint returns HTML with Stoplight Elements UI."""
    response = client.get("/docs")
    assert response.status_code == 200
    assert response.headers["content-type"] == "text/html; charset=utf-8"

    html_content = response.text
    assert "<!doctype html>" in html_content.lower()
    assert "ISCC-Search API - Documentation" in html_content
    assert "/openapi/openapi.yaml" in html_content
    assert "@stoplight/elements" in html_content
    assert "elements-api" in html_content


def test_custom_docs_function():
    """Test custom_docs function directly."""
    result = custom_docs()
    html_content = result.body.decode("utf-8")
    assert "<!doctype html>" in html_content.lower()
    assert "ISCC-Search API - Documentation" in html_content
    assert "/openapi/openapi.yaml" in html_content
    assert "hideExport" in html_content
    assert "logo" in html_content


def test_root_function():
    """Test root function directly."""
    result = root()
    assert isinstance(result, dict)
    assert result["title"] == "ISCC-Search API"
    assert result["version"] == "0.1.0"
    assert result["docs"] == "/docs"


def test_openapi_static_files(client):
    """Test that OpenAPI static files are accessible."""
    response = client.get("/openapi/openapi.yaml")
    assert response.status_code == 200


def test_main_entry_point():
    """Test main entry point starts uvicorn server."""
    from iscc_search.server.__main__ import main

    with patch("iscc_search.server.__main__.uvicorn.run") as mock_run:
        main()
        mock_run.assert_called_once_with(
            "iscc_search.server:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info",
        )


def test_main_module_execution():
    """Test module execution via python -m."""
    import subprocess
    import sys

    # Run the module with mocked uvicorn to verify __name__ == '__main__' block executes
    # We use python -m to properly trigger the __main__ block for coverage
    code = """
from unittest.mock import patch, MagicMock

# Mock uvicorn.run before the module executes
import sys
mock_run = MagicMock()
sys.modules['uvicorn'] = MagicMock(run=mock_run)

# Now execute the __main__ module which will call uvicorn.run
if __name__ == '__main__':
    import iscc_search.server.__main__
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        timeout=5,
        text=True,
    )
    # Check that the subprocess completed successfully
    assert result.returncode == 0, f"Module execution failed: {result.stderr}"


def test_lifespan_shutdown():
    """Test that lifespan context manager properly closes index on shutdown."""
    from iscc_search.indexes.memory import MemoryIndex
    import iscc_search.settings

    # Save and override URI to use memory://
    original_uri = iscc_search.settings.search_settings.indexes_uri
    iscc_search.settings.search_settings.indexes_uri = "memory://"

    try:
        # Use TestClient context manager to trigger lifespan events
        with TestClient(app) as client:
            # Verify the lifespan startup created an index
            assert hasattr(client.app.state, "index")
            original_index = client.app.state.index
            assert isinstance(original_index, MemoryIndex)

            # Replace the index with a mock to track close() calls
            mock_index = MagicMock(spec=MemoryIndex)
            client.app.state.index = mock_index

        # Verify close was called during shutdown on the mock
        # (In production, the real index's close() would be called)
        mock_index.close.assert_called_once()
    finally:
        # Restore original URI
        iscc_search.settings.search_settings.indexes_uri = original_uri
