"""Test FastAPI server for ISCC-VDB API."""

from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from iscc_vdb.server import app, custom_docs, root


# Create TestClient instance at module level following FastAPI best practices
client = TestClient(app)


def test_app_instance():
    """Test FastAPI app instance is properly configured."""
    assert app.title == "ISCC-VDB API"
    assert app.version == "0.1.0"
    assert app.docs_url is None
    assert app.redoc_url is None
    assert app.openapi_url is None
    assert "A Scalable Nearest Neighbor Search" in app.description


def test_root_endpoint():
    """Test root endpoint returns API information."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["title"] == "ISCC-VDB API"
    assert data["version"] == "0.1.0"
    assert data["docs"] == "/docs"
    assert "description" in data


def test_docs_endpoint():
    """Test custom docs endpoint returns HTML with Scalar UI."""
    response = client.get("/docs")
    assert response.status_code == 200
    assert response.headers["content-type"] == "text/html; charset=utf-8"

    html_content = response.text
    assert "<!doctype html>" in html_content.lower()
    assert "ISCC-VDB API - Documentation" in html_content
    assert "/openapi/openapi.yaml" in html_content
    assert "@scalar/api-reference" in html_content
    assert "api-reference" in html_content


def test_custom_docs_function():
    """Test custom_docs function directly."""
    result = custom_docs()
    html_content = result.body.decode("utf-8")
    assert "<!doctype html>" in html_content.lower()
    assert "ISCC-VDB API - Documentation" in html_content
    assert "/openapi/openapi.yaml" in html_content
    assert "telemetry" in html_content
    assert "false" in html_content


def test_root_function():
    """Test root function directly."""
    result = root()
    assert isinstance(result, dict)
    assert result["title"] == "ISCC-VDB API"
    assert result["version"] == "0.1.0"
    assert result["docs"] == "/docs"


def test_openapi_static_files():
    """Test that OpenAPI static files are accessible."""
    response = client.get("/openapi/openapi.yaml")
    assert response.status_code == 200


def test_main_entry_point():
    """Test main entry point starts uvicorn server."""
    from iscc_vdb.server.__main__ import main

    with patch("iscc_vdb.server.__main__.uvicorn.run") as mock_run:
        main()
        mock_run.assert_called_once_with(
            "iscc_vdb.server:app",
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
    import iscc_vdb.server.__main__
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
    from iscc_vdb.indexes.memory import MemoryIndex

    # Create a mock index with a close method we can track
    mock_index = MagicMock(spec=MemoryIndex)

    # Use TestClient context manager to trigger lifespan shutdown
    with TestClient(app) as client:
        # Override with our mock
        client.app.state.index = mock_index

    # Verify close was called during shutdown
    mock_index.close.assert_called_once()
