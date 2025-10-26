"""FastAPI server for ISCC-VDB API."""

from pathlib import Path
import yaml
from fastapi import FastAPI
from fastapi.responses import HTMLResponse


# Load OpenAPI spec from YAML file
def load_openapi_spec():
    # type: () -> dict
    """
    Load OpenAPI specification from YAML file.

    :return: OpenAPI specification as dictionary
    """
    spec_path = Path(__file__).parent.parent / "openapi" / "openapi.yaml"
    with open(spec_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# Create FastAPI app instance
app = FastAPI(
    title="ISCC-VDB API",
    description="A Scalable Nearest Neighbor Search Multi-Index for the International Standard Content Code (ISCC)",
    version="0.1.0",
    docs_url=None,  # Disable default docs
    redoc_url=None,  # Disable default redoc
    openapi_url=None,  # We'll serve our own OpenAPI spec
)


@app.get("/docs", response_class=HTMLResponse, include_in_schema=False)
def custom_docs():
    # type: () -> HTMLResponse
    """
    Render modern interactive API documentation using Scalar.

    :return: HTML response with Scalar UI
    """
    html = f"""
    <!doctype html>
    <html>
      <head>
        <title>{app.title} - Documentation</title>
        <meta charset="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
      </head>
      <body>
        <script
          id="api-reference"
          data-url="/openapi.yaml"
          data-configuration='{{"telemetry": false, "hideClientButton": true, "hideSearch": true, "defaultOpenAllTags": true, "layout": "classic"}}'></script>
        <script src="https://cdn.jsdelivr.net/npm/@scalar/api-reference"></script>
      </body>
    </html>
    """
    return HTMLResponse(content=html)


@app.get("/openapi.yaml", include_in_schema=False)
def get_openapi_spec():
    # type: () -> dict
    """
    Serve the OpenAPI specification.

    :return: OpenAPI specification as dictionary
    """
    return load_openapi_spec()


@app.get("/", include_in_schema=False)
def root():
    # type: () -> dict
    """
    Root endpoint with basic API information.

    :return: API information
    """
    return {
        "title": app.title,
        "description": app.description,
        "version": app.version,
        "docs": "/docs",
    }
