"""FastAPI server for ISCC-VDB API."""

from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles


# Create FastAPI app instance
app = FastAPI(
    title="ISCC-VDB API",
    description="A Scalable Nearest Neighbor Search Multi-Index for the International Standard Content Code (ISCC)",
    version="0.1.0",
    docs_url=None,  # Disable default docs
    redoc_url=None,  # Disable default redoc
    openapi_url=None,  # We'll serve our own OpenAPI spec
)

# Mount OpenAPI schema files as static directory
openapi_dir = Path(__file__).parent.parent / "openapi"
app.mount("/openapi", StaticFiles(directory=str(openapi_dir)), name="openapi")


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
          data-url="/openapi/openapi.yaml"
          data-configuration='{{"telemetry": false, "hideClientButton": true, "hideSearch": true, "defaultOpenAllTags": true, "layout": "classic"}}'></script>
        <script src="https://cdn.jsdelivr.net/npm/@scalar/api-reference"></script>
      </body>
    </html>
    """
    return HTMLResponse(content=html)


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
