"""FastAPI server for ISCC-VDB API."""

import typing  # noqa: F401
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from iscc_search.settings import get_index
from iscc_search.protocol import IsccIndexProtocol  # noqa: F401


@asynccontextmanager
async def lifespan(app):  # type: ignore
    # type: (FastAPI) -> typing.AsyncGenerator[None, None]
    """
    Manage ISCC index lifecycle across FastAPI app startup and shutdown.

    On startup: Creates index instance via factory and stores in app.state.
    On shutdown: Properly closes index and releases resources.

    :param app: FastAPI application instance
    :yield: Control to FastAPI application
    """
    # Startup: Create and store index instance
    app.state.index = get_index()
    yield
    # Shutdown: Close index and cleanup resources
    app.state.index.close()


def get_index_from_state(request: Request):
    # type: (...) -> IsccIndexProtocol
    """
    Dependency function to inject index instance from app state.

    Used with FastAPI's Depends() for clean dependency injection in route handlers.

    :param request: FastAPI request object
    :return: Index instance from app.state
    """
    return request.app.state.index


# Create FastAPI app instance with lifespan management
app = FastAPI(
    lifespan=lifespan,
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


# Include API routers
from iscc_search.server import indexes, assets, search  # noqa: E402

app.include_router(indexes.router)
app.include_router(assets.router)
app.include_router(search.router)
