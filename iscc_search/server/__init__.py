"""FastAPI server for ISCC-Search API."""

import typing  # noqa: F401
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from iscc_search.settings import get_index
from iscc_search.protocols.index import IsccIndexProtocol  # noqa: F401


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
    title="ISCC-Search API",
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
    Render modern interactive API documentation using Stoplight Elements.

    :return: HTML response with Stoplight Elements UI
    """
    html = rf"""
    <!doctype html>
    <html>
      <head>
        <title>{app.title} - Documentation</title>
        <meta charset="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />\
        <script>
          // Intercept fetch and XMLHttpRequest to add ngrok-skip-browser-warning header
          // This bypasses ngrok's interstitial page for both schema loading and Try It requests
          (function() {{
            // Intercept fetch API
            const originalFetch = window.fetch;
            window.fetch = function(...args) {{
              let [resource, config] = args;
              config = config || {{}};
              config.headers = config.headers || {{}};

              // Add ngrok bypass header to all requests
              if (config.headers instanceof Headers) {{
                config.headers.set('ngrok-skip-browser-warning', '1');
              }} else {{
                config.headers['ngrok-skip-browser-warning'] = '1';
              }}

              return originalFetch(resource, config);
            }};

            // Intercept XMLHttpRequest for Try It feature
            const originalXHROpen = XMLHttpRequest.prototype.open;
            const originalXHRSend = XMLHttpRequest.prototype.send;

            XMLHttpRequest.prototype.open = function(method, url, ...rest) {{
              this._url = url;
              this._method = method;
              return originalXHROpen.apply(this, [method, url, ...rest]);
            }};

            XMLHttpRequest.prototype.send = function(...args) {{
              // Add ngrok bypass header before sending
              this.setRequestHeader('ngrok-skip-browser-warning', '1');
              return originalXHRSend.apply(this, args);
            }};
          }})();
        </script>
        <script src="https://unpkg.com/@stoplight/elements/web-components.min.js"></script>
        <link rel="stylesheet" href="https://unpkg.com/@stoplight/elements/styles.min.css">
        <style>
          /* Override inline max-width on right panel to allow full width for long ISCC strings */
          .sl-elements .sl-relative.sl-w-2\/5.sl-ml-16 {{
            max-width: none !important;
          }}

          /* Enable horizontal scrolling for code blocks */
          .sl-elements pre {{
            overflow-x: auto !important;
            white-space: pre !important;
          }}

          .sl-elements code {{
            white-space: pre !important;
          }}
        </style>
      </head>
      <body>
        <elements-api
          apiDescriptionUrl="/openapi/openapi.yaml"
          router="hash"
          layout="sidebar"
          hideExport="true"
          logo="https://avatars.githubusercontent.com/u/47259639?s=400&u=d26d161a5e7391dd7e0011ca3d5b317b93e6ad4d&v=4"
        />
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
from iscc_search.server import indexes, assets, search, playground  # noqa: E402

app.include_router(indexes.router)
app.include_router(assets.router)
app.include_router(search.router)
app.include_router(playground.router)
