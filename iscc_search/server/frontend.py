"""Mode-aware HTML landing pages, JSON root, and public /status endpoint."""

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING
import msgspec
from fastapi import APIRouter, Request, Response
from fastapi.responses import FileResponse, RedirectResponse
from loguru import logger
from iscc_search import __version__
from iscc_search.options import search_opts

if TYPE_CHECKING:
    from iscc_search.aggregator.poller import HubStatus  # noqa: F401
    from iscc_search.protocols.index import IsccIndexProtocol  # noqa: F401


STATIC_DIR = Path(__file__).parent / "static"

# /status is public and polled (~10s/tab); index-size accounting walks shard files,
# so a snapshot is reused for this many seconds to bound disk I/O.
STATUS_INDEX_TTL = 30.0


def cached_index_info(cache, index, name, ttl=STATUS_INDEX_TTL):
    # type: (dict, IsccIndexProtocol, str, float) -> dict | None
    """
    Return an index's stats as a JSON dict, reusing a snapshot for ttl seconds.

    Caches per index name so the public /status does not re-walk shard files on
    every poll. A missing index or backend error is cached as None so /status
    degrades instead of 500-ing (matches /readyz).

    :param cache: Mutable name -> (monotonic_ts, info) snapshot store
    :param index: Index manager implementing IsccIndexProtocol
    :param name: Aggregator index name
    :param ttl: Seconds a cached snapshot stays fresh
    :return: Index stats dict, or None if the index is missing/errored
    """
    cached = cache.get(name)
    now = time.monotonic()
    if cached is not None and now - cached[0] < ttl:
        return cached[1]
    try:
        info = index.get_index(name).model_dump(mode="json")
    except Exception as exc:
        logger.warning(f"/status: get_index({name!r}) failed: {exc}")
        info = None
    cache[name] = (now, info)
    return info


router = APIRouter()


def hub_status_dict(hub_status):
    # type: (HubStatus) -> dict
    """
    Convert a HubStatus struct to a JSON-safe dict for the public /status response.

    Renders last_poll as ISO-8601 and replaces the raw error with a generic
    marker — the full exception is already in the server logs (poll_loop /
    poll_hub_once log it), so /status stays free of internal strings (paths,
    library internals) for anonymous callers.

    :param hub_status: Per-hub ingestion status from the poller
    :return: JSON-serializable dict for the /status response
    """
    data = msgspec.to_builtins(hub_status)
    if data["last_poll"] is not None:
        data["last_poll"] = datetime.fromtimestamp(data["last_poll"], tz=timezone.utc).isoformat()
    if data["error"] is not None:
        data["error"] = "poll failed"
    return data


@router.get("/", include_in_schema=False)
def root(request: Request, response: Response):
    # type: (Request, Response) -> FileResponse | dict
    """
    Serve the mode-specific landing page to browsers, JSON to API clients.

    Content negotiation: an Accept header containing text/html (any browser)
    gets the branded HTML landing page for the active deployment mode; all
    other clients (curl, httpx, monitoring) get the JSON API summary. Both
    variants set ``Vary: Accept`` so a shared cache/CDN never serves the HTML
    page to a JSON client (or vice versa).

    :param request: FastAPI request (Accept header + app metadata)
    :param response: Injected response, used to set Vary on the JSON variant
    :return: HTML landing page or JSON API information
    """
    if "text/html" in request.headers.get("accept", ""):
        page = "aggregator.html" if search_opts.aggregator_mode else "index.html"
        return FileResponse(STATIC_DIR / page, media_type="text/html", headers={"Vary": "Accept"})
    response.headers["Vary"] = "Accept"
    return {
        "title": request.app.title,
        "description": request.app.description,
        "version": request.app.version,
        "docs": "/docs",
        "mode": "aggregator" if search_opts.aggregator_mode else "normal",
        "network": search_opts.aggregator_network,
    }


@router.get("/playground", include_in_schema=False)
def playground_redirect():
    # type: () -> RedirectResponse
    """
    Permanent redirect from the retired playground URL to the landing page.

    :return: 301 redirect to /
    """
    return RedirectResponse(url="/", status_code=301)


@router.get("/status", include_in_schema=False)
def server_status(request: Request):
    # type: (Request) -> dict
    """
    Public deployment status: version, mode, and aggregator ingestion state.

    In aggregator mode the response additionally carries the configured
    aggregator index name, the index stats, and the per-hub ingestion table
    published by the poller. Public in both modes (like /healthz) — it feeds
    the landing pages and gives operators a curl-able health summary.

    :param request: FastAPI request (index and poller status on app.state)
    :return: Status dict
    """
    result = {
        "version": __version__,
        "mode": "aggregator" if search_opts.aggregator_mode else "normal",
        "network": search_opts.aggregator_network,
    }
    if not search_opts.aggregator_mode:
        return result
    result["index_name"] = search_opts.aggregator_index_name
    # Public status must degrade, not 500, on a backend hiccup (matches /readyz); the
    # cache bounds repeated shard-file walks from polling clients.
    result["index"] = cached_index_info(
        request.app.state.status_index_cache, request.app.state.index, search_opts.aggregator_index_name
    )
    # Copy before iterating: the poller inserts and prunes hubs from the event-loop thread.
    statuses = dict(getattr(request.app.state, "aggregator_status", {}))
    result["hubs"] = [hub_status_dict(statuses[hub_id]) for hub_id in sorted(statuses)]
    return result
