"""Tests for the frontend router: landing pages, JSON root, playground redirect, and /status."""

import time
import typing  # noqa: F401
import pytest
from fastapi.testclient import TestClient
import iscc_search.options
from iscc_search import __version__
from iscc_search.aggregator import poller
from iscc_search.server import app
from iscc_search.server.frontend import cached_index_info


ISCC_ID = "ISCC:MAIGIIFJRDGEQQAA"
ISCC_CODE = "ISCC:KECYCMZIOY36XXGZ7S6QJQ2AEEXPOVEHZYPK6GMSFLU3WF54UPZMTPY"
HTML_ACCEPT = {"Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"}


@pytest.fixture
def client_normal():
    # type: () -> typing.Generator[TestClient, None, None]
    """TestClient in normal mode on a memory backend."""
    opts = iscc_search.options.search_opts
    original_uri = opts.index_uri
    opts.index_uri = "memory://"
    try:
        with TestClient(app) as client:
            yield client
    finally:
        opts.index_uri = original_uri


@pytest.fixture
def client_aggregator():
    # type: () -> typing.Generator[TestClient, None, None]
    """
    TestClient with aggregator mode toggled on after startup (no poller running).

    Populates the idptest index with one asset before enabling aggregator mode
    so /status can report index stats.
    """
    opts = iscc_search.options.search_opts
    original = (opts.index_uri, opts.aggregator_network)
    opts.index_uri = "memory://"
    try:
        with TestClient(app) as client:
            assert client.post("/indexes", json={"name": "idptest"}).status_code == 201
            response = client.post(
                "/indexes/idptest/assets",
                json=[{"iscc_id": ISCC_ID, "iscc_code": ISCC_CODE}],
            )
            assert response.status_code == 201
            opts.aggregator_network = "testnet"
            yield client
    finally:
        opts.index_uri, opts.aggregator_network = original


@pytest.fixture
def client_aggregator_empty():
    # type: () -> typing.Generator[TestClient, None, None]
    """Aggregator-mode TestClient whose backend has no idptest index."""
    opts = iscc_search.options.search_opts
    original = (opts.index_uri, opts.aggregator_network)
    opts.index_uri = "memory://"
    try:
        with TestClient(app) as client:
            opts.aggregator_network = "testnet"
            yield client
    finally:
        opts.index_uri, opts.aggregator_network = original


def test_root_json_normal(client_normal):
    # type: (TestClient) -> None
    """Default Accept (*/*) gets the JSON API summary with mode and network."""
    response = client_normal.get("/")
    assert response.headers["vary"] == "Accept"  # negotiated route must not be cached across Accept
    data = response.json()
    assert data["title"] == "ISCC-Search API"
    assert data["version"] == __version__
    assert data["docs"] == "/docs"
    assert data["mode"] == "normal"
    assert data["network"] is None


def test_root_json_aggregator(client_aggregator):
    # type: (TestClient) -> None
    """In aggregator mode the JSON root reports mode and network."""
    data = client_aggregator.get("/").json()
    assert data["mode"] == "aggregator"
    assert data["network"] == "testnet"


def test_root_html_normal(client_normal):
    # type: (TestClient) -> None
    """Browsers (Accept: text/html) get the normal-mode landing page."""
    response = client_normal.get("/", headers=HTML_ACCEPT)
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert response.headers["vary"] == "Accept"
    assert 'data-mode="normal"' in response.text


def test_root_html_aggregator(client_aggregator):
    # type: (TestClient) -> None
    """Browsers get the aggregator landing page when aggregator mode is on."""
    response = client_aggregator.get("/", headers=HTML_ACCEPT)
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert 'data-mode="aggregator"' in response.text


def test_playground_redirect(client_normal):
    # type: (TestClient) -> None
    """The retired playground URL permanently redirects to the landing page."""
    response = client_normal.get("/playground", follow_redirects=False)
    assert response.status_code == 301
    assert response.headers["location"] == "/"


def test_status_normal_mode(client_normal):
    # type: (TestClient) -> None
    """In normal mode /status reports version and mode without aggregator details."""
    data = client_normal.get("/status").json()
    assert data == {"version": __version__, "mode": "normal", "network": None}


def test_status_aggregator_with_hubs(client_aggregator):
    # type: (TestClient) -> None
    """In aggregator mode /status reports index stats and the per-hub ingestion table."""
    saved = getattr(app.state, "aggregator_status", None)
    app.state.aggregator_status = {
        2: poller.HubStatus(hub_id=2, url="https://sb1.example.com"),
        0: poller.HubStatus(hub_id=0, url="https://sb0.iscc.id", cursor=3, last_poll=time.time(), counts={"ok": 3}),
    }
    try:
        data = client_aggregator.get("/status").json()
        assert data["mode"] == "aggregator"
        assert data["network"] == "testnet"
        assert data["index_name"] == "idptest"
        assert data["index"]["name"] == "idptest"
        assert data["index"]["assets"] == 1
        assert "size" in data["index"]
        assert "sizes" in data["index"]
        assert [hub["hub_id"] for hub in data["hubs"]] == [0, 2]
        polled, never_polled = data["hubs"]
        assert polled["cursor"] == 3
        assert polled["ok"] is True
        assert polled["error"] is None
        assert polled["counts"] == {"ok": 3}
        assert polled["last_poll"].endswith("+00:00")  # ISO-8601 UTC
        assert never_polled["last_poll"] is None
    finally:
        if saved is None:
            del app.state.aggregator_status
        else:
            app.state.aggregator_status = saved


def test_status_aggregator_no_poller_state(client_aggregator):
    # type: (TestClient) -> None
    """Without poller state on app.state (poller never started), hubs is empty."""
    saved = getattr(app.state, "aggregator_status", None)
    if saved is not None:
        del app.state.aggregator_status
    try:
        data = client_aggregator.get("/status").json()
        assert data["hubs"] == []
    finally:
        if saved is not None:
            app.state.aggregator_status = saved


def test_status_aggregator_index_missing(client_aggregator_empty):
    # type: (TestClient) -> None
    """A missing aggregator index reports as null but still names the configured index."""
    data = client_aggregator_empty.get("/status").json()
    assert data["mode"] == "aggregator"
    assert data["index"] is None
    assert data["index_name"] == "idptest"


def test_status_aggregator_sanitizes_hub_error(client_aggregator):
    # type: (TestClient) -> None
    """A hub poll error is reported generically on /status; raw detail stays in logs."""
    saved = getattr(app.state, "aggregator_status", None)
    app.state.aggregator_status = {
        0: poller.HubStatus(
            hub_id=0,
            url="https://sb0.iscc.id",
            ok=False,
            error="ConnectError to https://internal.example:5432/secret-path",
        ),
    }
    try:
        hub = client_aggregator.get("/status").json()["hubs"][0]
        assert hub["ok"] is False
        assert hub["error"] == "poll failed"
    finally:
        if saved is None:
            del app.state.aggregator_status
        else:
            app.state.aggregator_status = saved


def test_cached_index_info_reuses_snapshot_within_ttl(client_aggregator):
    # type: (TestClient) -> None
    """A cached index snapshot is reused within ttl and recomputed once expired."""
    cache = {}  # type: dict
    index = app.state.index
    cold = cached_index_info(cache, index, "idptest")
    assert cold["assets"] == 1
    assert cached_index_info(cache, index, "idptest") is cold  # warm hit reuses the snapshot
    assert cached_index_info(cache, index, "idptest", ttl=0) is not cold  # expired -> recompute


def test_static_css_served(client_normal):
    # type: (TestClient) -> None
    """The stylesheet is served from the static mount."""
    response = client_normal.get("/static/css/style.css")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/css")


def test_static_font_served(client_normal):
    # type: (TestClient) -> None
    """Self-hosted woff2 fonts are served with the registered MIME type."""
    response = client_normal.get("/static/fonts/JetBrainsMono-Regular.woff2")
    assert response.status_code == 200
    assert response.headers["content-type"] == "font/woff2"
    assert response.content[:4] == b"wOF2"
