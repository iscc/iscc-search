"""Tests for aggregator-mode endpoint gating (route reachability in both modes)."""

import asyncio
import typing  # noqa: F401
import pytest
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient
from loguru import logger
import iscc_search.options
from iscc_search.server import app, log_poller_crash
from iscc_search.server.auth import block_foreign_index_if_aggregator, block_if_aggregator


ISCC_ID = "ISCC:MAIGIIFJRDGEQQAA"
ISCC_CODE = "ISCC:KECYCMZIOY36XXGZ7S6QJQ2AEEXPOVEHZYPK6GMSFLU3WF54UPZMTPY"


@pytest.fixture
def client_aggregator():
    # type: () -> typing.Generator[TestClient, None, None]
    """
    TestClient with aggregator mode enabled via search_opts (read live by the guards).

    Populates the idptest index and a second index named "other" before
    enabling aggregator mode, so reads can be asserted against both the
    aggregator index and a foreign index. The lifespan ran with aggregator
    mode off, so no poller is started.
    """
    opts = iscc_search.options.search_opts
    original = (opts.index_uri, opts.aggregator_network)
    opts.index_uri = "memory://"

    try:
        with TestClient(app) as client:
            for name in ("idptest", "other"):
                assert client.post("/indexes", json={"name": name}).status_code == 201
                response = client.post(
                    f"/indexes/{name}/assets",
                    json=[{"iscc_id": ISCC_ID, "iscc_code": ISCC_CODE}],
                )
                assert response.status_code == 201
            opts.aggregator_network = "testnet"
            yield client
    finally:
        # search_opts is a module-level singleton — restore for tests that follow
        opts.index_uri, opts.aggregator_network = original


@pytest.fixture
def client_normal():
    # type: () -> typing.Generator[TestClient, None, None]
    """TestClient in normal mode with the idptest index populated."""
    original_uri = iscc_search.options.search_opts.index_uri
    iscc_search.options.search_opts.index_uri = "memory://"

    try:
        with TestClient(app) as client:
            assert client.post("/indexes", json={"name": "idptest"}).status_code == 201
            response = client.post(
                "/indexes/idptest/assets",
                json=[{"iscc_id": ISCC_ID, "iscc_code": ISCC_CODE}],
            )
            assert response.status_code == 201
            yield client
    finally:
        iscc_search.options.search_opts.index_uri = original_uri


def test_aggregator_read_surface_reachable(client_aggregator):
    # type: (TestClient) -> None
    """Search and get-asset endpoints stay reachable for the aggregator index."""
    response = client_aggregator.post("/indexes/idptest/search", json={"iscc_code": ISCC_CODE})
    assert response.status_code == 200

    response = client_aggregator.get("/indexes/idptest/search", params={"iscc_code": ISCC_CODE})
    assert response.status_code == 200

    response = client_aggregator.get(f"/indexes/idptest/assets/{ISCC_ID}")
    assert response.status_code == 200
    assert response.json()["iscc_id"] == ISCC_ID


def test_aggregator_suppressed_endpoints_404(client_aggregator):
    # type: (TestClient) -> None
    """Index management and asset add return 404; playground redirects."""
    assert client_aggregator.get("/indexes").status_code == 404
    assert client_aggregator.post("/indexes", json={"name": "newindex"}).status_code == 404
    assert client_aggregator.get("/indexes/idptest").status_code == 404
    assert client_aggregator.delete("/indexes/idptest").status_code == 404
    response = client_aggregator.post(
        "/indexes/idptest/assets",
        json=[{"iscc_id": ISCC_ID, "iscc_code": ISCC_CODE}],
    )
    assert response.status_code == 404
    response = client_aggregator.get("/playground", follow_redirects=False)
    assert response.status_code == 301
    assert response.headers["location"] == "/"


def test_all_index_routes_carry_an_aggregator_gate():
    # type: () -> None
    """Every /indexes* route declares a mode gate so a new endpoint can't ship ungated."""
    gates = {block_if_aggregator, block_foreign_index_if_aggregator}
    ungated = [
        f"{sorted(route.methods)} {route.path}"
        for route in app.routes
        if isinstance(route, APIRoute)
        and route.path.startswith("/indexes")
        and not ({dep.dependency for dep in route.dependencies} & gates)
    ]
    assert ungated == [], f"ungated /indexes routes reachable in aggregator mode: {ungated}"


def test_aggregator_suppressed_404_matches_unknown_route(client_aggregator):
    # type: (TestClient) -> None
    """Suppressed endpoints are indistinguishable from unknown routes."""
    suppressed = client_aggregator.get("/indexes")
    unknown = client_aggregator.get("/nosuchroute")
    assert suppressed.status_code == unknown.status_code == 404
    assert suppressed.json() == unknown.json()


def test_aggregator_foreign_index_reads_404(client_aggregator):
    # type: (TestClient) -> None
    """Reads against any index other than the aggregator index return 404."""
    response = client_aggregator.post("/indexes/other/search", json={"iscc_code": ISCC_CODE})
    assert response.status_code == 404

    response = client_aggregator.get("/indexes/other/search", params={"iscc_code": ISCC_CODE})
    assert response.status_code == 404

    response = client_aggregator.get(f"/indexes/other/assets/{ISCC_ID}")
    assert response.status_code == 404


def test_aggregator_suppressed_404_before_auth_401(client_aggregator):
    # type: (TestClient) -> None
    """With api_secret set and no key, suppressed endpoints return 404, not 401."""
    original_secret = iscc_search.options.search_opts.api_secret
    iscc_search.options.search_opts.api_secret = "secret-key"
    try:
        assert client_aggregator.get("/indexes").status_code == 404
        assert client_aggregator.post("/indexes", json={"name": "newindex"}).status_code == 404
        response = client_aggregator.post("/indexes/other/search", json={"iscc_code": ISCC_CODE})
        assert response.status_code == 404
        # The exposed read surface still enforces auth
        response = client_aggregator.post("/indexes/idptest/search", json={"iscc_code": ISCC_CODE})
        assert response.status_code == 401
    finally:
        iscc_search.options.search_opts.api_secret = original_secret


def test_aggregator_infra_routes_reachable(client_aggregator):
    # type: (TestClient) -> None
    """Infra endpoints (root, healthz, readyz, docs, status) remain available."""
    assert client_aggregator.get("/").status_code == 200
    assert client_aggregator.get("/healthz").status_code == 200
    assert client_aggregator.get("/readyz").status_code == 200
    assert client_aggregator.get("/docs").status_code == 200
    assert client_aggregator.get("/status").status_code == 200


def test_normal_mode_all_endpoints_reachable(client_normal):
    # type: (TestClient) -> None
    """With aggregator mode off, all endpoints behave as before."""
    assert client_normal.get("/indexes").status_code == 200
    assert client_normal.get("/indexes/idptest").status_code == 200
    response = client_normal.post(
        "/indexes/idptest/assets",
        json=[{"iscc_id": ISCC_ID, "iscc_code": ISCC_CODE}],
    )
    assert response.status_code == 201
    response = client_normal.post("/indexes/idptest/search", json={"iscc_code": ISCC_CODE})
    assert response.status_code == 200
    response = client_normal.get("/indexes/idptest/search", params={"iscc_code": ISCC_CODE})
    assert response.status_code == 200
    assert client_normal.get(f"/indexes/idptest/assets/{ISCC_ID}").status_code == 200
    response = client_normal.get("/playground", follow_redirects=False)
    assert response.status_code == 301
    assert response.headers["location"] == "/"
    assert client_normal.delete("/indexes/idptest").status_code == 204


def test_lifespan_aggregator_starts_and_stops_poller(tmp_path):
    # type: (typing.Any) -> None
    """Aggregator mode creates the index, runs the poller, and stops it on shutdown."""
    hub_file = tmp_path / "hubs.yaml"
    hub_file.write_text("version: 1\nnetwork: testnet\nhubs:\n", encoding="utf-8")
    opts = iscc_search.options.search_opts
    original = (opts.index_uri, opts.aggregator_network, opts.aggregator_hub_list_url)
    opts.index_uri = "memory://"
    opts.aggregator_network = "testnet"
    opts.aggregator_hub_list_url = str(hub_file)
    try:
        with TestClient(app) as client:
            assert not app.state.aggregator_poller_task.done()
            # The aggregator index exists on a fresh deployment: search works before any asset
            response = client.get("/indexes/idptest/search", params={"iscc_code": ISCC_CODE})
            assert response.status_code == 200
        assert app.state.aggregator_poller_task.done()
    finally:
        opts.index_uri, opts.aggregator_network, opts.aggregator_hub_list_url = original


def test_lifespan_aggregator_existing_index(tmp_path):
    # type: (typing.Any) -> None
    """A restart with an existing aggregator index passes ensure-create (FileExistsError)."""
    hub_file = tmp_path / "hubs.yaml"
    hub_file.write_text("version: 1\nnetwork: testnet\nhubs:\n", encoding="utf-8")
    opts = iscc_search.options.search_opts
    original = (opts.index_uri, opts.aggregator_network, opts.aggregator_hub_list_url)
    opts.index_uri = f"lmdb:///{(tmp_path / 'data').as_posix()}"
    opts.aggregator_network = "testnet"
    opts.aggregator_hub_list_url = str(hub_file)
    try:
        with TestClient(app):
            pass  # first startup creates the idptest index
        with TestClient(app) as client:
            response = client.get("/indexes/idptest/search", params={"iscc_code": ISCC_CODE})
            assert response.status_code == 200
    finally:
        opts.index_uri, opts.aggregator_network, opts.aggregator_hub_list_url = original


async def crashing_poller():
    # type: () -> None
    """Stand-in poller coroutine that crashes immediately."""
    raise RuntimeError("kaboom")


def test_log_poller_crash_callback():
    # type: () -> None
    """log_poller_crash logs escaped exceptions and stays silent for cancellation."""
    messages = []
    sink_id = logger.add(lambda message: messages.append(str(message)), level="ERROR")

    async def main():
        # type: () -> None
        crashed = asyncio.create_task(crashing_poller())
        await asyncio.gather(crashed, return_exceptions=True)
        log_poller_crash(crashed)
        cancelled = asyncio.create_task(asyncio.sleep(10))
        cancelled.cancel()
        await asyncio.gather(cancelled, return_exceptions=True)
        log_poller_crash(cancelled)

    try:
        asyncio.run(main())
    finally:
        logger.remove(sink_id)

    assert len(messages) == 1
    assert "kaboom" in messages[0]
