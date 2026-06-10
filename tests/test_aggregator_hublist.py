"""Tests for hub-list parsing and loading."""

import asyncio
from pathlib import Path
import httpx
import pytest
from iscc_search.aggregator.hublist import Hub, fetch_hub_list, parse_hub_list


TESTNET_YAML = (Path(__file__).parent / "data" / "testnet.yaml").read_bytes()


def test_parse_hub_list_real_testnet():
    """The real testnet.yaml parses into the two active sandbox hubs."""
    hubs = parse_hub_list(TESTNET_YAML, "testnet")
    assert hubs == [Hub(hub_id=0, url="https://sb0.iscc.id"), Hub(hub_id=1, url="https://sb1.amlet.id")]


def test_parse_hub_list_filters_inactive():
    """Hubs with active=false are excluded."""
    data = """
network: testnet
hubs:
    - {hub_id: 0, url: "https://sb0.iscc.id", active: true}
    - {hub_id: 1, url: "https://gone.example.com", active: false}
"""
    assert parse_hub_list(data, "testnet") == [Hub(hub_id=0, url="https://sb0.iscc.id")]


def test_parse_hub_list_network_mismatch():
    """A hub list for another network is rejected."""
    with pytest.raises(ValueError, match="network mismatch"):
        parse_hub_list(TESTNET_YAML, "mainnet")


def test_parse_hub_list_not_mapping():
    """Non-mapping YAML (e.g. an error body) is rejected."""
    with pytest.raises(ValueError, match="not a mapping"):
        parse_hub_list("Not Found", "testnet")
    with pytest.raises(ValueError, match="not a mapping"):
        parse_hub_list("- a\n- b\n", "testnet")


def test_parse_hub_list_missing_hubs():
    """A hub list without hubs (mainnet not yet published) yields an empty list."""
    assert parse_hub_list("version: 1\nnetwork: mainnet\n", "mainnet") == []
    assert parse_hub_list("version: 1\nnetwork: mainnet\nhubs:\n", "mainnet") == []


async def fetch_with_transport(source, network, handler):
    # type: (str, str, typing.Callable) -> list[Hub]
    """Run fetch_hub_list against a MockTransport-backed AsyncClient."""
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        return await fetch_hub_list(source, network, client)


def test_fetch_hub_list_url():
    """An http(s) source is fetched and parsed."""
    handler = lambda request: httpx.Response(200, content=TESTNET_YAML)  # noqa: E731
    hubs = asyncio.run(fetch_with_transport("https://example.com/testnet.yaml", "testnet", handler))
    assert len(hubs) == 2


def test_fetch_hub_list_url_http_error():
    """An HTTP error status raises instead of parsing the error body."""
    handler = lambda request: httpx.Response(404, content=b"Not Found")  # noqa: E731
    with pytest.raises(httpx.HTTPStatusError):
        asyncio.run(fetch_with_transport("https://example.com/mainnet.yaml", "mainnet", handler))


def test_fetch_hub_list_local_file(tmp_path):
    """A non-URL source is read as a local file path."""
    path = tmp_path / "hubs.yaml"
    path.write_bytes(TESTNET_YAML)
    handler = lambda request: httpx.Response(500)  # noqa: E731
    hubs = asyncio.run(fetch_with_transport(str(path), "testnet", handler))
    assert len(hubs) == 2
