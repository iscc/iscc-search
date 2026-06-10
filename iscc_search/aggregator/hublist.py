"""
Hub-list loading for the IDP aggregator.

Parses the authoritative ``{network}.yaml`` hub list (pure) and loads it from
an http(s) URL or a local file path (thin I/O shell).
"""

from pathlib import Path
import msgspec
import yaml


class Hub(msgspec.Struct, frozen=True):
    """Active hub entry from the authoritative {network}.yaml hub list."""

    hub_id: int
    url: str


def parse_hub_list(data, network):
    # type: (bytes | str, str) -> list[Hub]
    """
    Parse hub-list YAML and return the active hubs.

    :param data: Raw YAML text/bytes ({version, network, hubs: [...]})
    :param network: Expected network name; a mismatch is rejected
    :return: Hubs with active=true, in list order
    :raises ValueError: If the YAML is not a mapping or the network mismatches
    """
    parsed = yaml.safe_load(data)
    if not isinstance(parsed, dict):
        raise ValueError("hub list is not a mapping")
    if parsed.get("network") != network:
        raise ValueError(f"hub list network mismatch: expected {network}, got {parsed.get('network')}")
    hubs = parsed.get("hubs") or []
    return [Hub(hub_id=hub["hub_id"], url=hub["url"]) for hub in hubs if hub.get("active")]


async def fetch_hub_list(source, network, client):
    # type: (str, str, httpx.AsyncClient) -> list[Hub]
    """
    Load the hub list from an http(s) URL or a local file path.

    A source starting with http:// or https:// is fetched over the network;
    any other value is read as a local filesystem path.

    :param source: Hub-list URL or file path
    :param network: Expected network name
    :param client: httpx AsyncClient used for URL sources
    :return: Hubs with active=true
    :raises ValueError: If the hub list is malformed or the network mismatches
    """
    if source.startswith(("http://", "https://")):
        response = await client.get(source)
        response.raise_for_status()
        data = response.text
    else:
        data = Path(source).read_text(encoding="utf-8")
    return parse_hub_list(data, network)
