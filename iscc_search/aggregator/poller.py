"""
Transparency-log poller for the IDP aggregator.

Orchestrates ingestion: plan which entry bundles cover the new records (pure
``plan_bundles``), fetch and convert them per hub (``poll_hub_once``), and
loop over the refreshed hub list until stopped (``poll_loop``/``run``).
Ingestion is at-least-once and safe by construction: ``add_assets`` is an
idempotent upsert and cursors are in-memory, so a restart re-backfills from
leaf 0.
"""

import asyncio
import time
from typing import TYPE_CHECKING
import httpx
import msgspec
from loguru import logger
from iscc_search.aggregator import hublist, tlog
from iscc_search.aggregator.entry import REASONS, record_to_entry

if TYPE_CHECKING:
    from iscc_search.options import SearchOptions  # noqa: F401
    from iscc_search.protocols.index import IsccIndexProtocol  # noqa: F401

# Per-HTTP-request timeout in seconds (module constant; promote to an option on demand).
HTTP_TIMEOUT = 30.0


class PollResult(msgspec.Struct, frozen=True):
    """
    Outcome of polling one hub: new cursor position and per-reason record counts.

    ``error`` carries a bundle-level failure (fetch, validation, or indexing)
    that stopped the poll early — the cursor keeps the progress made, but the
    hub must not be reported as healthy.
    """

    last_size: int
    counts: dict[str, int]
    error: str | None = None


class HubStatus(msgspec.Struct):
    """
    Mutable per-hub ingestion status shared between poll_loop and the /status endpoint.

    ``cursor`` is the single source of truth for poll progress. ``counts`` are
    cumulative per-reason record counts since process start and double-count
    after a checkpoint-regression re-backfill (accepted; the cursor stays
    truthful). ``last_poll`` is wall-clock epoch seconds (time.time()).
    """

    hub_id: int
    url: str
    cursor: int = 0
    last_poll: float | None = None
    ok: bool = True
    error: str | None = None
    counts: dict[str, int] = msgspec.field(default_factory=dict)


def plan_bundles(last_size, tree_size):
    # type: (int, int) -> list[tuple[int, int]]
    """
    Plan the entry-bundle fetches covering records [last_size, tree_size).

    Returns (bundle_index, width) pairs where width=0 requests a full bundle
    and width>0 requests the in-progress partial bundle as .p/<width>. Empty
    when the tree has not grown — including a tree-size regression, which the
    caller handles separately.

    :param last_size: Number of records already processed
    :param tree_size: Tree size reported by the hub checkpoint
    :return: Bundle fetch plan in index order
    """
    if tree_size <= last_size:
        return []
    first = last_size // tlog.TILE_WIDTH
    last = (tree_size - 1) // tlog.TILE_WIDTH
    plans = []
    for index in range(first, last + 1):
        if (index + 1) * tlog.TILE_WIDTH <= tree_size:
            plans.append((index, 0))
        else:
            plans.append((index, tree_size - index * tlog.TILE_WIDTH))
    return plans


async def poll_hub_once(client, hub, last_size, index, index_name, network, stop_event):
    # type: (httpx.AsyncClient, hublist.Hub, int, IsccIndexProtocol, str, str, asyncio.Event) -> PollResult
    """
    Poll one hub once: read its checkpoint and ingest any new log records.

    Fetches bundles one at a time and awaits between them (bounded on-loop
    bursts of <=256 records — do not batch bundles before yielding). A
    checkpoint tree-size regression (e.g. a hub database reset) logs a warning
    and resets the cursor to 0 so the next poll re-backfills. A failing bundle
    (fetch error, short bundle, or indexing error) stops the poll but keeps
    the progress made so far, so the next poll resumes at the failed bundle;
    the failure is surfaced via ``PollResult.error`` for hub-health reporting.

    :param client: httpx AsyncClient (injectable for tests)
    :param hub: Hub to poll
    :param last_size: Records already processed for this hub
    :param index: Index manager implementing IsccIndexProtocol
    :param index_name: Target index name (idp/idptest)
    :param network: Deployment network for realm checks
    :param stop_event: Honored between bundle fetches for fast shutdown
    :return: PollResult with the new cursor and per-reason record counts
    """
    response = await client.get(f"{hub.url}/log/checkpoint")
    response.raise_for_status()
    tree_size = tlog.parse_checkpoint(response.text)
    counts = dict.fromkeys(REASONS, 0)
    if tree_size < last_size:
        logger.warning(f"aggregator: {hub.url}: checkpoint regression {last_size} -> {tree_size}, re-backfilling")
        return PollResult(last_size=0, counts=counts)
    processed = last_size
    for bundle_index, width in plan_bundles(last_size, tree_size):
        if stop_event.is_set():
            break
        try:
            path = tlog.entries_path(bundle_index, width)
            response = await client.get(f"{hub.url}/log/{path}")
            response.raise_for_status()
            records = tlog.parse_entry_bundle(response.content)
            expected = width or tlog.TILE_WIDTH
            if len(records) != expected:
                raise ValueError(f"{path} has {len(records)} records, expected {expected}")
            bundle_start = bundle_index * tlog.TILE_WIDTH
            entries = []
            for record in records[max(last_size - bundle_start, 0) :]:
                converted, reason = record_to_entry(record, network)
                counts[reason] += 1
                if converted is not None:
                    entries.append(converted)
                elif reason != "deletion":
                    logger.warning(f"aggregator: {hub.url}: skipped record ({reason})")
            await asyncio.to_thread(index.add_assets, index_name, entries)
        except Exception as exc:
            logger.warning(f"aggregator: {hub.url}: bundle {bundle_index} failed, retrying next poll: {exc}")
            return PollResult(last_size=processed, counts=counts, error=f"bundle {bundle_index} failed: {exc}")
        processed = min((bundle_index + 1) * tlog.TILE_WIDTH, tree_size)
    return PollResult(last_size=processed, counts=counts)


async def poll_loop(index, opts, stop_event, client, status):
    # type: (IsccIndexProtocol, SearchOptions, asyncio.Event, httpx.AsyncClient, dict[int, HubStatus]) -> None
    """
    Aggregator loop: refresh the hub list and poll each hub until stopped.

    A hub-list refresh failure keeps the last-known-good list and retries
    after the poll interval instead of the full refresh interval; an empty hub
    list is a benign idle state (mainnet.yaml may not exist yet). Each hub's
    poll failure is isolated. Per-hub cursors live in the shared ``status``
    mapping, which the /status endpoint reads concurrently for observability;
    hubs that drop out of a non-empty refreshed list are pruned from it so
    /status never reports decommissioned hubs as live. An empty refresh prunes
    nothing, so a transient empty/all-inactive hub list cannot discard cursors
    and force a full re-backfill.

    :param index: Index manager implementing IsccIndexProtocol
    :param opts: Deployment options (network, intervals, hub-list source)
    :param stop_event: Set to request shutdown
    :param client: httpx AsyncClient shared across all requests
    :param status: Mutable hub_id -> HubStatus mapping updated in place
    """
    network = opts.aggregator_network
    index_name = opts.aggregator_index_name
    hubs = []  # type: list[hublist.Hub]
    next_refresh = 0.0
    while not stop_event.is_set():
        now = time.monotonic()
        if now >= next_refresh:
            try:
                hubs = await hublist.fetch_hub_list(opts.aggregator_hub_list_source, network, client)
                if not hubs:
                    # Re-check at the short poll interval: a transient empty/all-inactive list
                    # must not stall ingestion for a full refresh interval (cursors survive, but
                    # nothing would be polled until the next hourly refresh).
                    logger.warning("aggregator: hub list is empty, nothing to poll")
                    next_refresh = now + opts.aggregator_poll_interval
                else:
                    for stale_id in set(status) - {hub.hub_id for hub in hubs}:
                        del status[stale_id]
                    next_refresh = now + opts.aggregator_hub_refresh_interval
            except Exception as exc:
                logger.warning(f"aggregator: hub-list refresh failed, keeping previous list: {exc}")
                next_refresh = now + opts.aggregator_poll_interval
        for hub in hubs:
            if stop_event.is_set():
                break
            if hub.hub_id not in status:
                status[hub.hub_id] = HubStatus(hub_id=hub.hub_id, url=hub.url)
            hub_status = status[hub.hub_id]
            hub_status.url = hub.url  # a hub-list refresh may relocate the hub
            hub_status.last_poll = time.time()
            try:
                result = await poll_hub_once(client, hub, hub_status.cursor, index, index_name, network, stop_event)
                hub_status.cursor = result.last_size
                for reason, count in result.counts.items():
                    hub_status.counts[reason] = hub_status.counts.get(reason, 0) + count
                hub_status.ok = result.error is None
                hub_status.error = result.error
                if result.counts["ok"]:
                    logger.info(
                        f"aggregator: {hub.url}: indexed {result.counts['ok']} records, cursor {result.last_size}"
                    )
            except Exception as exc:
                hub_status.ok = False
                hub_status.error = str(exc)
                logger.warning(f"aggregator: poll of {hub.url} failed: {exc}")
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=opts.aggregator_poll_interval)
        except TimeoutError:
            pass


async def run(index, opts, stop_event, status):
    # type: (IsccIndexProtocol, SearchOptions, asyncio.Event, dict[int, HubStatus]) -> None
    """
    Aggregator entry point used by the server lifespan: owns the HTTP client.

    :param index: Index manager implementing IsccIndexProtocol
    :param opts: Deployment options
    :param stop_event: Set to request shutdown
    :param status: Mutable hub_id -> HubStatus mapping updated in place
    """
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT, follow_redirects=True) as client:
        await poll_loop(index, opts, stop_event, client, status)
