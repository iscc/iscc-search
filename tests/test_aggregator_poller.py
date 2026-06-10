"""Tests for the aggregator poller: plan_bundles edge cases and MockTransport-driven I/O."""

import asyncio
import struct
import httpx
import iscc_core as ic
import pytest
from iscc_search.aggregator import tlog
from iscc_search.aggregator.hublist import Hub
from iscc_search.aggregator.poller import plan_bundles, poll_hub_once, poll_loop, run
from iscc_search.indexes.memory import MemoryIndex
from iscc_search.options import SearchOptions
from iscc_search.schema import IsccIndex


HUB = Hub(hub_id=0, url="https://sb0.iscc.id")
HUB2 = Hub(hub_id=1, url="https://sb1.amlet.id")
HUBS_YAML = """
version: 1
network: testnet
hubs:
    - {hub_id: 0, url: "https://sb0.iscc.id", active: true}
"""


def make_iscc_id(i, realm_id=0):
    # type: (int, int) -> str
    """Generate a distinct ISCC-ID for leaf index i."""
    return ic.gen_iscc_id(timestamp=1750000000000000 + i, hub_id=0, realm_id=realm_id)["iscc"]


class FakeLog:
    """In-memory tlog-tiles server backing an httpx.MockTransport handler."""

    def __init__(self, records=()):
        # type: (typing.Iterable[bytes]) -> None
        self.records = list(records)
        self.requests = []  # type: list[str]

    def handler(self, request):
        # type: (httpx.Request) -> httpx.Response
        """Serve /log/checkpoint and /log/tile/entries/* from current records."""
        path = request.url.path
        self.requests.append(f"{request.url.host}{path}")
        if path == "/log/checkpoint":
            return httpx.Response(200, text=f"sb0.iscc.id\n{len(self.records)}\ncm9vdA==\n")
        bundles = self.bundles()
        if path in bundles:
            return httpx.Response(200, content=bundles[path])
        return httpx.Response(404)

    def bundles(self):
        # type: () -> dict[str, bytes]
        """Frame the current records into entry-bundle bytes keyed by request path."""
        out = {}
        for index, width in plan_bundles(0, len(self.records)):
            start = index * tlog.TILE_WIDTH
            end = min(start + tlog.TILE_WIDTH, len(self.records))
            framed = b"".join(struct.pack(">H", len(r)) + r for r in self.records[start:end])
            out["/log/" + tlog.entries_path(index, width)] = framed
        return out


async def poll_once(handler, last_size, index, network="testnet", stop_set=False):
    # type: (typing.Callable, int, MemoryIndex, str, bool) -> PollResult
    """Run poll_hub_once against a MockTransport-backed client."""
    stop = asyncio.Event()
    if stop_set:
        stop.set()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        return await poll_hub_once(client, HUB, last_size, index, "idptest", network, stop)


@pytest.fixture
def agg_index():
    # type: () -> MemoryIndex
    """MemoryIndex with the idptest aggregator index pre-created."""
    index = MemoryIndex()
    index.create_index(IsccIndex(name="idptest"))
    return index


# --- plan_bundles (pure) ---


def test_plan_bundles_no_growth():
    """No growth and tree-size regression both yield an empty plan."""
    assert plan_bundles(0, 0) == []
    assert plan_bundles(5, 5) == []
    assert plan_bundles(10, 5) == []


def test_plan_bundles_first_run_partial():
    """A small first backfill requests one partial bundle."""
    assert plan_bundles(0, 3) == [(0, 3)]
    assert plan_bundles(0, 255) == [(0, 255)]


def test_plan_bundles_exact_boundary():
    """Exactly full bundles carry no partial width."""
    assert plan_bundles(0, 256) == [(0, 0)]
    assert plan_bundles(256, 512) == [(1, 0)]


def test_plan_bundles_multi_bundle():
    """Backfills spanning bundles request fulls plus the trailing partial."""
    assert plan_bundles(0, 600) == [(0, 0), (1, 0), (2, 88)]
    assert plan_bundles(0, 257) == [(0, 0), (1, 1)]


def test_plan_bundles_resume_mid_bundle():
    """Resuming inside a bundle re-requests that bundle (caller slices records)."""
    assert plan_bundles(100, 200) == [(0, 200)]
    assert plan_bundles(100, 300) == [(0, 0), (1, 44)]
    assert plan_bundles(256, 257) == [(1, 1)]


# --- poll_hub_once (MockTransport I/O against a real MemoryIndex) ---


def test_poll_hub_once_backfill(agg_index, make_log_record):
    """A first poll backfills all declarations with gateway metadata."""
    log = FakeLog([
        make_log_record(iscc_id=make_iscc_id(0), gateway="https://example.com/{iscc_id}"),
        make_log_record(iscc_id=make_iscc_id(1)),
        make_log_record(iscc_id=make_iscc_id(2)),
    ])
    result = asyncio.run(poll_once(log.handler, 0, agg_index))
    assert result.last_size == 3
    assert result.counts["ok"] == 3
    assert agg_index.get_index("idptest").assets == 3
    asset = agg_index.get_asset("idptest", make_iscc_id(0))
    assert asset.metadata == {"gateway": f"https://example.com/{make_iscc_id(0).removeprefix('ISCC:')}"}
    assert agg_index.get_asset("idptest", make_iscc_id(1)).metadata is None


def test_poll_hub_once_idempotent_repoll(agg_index, make_log_record):
    """Re-polling from 0 upserts without duplicates (at-least-once safety, C5)."""
    log = FakeLog([make_log_record(iscc_id=make_iscc_id(i)) for i in range(3)])
    first = asyncio.run(poll_once(log.handler, 0, agg_index))
    second = asyncio.run(poll_once(log.handler, 0, agg_index))
    assert first.counts["ok"] == second.counts["ok"] == 3
    assert agg_index.get_index("idptest").assets == 3


def test_poll_hub_once_no_growth(agg_index, make_log_record):
    """An unchanged checkpoint fetches no bundles and keeps the cursor."""
    log = FakeLog([make_log_record(iscc_id=make_iscc_id(i)) for i in range(3)])
    result = asyncio.run(poll_once(log.handler, 3, agg_index))
    assert result.last_size == 3
    assert sum(result.counts.values()) == 0
    assert log.requests == ["sb0.iscc.id/log/checkpoint"]


def test_poll_hub_once_multi_bundle(agg_index, make_log_record):
    """A backfill spanning bundles ingests full and partial bundles."""
    log = FakeLog([make_log_record(iscc_id=make_iscc_id(i)) for i in range(260)])
    result = asyncio.run(poll_once(log.handler, 0, agg_index))
    assert result.last_size == 260
    assert result.counts["ok"] == 260
    assert agg_index.get_index("idptest").assets == 260


def test_poll_hub_once_resume(agg_index, make_log_record):
    """Resuming from a cursor ingests only the new records."""
    records = [make_log_record(iscc_id=make_iscc_id(i)) for i in range(5)]
    log = FakeLog(records[:3])
    first = asyncio.run(poll_once(log.handler, 0, agg_index))
    assert first.last_size == 3
    log.records = records
    second = asyncio.run(poll_once(log.handler, 3, agg_index))
    assert second.last_size == 5
    assert second.counts["ok"] == 2
    assert agg_index.get_index("idptest").assets == 5


def test_poll_hub_once_skip_reasons(agg_index, make_log_record):
    """Deletions and unknown/malformed/realm-mismatched records are counted, not indexed."""
    log = FakeLog([
        make_log_record(iscc_id=make_iscc_id(0)),
        make_log_record(iscc_id=make_iscc_id(1), deletion=True),
        make_log_record(iscc_id=make_iscc_id(2), note_schema="http://purl.org/iscc/schema/iscc-note-9.9.9.json"),
        b"not json",
        make_log_record(iscc_id=make_iscc_id(4, realm_id=1)),
    ])
    result = asyncio.run(poll_once(log.handler, 0, agg_index))
    assert result.last_size == 5
    assert result.counts == {"ok": 1, "deletion": 1, "unknown_schema": 1, "malformed": 1, "realm_mismatch": 1}
    assert agg_index.get_index("idptest").assets == 1


def test_poll_hub_once_regression_resets_cursor(agg_index, make_log_record):
    """A checkpoint below the cursor logs a warning and resets to 0 (C10)."""
    log = FakeLog([make_log_record(iscc_id=make_iscc_id(i)) for i in range(3)])
    result = asyncio.run(poll_once(log.handler, 10, agg_index))
    assert result.last_size == 0
    assert sum(result.counts.values()) == 0


def test_poll_hub_once_stop_event_breaks(agg_index, make_log_record):
    """A pre-armed stop event skips bundle fetching for fast shutdown."""
    log = FakeLog([make_log_record(iscc_id=make_iscc_id(i)) for i in range(3)])
    result = asyncio.run(poll_once(log.handler, 0, agg_index, stop_set=True))
    assert result.last_size == 0
    assert log.requests == ["sb0.iscc.id/log/checkpoint"]


def test_poll_hub_once_http_error(agg_index):
    """An HTTP error on the checkpoint propagates to the caller (isolated per hub in poll_loop)."""
    handler = lambda request: httpx.Response(500)  # noqa: E731
    with pytest.raises(httpx.HTTPStatusError):
        asyncio.run(poll_once(handler, 0, agg_index))


def test_poll_hub_once_short_bundle_keeps_cursor(agg_index, make_log_record):
    """A frame-complete but short bundle fails the bundle instead of silently skipping records."""
    log = FakeLog([make_log_record(iscc_id=make_iscc_id(i)) for i in range(3)])

    def handler(request):
        # type: (httpx.Request) -> httpx.Response
        response = log.handler(request)
        if "tile/entries" in request.url.path:
            records = tlog.parse_entry_bundle(response.content)[:-1]  # drop the last record
            return httpx.Response(200, content=b"".join(struct.pack(">H", len(r)) + r for r in records))
        return response

    result = asyncio.run(poll_once(handler, 0, agg_index))
    assert result.last_size == 0
    assert agg_index.get_index("idptest").assets == 0


def test_poll_hub_once_bundle_failure_keeps_progress(agg_index, make_log_record):
    """A failing bundle stops the poll but commits the bundles ingested before it."""
    log = FakeLog([make_log_record(iscc_id=make_iscc_id(i)) for i in range(260)])

    def handler(request):
        # type: (httpx.Request) -> httpx.Response
        if request.url.path.endswith(".p/4"):  # fail the trailing partial bundle
            return httpx.Response(500)
        return log.handler(request)

    result = asyncio.run(poll_once(handler, 0, agg_index))
    assert result.last_size == 256
    assert result.counts["ok"] == 256
    assert agg_index.get_index("idptest").assets == 256


# --- poll_loop / run ---


def make_opts(hub_list_source, poll_interval=1, refresh_interval=3600):
    # type: (str, int, int) -> SearchOptions
    """Aggregator-mode options pointing at the given hub-list source."""
    return SearchOptions(
        aggregator_network="testnet",
        aggregator_hub_list_url=hub_list_source,
        aggregator_poll_interval=poll_interval,
        aggregator_hub_refresh_interval=refresh_interval,
    )


async def run_loop_until(index, opts, handler, condition, timeout=5.0, extra_sleep=0.0):
    # type: (MemoryIndex, SearchOptions, typing.Callable, typing.Callable, float, float) -> None
    """Run poll_loop until condition() holds (polling), then stop it."""
    stop = asyncio.Event()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        task = asyncio.create_task(poll_loop(index, opts, stop, client))
        deadline = asyncio.get_event_loop().time() + timeout
        while not condition() and asyncio.get_event_loop().time() < deadline:
            await asyncio.sleep(0.02)
        if extra_sleep:
            await asyncio.sleep(extra_sleep)
        stop.set()
        await asyncio.wait_for(task, timeout=5.0)
    assert condition()


def test_poll_loop_ingests_and_stops(agg_index, make_log_record, tmp_path):
    """The loop loads the hub list, polls the hub, indexes records, and honors stop."""
    hub_file = tmp_path / "hubs.yaml"
    hub_file.write_text(HUBS_YAML, encoding="utf-8")
    log = FakeLog([make_log_record(iscc_id=make_iscc_id(i)) for i in range(3)])
    opts = make_opts(str(hub_file))
    asyncio.run(run_loop_until(agg_index, opts, log.handler, lambda: agg_index.get_index("idptest").assets == 3))


def test_poll_loop_refresh_interval_respected(agg_index, make_log_record):
    """Across two poll iterations the hub list is fetched only once."""
    log = FakeLog([make_log_record(iscc_id=make_iscc_id(i)) for i in range(2)])
    hub_list_requests = []

    def handler(request):
        # type: (httpx.Request) -> httpx.Response
        if request.url.host == "hubs.example.com":
            hub_list_requests.append(request.url.path)
            return httpx.Response(200, text=HUBS_YAML)
        return log.handler(request)

    opts = make_opts("https://hubs.example.com/testnet.yaml", poll_interval=1, refresh_interval=3600)
    checkpoints = lambda: sum("checkpoint" in r for r in log.requests)  # noqa: E731
    asyncio.run(run_loop_until(agg_index, opts, handler, lambda: checkpoints() >= 2))
    assert hub_list_requests == ["/testnet.yaml"]


def test_poll_loop_empty_hub_list(agg_index, tmp_path):
    """An empty hub list is a benign idle state (C1)."""
    hub_file = tmp_path / "hubs.yaml"
    hub_file.write_text("version: 1\nnetwork: mainnet\nhubs:\n", encoding="utf-8")
    opts = SearchOptions(aggregator_network="mainnet", aggregator_hub_list_url=str(hub_file))

    async def main():
        stop = asyncio.Event()
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(500))) as client:
            task = asyncio.create_task(poll_loop(agg_index, opts, stop, client))
            await asyncio.sleep(0.05)
            stop.set()
            await asyncio.wait_for(task, timeout=5.0)

    asyncio.run(main())


def test_poll_loop_refresh_failure_keeps_previous_list(agg_index, make_log_record):
    """A failed refresh keeps the last-known-good hub list and polling continues."""
    log = FakeLog([make_log_record(iscc_id=make_iscc_id(i)) for i in range(2)])
    hub_list_requests = []

    def handler(request):
        # type: (httpx.Request) -> httpx.Response
        if request.url.host == "hubs.example.com":
            hub_list_requests.append(request.url.path)
            if len(hub_list_requests) == 1:
                return httpx.Response(200, text=HUBS_YAML)
            return httpx.Response(500)
        return log.handler(request)

    opts = make_opts("https://hubs.example.com/testnet.yaml", poll_interval=1, refresh_interval=1)
    condition = lambda: len(hub_list_requests) >= 2 and sum("checkpoint" in r for r in log.requests) >= 2  # noqa: E731
    asyncio.run(run_loop_until(agg_index, opts, handler, condition, timeout=10.0))


def test_poll_loop_refresh_failure_retries_quickly(agg_index, make_log_record):
    """A failed initial hub-list fetch is retried after the poll interval, not the refresh interval."""
    log = FakeLog([make_log_record(iscc_id=make_iscc_id(i)) for i in range(2)])
    hub_list_requests = []

    def handler(request):
        # type: (httpx.Request) -> httpx.Response
        if request.url.host == "hubs.example.com":
            hub_list_requests.append(request.url.path)
            if len(hub_list_requests) == 1:
                return httpx.Response(500)
            return httpx.Response(200, text=HUBS_YAML)
        return log.handler(request)

    opts = make_opts("https://hubs.example.com/testnet.yaml", poll_interval=1, refresh_interval=3600)
    condition = lambda: agg_index.get_index("idptest").assets == 2  # noqa: E731
    asyncio.run(run_loop_until(agg_index, opts, handler, condition, timeout=10.0))
    assert len(hub_list_requests) >= 2


def test_poll_loop_hub_failure_isolated(agg_index, make_log_record, tmp_path):
    """One unreachable hub does not prevent ingestion from the other."""
    hub_file = tmp_path / "hubs.yaml"
    hub_file.write_text(
        "version: 1\nnetwork: testnet\nhubs:\n"
        '    - {hub_id: 0, url: "https://sb0.iscc.id", active: true}\n'
        '    - {hub_id: 1, url: "https://sb1.amlet.id", active: true}\n',
        encoding="utf-8",
    )
    log = FakeLog([make_log_record(iscc_id=make_iscc_id(i)) for i in range(2)])

    def handler(request):
        # type: (httpx.Request) -> httpx.Response
        if request.url.host == "sb0.iscc.id":
            raise httpx.ConnectError("connection refused", request=request)
        return log.handler(request)

    opts = make_opts(str(hub_file))
    asyncio.run(run_loop_until(agg_index, opts, handler, lambda: agg_index.get_index("idptest").assets == 2))


def test_poll_loop_stops_between_hubs(agg_index, make_log_record, tmp_path):
    """A stop request during one hub's poll skips the remaining hubs."""
    hub_file = tmp_path / "hubs.yaml"
    hub_file.write_text(
        "version: 1\nnetwork: testnet\nhubs:\n"
        '    - {hub_id: 0, url: "https://sb0.iscc.id", active: true}\n'
        '    - {hub_id: 1, url: "https://sb1.amlet.id", active: true}\n',
        encoding="utf-8",
    )
    log = FakeLog([make_log_record(iscc_id=make_iscc_id(0))])
    opts = make_opts(str(hub_file))
    seen_hosts = []

    async def main():
        stop = asyncio.Event()

        def handler(request):
            # type: (httpx.Request) -> httpx.Response
            seen_hosts.append(request.url.host)
            stop.set()  # request shutdown while the first hub is being polled
            return log.handler(request)

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            await asyncio.wait_for(poll_loop(agg_index, opts, stop, client), timeout=5.0)

    asyncio.run(main())
    assert seen_hosts == ["sb0.iscc.id"]


def test_run_smoke(agg_index, tmp_path):
    """run() owns a real AsyncClient and exits promptly on a pre-armed stop event."""
    hub_file = tmp_path / "hubs.yaml"
    hub_file.write_text(HUBS_YAML, encoding="utf-8")
    opts = make_opts(str(hub_file))

    async def main():
        stop = asyncio.Event()
        stop.set()
        await asyncio.wait_for(run(agg_index, opts, stop), timeout=5.0)

    asyncio.run(main())
