"""How the arXiv tools pace their requests (``ArxivSearchSource``).

arXiv asks for one request every three seconds, on one connection at a time,
and answers HTTP 429 (with ``Retry-After``) when a client goes faster. Every
request the source makes (a search, a paper's metadata, a PDF) waits its
turn behind the others, a refused one is retried after the wait arXiv asks
for, and the calls queued behind it wait too rather than each meeting the
same refusal. A search runs on the event loop without stalling it, and a
cancelled one frees arXiv for the next at once.

``tests/tools/arxiv_fakes.py`` serves the API; the waits are shortened.
"""

from __future__ import annotations

import asyncio
import email.utils
import inspect
import threading
import time
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from agentic_cli.tools import arxiv_source
from agentic_cli.tools.arxiv_source import ArxivSearchSource
from tests.tools.arxiv_fakes import ArxivAPI, Paper, atom_feed

#: The shortened gap between two requests' starts.
GAP = 0.05
#: A margin for the timer's resolution.
SLACK = 0.01


@pytest.fixture(autouse=True)
def fast(monkeypatch):
    """Short waits: GAP between requests, retries after 0.02, 0.04, 0.08 s."""
    monkeypatch.setattr(ArxivSearchSource, "rate_limit", property(lambda self: GAP))
    monkeypatch.setattr(arxiv_source, "_RETRY_DELAYS", (0.02, 0.04, 0.08))


@pytest.fixture
def api(arxiv_api) -> ArxivAPI:
    """The fake ``arxiv_source_ctx``'s source talks to, holding two papers."""
    arxiv_api.papers[:] = [Paper("2401.00001", title="First"), Paper("2401.00002")]
    return arxiv_api


@pytest.fixture
def source(api) -> ArxivSearchSource:
    return ArxivSearchSource(transport=api.transport)


def _queries(api: ArxivAPI) -> list[str]:
    """Each request's search terms (or id list), in the order they were sent."""
    found = []
    for url in api.requests:
        query = httpx.URL(url).params
        found.append(query.get("search_query") or query.get("id_list"))
    return found


def _gaps(starts: list[float]) -> list[float]:
    return [b - a for a, b in zip(starts, starts[1:])]


async def test_requests_go_one_at_a_time_each_start_a_gap_after_the_last(api, source):
    """Three searches and a metadata fetch at once, each reply slower than
    the gap: arXiv sees one connection at a time."""
    api.delay = 2 * GAP

    await asyncio.gather(
        source.asearch("alpha"),
        source.asearch("beta"),
        source.asearch("gamma"),
        source.fetch_by_id("2401.00002"),
    )

    assert len(api.requests) == 4
    assert api.max_in_flight == 1
    assert min(_gaps(api.starts)) >= GAP - SLACK


async def test_a_refused_request_is_retried_after_the_wait_arxiv_asks_for(api, source):
    api.refuse(429, retry_after="1")

    results = await source.asearch("alpha")

    assert [r.title for r in results] == ["First", "A paper"]
    assert _queries(api) == ["all:alpha", "all:alpha"]
    assert _gaps(api.starts)[0] >= 1.0 - SLACK


async def test_without_retry_after_each_retry_waits_longer(api, source):
    api.refuse(429, times=2)

    await source.asearch("alpha")

    assert len(api.requests) == 3
    first, second = _gaps(api.starts)
    assert first >= 0.02 - SLACK
    assert second >= 0.04 - SLACK


async def test_calls_queued_behind_a_refusal_wait_for_its_retry(api, source):
    """The refused call keeps its turn while it waits, so the next call does
    not meet the same refusal."""
    api.refuse(503)

    await asyncio.gather(source.asearch("alpha"), source.asearch("beta"))

    assert _queries(api) == ["all:alpha", "all:alpha", "all:beta"]


@pytest.mark.parametrize("status", [429, 503])
async def test_once_the_retries_run_out_the_call_fails_and_the_next_fails_at_once(
    api, source, status
):
    api.refuse(status, times=4)

    with pytest.raises(arxiv_source.ArxivRequestError, match=f"rate limit.*HTTP {status}"):
        await source.asearch("alpha")
    assert len(api.requests) == 4

    with pytest.raises(arxiv_source.ArxivRequestError, match="try again in"):
        await source.asearch("beta")
    assert len(api.requests) == 4  # nothing sent


async def test_a_wait_longer_than_the_tool_can_give_fails_at_once_naming_it(api, source):
    api.refuse(429, retry_after="3600")

    with pytest.raises(arxiv_source.ArxivRequestError, match="3600"):
        await source.asearch("alpha")
    assert len(api.requests) == 1

    with pytest.raises(arxiv_source.ArxivRequestError, match="try again in 3600 s"):
        await source.asearch("beta")
    assert len(api.requests) == 1


async def test_a_wait_of_days_refuses_calls_for_an_hour_at_most(api, source):
    api.refuse(429, retry_after="900000")

    with pytest.raises(arxiv_source.ArxivRequestError):
        await source.asearch("alpha")

    with pytest.raises(arxiv_source.ArxivRequestError, match="try again in 3600 s"):
        await source.asearch("beta")


async def test_a_refusal_that_is_not_a_rate_limit_is_not_retried(api, source):
    api.refuse(403)

    with pytest.raises(arxiv_source.ArxivRequestError, match="HTTP 403"):
        await source.asearch("alpha")
    assert len(api.requests) == 1


async def test_a_request_arxiv_does_not_answer_in_time_fails_naming_the_timeout(api, source):
    api.fail(httpx.ReadTimeout("no answer"))

    with pytest.raises(arxiv_source.ArxivRequestError, match="did not answer within 30"):
        await source.asearch("alpha")


@pytest.mark.parametrize(
    "error", [httpx.ReadTimeout("no answer"), httpx.ConnectError("unreachable")], ids=type
)
async def test_calls_queued_behind_an_unanswered_request_fail_at_once(api, source, error):
    """Five calls queued behind an arXiv that does not answer would each
    wait out the timeout in turn: 150 s for the fifth."""
    api.fail(error)

    outcomes = await asyncio.gather(
        *(source.asearch(query) for query in ("alpha", "beta", "gamma")),
        return_exceptions=True,
    )

    assert len(api.requests) == 1
    assert all(isinstance(o, arxiv_source.ArxivRequestError) for o in outcomes)
    assert all("try again in" in str(o) for o in outcomes[1:])


async def test_each_of_two_parallel_searches_reports_its_own_outcome(api, source):
    """A failure belongs to its call: the shared last_error let one search
    report another's error, or success with no papers."""
    from agentic_cli.tools.arxiv_tools import _search_arxiv_with_source

    api.refuse(403)

    failed, found = await asyncio.gather(
        _search_arxiv_with_source(source, query="alpha"),
        _search_arxiv_with_source(source, query="beta"),
    )

    assert failed["success"] is False and "HTTP 403" in failed["error"]
    assert failed["query"] == "alpha"
    assert found["success"] is True and found["total_found"] == 2


async def test_search_arxiv_is_awaited_and_does_not_stall_the_event_loop(
    api, arxiv_source_ctx
):
    """ADK runs a synchronous tool on the event loop, so a search that waited
    and fetched there froze the UI (the spinner, Ctrl+C) for seconds."""
    from agentic_cli.tools import arxiv_tools, factories

    assert inspect.iscoroutinefunction(arxiv_tools.search_arxiv)
    bound = factories.make_arxiv_tools(arxiv_source_ctx)[0]
    assert bound.__name__ == "search_arxiv" and inspect.iscoroutinefunction(bound)

    api.delay = 0.3
    ticks: list[float] = []

    async def tick() -> None:
        while True:
            ticks.append(time.monotonic())
            await asyncio.sleep(0.01)

    ticker = asyncio.create_task(tick())
    await asyncio.sleep(0.02)  # the ticker runs first, so a block at the call's start shows too
    try:
        result = await arxiv_tools.search_arxiv("alpha")
    finally:
        ticker.cancel()

    assert result["success"] is True
    # A search that waited on the loop stalls it for the whole delay; a busy
    # machine's own hiccups reach about a tenth of a second (0.12 s seen), so
    # the bound sits between the two.
    assert max(_gaps(ticks)) < api.delay / 2


async def test_a_cancelled_search_frees_arxiv_for_the_next_at_once(api, source):
    """Ctrl+C cancels the turn: its search stops waiting, and the next turn's
    search is not held behind it."""
    api.delay = 5.0
    stuck = asyncio.create_task(source.asearch("alpha"))
    await asyncio.sleep(GAP)
    stuck.cancel()
    with pytest.raises(asyncio.CancelledError):
        await stuck

    api.delay = 0.0
    started = time.monotonic()
    await source.asearch("beta")

    assert time.monotonic() - started < 2 * GAP + 0.5


async def test_a_pdf_download_waits_its_turn_too(api, source, monkeypatch):
    """arXiv's policy covers arxiv.org's PDFs too: a download and a search
    never overlap."""
    from agentic_cli.tools import webfetch_tool
    from agentic_cli.tools.webfetch.fetcher import FetchResult

    spans: list[tuple[str, float, float]] = []

    async def slow_fetch(url, **kwargs):
        start = time.monotonic()
        await asyncio.sleep(2 * GAP)
        spans.append(("pdf", start, time.monotonic()))
        return FetchResult(success=True, content=b"%PDF-1.4", raw=b"%PDF-1.4")

    fetcher = MagicMock()
    fetcher.fetch = AsyncMock(side_effect=slow_fetch)
    monkeypatch.setattr(webfetch_tool, "get_or_create_fetcher", lambda: fetcher)
    api.delay = 2 * GAP

    await asyncio.gather(
        source.download_pdf("https://arxiv.org/pdf/2401.00001"),
        source.asearch("alpha"),
    )

    (_, pdf_start, pdf_end), search_start = spans[0], api.starts[0]
    search_end = search_start + 2 * GAP
    assert pdf_end <= search_start + SLACK or search_end <= pdf_start + SLACK


async def test_a_reply_that_names_a_local_file_is_not_read(api, source, tmp_path):
    """feedparser opens a string that names a file: a reply (from a proxy in
    the middle, say) must not make the source read one of the user's files."""
    local = tmp_path / "local.xml"
    local.write_text(atom_feed([Paper("9999.99999", title="FROM LOCAL FILE")]))
    api.garble(body=str(local))

    with pytest.raises(arxiv_source.ArxivRequestError, match="feed error"):
        await source.asearch("alpha")


async def test_a_feed_is_parsed_off_the_event_loop(api, source, monkeypatch):
    """A long feed takes feedparser a noticeable time (0.2 s for 2,000
    entries): not on the loop, where it stalls the UI."""
    import feedparser

    threads: list[int] = []
    parse = feedparser.parse

    def recording(*args, **kwargs):
        threads.append(threading.get_ident())
        return parse(*args, **kwargs)

    monkeypatch.setattr(feedparser, "parse", recording)

    await source.asearch("alpha")
    source.clear_cache()  # so the fetch asks arXiv
    await source.fetch_by_id("2401.00002")

    assert len(threads) == 2
    assert threading.get_ident() not in threads


async def test_requests_name_the_client(api, source):
    """arXiv asks an API client to identify itself."""
    await source.asearch("alpha")

    assert api.user_agents[0].startswith("agentic-cli")


async def test_the_synchronous_search_refuses_to_block_a_running_loop(source):
    with pytest.raises(RuntimeError, match="await asearch"):
        source.search("alpha")


def test_the_synchronous_search_still_works_outside_an_event_loop(api, source):
    """``SearchSource.search`` stays synchronous (examples/arxiv_demo.py)."""
    assert [r.title for r in source.search("alpha")] == ["First", "A paper"]
    assert source.last_error is None

    api.refuse(403)
    assert source.search("beta") == []
    assert "HTTP 403" in source.last_error


@pytest.mark.parametrize(
    "header, wait",
    [
        ("5", 5.0), ("0", 0.0), ("-3", 0.0), ("soon", None), (None, None),
        ("inf", None), ("1e400", None), ("nan", None),
    ],
)
def test_retry_after_is_read_in_seconds(header, wait):
    headers = {} if header is None else {"Retry-After": header}

    assert arxiv_source._retry_after(httpx.Response(429, headers=headers)) == wait


@pytest.mark.parametrize("zone", ["GMT", "-0000", "+0000"])
def test_retry_after_is_read_as_a_date(zone):
    """``-0000`` (no zone) is still UTC, not the machine's local time."""
    when = email.utils.formatdate(time.time() + 30, usegmt=True).replace("GMT", zone)

    wait = arxiv_source._retry_after(httpx.Response(429, headers={"Retry-After": when}))

    assert 25 <= wait <= 31
