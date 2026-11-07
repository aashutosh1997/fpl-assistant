"""Collect the decisions of the managers at the top: who they are, what they bought and sold.

Everything comes from public endpoints. The overall league's standings say who is where now;
each manager's history gives their rank after every gameweek, the chips they played and how
they finished in past seasons; their transfers list every sale and purchase with its price; and
their picks give the squad, captain and bench at a gameweek.

Two populations are read, because the obvious one is biased. Today's top 10,000 are the
managers whose decisions worked: judging transfers by them flatters the transfers. Managers who
did well *last* season were chosen without this season's results, but they are spread through
the rankings — about 7% of today's top 10,000, and about 1% of ranks 10,000 to 250,000. So the
crawl reads the top 10,000 in full and every ``band_every``-th standings page below it to
``band_to``, reads every one of those managers' histories, and then fetches decisions for the
band managers who either finished last season well or sat inside the top 10,000 at some
deadline this season. A band manager stands for ``band_every`` managers.

Requests are spread over a few threads under one shared rate limit. Results are written in
batches from the main thread, and every stage skips what it has already fetched for the
current gameweek, so an interrupted crawl resumes rather than restarts.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from ..api import FPLAPIError, FPLClient
from ..ingest.sources import CURRENT_SEASON
from ..ingest.warehouse import upsert
from .store import OVERALL_LEAGUE, connect_managers

log = logging.getLogger(__name__)

PAGE_SIZE = 50  # entries per classic-league standings page
TRANSFER_CAP = 500  # records entry/{id}/transfers/ returns at most


@dataclass(slots=True)
class CrawlConfig:
    season: str = CURRENT_SEASON
    top: int = 10_000
    band: bool = True
    band_to: int = 250_000
    band_every: int = 4
    leagues: tuple[int, ...] = ()
    me: int | None = None
    # Band managers whose decisions are fetched: last season's rank inside ``prior_rank``, or
    # this season's rank inside ``deadline_rank`` after some finished gameweek.
    prior_rank: int = 50_000
    deadline_rank: int = 10_000
    rate: float = 10.0
    workers: int = 8
    batch: int = 1_000
    stages: tuple[str, ...] = ("standings", "history", "transfers", "picks")
    # Weekly refresh: read the top pages again but not the band, and refresh histories only for
    # managers whose decisions are studied (or who are new). A band manager who climbs into the
    # top 10,000 is found on the top pages.
    incremental: bool = False


class RateLimiter:
    """At most ``per_second`` calls a second across every thread that shares it."""

    def __init__(self, per_second: float) -> None:
        self.interval = 1.0 / per_second
        self._lock = threading.Lock()
        self._next = time.monotonic()

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            self._next = max(self._next, now)
            delay = self._next - now
            self._next += self.interval
        if delay > 0:
            time.sleep(delay)


@dataclass(slots=True)
class Fetcher:
    """Threaded, rate-limited calls; results come back to the caller's thread in batches."""

    client: FPLClient
    limiter: RateLimiter
    workers: int
    errors: dict[str, int] = field(default_factory=dict)

    def run(
        self, call: Callable[[Any], Any], items: Iterable[Any], *, batch: int, label: str
    ) -> Iterator[list[tuple[Any, Any]]]:
        """Yield lists of ``(item, result)``; a failed call gives ``None`` and is counted."""
        items = list(items)
        if not items:
            return

        def guarded(item: Any) -> tuple[Any, Any]:
            self.limiter.wait()
            try:
                return item, call(item)
            except FPLAPIError as exc:
                key = f"{label}: {str(exc).split('->')[-1].strip()}"
                self.errors[key] = self.errors.get(key, 0) + 1
                log.debug("%s %s failed: %s", label, item, exc)
                return item, None

        started = time.time()
        done = 0
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            for start in range(0, len(items), batch):
                chunk = items[start : start + batch]
                results = list(pool.map(guarded, chunk))
                done += len(chunk)
                elapsed = time.time() - started
                remaining = (len(items) - done) * elapsed / done
                log.info(
                    "%s: %d/%d (%.1f/s, %d min left)",
                    label, done, len(items), done / elapsed, remaining / 60,
                )
                yield results


# ------------------------------------------------------------------------------- parsing


def previous_season_name(season: str) -> str:
    """'2026-27' -> '2025/26', the form entry history's ``past`` uses."""
    start = int(season[:4]) - 1
    return f"{start}/{str(start + 1)[-2:]}"


def parse_history(entry: int, payload: dict, season: str) -> tuple[dict, pd.DataFrame]:
    """The entry's past-season summary and its per-gameweek rows, chips attached."""
    past = payload.get("past") or []
    last = previous_season_name(season)
    last_rank = next((p.get("rank") for p in past if p.get("season_name") == last), None)
    ranks = [p.get("rank") for p in past if p.get("rank")]
    chips = {c.get("event"): c.get("name") for c in payload.get("chips") or []}
    rows = [
        {
            "season": season,
            "entry": entry,
            "gw": r["event"],
            "points": r.get("points"),
            "total_points": r.get("total_points"),
            "overall_rank": r.get("overall_rank"),
            "bank": r.get("bank"),
            "value": r.get("value"),
            "event_transfers": r.get("event_transfers"),
            "event_transfers_cost": r.get("event_transfers_cost"),
            "points_on_bench": r.get("points_on_bench"),
            "chip": chips.get(r["event"]),
        }
        for r in payload.get("current") or []
    ]
    summary = {
        "entry": entry,
        "started_gw": min((r["gw"] for r in rows), default=None),
        "past_seasons": len(past),
        "past_rank_last": last_rank,
        "past_top10k": sum(r <= 10_000 for r in ranks),
        "past_top100k": sum(r <= 100_000 for r in ranks),
    }
    return summary, pd.DataFrame.from_records(rows)


def parse_transfers(entry: int, payload: list[dict], season: str) -> pd.DataFrame:
    frame = pd.DataFrame.from_records(
        [
            {
                "season": season,
                "entry": entry,
                "gw": t.get("event"),
                "element_in": t.get("element_in"),
                "element_in_cost": t.get("element_in_cost"),
                "element_out": t.get("element_out"),
                "element_out_cost": t.get("element_out_cost"),
                "time": t.get("time"),
            }
            for t in payload or []
        ],
        columns=[
            "season", "entry", "gw", "element_in", "element_in_cost", "element_out",
            "element_out_cost", "time",
        ],
    )
    # ISO8601 explicitly: the feed mixes whole and fractional seconds, and pandas otherwise
    # infers one format from the first value and turns the rest into NaT.
    frame["time"] = pd.to_datetime(frame["time"], utc=True, format="ISO8601", errors="coerce").dt.tz_localize(None)
    bad = frame["time"].isna() | frame["gw"].isna() | frame["element_in"].isna() | frame["element_out"].isna()
    if bad.any():
        log.warning("transfers: dropped %d unreadable records for entry %d", int(bad.sum()), entry)
        frame = frame[~bad].reset_index(drop=True)
    return frame


def parse_picks(entry: int, gw: int, payload: dict, season: str) -> pd.DataFrame:
    return pd.DataFrame.from_records(
        [
            {
                "season": season,
                "entry": entry,
                "gw": gw,
                "element": p.get("element"),
                "slot": p.get("position"),
                "multiplier": p.get("multiplier"),
                "is_captain": p.get("is_captain"),
                "is_vice_captain": p.get("is_vice_captain"),
            }
            for p in payload.get("picks") or []
        ]
    )


# -------------------------------------------------------------------------------- stages


def gameweeks_now(client: FPLClient) -> tuple[int, int]:
    """(last finished gameweek, last gameweek whose deadline has passed).

    Ranks and points are final only for the first; transfers and picks are fixed, and public,
    from the second's deadline. Between a deadline and the final whistle they differ by one.
    """
    events = client.bootstrap(ttl=0)["events"]
    finished = max((e["id"] for e in events if e.get("finished")), default=0)
    started = max(
        (e["id"] for e in events if e.get("finished") or e.get("is_current")), default=finished
    )
    return finished, started


def _record_entries(con, frame: pd.DataFrame, season: str) -> None:
    """Insert newly seen managers and refresh the standing of known ones, keeping their data.

    A standings read decides ``source`` and ``weight`` for the managers on it; a league read
    never demotes a manager the standings found.
    """
    if frame.empty:
        return
    con.register("_seen", frame)
    try:
        con.execute(
            """
            INSERT INTO manager_entries (season, entry, source, weight, standing_rank, standing_gw)
            SELECT ?, entry, source, weight, standing_rank, standing_gw FROM _seen
            ON CONFLICT (season, entry) DO UPDATE SET
                source = CASE WHEN excluded.source IN ('top', 'band') THEN excluded.source
                              ELSE manager_entries.source END,
                weight = CASE WHEN excluded.source IN ('top', 'band') THEN excluded.weight
                              ELSE manager_entries.weight END,
                standing_rank = COALESCE(excluded.standing_rank, manager_entries.standing_rank),
                standing_gw = COALESCE(excluded.standing_gw, manager_entries.standing_gw)
            """,
            [season],
        )
    finally:
        con.unregister("_seen")


def crawl_standings(con, fetcher: Fetcher, config: CrawlConfig, gw: int) -> None:
    """Read the overall league's top pages and the sampled band below them."""
    top_pages = math.ceil(config.top / PAGE_SIZE)
    pages = [(p, "top", 1.0) for p in range(1, top_pages + 1)]
    if config.band and not config.incremental:
        last_page = math.ceil(config.band_to / PAGE_SIZE)
        pages += [
            (p, "band", float(config.band_every))
            for p in range(top_pages + 1, last_page + 1, config.band_every)
        ]
    done = {
        r[0]
        for r in con.execute(
            "SELECT page FROM crawl_pages WHERE season = ? AND league = ? AND gw = ?",
            [config.season, OVERALL_LEAGUE, gw],
        ).fetchall()
    }
    todo = [p for p in pages if p[0] not in done]
    log.info("standings: %d of %d pages to read", len(todo), len(pages))

    def call(page: tuple[int, str, float]) -> dict:
        return fetcher.client.league_standings(OVERALL_LEAGUE, page[0])

    for results in fetcher.run(call, todo, batch=config.batch // 10 or 50, label="standings"):
        rows, read = [], []
        for (page, source, weight), payload in results:
            if payload is None:
                continue
            read.append(page)
            for r in payload.get("standings", {}).get("results", []):
                rows.append(
                    {"entry": r["entry"], "source": source, "weight": weight,
                     "standing_rank": r.get("rank"), "standing_gw": gw}
                )
        _record_entries(con, pd.DataFrame(rows), config.season)
        upsert(
            con,
            "crawl_pages",
            pd.DataFrame({"season": config.season, "league": OVERALL_LEAGUE, "page": read, "gw": gw}),
            ("season", "league", "page", "gw"),
        )


def crawl_leagues(con, fetcher: Fetcher, config: CrawlConfig) -> None:
    """Members of the mini-leagues being played, and the manager being advised."""
    members: list[dict] = []
    for league in config.leagues:
        page = 1
        while True:
            try:
                payload = fetcher.client.league_standings(league, page)
            except FPLAPIError as exc:
                # A mini-league is a comparison, not the study: lose it for this run, not the run.
                log.warning("league %d page %d unreadable, skipped: %s", league, page, exc)
                break
            standings = payload.get("standings", {})
            for r in standings.get("results", []):
                members.append({"league": league, "entry": r["entry"]})
            if not standings.get("has_next"):
                break
            page += 1
    if members:
        frame = pd.DataFrame(members)
        frame.insert(0, "season", config.season)
        upsert(con, "manager_leagues", frame, ("season", "league", "entry"))
        _record_entries(
            con,
            frame[["entry"]].drop_duplicates().assign(
                source="league", weight=1.0, standing_rank=None, standing_gw=None
            ),
            config.season,
        )
    if config.me:
        _record_entries(
            con,
            pd.DataFrame([{"entry": config.me, "source": "us", "weight": 1.0,
                           "standing_rank": None, "standing_gw": None}]),
            config.season,
        )


def crawl_histories(con, fetcher: Fetcher, config: CrawlConfig, gw: int) -> None:
    todo = [
        r[0]
        for r in con.execute(
            "SELECT entry FROM manager_entries WHERE season = ? "
            "AND (history_gw IS NULL OR history_gw < ?) ORDER BY entry",
            [config.season, gw],
        ).fetchall()
    ]
    if config.incremental:
        studied = set(decision_members(con, config))
        known = {
            r[0]
            for r in con.execute(
                "SELECT entry FROM manager_entries WHERE season = ? AND history_gw IS NOT NULL",
                [config.season],
            ).fetchall()
        }
        todo = [e for e in todo if e in studied or e not in known]
    log.info("history: %d managers to read", len(todo))
    for results in fetcher.run(fetcher.client.entry_history, todo, batch=config.batch, label="history"):
        summaries, frames = [], []
        for entry, payload in results:
            if payload is None:
                continue
            summary, rows = parse_history(entry, payload, config.season)
            summaries.append(summary)
            if not rows.empty:
                frames.append(rows)
        if frames:
            upsert(con, "manager_gw", pd.concat(frames, ignore_index=True), ("season", "entry"))
        if summaries:
            staged = pd.DataFrame(summaries)
            con.register("_summary", staged)
            try:
                con.execute(
                    """
                    UPDATE manager_entries SET
                        started_gw = s.started_gw, past_seasons = s.past_seasons,
                        past_rank_last = s.past_rank_last, past_top10k = s.past_top10k,
                        past_top100k = s.past_top100k, history_gw = ?
                    FROM _summary s
                    WHERE manager_entries.season = ? AND manager_entries.entry = s.entry
                    """,
                    [gw, config.season],
                )
            finally:
                con.unregister("_summary")


def decision_members(con, config: CrawlConfig) -> list[int]:
    """Managers whose transfers and picks are fetched.

    Everyone found on the top pages, in the leagues or advised; from the band, those who
    finished last season inside ``prior_rank`` or sat inside ``deadline_rank`` after some
    finished gameweek this season.
    """
    rows = con.execute(
        """
        SELECT e.entry FROM manager_entries e
        WHERE e.season = ? AND (
            e.source <> 'band'
            OR e.past_rank_last <= ?
            OR EXISTS (
                SELECT 1 FROM manager_gw g
                WHERE g.season = e.season AND g.entry = e.entry AND g.overall_rank <= ?
            )
        )
        ORDER BY e.entry
        """,
        [config.season, config.prior_rank, config.deadline_rank],
    ).fetchall()
    return [r[0] for r in rows]


def crawl_transfers(con, fetcher: Fetcher, config: CrawlConfig, gw: int, members: list[int]) -> None:
    fetched = {
        r[0]
        for r in con.execute(
            "SELECT entry FROM manager_entries WHERE season = ? AND transfers_gw >= ?",
            [config.season, gw],
        ).fetchall()
    }
    todo = [e for e in members if e not in fetched]
    log.info("transfers: %d managers to read", len(todo))
    for results in fetcher.run(fetcher.client.entry_transfers, todo, batch=config.batch, label="transfers"):
        frames, done = [], []
        for entry, payload in results:
            if payload is None:
                continue
            done.append(entry)
            frame = parse_transfers(entry, payload, config.season)
            if not frame.empty:
                frames.append(frame)
        if done:
            # Replace each manager's transfers wholesale: the list is complete on every read.
            con.execute(
                f"DELETE FROM manager_transfers WHERE season = ? AND entry IN ({','.join(map(str, done))})",
                [config.season],
            )
            if frames:
                staged = pd.concat(frames, ignore_index=True)
                con.register("_transfers", staged)
                try:
                    con.execute("INSERT OR REPLACE INTO manager_transfers SELECT * FROM _transfers")
                finally:
                    con.unregister("_transfers")
            con.execute(
                f"UPDATE manager_entries SET transfers_gw = ? WHERE season = ? "
                f"AND entry IN ({','.join(map(str, done))})",
                [gw, config.season],
            )


def crawl_picks(con, fetcher: Fetcher, config: CrawlConfig, gw: int, members: list[int]) -> None:
    """Each member's squad at their first gameweek and at the latest finished one.

    The first is the base the transfers are replayed on; the latest checks that replay and
    records the captain and bench. Run weekly, the latest accumulates into every gameweek.
    Managers whose transfer list hit the endpoint's cap get every gameweek.
    """
    started = dict(
        con.execute(
            "SELECT entry, started_gw FROM manager_entries WHERE season = ?", [config.season]
        ).fetchall()
    )
    have = {
        (r[0], r[1])
        for r in con.execute(
            "SELECT DISTINCT entry, gw FROM manager_picks WHERE season = ?", [config.season]
        ).fetchall()
    }
    # The transfers endpoint returns at most TRANSFER_CAP records, newest first; a manager at
    # the cap has lost their earliest transfers, so every gameweek's squad is read instead.
    capped = {
        r[0]
        for r in con.execute(
            "SELECT entry FROM manager_transfers WHERE season = ? GROUP BY entry "
            "HAVING count(*) >= ?",
            [config.season, TRANSFER_CAP],
        ).fetchall()
    }
    todo = []
    for entry in members:
        first = started.get(entry)
        if first is None:
            continue
        weeks = range(first, gw + 1) if entry in capped else sorted({first, gw})
        for g in weeks:
            if g <= gw and (entry, g) not in have:
                todo.append((entry, g))
    log.info("picks: %d squads to read", len(todo))

    def call(item: tuple[int, int]) -> dict:
        # Straight through the session: entry_picks caches forever, which the crawl does not want.
        return fetcher.client.get(f"entry/{item[0]}/event/{item[1]}/picks/", ttl=0)

    for results in fetcher.run(call, todo, batch=config.batch, label="picks"):
        frames = [
            parse_picks(entry, g, payload, config.season)
            for (entry, g), payload in results
            if payload is not None
        ]
        frames = [f for f in frames if not f.empty]
        if frames:
            upsert(con, "manager_picks", pd.concat(frames, ignore_index=True), ("season", "entry", "gw"))


def crawl(config: CrawlConfig) -> dict[str, object]:
    """Run the crawl's stages in order and report what is now stored."""
    client = FPLClient(cache_dir=None)
    client._http()  # create the shared session before the threads race to
    fetcher = Fetcher(client, RateLimiter(config.rate), config.workers)
    con = connect_managers()
    try:
        gw, current = gameweeks_now(client)
        log.info(
            "crawl for %s after GW%d (deadline of GW%d passed): stages %s",
            config.season, gw, current, ", ".join(config.stages),
        )
        if "standings" in config.stages:
            crawl_standings(con, fetcher, config, gw)
            crawl_leagues(con, fetcher, config)
        if "history" in config.stages:
            crawl_histories(con, fetcher, config, gw)
        members = decision_members(con, config)
        # Decisions are fixed at a deadline, so they are read up to the current gameweek.
        if "transfers" in config.stages:
            crawl_transfers(con, fetcher, config, current, members)
        if "picks" in config.stages:
            crawl_picks(con, fetcher, config, current, members)
        counts = {
            t: con.execute(f"SELECT count(*) FROM {t} WHERE season = ?", [config.season]).fetchone()[0]
            for t in ("manager_entries", "manager_gw", "manager_transfers", "manager_picks")
        }
        return {
            "finished_gw": gw, "current_gw": current, "members": len(members),
            "errors": dict(fetcher.errors), **counts,
        }
    finally:
        con.close()
        client.close()
