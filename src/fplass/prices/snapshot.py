"""Append-only price snapshots.

FPL 2026/27 added an official price-change API on each element:

    price_change_percent       progress toward the next move, as a string percentage
    price_change_hourly_rate   rate that percentage is currently moving
    price_change_locked_until  timestamp before which the price cannot move
    price_change_calibrating   True while FPL's own model is still warming up
    price_change_projections   [{offset, projected_percent, likelihood}] for the next 3 days

The exact semantics are not documented, and every field reads 0 pre-season. So we log the raw
values verbatim, hourly, and let ``prices/calibrate.py`` learn what they mean by regressing
realised ``cost_change_event`` moves against what was showing beforehand. That means the value of
this module is entirely in *not missing hours* — an unlogged hour is calibration data that cannot
be recovered later.

We also snapshot availability (``status``, ``news``, ``chance_of_playing_next_round``) in the same
row. That is not incidental: the same time series is what lets ``prices/decide.py`` estimate the
option value of waiting for team news before committing a transfer.

One row per element per run, appended to ``data/snapshots/prices/YYYY-MM.csv``.
"""

from __future__ import annotations

import csv
import datetime as dt
import logging
from pathlib import Path
from typing import Any

from ..api import FPLClient
from ..paths import PRICE_SNAPSHOTS

log = logging.getLogger(__name__)

N_PROJECTIONS = 3

COLUMNS = [
    "ts",  # snapshot time, UTC ISO-8601
    "element",  # FPL element id (reassigned each season)
    "code",  # stable cross-season player code
    "web_name",
    "team",
    "element_type",
    "now_cost",
    "cost_change_event",
    "cost_change_start",
    "price_change_percent",
    "price_change_hourly_rate",
    "price_change_locked_until",
    "price_change_calibrating",
    *[f"proj{i}_percent" for i in range(N_PROJECTIONS)],
    *[f"proj{i}_likelihood" for i in range(N_PROJECTIONS)],
    "transfers_in_event",
    "transfers_out_event",
    "transfers_in",
    "transfers_out",
    "selected_by_percent",
    "status",
    "chance_of_playing_next_round",
    "news",
    "news_added",
    "form",
    "event_points",
    "total_points",
    "minutes",
]


def snapshot_path(when: dt.datetime, root: Path = PRICE_SNAPSHOTS) -> Path:
    """Monthly shard, so any single CSV stays small enough to diff and commit cheaply."""
    return root / f"{when:%Y-%m}.csv"


def _row(element: dict[str, Any], ts: str) -> dict[str, Any]:
    """Flatten one element into a snapshot row.

    The nested ``price_change_projections`` list is flattened by ``offset`` rather than by list
    position — FPL is free to reorder or omit entries, and silently shifting day-2's projection
    into day-0's column would poison the calibration regression in a way that is very hard to
    notice after the fact.
    """
    row: dict[str, Any] = {
        "ts": ts,
        "element": element.get("id"),
        "code": element.get("code"),
        "web_name": element.get("web_name"),
        "team": element.get("team"),
        "element_type": element.get("element_type"),
        "now_cost": element.get("now_cost"),
        "cost_change_event": element.get("cost_change_event"),
        "cost_change_start": element.get("cost_change_start"),
        "price_change_percent": element.get("price_change_percent"),
        "price_change_hourly_rate": element.get("price_change_hourly_rate"),
        "price_change_locked_until": element.get("price_change_locked_until"),
        "price_change_calibrating": element.get("price_change_calibrating"),
        "transfers_in_event": element.get("transfers_in_event"),
        "transfers_out_event": element.get("transfers_out_event"),
        "transfers_in": element.get("transfers_in"),
        "transfers_out": element.get("transfers_out"),
        "selected_by_percent": element.get("selected_by_percent"),
        "status": element.get("status"),
        "chance_of_playing_next_round": element.get("chance_of_playing_next_round"),
        # Newlines in news would break the CSV row; collapse them.
        "news": (element.get("news") or "").replace("\n", " ").strip(),
        "news_added": element.get("news_added"),
        "form": element.get("form"),
        "event_points": element.get("event_points"),
        "total_points": element.get("total_points"),
        "minutes": element.get("minutes"),
    }

    for i in range(N_PROJECTIONS):
        row[f"proj{i}_percent"] = None
        row[f"proj{i}_likelihood"] = None
    for proj in element.get("price_change_projections") or []:
        offset = proj.get("offset")
        if isinstance(offset, int) and 0 <= offset < N_PROJECTIONS:
            row[f"proj{offset}_percent"] = proj.get("projected_percent")
            row[f"proj{offset}_likelihood"] = proj.get("likelihood")

    return row


def take_snapshot(
    client: FPLClient | None = None,
    *,
    root: Path = PRICE_SNAPSHOTS,
    now: dt.datetime | None = None,
) -> tuple[Path, int]:
    """Fetch a fresh bootstrap and append one row per element.

    Returns:
        The CSV written to, and the number of rows appended.
    """
    owns_client = client is None
    client = client or FPLClient()
    try:
        # ttl=0 forces a live read: a cached bootstrap would silently duplicate the previous
        # hour's price fields, which is worse than a missing snapshot because it looks valid.
        bootstrap = client.bootstrap(ttl=0)
    finally:
        if owns_client:
            client.close()

    ts_dt = now or dt.datetime.now(dt.UTC)
    ts = ts_dt.replace(microsecond=0).isoformat()

    elements = bootstrap.get("elements", [])
    rows = [_row(e, ts) for e in elements]

    path = snapshot_path(ts_dt, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists() or path.stat().st_size == 0

    with path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS, extrasaction="ignore")
        if is_new:
            writer.writeheader()
        writer.writerows(rows)

    log.info("snapshot %s: %d rows -> %s", ts, len(rows), path)
    return path, len(rows)


def snapshot_metadata(bootstrap: dict[str, Any]) -> dict[str, Any]:
    """Pull the game-level price context that is not per-element.

    ``price_change_deadlines`` gives the exact timestamps at which prices move (00:00 UK). The
    calibration step needs these to know which snapshot was the last one *before* each move.
    """
    settings = bootstrap.get("game_config", {}).get("settings", {})
    events = bootstrap.get("events", [])
    current = next((e["id"] for e in events if e.get("is_current")), None)
    nxt = next((e["id"] for e in events if e.get("is_next")), None)
    return {
        "price_change_deadlines": settings.get("price_change_deadlines", []),
        "current_event": current,
        "next_event": nxt,
        "total_players": bootstrap.get("total_players"),
    }
