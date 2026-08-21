"""Refresh the current season from the live FPL API.

The historical dataset lags by a day or two and does not carry the things that change during a
gameweek — availability, prices, and the chip and scoring configuration. This module reads those
straight from ``bootstrap-static`` and ``fixtures``.

Two tables here exist specifically so that nothing downstream hardcodes rules that FPL changes
between seasons:

``chips``
    The legal gameweek window for each chip. For 2026/27 these are asymmetric — Bench Boost and
    Triple Captain are legal in GW1, Wildcard and Free Hit only from GW2 — and both sets repeat in
    the second half of the season. A planner that assumed all four chips share a window would
    quietly propose an illegal GW1 Wildcard.

``scoring_rules``
    Points per stat per position. Keeping this per-season means a backtest of 2019-20 is scored
    with 2019-20's rules rather than this season's, which matters because defensive contribution
    did not exist then and goalkeeper goals are worth 10 now.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from ..api import FPLClient
from .sources import CURRENT_SEASON
from .warehouse import connect, create_as_of_view, table_counts, upsert

log = logging.getLogger(__name__)

POSITION_BY_ELEMENT_TYPE = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}


def _teams_frame(bootstrap: dict[str, Any], season: str) -> pd.DataFrame:
    frame = pd.DataFrame(bootstrap["teams"])
    out = pd.DataFrame({"season": season}, index=frame.index)
    out["team_id"] = frame["id"].astype("Int64")
    for col in (
        "code",
        "name",
        "short_name",
        "strength",
        "strength_overall_home",
        "strength_overall_away",
        "strength_attack_home",
        "strength_attack_away",
        "strength_defence_home",
        "strength_defence_away",
    ):
        out[col] = frame[col] if col in frame.columns else None
    return out


def _players_frame(bootstrap: dict[str, Any], season: str) -> pd.DataFrame:
    frame = pd.DataFrame(bootstrap["elements"])
    return pd.DataFrame(
        {
            "season": season,
            "element": frame["id"].astype("Int64"),
            "code": frame["code"].astype("Int64"),
            "first_name": frame["first_name"],
            "second_name": frame["second_name"],
            "web_name": frame["web_name"],
            "team_id": frame["team"].astype("Int64"),
            "element_type": frame["element_type"].astype("Int64"),
            "start_cost": (frame["now_cost"] - frame["cost_change_start"]).astype("Int64"),
            "birth_date": pd.to_datetime(frame.get("birth_date"), errors="coerce").dt.date,
        }
    )


def _fixtures_frame(fixtures: list[dict], season: str) -> pd.DataFrame:
    frame = pd.DataFrame(fixtures)
    out = pd.DataFrame({"season": season}, index=frame.index)
    out["fixture_id"] = frame["id"].astype("Int64")
    out["code"] = frame["code"].astype("Int64")
    out["event"] = pd.to_numeric(frame["event"], errors="coerce").astype("Int64")
    out["kickoff_time"] = pd.to_datetime(
        frame["kickoff_time"], errors="coerce", utc=True
    ).dt.tz_localize(None)
    for col in ("team_h", "team_a", "team_h_score", "team_a_score", "minutes"):
        out[col] = pd.to_numeric(frame[col], errors="coerce").astype("Int64")
    out["team_h_difficulty"] = pd.to_numeric(
        frame["team_h_difficulty"], errors="coerce"
    ).astype("Int64")
    out["team_a_difficulty"] = pd.to_numeric(
        frame["team_a_difficulty"], errors="coerce"
    ).astype("Int64")
    out["finished"] = frame["finished"]
    return out


def _events_frame(bootstrap: dict[str, Any], season: str) -> pd.DataFrame:
    frame = pd.DataFrame(bootstrap["events"])
    out = pd.DataFrame({"season": season}, index=frame.index)
    out["event"] = frame["id"].astype("Int64")
    out["name"] = frame["name"]
    out["deadline_time"] = pd.to_datetime(
        frame["deadline_time"], errors="coerce", utc=True
    ).dt.tz_localize(None)
    out["finished"] = frame["finished"]
    out["data_checked"] = frame["data_checked"]
    for col in (
        "average_entry_score",
        "highest_score",
        "most_selected",
        "most_transferred_in",
        "most_captained",
        "transfers_made",
    ):
        out[col] = pd.to_numeric(frame.get(col), errors="coerce").astype("Int64")
    return out


def _chips_frame(bootstrap: dict[str, Any], season: str) -> pd.DataFrame:
    chips = bootstrap.get("chips") or []
    if not chips:
        return pd.DataFrame()
    frame = pd.DataFrame(chips)
    return pd.DataFrame(
        {
            "season": season,
            "chip_id": frame["id"].astype("Int64"),
            "name": frame["name"],
            "chip_type": frame.get("chip_type"),
            "start_event": pd.to_numeric(frame["start_event"], errors="coerce").astype("Int64"),
            "stop_event": pd.to_numeric(frame["stop_event"], errors="coerce").astype("Int64"),
        }
    )


def _scoring_frame(bootstrap: dict[str, Any], season: str) -> pd.DataFrame:
    """Flatten ``game_config.scoring`` into (stat, position, points) rows.

    Values are either a scalar (same for every position) or a dict keyed by position. Scalars are
    stored against position ``ALL`` so a lookup can fall back to it.
    """
    scoring = bootstrap.get("game_config", {}).get("scoring", {})
    records: list[dict[str, Any]] = []
    for stat, value in scoring.items():
        if isinstance(value, dict):
            for position, points in value.items():
                records.append(
                    {"season": season, "stat": stat, "position": position, "points": float(points)}
                )
        elif isinstance(value, (int, float)):
            records.append(
                {"season": season, "stat": stat, "position": "ALL", "points": float(value)}
            )
    return pd.DataFrame.from_records(records)


def refresh_current(
    client: FPLClient | None = None, season: str = CURRENT_SEASON
) -> dict[str, int]:
    """Refresh players, teams, fixtures, events, chips and scoring rules from the live API."""
    owns_client = client is None
    client = client or FPLClient()
    try:
        bootstrap = client.bootstrap(ttl=0)
        fixtures = client.fixtures(ttl=0)
    finally:
        if owns_client:
            client.close()

    con = connect()
    try:
        upsert(con, "teams", _teams_frame(bootstrap, season), ("season", "team_id"))
        upsert(con, "players", _players_frame(bootstrap, season), ("season", "element"))
        upsert(con, "fixtures", _fixtures_frame(fixtures, season), ("season", "fixture_id"))
        upsert(con, "events", _events_frame(bootstrap, season), ("season", "event"))
        upsert(con, "chips", _chips_frame(bootstrap, season), ("season", "chip_id"))
        upsert(
            con,
            "scoring_rules",
            _scoring_frame(bootstrap, season),
            ("season", "stat", "position"),
        )
        create_as_of_view(con)
        return table_counts(con)
    finally:
        con.close()


def chip_windows(con, season: str = CURRENT_SEASON) -> dict[str, list[tuple[int, int]]]:
    """Legal gameweek windows per chip name, e.g. ``{"bboost": [(1, 19), (20, 38)]}``.

    Read this rather than hardcoding. In 2026/27 ``wildcard`` and ``freehit`` start at GW2 while
    ``bboost`` and ``3xc`` start at GW1.
    """
    rows = con.execute(
        "SELECT name, start_event, stop_event FROM chips "
        "WHERE season = ? ORDER BY name, start_event",
        [season],
    ).fetchall()
    windows: dict[str, list[tuple[int, int]]] = {}
    for name, start, stop in rows:
        windows.setdefault(name, []).append((int(start), int(stop)))
    return windows


def double_and_blank_gameweeks(con, season: str = CURRENT_SEASON) -> dict[str, dict[int, list[int]]]:
    """Find teams with two fixtures (doubles) or none (blanks) in each gameweek.

    Recomputed on every call rather than cached, because doubles and blanks do not exist at the
    start of a season — they appear as cup progress forces postponements, and chip value depends
    almost entirely on them.
    """
    rows = con.execute(
        """
        WITH per_team AS (
            SELECT event, team_h AS team FROM fixtures WHERE season = ? AND event IS NOT NULL
            UNION ALL
            SELECT event, team_a AS team FROM fixtures WHERE season = ? AND event IS NOT NULL
        ),
        counts AS (
            SELECT event, team, count(*) AS n FROM per_team GROUP BY event, team
        ),
        events AS (SELECT DISTINCT event FROM per_team),
        teams AS (SELECT DISTINCT team_id AS team FROM teams WHERE season = ?)
        SELECT e.event, t.team, COALESCE(c.n, 0) AS n
        FROM events e CROSS JOIN teams t
        LEFT JOIN counts c ON c.event = e.event AND c.team = t.team
        WHERE COALESCE(c.n, 0) <> 1
        ORDER BY e.event, t.team
        """,
        [season, season, season],
    ).fetchall()

    doubles: dict[int, list[int]] = {}
    blanks: dict[int, list[int]] = {}
    for event, team, n in rows:
        target = blanks if n == 0 else doubles
        target.setdefault(int(event), []).append(int(team))
    return {"doubles": doubles, "blanks": blanks}
