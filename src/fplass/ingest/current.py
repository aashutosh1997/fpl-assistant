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


LIVE_STAT_COLUMNS = (
    "minutes",
    "goals_scored",
    "assists",
    "clean_sheets",
    "goals_conceded",
    "own_goals",
    "penalties_saved",
    "penalties_missed",
    "yellow_cards",
    "red_cards",
    "saves",
    "bonus",
    "bps",
    "influence",
    "creativity",
    "threat",
    "ict_index",
    "expected_goals",
    "expected_assists",
    "expected_goal_involvements",
    "expected_goals_conceded",
    "tackles",
    "recoveries",
    "clearances_blocks_interceptions",
    "defensive_contribution",
    "starts",
    "total_points",
)


def ingest_live_gameweek(
    con, gameweek: int, client: FPLClient | None = None, season: str = CURRENT_SEASON
) -> int:
    """Load a played gameweek's results from the live endpoint into ``player_gw``.

    The community historical dataset lags by a day or more, and the calibration layer needs
    outcomes as soon as they exist. The live endpoint carries the same per-player stats keyed by
    element, and its ``explain`` block gives the fixture each set of stats belongs to — which is
    what makes double gameweeks resolve into two rows rather than one merged one.

    Safe to re-run: bonus and defensive contributions are revised for a day or two after the final
    whistle, and the upsert replaces rather than duplicates.
    """
    owns_client = client is None
    client = client or FPLClient()
    try:
        live = client.live(gameweek)
        fixtures = client.fixtures(event=gameweek, ttl=0)
        # Refresh the player reference first. FPL adds players mid-season (loan returns, late
        # registrations), so the live feed routinely carries elements the warehouse has never
        # seen — which would otherwise land as rows with no position or club.
        bootstrap = client.bootstrap(ttl=0)
    finally:
        if owns_client:
            client.close()

    upsert(con, "teams", _teams_frame(bootstrap, season), ("season", "team_id"))
    upsert(con, "players", _players_frame(bootstrap, season), ("season", "element"))
    # The scores too: team strength and clean-sheet history read them from the fixtures table,
    # and without this a played gameweek stayed unscored there until a full `ingest current`.
    upsert(con, "fixtures", _fixtures_frame(fixtures, season), ("season", "fixture_id"))

    fixture_sides: dict[int, dict[int, bool]] = {}
    fixture_opponent: dict[int, dict[int, int]] = {}
    kickoffs: dict[int, str] = {}
    for fixture in fixtures:
        fid = fixture["id"]
        kickoffs[fid] = fixture.get("kickoff_time")
        fixture_sides[fid] = {fixture["team_h"]: True, fixture["team_a"]: False}
        fixture_opponent[fid] = {
            fixture["team_h"]: fixture["team_a"],
            fixture["team_a"]: fixture["team_h"],
        }

    teams = dict(
        con.execute(
            "SELECT element, team_id FROM players WHERE season = ?", [season]
        ).fetchall()
    )
    positions = dict(
        con.execute(
            """
            SELECT element, CASE element_type WHEN 1 THEN 'GKP' WHEN 2 THEN 'DEF'
                                              WHEN 3 THEN 'MID' ELSE 'FWD' END
            FROM players WHERE season = ?
            """,
            [season],
        ).fetchall()
    )
    names = dict(
        con.execute("SELECT element, web_name FROM players WHERE season = ?", [season]).fetchall()
    )

    records: list[dict[str, Any]] = []
    for element in live.get("elements", []):
        eid = element["id"]
        stats = element.get("stats", {})
        explains = element.get("explain") or []
        # One row per fixture the player featured in; fall back to a single row when the live feed
        # gives no explain block (which happens for players who did not appear).
        fixture_ids = [x.get("fixture") for x in explains if x.get("fixture")]
        if not fixture_ids:
            continue

        team = teams.get(eid)
        for fid in fixture_ids:
            row: dict[str, Any] = {
                "season": season,
                "element": eid,
                "fixture_id": fid,
                "gw": gameweek,
                "web_name": names.get(eid),
                "position": positions.get(eid),
                "was_home": fixture_sides.get(fid, {}).get(team),
                "opponent_team": fixture_opponent.get(fid, {}).get(team),
                "kickoff_time": kickoffs.get(fid),
            }
            for column in LIVE_STAT_COLUMNS:
                row[column] = stats.get(column)
            records.append(row)

    if not records:
        log.warning("no live data for GW%d", gameweek)
        return 0

    frame = pd.DataFrame.from_records(records)
    frame["kickoff_time"] = pd.to_datetime(
        frame["kickoff_time"], errors="coerce", utc=True
    ).dt.tz_localize(None)
    for column in LIVE_STAT_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame = _keep_market_columns(con, frame, season)
    written = upsert(con, "player_gw", frame, ("season", "element", "fixture_id"))
    create_as_of_view(con)
    log.info("ingested %d player rows for %s GW%d", written, season, gameweek)
    return written


# The market's columns: price, owners and the week's transfers. The live feed has none of them,
# and they are what the order-flow layer and the price tiers read. See ingest_market_history.
MARKET_COLUMNS = ("value", "selected", "transfers_in", "transfers_out", "transfers_balance")


def _keep_market_columns(con, frame: pd.DataFrame, season: str) -> pd.DataFrame:
    """Carry already-ingested market columns onto rows about to be re-upserted.

    ``upsert`` replaces whole rows, so re-ingesting a gameweek as its bonus settles would
    otherwise blank the market history filled in by :func:`ingest_market_history`.
    """
    existing = con.execute(
        f"""
        SELECT element, fixture_id, {", ".join(MARKET_COLUMNS)}
        FROM player_gw WHERE season = ?
        """,
        [season],
    ).fetchdf()
    if existing.empty:
        return frame
    frame = frame.drop(columns=[c for c in MARKET_COLUMNS if c in frame.columns])
    return frame.merge(existing, on=["element", "fixture_id"], how="left")


def _market_frame(summaries: dict[int, dict[str, Any]]) -> pd.DataFrame:
    """One row per (element, fixture) from each player's ``element-summary`` history."""
    records = [
        {
            "element": element,
            "fixture_id": row.get("fixture"),
            **{c: row.get(c) for c in MARKET_COLUMNS},
        }
        for element, summary in summaries.items()
        for row in summary.get("history", [])
        if row.get("fixture") is not None
    ]
    frame = pd.DataFrame.from_records(
        records, columns=["element", "fixture_id", *MARKET_COLUMNS]
    )
    for column in ("element", "fixture_id", *MARKET_COLUMNS):
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("Int64")
    return frame


def apply_market_history(con, frame: pd.DataFrame, season: str) -> int:
    """Write market columns onto existing ``player_gw`` rows, leaving every other column alone.

    An UPDATE rather than :func:`upsert`: the upsert deletes each matching row and inserts only
    the staged columns, which would blank the match statistics this frame does not carry.
    """
    if frame.empty:
        return 0
    con.register("_market", frame)
    try:
        assignments = ", ".join(f"{c} = m.{c}" for c in MARKET_COLUMNS)
        updated = con.execute(
            f"""
            UPDATE player_gw SET {assignments}
            FROM _market m
            WHERE player_gw.season = ? AND player_gw.element = m.element
              AND player_gw.fixture_id = m.fixture_id
            """,
            [season],
        ).fetchone()[0]
    finally:
        con.unregister("_market")
    create_as_of_view(con)
    return int(updated)


def ingest_market_history(
    con, client: FPLClient | None = None, season: str = CURRENT_SEASON, *, workers: int = 4
) -> int:
    """Fill this season's price, ownership and transfer columns from each player's history.

    The live endpoint records what players did, not what managers did, so for the season being
    played ``player_gw`` had no ``value``, ``selected`` or ``transfers_in``/``_out``. Without
    them a replay of this season has no order-flow layer (``flow.historical_flow`` reads them)
    and no deadline prices, and the current season's price tiers fall back to the starting
    price. ``element-summary/{id}`` carries all five per fixture: one call a player.
    """
    from concurrent.futures import ThreadPoolExecutor

    elements = [
        int(r[0])
        for r in con.execute(
            "SELECT DISTINCT element FROM player_gw WHERE season = ? ORDER BY element", [season]
        ).fetchall()
    ]
    owns_client = client is None
    client = client or FPLClient()
    try:
        # httpx clients are safe to share across threads; a few at once keeps this to a minute.
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            summaries = dict(
                zip(elements, pool.map(client.element_summary, elements), strict=True)
            )
    finally:
        if owns_client:
            client.close()
    updated = apply_market_history(con, _market_frame(summaries), season)
    log.info("market history: %d player_gw rows updated for %s", updated, season)
    return updated


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
