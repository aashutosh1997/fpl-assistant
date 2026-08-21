"""Load historical seasons into the warehouse.

The upstream per-gameweek schema changed several times in eleven seasons, and the changes matter
for what can actually be trained on:

===============  ===================================================================
2016-17..2018-19 Rich Opta feed: clearances/blocks/interceptions, tackles, recoveries,
                 key passes, big chances. No ``position`` or ``team`` column, no xG.
2019-20          Minimal. No position, no team, no xG, no defensive counts.
2020-21..2021-22 ``position``, ``team`` and ``xP`` appear. Still no xG.
2022-23..2023-24 Expected goals/assists/involvements/conceded, and ``starts``.
2024-25          Assistant Manager chip columns (``mng_*``), since removed.
2025-26..        ``defensive_contribution`` and its components return.
===============  ===================================================================

The consequence worth being explicit about: **defensive-contribution component counts exist for
2016-19 and again from 2025-26, but not for the six seasons in between.** DEFCON points did not
exist before 2025-26, but the underlying CBIT/CBIRT *counts* from the Opta-era seasons are still
usable for fitting how often a given player type racks up defensive actions. That turns one season
of DEFCON-rate training data into four, which matters because DEFCON is worth 2 points a match to
roughly a third of all outfield players. The era is recorded per row so a model can down-weight
the older feed if its definitions turn out to drift.

Anything a season genuinely does not have is loaded as NULL rather than zero. Zero-filling here
would be silently wrong: it would tell the xG model that every player in 2019-20 had exactly zero
expected goals, rather than that the number is unknown.
"""

from __future__ import annotations

import ast
import logging
from typing import Any

import pandas as pd

from .sources import SEASONS, fetch_season
from .warehouse import (
    connect,
    create_as_of_view,
    drop_all,
    table_counts,
    upsert,
)

log = logging.getLogger(__name__)

POSITION_BY_ELEMENT_TYPE = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}

# The early seasons' files are not UTF-8 — accented player names were written in a Windows/Latin
# encoding. Try these in order rather than mangling names with errors="replace", because player
# names are how we reconcile against Understat.
ENCODINGS = ("utf-8", "cp1252", "latin-1")


def read_csv(path, **kwargs) -> pd.DataFrame:
    """Read a source CSV, tolerating the encoding drift across eleven seasons."""
    last: UnicodeDecodeError | None = None
    for encoding in ENCODINGS:
        try:
            return pd.read_csv(path, encoding=encoding, **kwargs)
        except UnicodeDecodeError as exc:
            last = exc
    raise UnicodeDecodeError(  # pragma: no cover - all three failing is not a real case
        "utf-8", b"", 0, 1, f"could not decode {path} with any of {ENCODINGS}"
    ) from last

# Warehouse column <- upstream column. Upstream names are stable where they exist at all, so a
# single mapping serves every season; absent columns become NULL.
PLAYER_GW_COLUMNS: dict[str, str] = {
    "element": "element",
    "fixture_id": "fixture",
    "gw": "GW",
    "web_name": "name",
    "position": "position",
    "team_name": "team",
    "opponent_team": "opponent_team",
    "was_home": "was_home",
    "kickoff_time": "kickoff_time",
    "minutes": "minutes",
    "starts": "starts",
    "total_points": "total_points",
    "goals_scored": "goals_scored",
    "assists": "assists",
    "clean_sheets": "clean_sheets",
    "goals_conceded": "goals_conceded",
    "own_goals": "own_goals",
    "penalties_saved": "penalties_saved",
    "penalties_missed": "penalties_missed",
    "yellow_cards": "yellow_cards",
    "red_cards": "red_cards",
    "saves": "saves",
    "bonus": "bonus",
    "bps": "bps",
    "influence": "influence",
    "creativity": "creativity",
    "threat": "threat",
    "ict_index": "ict_index",
    "expected_goals": "expected_goals",
    "expected_assists": "expected_assists",
    "expected_goal_involvements": "expected_goal_involvements",
    "expected_goals_conceded": "expected_goals_conceded",
    "tackles": "tackles",
    "recoveries": "recoveries",
    "clearances_blocks_interceptions": "clearances_blocks_interceptions",
    "defensive_contribution": "defensive_contribution",
    "team_h_score": "team_h_score",
    "team_a_score": "team_a_score",
    "value": "value",
    "selected": "selected",
    "transfers_in": "transfers_in",
    "transfers_out": "transfers_out",
    "transfers_balance": "transfers_balance",
    "xP": "xP",
}

INT_COLUMNS = (
    "element",
    "fixture_id",
    "gw",
    "opponent_team",
    "minutes",
    "starts",
    "total_points",
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
    "tackles",
    "recoveries",
    "clearances_blocks_interceptions",
    "defensive_contribution",
    "team_h_score",
    "team_a_score",
    "value",
    "selected",
    "transfers_in",
    "transfers_out",
    "transfers_balance",
)

FLOAT_COLUMNS = (
    "influence",
    "creativity",
    "threat",
    "ict_index",
    "expected_goals",
    "expected_assists",
    "expected_goal_involvements",
    "expected_goals_conceded",
    "xP",
)


def _coerce(frame: pd.DataFrame) -> pd.DataFrame:
    """Cast to warehouse types, keeping missing values missing."""
    for col in INT_COLUMNS:
        if col in frame.columns:
            # Int64 (nullable) not int64: a genuinely absent count must stay NA.
            frame[col] = pd.to_numeric(frame[col], errors="coerce").astype("Int64")
    for col in FLOAT_COLUMNS:
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce").astype("float64")
    if "kickoff_time" in frame.columns:
        frame["kickoff_time"] = pd.to_datetime(
            frame["kickoff_time"], errors="coerce", utc=True
        ).dt.tz_localize(None)
    if "was_home" in frame.columns:
        frame["was_home"] = frame["was_home"].map(
            {True: True, False: False, "True": True, "False": False, "true": True, "false": False}
        )
    return frame


def load_players(season: str, path) -> pd.DataFrame:
    """Season player reference table, and the source of ``code`` — the cross-season id."""
    raw = read_csv(path)
    frame = pd.DataFrame(
        {
            "season": season,
            "element": pd.to_numeric(raw["id"], errors="coerce").astype("Int64"),
            "code": pd.to_numeric(raw["code"], errors="coerce").astype("Int64"),
            "first_name": raw.get("first_name"),
            "second_name": raw.get("second_name"),
            "web_name": raw.get("web_name"),
            "team_id": pd.to_numeric(raw.get("team"), errors="coerce").astype("Int64"),
            "element_type": pd.to_numeric(raw["element_type"], errors="coerce").astype("Int64"),
            "start_cost": pd.to_numeric(raw.get("now_cost"), errors="coerce").astype("Int64"),
        }
    )
    if "birth_date" in raw.columns:
        frame["birth_date"] = pd.to_datetime(raw["birth_date"], errors="coerce").dt.date
    else:
        frame["birth_date"] = pd.NaT
    return frame


def load_teams(season: str, path) -> pd.DataFrame:
    """Season team reference table. 2016-17 has no ``teams.csv`` upstream."""
    raw = read_csv(path)
    frame = pd.DataFrame({"season": season}, index=raw.index)
    frame["team_id"] = pd.to_numeric(raw["id"], errors="coerce").astype("Int64")
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
        frame[col] = raw[col] if col in raw.columns else None
    return frame


def teams_from_players(season: str, path) -> pd.DataFrame:
    """Derive the team table from ``players_raw.csv`` when ``teams.csv`` is missing upstream.

    2016-17 through 2018-19 have no ``teams.csv``, which would otherwise cost us three seasons:
    without a ``team_id -> code`` mapping their fixtures cannot be tied to a club identity, and
    the team-strength model keys on club code so that ratings survive across seasons.

    ``players_raw.csv`` carries both ``team`` and ``team_code`` on every player row, so the
    mapping is recoverable exactly. Names and FPL's strength ratings are genuinely unavailable
    and stay null — nothing depends on them, since we fit our own ratings.
    """
    raw = read_csv(path)
    pairs = (
        raw[["team", "team_code"]]
        .dropna()
        .drop_duplicates()
        .sort_values("team")
        .reset_index(drop=True)
    )
    frame = pd.DataFrame({"season": season}, index=pairs.index)
    frame["team_id"] = pd.to_numeric(pairs["team"], errors="coerce").astype("Int64")
    frame["code"] = pd.to_numeric(pairs["team_code"], errors="coerce").astype("Int64")
    for col in (
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
        frame[col] = None
    log.info("%s: derived %d teams from players_raw (no teams.csv upstream)", season, len(frame))
    return frame


def reconstruct_fixtures(con, season: str) -> int:
    """Rebuild a season's fixture list from its per-player gameweek rows.

    2016-17 and 2017-18 have no ``fixtures.csv`` upstream at all, which would exclude 760 matches
    from the team-strength model — and those are two of only four seasons with defensive-action
    data, so they are disproportionately valuable.

    Every ``player_gw`` row already carries the fixture id, the opponent, which side the player
    was on, the kickoff time and the final score. That is enough to invert into a fixture: the
    player's own team comes from the player reference table, so ``was_home`` decides which side of
    the fixture each team belongs on. Difficulty ratings are unavailable and left null; we do not
    use FPL's difficulty anyway.
    """
    inserted = con.execute(
        """
        INSERT INTO fixtures
            (season, fixture_id, code, event, kickoff_time, team_h, team_a,
             team_h_score, team_a_score, team_h_difficulty, team_a_difficulty, finished, minutes)
        WITH sides AS (
            SELECT
                p.fixture_id,
                any_value(p.gw)           AS event,
                any_value(p.kickoff_time) AS kickoff_time,
                any_value(p.team_h_score) AS team_h_score,
                any_value(p.team_a_score) AS team_a_score,
                -- was_home decides which of (player's team, opponent) is the home side.
                any_value(CASE WHEN p.was_home THEN pl.team_id ELSE p.opponent_team END) AS team_h,
                any_value(CASE WHEN p.was_home THEN p.opponent_team ELSE pl.team_id END) AS team_a
            FROM player_gw p
            JOIN players pl ON pl.season = p.season AND pl.element = p.element
            WHERE p.season = ?
              AND p.was_home IS NOT NULL
              AND pl.team_id IS NOT NULL
              AND p.opponent_team IS NOT NULL
            GROUP BY p.fixture_id
        )
        SELECT ?, fixture_id, NULL, event, kickoff_time, team_h, team_a,
               team_h_score, team_a_score, NULL, NULL, TRUE, 90
        FROM sides
        WHERE team_h IS NOT NULL AND team_a IS NOT NULL AND team_h <> team_a
          AND NOT EXISTS (
              SELECT 1 FROM fixtures f WHERE f.season = ? AND f.fixture_id = sides.fixture_id
          )
        """,
        [season, season, season],
    ).fetchall()
    count = con.execute(
        "SELECT count(*) FROM fixtures WHERE season = ?", [season]
    ).fetchone()[0]
    log.info("%s: reconstructed fixtures from player_gw, now %d fixtures", season, count)
    del inserted
    return count


def load_fixtures(season: str, path) -> pd.DataFrame:
    """Season fixture list, including final scores."""
    raw = read_csv(path)
    frame = pd.DataFrame(
        {
            "season": season,
            "fixture_id": pd.to_numeric(raw["id"], errors="coerce").astype("Int64"),
            "code": pd.to_numeric(raw.get("code"), errors="coerce").astype("Int64"),
            "event": pd.to_numeric(raw.get("event"), errors="coerce").astype("Int64"),
            "team_h": pd.to_numeric(raw["team_h"], errors="coerce").astype("Int64"),
            "team_a": pd.to_numeric(raw["team_a"], errors="coerce").astype("Int64"),
            "team_h_score": pd.to_numeric(raw.get("team_h_score"), errors="coerce").astype("Int64"),
            "team_a_score": pd.to_numeric(raw.get("team_a_score"), errors="coerce").astype("Int64"),
            "team_h_difficulty": pd.to_numeric(
                raw.get("team_h_difficulty"), errors="coerce"
            ).astype("Int64"),
            "team_a_difficulty": pd.to_numeric(
                raw.get("team_a_difficulty"), errors="coerce"
            ).astype("Int64"),
            "minutes": pd.to_numeric(raw.get("minutes"), errors="coerce").astype("Int64"),
        }
    )
    frame["kickoff_time"] = pd.to_datetime(
        raw.get("kickoff_time"), errors="coerce", utc=True
    ).dt.tz_localize(None)
    frame["finished"] = raw.get("finished")
    return frame


def _parse_stats_blob(blob: Any) -> list[dict] | None:
    """Parse a fixture ``stats`` cell.

    The upstream CSV stores it as a Python ``repr`` (single-quoted keys), not JSON, so
    ``ast.literal_eval`` is the correct reader and ``json.loads`` would fail on every row.
    """
    if not isinstance(blob, str) or not blob.strip() or blob.strip() in {"[]", "nan"}:
        return None
    try:
        parsed = ast.literal_eval(blob)
    except (ValueError, SyntaxError):
        return None
    return parsed if isinstance(parsed, list) else None


def load_match_player_stats(season: str, path) -> pd.DataFrame:
    """Explode every fixture's ``stats`` blob into one row per player, per stat.

    This is the only source that pairs a player's raw event counts in a *specific match* with the
    BPS the game awarded for it, which is what makes fitting the reworked 2026/27 BPS possible.
    """
    raw = read_csv(path)
    if "stats" not in raw.columns:
        return pd.DataFrame()

    records: list[dict[str, Any]] = []
    for fixture_id, blob in zip(raw["id"], raw["stats"], strict=True):
        parsed = _parse_stats_blob(blob)
        if not parsed:
            continue
        for stat in parsed:
            identifier = stat.get("identifier")
            for side, is_home in (("h", True), ("a", False)):
                for entry in stat.get(side) or []:
                    records.append(
                        {
                            "season": season,
                            "fixture_id": int(fixture_id),
                            "element": int(entry["element"]),
                            "identifier": identifier,
                            "is_home": is_home,
                            "value": int(entry["value"]),
                        }
                    )

    if not records:
        return pd.DataFrame()
    frame = pd.DataFrame.from_records(records)
    # A player can appear twice for one identifier in a fixture in rare upstream duplicates;
    # the primary key forbids it, so collapse by summing.
    return (
        frame.groupby(["season", "fixture_id", "element", "identifier"], as_index=False)
        .agg(is_home=("is_home", "first"), value=("value", "sum"))
    )


def load_player_gw(season: str, path, players: pd.DataFrame) -> pd.DataFrame:
    """Load a season's per-player-per-gameweek rows, normalised to the warehouse schema."""
    raw = read_csv(path, low_memory=False)

    frame = pd.DataFrame({"season": season}, index=raw.index)
    for target, source in PLAYER_GW_COLUMNS.items():
        frame[target] = raw[source] if source in raw.columns else None

    frame = _coerce(frame)

    # `code`, and `position`/`team_name` for the seasons whose gameweek files omit them, come
    # from the season's player reference table.
    reference = players[["element", "code", "element_type", "team_id"]].copy()
    frame = frame.merge(reference, on="element", how="left")

    derived_position = frame["element_type"].map(POSITION_BY_ELEMENT_TYPE)
    frame["position"] = frame["position"].where(frame["position"].notna(), derived_position)
    frame = frame.drop(columns=["element_type", "team_id"])

    # Rows without a fixture cannot be keyed and are not usable for anything downstream.
    before = len(frame)
    frame = frame.dropna(subset=["element", "fixture_id", "gw"])
    if len(frame) != before:
        log.warning("%s: dropped %d rows lacking element/fixture/gw", season, before - len(frame))

    return _dedupe_player_fixture(frame, season)


def _dedupe_player_fixture(frame: pd.DataFrame, season: str) -> pd.DataFrame:
    """Collapse duplicate (element, fixture) rows from the upstream dataset.

    Upstream builds ``merged_gw.csv`` by concatenating per-player folders, so a player who is
    renamed mid-season (Ben Doak -> Ben Gannon-Doak) ends up with two folders and every one of
    their gameweeks appears twice.

    A real double gameweek gives a player two *different* fixtures, so (element, fixture) stays
    unique and is the right key. That means any duplicate here is an artifact — but we check that
    the copies actually agree before discarding one, because silently keeping the first of two
    *disagreeing* rows would bury a real data problem.
    """
    key = ["element", "fixture_id"]
    duplicated = frame.duplicated(subset=key, keep=False)
    if not duplicated.any():
        return frame

    dupes = frame[duplicated]
    # Compare on the outcome columns; identity columns like web_name are expected to differ,
    # since a rename is the usual cause.
    compare = [
        c
        for c in ("minutes", "total_points", "bps", "goals_scored", "assists", "value", "selected")
        if c in dupes.columns
    ]
    conflicting = dupes.groupby(key)[compare].nunique().gt(1).any(axis=1)
    n_conflicts = int(conflicting.sum())

    if n_conflicts:
        offenders = conflicting[conflicting].index.tolist()[:5]
        log.error(
            "%s: %d duplicate (element, fixture) pairs DISAGREE on outcomes, e.g. %s; "
            "keeping the higher-minutes row",
            season,
            n_conflicts,
            offenders,
        )
        # Prefer the row that recorded appearance data; a zero-minute stub is the likelier stale
        # copy when two rows genuinely differ.
        frame = frame.sort_values("minutes", ascending=False, kind="stable")
    else:
        log.info(
            "%s: collapsed %d identical duplicate rows across %d (element, fixture) pairs "
            "(upstream per-player-folder artifact)",
            season,
            len(dupes) - dupes[key].drop_duplicates().shape[0],
            dupes[key].drop_duplicates().shape[0],
        )

    return frame.drop_duplicates(subset=key, keep="first").sort_index()


def load_season(con, season: str, *, refresh: bool = False) -> dict[str, int]:
    """Load one season's four source files into the warehouse."""
    files = fetch_season(season, refresh=refresh)
    counts: dict[str, int] = {}

    if files["players_raw"] is None:
        log.error("%s: no players_raw.csv; skipping season", season)
        return counts
    players = load_players(season, files["players_raw"])
    counts["players"] = upsert(con, "players", players, ("season", "element"))

    if files["teams"] is not None:
        teams = load_teams(season, files["teams"])
    else:
        # Recoverable from players_raw; see teams_from_players.
        teams = teams_from_players(season, files["players_raw"])
    counts["teams"] = upsert(con, "teams", teams, ("season", "team_id"))

    if files["fixtures"] is not None:
        fixtures = load_fixtures(season, files["fixtures"])
        counts["fixtures"] = upsert(con, "fixtures", fixtures, ("season", "fixture_id"))

        stats = load_match_player_stats(season, files["fixtures"])
        if len(stats):
            counts["match_player_stats"] = upsert(
                con, "match_player_stats", stats, ("season", "fixture_id", "element", "identifier")
            )

    if files["merged_gw"] is not None:
        player_gw = load_player_gw(season, files["merged_gw"], players)
        counts["player_gw"] = upsert(
            con, "player_gw", player_gw, ("season", "element", "fixture_id")
        )
    else:
        log.info("%s: no merged_gw.csv upstream (season not started)", season)

    # Seasons with no fixtures.csv upstream get their fixture list inverted out of player_gw.
    # Must run after player_gw and players are loaded.
    if files["fixtures"] is None and counts.get("player_gw"):
        counts["fixtures"] = reconstruct_fixtures(con, season)

    log.info("%s loaded: %s", season, counts)
    return counts


def load_seasons(
    seasons: list[str] | None = None, *, rebuild: bool = False, refresh: bool = False
) -> dict[str, int]:
    """Load every requested season and return final warehouse row counts."""
    seasons = list(seasons) if seasons else list(SEASONS)
    unknown = set(seasons) - set(SEASONS)
    if unknown:
        raise ValueError(f"unknown season(s): {sorted(unknown)}; known: {list(SEASONS)}")

    con = connect()
    try:
        if rebuild:
            log.info("rebuilding warehouse from scratch")
            drop_all(con)
        for season in seasons:
            load_season(con, season, refresh=refresh)
        create_as_of_view(con)
        return table_counts(con)
    finally:
        con.close()
