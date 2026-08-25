"""DuckDB warehouse: schema, connections, and the leakage-proof ``as_of`` view.

The single most important thing in this file is :func:`create_as_of_view`.

The historical per-gameweek data contains columns that are only knowable *after* the gameweek was
played — ``selected``, ``transfers_in``, ``transfers_out``, ``transfers_balance``, ``value`` and
``xP``. A backtest that reads those for the gameweek it is predicting is not a backtest; it is a
model with the answer in its features, and it will look excellent right up until it is used for
real. Rather than trusting every future feature builder to remember to lag them, we expose one
view that structurally cannot leak, and point all training code at that.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import duckdb

from ..paths import DB_PATH, ensure_dirs

log = logging.getLogger(__name__)

# Columns in player_gw that describe the gameweek's *outcome* or its post-hoc market reaction.
# Available when predicting gameweek N only for gameweeks <= N-1.
POST_HOC_COLUMNS = (
    "selected",
    "transfers_in",
    "transfers_out",
    "transfers_balance",
    "value",
    "xP",
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS teams (
    season        VARCHAR NOT NULL,
    team_id       INTEGER NOT NULL,   -- FPL team id, reassigned each season
    code          INTEGER,            -- stable cross-season club code
    name          VARCHAR,
    short_name    VARCHAR,
    strength      INTEGER,
    strength_overall_home  INTEGER,
    strength_overall_away  INTEGER,
    strength_attack_home   INTEGER,
    strength_attack_away   INTEGER,
    strength_defence_home  INTEGER,
    strength_defence_away  INTEGER,
    PRIMARY KEY (season, team_id)
);

CREATE TABLE IF NOT EXISTS players (
    season        VARCHAR NOT NULL,
    element       INTEGER NOT NULL,   -- FPL element id, reassigned each season
    code          INTEGER,            -- STABLE cross-season player id: join on this
    first_name    VARCHAR,
    second_name   VARCHAR,
    web_name      VARCHAR,
    team_id       INTEGER,
    element_type  INTEGER,            -- 1 GKP, 2 DEF, 3 MID, 4 FWD
    start_cost    INTEGER,
    birth_date    DATE,
    PRIMARY KEY (season, element)
);

CREATE TABLE IF NOT EXISTS fixtures (
    season        VARCHAR NOT NULL,
    fixture_id    INTEGER NOT NULL,
    code          BIGINT,
    event         INTEGER,            -- gameweek; NULL when unscheduled
    kickoff_time  TIMESTAMP,
    team_h        INTEGER,
    team_a        INTEGER,
    team_h_score  INTEGER,
    team_a_score  INTEGER,
    team_h_difficulty INTEGER,
    team_a_difficulty INTEGER,
    finished      BOOLEAN,
    minutes       INTEGER,
    PRIMARY KEY (season, fixture_id)
);

CREATE TABLE IF NOT EXISTS player_gw (
    season        VARCHAR NOT NULL,
    element       INTEGER NOT NULL,
    fixture_id    INTEGER NOT NULL,
    gw            INTEGER NOT NULL,
    code          INTEGER,
    web_name      VARCHAR,
    position      VARCHAR,
    team_name     VARCHAR,
    opponent_team INTEGER,
    was_home      BOOLEAN,
    kickoff_time  TIMESTAMP,
    minutes       INTEGER,
    starts        INTEGER,
    total_points  INTEGER,
    goals_scored  INTEGER,
    assists       INTEGER,
    clean_sheets  INTEGER,
    goals_conceded INTEGER,
    own_goals     INTEGER,
    penalties_saved INTEGER,
    penalties_missed INTEGER,
    yellow_cards  INTEGER,
    red_cards     INTEGER,
    saves         INTEGER,
    bonus         INTEGER,
    bps           INTEGER,
    influence     DOUBLE,
    creativity    DOUBLE,
    threat        DOUBLE,
    ict_index     DOUBLE,
    expected_goals DOUBLE,
    expected_assists DOUBLE,
    expected_goal_involvements DOUBLE,
    expected_goals_conceded DOUBLE,
    tackles       INTEGER,
    recoveries    INTEGER,
    clearances_blocks_interceptions INTEGER,
    defensive_contribution INTEGER,
    team_h_score  INTEGER,
    team_a_score  INTEGER,
    -- post-hoc columns: see POST_HOC_COLUMNS
    value         INTEGER,
    selected      BIGINT,
    transfers_in  BIGINT,
    transfers_out BIGINT,
    transfers_balance BIGINT,
    xP            DOUBLE,
    PRIMARY KEY (season, element, fixture_id)
);

-- Per-player, per-match event counts exploded from each fixture's `stats` blob. This is the
-- training set for the BPS model: it is the only source that pairs a player's raw event counts
-- with the BPS the game actually awarded them in that specific match.
CREATE TABLE IF NOT EXISTS match_player_stats (
    season      VARCHAR NOT NULL,
    fixture_id  INTEGER NOT NULL,
    element     INTEGER NOT NULL,
    identifier  VARCHAR NOT NULL,   -- goals_scored, bps, defensive_contribution, ...
    is_home     BOOLEAN,
    value       INTEGER,
    PRIMARY KEY (season, fixture_id, element, identifier)
);

CREATE TABLE IF NOT EXISTS events (
    season        VARCHAR NOT NULL,
    event         INTEGER NOT NULL,
    name          VARCHAR,
    deadline_time TIMESTAMP,
    finished      BOOLEAN,
    data_checked  BOOLEAN,
    average_entry_score INTEGER,
    highest_score INTEGER,
    most_selected INTEGER,
    most_transferred_in INTEGER,
    most_captained INTEGER,
    transfers_made BIGINT,
    PRIMARY KEY (season, event)
);

-- Chip definitions with their legal gameweek windows, straight from the API. Read from here
-- rather than hardcoding: the 2026/27 windows are asymmetric (Bench Boost and Triple Captain
-- are legal in GW1, Wildcard and Free Hit are not), and they have changed between seasons.
CREATE TABLE IF NOT EXISTS chips (
    season      VARCHAR NOT NULL,
    chip_id     INTEGER NOT NULL,
    name        VARCHAR,
    chip_type   VARCHAR,
    start_event INTEGER,
    stop_event  INTEGER,
    PRIMARY KEY (season, chip_id)
);

-- The scoring rules in force for a season, so the scoring engine is never hardcoded and a
-- backtest of 2019-20 scores with 2019-20's rules rather than this season's.
CREATE TABLE IF NOT EXISTS scoring_rules (
    season   VARCHAR NOT NULL,
    stat     VARCHAR NOT NULL,
    position VARCHAR NOT NULL,   -- GKP/DEF/MID/FWD, or 'ALL'
    points   DOUBLE,
    PRIMARY KEY (season, stat, position)
);

-- Preseason friendlies. The only observation of the *current* squad hierarchy that exists before
-- a competitive ball is kicked, and the single strongest predictor of gameweek 1 minutes.
-- Published for 2026-27 only, so every consumer must tolerate these tables being empty.
CREATE TABLE IF NOT EXISTS preseason_player (
    season                  VARCHAR NOT NULL,
    element                 INTEGER NOT NULL,
    preseason_matches       INTEGER,
    preseason_minutes       DOUBLE,
    preseason_minutes_avg   DOUBLE,
    preseason_minutes_max   DOUBLE,
    preseason_minutes_share DOUBLE,
    preseason_goals         DOUBLE,
    preseason_assists       DOUBLE,
    preseason_xg            DOUBLE,
    preseason_xa            DOUBLE,
    PRIMARY KEY (season, element)
);

CREATE TABLE IF NOT EXISTS preseason_team (
    season                  VARCHAR NOT NULL,
    code                    INTEGER NOT NULL,   -- stable club code
    preseason_played        INTEGER,
    preseason_fixtures      INTEGER,
    preseason_goals_for     DOUBLE,
    preseason_goals_against DOUBLE,
    preseason_xg_for        DOUBLE,
    preseason_xg_against    DOUBLE,
    PRIMARY KEY (season, code)
);

-- Every projection we make, stored *before* the deadline so it can be scored afterwards.
--
-- Without this there is nothing to calibrate against: projections were previously recomputed on
-- demand and discarded, so after gameweek 1 the only way to ask "was the model right" was to
-- reconstruct the prediction by hand. This table is what makes the model self-correcting.
CREATE TABLE IF NOT EXISTS projections (
    season        VARCHAR NOT NULL,
    gw            INTEGER NOT NULL,
    element       INTEGER NOT NULL,
    made_at       TIMESTAMP NOT NULL,   -- must be before the deadline to be a valid prediction
    model_version VARCHAR,
    p_none        DOUBLE,
    p_cameo       DOUBLE,
    p_full        DOUBLE,
    expected_points DOUBLE,
    ep_p10        DOUBLE,
    ep_p90        DOUBLE,
    PRIMARY KEY (season, gw, element, made_at)
);
"""


def connect(path: Path | str = DB_PATH, *, read_only: bool = False) -> duckdb.DuckDBPyConnection:
    """Open the warehouse, creating its directory and schema if needed."""
    ensure_dirs()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(path), read_only=read_only)
    if not read_only:
        con.execute(SCHEMA)
    return con


@contextmanager
def warehouse(
    path: Path | str = DB_PATH, *, read_only: bool = False
) -> Iterator[duckdb.DuckDBPyConnection]:
    con = connect(path, read_only=read_only)
    try:
        yield con
    finally:
        con.close()


def drop_all(con: duckdb.DuckDBPyConnection) -> None:
    """Drop every warehouse table, for a clean rebuild."""
    for table in (
        "player_gw",
        "match_player_stats",
        "fixtures",
        "players",
        "teams",
        "events",
        "chips",
        "scoring_rules",
        "preseason_player",
        "preseason_team",
        "projections",
    ):
        con.execute(f"DROP TABLE IF EXISTS {table}")
    con.execute(SCHEMA)


def upsert(
    con: duckdb.DuckDBPyConnection, table: str, frame, key_columns: tuple[str, ...]
) -> int:
    """Idempotently insert a DataFrame, replacing rows that match on ``key_columns``.

    Re-running an ingest must not duplicate rows — the gameweek workflow re-ingests the current
    season on every run, and a season's data is revised for days afterwards as Opta review
    settles bonus points and defensive contributions.
    """
    if frame is None or len(frame) == 0:
        return 0

    schema_cols = [r[1] for r in con.execute(f"PRAGMA table_info('{table}')").fetchall()]
    keep = [c for c in schema_cols if c in frame.columns]
    staged = frame[keep]

    con.register("_staged", staged)
    predicate = " AND ".join(f"t.{k} = s.{k}" for k in key_columns)
    con.execute(
        f"DELETE FROM {table} t WHERE EXISTS (SELECT 1 FROM _staged s WHERE {predicate})"
    )
    columns = ", ".join(keep)
    con.execute(f"INSERT INTO {table} ({columns}) SELECT {columns} FROM _staged")
    con.unregister("_staged")
    return len(staged)


# DEFCON thresholds, verified against 2025-26 actuals: defenders need 10 CBIT, midfielders and
# forwards 12 CBIRT (which additionally counts ball recoveries). Goalkeepers score no DEFCON.
DEFCON_THRESHOLDS = {"DEF": 10, "MID": 12, "FWD": 12}


def create_derived_view(con: duckdb.DuckDBPyConnection) -> None:
    """Create ``player_gw_derived``: ``player_gw`` plus columns that need cross-season logic.

    Three things are computed here rather than at load time, because each depends on knowing how
    the source data differs between seasons:

    ``defcon_count``
        Defensive contribution actions. ``defensive_contribution`` only exists from 2025-26, but
        its components do exist for 2016-17..2018-19, and the identity was verified exactly
        against 2025-26: CBIT for defenders, CBIRT for midfielders and forwards. So the count is
        *reconstructed* for the Opta-era seasons rather than left null, which turns one season of
        DEFCON training data into four. Still null for 2019-20..2024-25, where the components
        were simply not published.

    ``hit_defcon``
        Whether the 10/12 threshold was met. Note this is computed for every season, including
        ones played before DEFCON points existed — the *rate* at which a player type racks up
        defensive actions is what we are modelling, and that is observable long before FPL
        started paying for it.

    ``gw_seq``
        A dense, gap-free sequence of a season's gameweeks. 2019-20 was suspended for COVID and
        its gameweeks jump from 29 straight to 39, so any rolling window keyed on ``gw`` would
        silently span a four-month break. Rolling features must use this.
    """
    con.execute(
        f"""
        CREATE OR REPLACE VIEW player_gw_derived AS
        WITH seq AS (
            SELECT season, gw, DENSE_RANK() OVER (PARTITION BY season ORDER BY gw) AS gw_seq
            FROM (SELECT DISTINCT season, gw FROM player_gw)
        )
        SELECT
            p.*,
            s.gw_seq,
            COALESCE(
                p.defensive_contribution,
                CASE
                    WHEN p.position = 'DEF'
                        THEN p.clearances_blocks_interceptions + p.tackles
                    WHEN p.position IN ('MID', 'FWD')
                        THEN p.clearances_blocks_interceptions + p.tackles + p.recoveries
                END
            ) AS defcon_count,
            CASE p.position
                WHEN 'DEF' THEN {DEFCON_THRESHOLDS["DEF"]}
                WHEN 'MID' THEN {DEFCON_THRESHOLDS["MID"]}
                WHEN 'FWD' THEN {DEFCON_THRESHOLDS["FWD"]}
            END AS defcon_threshold,
            CASE
                WHEN p.position = 'DEF' THEN
                    COALESCE(p.defensive_contribution,
                             p.clearances_blocks_interceptions + p.tackles)
                    >= {DEFCON_THRESHOLDS["DEF"]}
                WHEN p.position IN ('MID', 'FWD') THEN
                    COALESCE(p.defensive_contribution,
                             p.clearances_blocks_interceptions + p.tackles + p.recoveries)
                    >= {DEFCON_THRESHOLDS["MID"]}
            END AS hit_defcon
        FROM player_gw p
        JOIN seq s USING (season, gw)
        """
    )
    log.debug("created player_gw_derived")


def create_as_of_view(con: duckdb.DuckDBPyConnection) -> None:
    """Create ``player_gw_as_of``: the same rows, with post-hoc columns lagged by one gameweek.

    For each (season, element, gw) row this exposes ``prev_selected``, ``prev_value`` and so on,
    carrying the values from that player's *previous* gameweek, and does not expose the current
    gameweek's versions at all. Feature builders and the backtest read this view; nothing reads
    the post-hoc columns of ``player_gw`` directly except the scoring-engine test, which is
    checking outcomes on purpose.
    """
    create_derived_view(con)
    lagged = ",\n                ".join(
        f"LAG({c}) OVER w AS prev_{c}" for c in POST_HOC_COLUMNS
    )
    con.execute(
        f"""
        CREATE OR REPLACE VIEW player_gw_as_of AS
        WITH lagged AS (
            SELECT
                season, element, fixture_id, gw,
                {lagged}
            FROM player_gw_derived
            WINDOW w AS (PARTITION BY season, element ORDER BY gw_seq, fixture_id)
        )
        SELECT
            p.* EXCLUDE ({", ".join(POST_HOC_COLUMNS)}),
            l.* EXCLUDE (season, element, fixture_id, gw)
        FROM player_gw_derived p
        JOIN lagged l USING (season, element, fixture_id, gw)
        """
    )
    log.debug("created player_gw_as_of (lagged: %s)", ", ".join(POST_HOC_COLUMNS))


def table_counts(con: duckdb.DuckDBPyConnection) -> dict[str, int]:
    """Row count per warehouse table, for ingest summaries and tests."""
    tables = [
        r[0]
        for r in con.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'main' AND table_type = 'BASE TABLE'"
        ).fetchall()
    ]
    return {t: con.execute(f"SELECT count(*) FROM {t}").fetchone()[0] for t in tables}
