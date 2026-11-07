"""Where the managers' decisions live: a DuckDB file of its own beside the warehouse.

A full crawl takes hours. Holding the warehouse open read-write that long would lock out every
reader — the planner, the panel replay, the tests — because DuckDB refuses a read-only open while
another process writes. So the crawl writes here, and the analysis attaches this file read-only
next to the warehouse.

Only ids are stored: no team or manager names. FPL serves a season's picks and transfers only
while that season is live, so this file (and its export under ``data/snapshots/managers``) is
the only lasting record of them.
"""

from __future__ import annotations

from pathlib import Path

import duckdb

from ..paths import WAREHOUSE

MANAGERS_DB = WAREHOUSE / "managers.duckdb"

# FPL's overall league: every manager, ranked.
OVERALL_LEAGUE = 314

SCHEMA = """
-- One row per manager the crawl has seen. ``source`` is how it was found: 'top' (the overall
-- standings from rank 1), 'band' (a sampled page deeper down; ``weight`` is the inverse of the
-- sampling rate), 'league' or 'us'. Past-season fields come from entry/{id}/history's ``past``.
CREATE TABLE IF NOT EXISTS manager_entries (
    season          VARCHAR NOT NULL,
    entry           INTEGER NOT NULL,
    source          VARCHAR NOT NULL,
    weight          DOUBLE NOT NULL,
    standing_rank   INTEGER,     -- overall rank when the standings were read
    standing_gw     INTEGER,     -- the last finished gameweek at that read
    started_gw      INTEGER,     -- the manager's first gameweek this season
    past_seasons    INTEGER,
    past_rank_last  INTEGER,     -- final overall rank in the previous season, if played
    past_top10k     INTEGER,     -- past seasons finished inside the top 10,000
    past_top100k    INTEGER,
    history_gw      INTEGER,     -- the last finished gameweek when history was fetched
    transfers_gw    INTEGER,     -- likewise for transfers
    PRIMARY KEY (season, entry)
);

CREATE TABLE IF NOT EXISTS manager_leagues (
    season  VARCHAR NOT NULL,
    league  INTEGER NOT NULL,
    entry   INTEGER NOT NULL,
    PRIMARY KEY (season, league, entry)
);

CREATE TABLE IF NOT EXISTS manager_gw (
    season               VARCHAR NOT NULL,
    entry                INTEGER NOT NULL,
    gw                   INTEGER NOT NULL,
    points               INTEGER,
    total_points         INTEGER,
    overall_rank         INTEGER,
    bank                 INTEGER,
    value                INTEGER,
    event_transfers      INTEGER,
    event_transfers_cost INTEGER,
    points_on_bench      INTEGER,
    chip                 VARCHAR,
    PRIMARY KEY (season, entry, gw)
);

CREATE TABLE IF NOT EXISTS manager_transfers (
    season           VARCHAR NOT NULL,
    entry            INTEGER NOT NULL,
    gw               INTEGER NOT NULL,
    element_in       INTEGER NOT NULL,
    element_in_cost  INTEGER,
    element_out      INTEGER NOT NULL,
    element_out_cost INTEGER,
    time             TIMESTAMP NOT NULL,
    PRIMARY KEY (season, entry, time, element_in, element_out)
);

-- Standings pages already read for a gameweek, so an interrupted crawl resumes where it stopped.
CREATE TABLE IF NOT EXISTS crawl_pages (
    season  VARCHAR NOT NULL,
    league  INTEGER NOT NULL,
    page    INTEGER NOT NULL,
    gw      INTEGER NOT NULL,
    PRIMARY KEY (season, league, page, gw)
);

CREATE TABLE IF NOT EXISTS manager_picks (
    season          VARCHAR NOT NULL,
    entry           INTEGER NOT NULL,
    gw              INTEGER NOT NULL,
    element         INTEGER NOT NULL,
    slot            INTEGER,     -- 1-11 the starting eleven in order, 12-15 the bench order
    multiplier      INTEGER,
    is_captain      BOOLEAN,
    is_vice_captain BOOLEAN,
    PRIMARY KEY (season, entry, gw, element)
);
"""


def connect_managers(
    path: Path | str = MANAGERS_DB, *, read_only: bool = False
) -> duckdb.DuckDBPyConnection:
    """Open the managers database, creating it and its schema if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if read_only and not path.exists():
        raise FileNotFoundError(f"no managers database at {path}; run `fpl study crawl` first")
    con = duckdb.connect(str(path), read_only=read_only)
    if not read_only:
        con.execute(SCHEMA)
    return con


def attach_managers(con: duckdb.DuckDBPyConnection, path: Path | str = MANAGERS_DB) -> None:
    """Attach the managers database read-only to a warehouse connection as ``m``."""
    attached = {r[0] for r in con.execute("SELECT database_name FROM duckdb_databases()").fetchall()}
    if "m" not in attached:
        con.execute(f"ATTACH '{Path(path)}' AS m (READ_ONLY)")
