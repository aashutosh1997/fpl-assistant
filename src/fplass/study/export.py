"""The lasting copy: cohort aggregates and the raw decisions, written under ``data/snapshots``.

FPL serves a season's picks and transfers only while the season is live; when the next one
starts, these entries are gone. Next season's models can only learn from the top managers'
decisions if they were kept, so ``fpl study export`` writes:

``data/snapshots/elite/<season>.csv``
    Per cohort, gameweek and player: ownership before and after the deadline, the share buying
    and selling, and the cohort's size. Small, and what a layer would read.
``data/snapshots/managers/<season>/<table>.csv.gz``
    The managers database's tables, ids only, for anything that needs single decisions —
    the imitation stage of a reinforcement-learning planner among them.
"""

from __future__ import annotations

import logging
from pathlib import Path

from ..paths import SNAPSHOTS
from . import cohorts as cohorts_module
from .store import attach_managers

log = logging.getLogger(__name__)

ELITE = SNAPSHOTS / "elite"
MANAGERS = SNAPSHOTS / "managers"
RAW_TABLES = ("manager_entries", "manager_leagues", "manager_gw", "manager_transfers", "manager_picks")
EXPORTED_COHORTS = ("prior10k", "prior50k", "deadline", "now")


def export_aggregates(con, season: str, last_gw: int, *, out_dir: Path = ELITE) -> Path:
    """Write every cohort's flows for gameweeks 2..``last_gw``."""
    squads = cohorts_module.load_squads(con, season, last_gw)
    chips = cohorts_module.chip_table(con, season)
    members = cohorts_module.memberships(con, season, list(range(2, last_gw + 1)))
    members = members[members["cohort"].isin(EXPORTED_COHORTS)]
    flows = cohorts_module.cohort_flows(members, squads, chips)
    flows = flows[(flows["own_before"] > 0) | (flows["own_after"] > 0) | (flows["bought"] > 0)]
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{season}.csv"
    flows.round(5).to_csv(path, index=False)
    log.info("cohort flows: %d rows -> %s", len(flows), path)
    return path


def export_raw(con, season: str, *, out_dir: Path = MANAGERS) -> list[Path]:
    """Copy the managers database's tables for one season to gzipped CSV."""
    attach_managers(con)
    target = out_dir / season
    target.mkdir(parents=True, exist_ok=True)
    paths = []
    for table in RAW_TABLES:
        path = target / f"{table}.csv.gz"
        con.execute(
            f"COPY (SELECT * FROM m.{table} WHERE season = '{season}' ORDER BY ALL) "
            f"TO '{path}' (FORMAT CSV, HEADER, COMPRESSION GZIP)"
        )
        paths.append(path)
        log.info("%s -> %s (%.1f MB)", table, path, path.stat().st_size / 1e6)
    return paths
