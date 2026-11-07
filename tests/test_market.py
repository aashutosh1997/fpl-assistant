"""The current season's market columns: filled from each player's history without touching stats.

The live feed records what players did, not what managers did, so this season's ``player_gw``
had no price, owners or transfers. They are filled per fixture from ``element-summary``; the
risk is the write, because the warehouse's upsert replaces whole rows.
"""

from __future__ import annotations

import pandas as pd

from fplass.ingest import current
from fplass.ingest.warehouse import connect


def _warehouse(tmp_path):
    con = connect(tmp_path / "w.duckdb")
    con.execute(
        """
        INSERT INTO player_gw (season, element, fixture_id, gw, minutes, total_points, goals_scored)
        VALUES ('2026-27', 1, 10, 1, 90, 8, 1), ('2026-27', 1, 20, 2, 90, 2, 0),
               ('2026-27', 2, 10, 1, 0, 0, 0)
        """
    )
    return con


SUMMARIES = {
    1: {
        "history": [
            {"fixture": 10, "value": 80, "selected": 1000, "transfers_in": 0,
             "transfers_out": 0, "transfers_balance": 0},
            {"fixture": 20, "value": 81, "selected": 1500, "transfers_in": 600,
             "transfers_out": 100, "transfers_balance": 500},
            # A fixture the warehouse has no row for is ignored, not inserted.
            {"fixture": 30, "value": 82, "selected": 1600, "transfers_in": 1,
             "transfers_out": 1, "transfers_balance": 0},
        ]
    },
}


def test_market_fill_writes_the_market_and_leaves_the_stats(tmp_path):
    con = _warehouse(tmp_path)
    updated = current.apply_market_history(con, current._market_frame(SUMMARIES), "2026-27")
    assert updated == 2
    rows = con.execute(
        "SELECT element, fixture_id, minutes, total_points, goals_scored, value, selected, "
        "transfers_in FROM player_gw ORDER BY element, fixture_id"
    ).fetchdf()
    assert len(rows) == 3
    assert rows["minutes"].tolist() == [90, 90, 0]
    assert rows["total_points"].tolist() == [8, 2, 0]
    assert rows.loc[1, "value"] == 81 and rows.loc[1, "transfers_in"] == 600
    assert pd.isna(rows.loc[2, "value"])
    con.close()


def test_reingesting_a_gameweek_keeps_its_market_columns(tmp_path):
    con = _warehouse(tmp_path)
    current.apply_market_history(con, current._market_frame(SUMMARIES), "2026-27")
    # What ingest_live_gameweek stages: stats, no market columns.
    restaged = pd.DataFrame(
        {"season": ["2026-27"], "element": [1], "fixture_id": [20], "gw": [2],
         "minutes": [90], "total_points": [3], "goals_scored": [0]}
    )
    kept = current._keep_market_columns(con, restaged, "2026-27")
    assert kept.loc[0, "value"] == 81 and kept.loc[0, "selected"] == 1500
    assert kept.loc[0, "total_points"] == 3
    con.close()
