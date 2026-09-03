"""Tests for serving-time form features across the season boundary.

Pinned against the gameweek 2 failure: after Kinsky played ninety minutes in gameweek 1 the base
model still had him at 0.06 to play an hour, because ``minutes_last`` was ninety percent last
season's average. The minutes model is trained on within-season windows, so once a match exists
this season the features must be that match, not a blend.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fplass.sim.project import _blend_form, player_form


def _frame(n_current, cur_last, prior):
    return pd.DataFrame(
        {
            "element": [1],
            "n_current": [n_current],
            "n_prior": [30],
            "cur_minutes_3": [cur_last],
            "cur_minutes_5": [cur_last],
            "cur_minutes_10": [cur_last],
            "cur_start_rate": [1.0 if cur_last else 0.0],
            "cur_played_rate": [1.0 if cur_last else 0.0],
            "cur_minutes_last": [cur_last],
            "cur_minutes_prev": [np.nan],
            "cur_started_last": [1.0 if cur_last else 0.0],
            "cur_last_kickoff": [pd.Timestamp("2026-08-15")],
            "cur_last_value": [45],
            "prior_minutes": [prior],
            "prior_start_rate": [1.0 if prior > 60 else 0.0],
            "prior_played_rate": [1.0 if prior > 0 else 0.0],
            "prior_last_kickoff": [pd.Timestamp("2026-05-20")],
            "prior_last_value": [45],
        }
    )


def test_before_the_first_match_last_season_stands_in():
    out = _blend_form(_frame(0, np.nan, 80.0), lookback=10)
    assert out["minutes_last"].iloc[0] == 80.0
    assert out["roll_minutes_5"].iloc[0] == 80.0
    assert out["n_current"].iloc[0] == 0


def test_one_match_in_the_current_season_replaces_it_entirely():
    """A new club's first choice keeper who was a backup last year: ninety minutes means ninety."""
    out = _blend_form(_frame(1, 90.0, 5.0), lookback=10)
    assert out["minutes_last"].iloc[0] == 90.0
    assert out["started_last"].iloc[0] == 1.0
    assert out["roll_minutes_5"].iloc[0] == 90.0
    # And the reverse: last year's starter who did not play this season's opener.
    out = _blend_form(_frame(1, 0.0, 85.0), lookback=10)
    assert out["minutes_last"].iloc[0] == 0.0
    assert out["roll_minutes_5"].iloc[0] == 0.0


def test_the_match_before_last_waits_for_a_second_match():
    out = _blend_form(_frame(1, 90.0, 40.0), lookback=10)
    assert out["minutes_prev"].iloc[0] == 40.0


def test_spurs_goalkeepers_after_gameweek_one(con):
    """Kinsky played ninety, Dubravka none. The features for gameweek 2 must say exactly that."""
    played = con.execute(
        "SELECT count(DISTINCT gw) FROM player_gw WHERE season = '2026-27'"
    ).fetchone()[0]
    if played < 1:
        pytest.skip("gameweek 1 not ingested")
    names = con.execute(
        "SELECT element, web_name FROM players WHERE season = '2026-27' "
        "AND web_name IN ('Kinsky', 'Dubravka')"
    ).fetchdf()
    if len(names) < 2:
        pytest.skip("Spurs goalkeepers not in this season's pool")

    form = player_form(con, "2026-27", before_gw=2).merge(names, on="element")
    by_name = form.set_index("web_name")
    assert by_name.at["Kinsky", "minutes_last"] == 90
    assert by_name.at["Kinsky", "n_current"] == 1
    assert by_name.at["Dubravka", "minutes_last"] == 0

    # And the as-of guard: rebuilding gameweek 1's inputs must not see gameweek 1's minutes.
    before_one = player_form(con, "2026-27", before_gw=1).merge(names, on="element")
    assert (before_one["n_current"] == 0).all()
