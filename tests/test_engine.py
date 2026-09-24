"""Tests for the simulator's goals-conceded and clean-sheet accounting.

Pinned against the GW3-5 review: every row was charged its team's full-match goals conceded, so
a defender the model gave a 99.99% chance of not playing projected at -0.99 points a week, and a
twenty-minute substitute paid for goals conceded before he came on. FPL counts both only while
the player is on the pitch.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from fplass.features.bps import BPSModel
from fplass.scoring import POSITIONS, ScoringRules
from fplass.sim.engine import simulate

# The home side concedes the away side's goals: Poisson(1.5), so a clean sheet 22% of the time.
LAM_AWAY = 1.5


def _rules(**per_position) -> ScoringRules:
    """Rules that score nothing but the stats given, so a total isolates one component."""
    rules = ScoringRules(season="test", long_play=0.0, short_play=0.0, saves=0.0,
                         yellow_cards=0.0, bonus=0.0)
    for stat in ("goals_scored", "assists", "clean_sheets", "goals_conceded",
                 "defensive_contribution"):
        setattr(rules, stat, per_position.get(stat, dict.fromkeys(POSITIONS, 0.0)))
    return rules


def _run(probabilities, rules, n_draws=20_000):
    n = len(probabilities)
    frame = pd.DataFrame(
        {
            "element": np.arange(1, n + 1),
            "event": 1,
            "fixture_id": 1,
            "is_home": True,
            "position": "DEF",
            "goal_rate": 0.0,
            "assist_rate": 0.0,
            "defcon_rate": 0.0,
            "save_rate": 0.0,
            "card_rate": 0.0,
            "xg_home": 1.2,
            "xg_away": LAM_AWAY,
        }
    )
    result = simulate(
        frame, np.asarray(probabilities, dtype="float64"), rules, BPSModel(),
        rho=0.0, n_draws=n_draws, seed=7,
    )
    return result.points[:, :, 0].astype("float64"), result.minutes_played[:, :, 0]


def test_a_defender_who_does_not_play_concedes_nothing():
    rules = _rules(goals_conceded={"GKP": -1.0, "DEF": -1.0, "MID": 0.0, "FWD": 0.0})
    points, _ = _run([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], rules)
    assert (points[:, 0] == 0).all()
    # The starter still pays: roughly -E[floor(conceded / 2)], a little less for early exits.
    assert points[:, 1].mean() < -0.3


def test_a_substitute_pays_only_for_his_share_of_the_match():
    rules = _rules(goals_conceded={"GKP": -1.0, "DEF": -1.0, "MID": 0.0, "FWD": 0.0})
    points, minutes = _run([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], rules)
    cameo, full = points[:, 0].mean(), points[:, 1].mean()
    assert full < cameo < 0
    # A thirty-minute average cameo concedes about a third of the goals the starter does.
    assert abs(cameo) < 0.6 * abs(full)
    assert minutes[:, 0].max() < 60


def test_clean_sheets_are_judged_on_goals_while_on_the_pitch():
    rules = _rules(clean_sheets={"GKP": 4.0, "DEF": 1.0, "MID": 0.0, "FWD": 0.0})
    points, minutes = _run([[0.0, 0.0, 1.0]], rules)
    sheet_rate = points[:, 0].mean()
    team_rate = math.exp(-LAM_AWAY)
    # Some starters leave between the hour and ninety minutes, before a late goal.
    assert sheet_rate > team_rate + 0.02
    # Ninety-minute rows keep their team's rate exactly: every goal conceded counts.
    ninety = minutes[:, 0] == 90
    assert abs(points[ninety, 0].mean() - team_rate) < 0.04
