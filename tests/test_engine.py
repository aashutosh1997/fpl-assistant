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
from fplass.features.minutes import MinutesProfile, measure_profile
from fplass.features.rates import TeamReturns, measure_team_returns
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


def _run(probabilities, rules, n_draws=20_000, profile=None, returns=None, assist_rate=0.0):
    n = len(probabilities)
    frame = pd.DataFrame(
        {
            "element": np.arange(1, n + 1),
            "event": 1,
            "fixture_id": 1,
            "is_home": True,
            "position": "DEF",
            "goal_rate": 0.0,
            "assist_rate": assist_rate,
            "defcon_rate": 0.0,
            "save_rate": 0.0,
            "card_rate": 0.0,
            "xg_home": 1.2,
            "xg_away": LAM_AWAY,
        }
    )
    result = simulate(
        frame, np.asarray(probabilities, dtype="float64"), rules, BPSModel(),
        rho=0.0, n_draws=n_draws, seed=7, minutes_profile=profile, team_returns=returns,
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


def _point_mass(size: int, index: int) -> np.ndarray:
    pmf = np.zeros(size)
    pmf[index] = 1.0
    return pmf


def test_a_defender_who_always_plays_ninety_keeps_his_teams_clean_sheet_rate():
    """With the measured profile a starting defender rarely leaves early, so his clean sheets
    are his team's, not the inflated rate a uniform 60-90 draw gave him."""
    profile = MinutesProfile(cameo={"DEF": _point_mass(59, 19)}, full={"DEF": _point_mass(31, 30)})
    rules = _rules(clean_sheets={"GKP": 4.0, "DEF": 1.0, "MID": 0.0, "FWD": 0.0})
    points, minutes = _run([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]], rules, profile=profile)
    assert (minutes[:, 0] == 90).all()
    assert (minutes[:, 1] == 20).all()
    assert abs(points[:, 0].mean() - math.exp(-LAM_AWAY)) < 0.02
    assert (points[:, 1] == 0).all()


def test_the_measured_profile_matches_how_long_players_stay_on(con, complete_seasons):
    profile = measure_profile(con, [complete_seasons[-1]])
    full_ninety = {pos: pmf[-1] for pos, pmf in profile.full.items()}
    assert full_ninety["GKP"] > 0.95
    assert full_ninety["DEF"] > 0.75
    assert full_ninety["FWD"] < full_ninety["MID"] < full_ninety["DEF"]
    for table in (profile.cameo, profile.full):
        for pmf in table.values():
            assert abs(pmf.sum() - 1.0) < 1e-9
    # Outfield cameos are mostly late substitutions: well under the uniform draw's thirty.
    cameo_mean = float(np.arange(1, 60) @ profile.cameo["DEF"])
    assert cameo_mean < 28


def test_assists_follow_the_measured_share_of_team_goals():
    returns = TeamReturns(scored=1.0, assisted=0.9, save_intercept=2.6, save_slope=0.2,
                          mean_conceded=1.4)
    rules = _rules(assists=dict.fromkeys(POSITIONS, 1.0))
    points, _ = _run([[0.0, 0.0, 1.0]] * 3, rules, returns=returns, assist_rate=0.2)
    # The home side scores Poisson(1.2); nine in ten of its goals carry an FPL assist.
    assert abs(points.sum(axis=1).mean() - 0.9 * 1.2) < 0.03


def test_the_measured_team_returns_match_the_game(con, complete_seasons):
    returns = measure_team_returns(con, [complete_seasons[-1]])
    assert 0.8 < returns.assisted < 0.95
    assert 0.94 < returns.scored < 0.99
    # Saves barely move with goals conceded; the response averages one at the league's mean.
    assert 0.0 <= returns.save_slope < 0.4 < 2.0 < returns.save_intercept
    assert abs(float(returns.save_response(np.array([returns.mean_conceded]))[0]) - 1.0) < 1e-9
