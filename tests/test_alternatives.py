"""Tests for ranking the planner's alternatives.

Hand-built plans over a handful of players, so every share and every ordering is knowable. The
solver side (distinct first-week squads, the roll reference) is tested in test_milp.py, where
the synthetic league lives.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from fplass.advise import Recommendation, format_report, move_summary
from fplass.optimise.alternatives import (
    chip_calls,
    economy_credit,
    is_roll,
    rank,
    robustness,
)
from fplass.optimise.chips import ChipRoadmap
from fplass.optimise.league import LeagueState, Rival
from fplass.optimise.milp import Plan

GWS = [11, 12]
ELEMENTS = np.array([1, 2, 3, 4])


def make_plan(
    lineup: list[int],
    *,
    captain: int | None = None,
    bought: list[int] | None = None,
    sold: list[int] | None = None,
    hits: int = 0,
    credit: float = 0.0,
    expected: float = 10.0,
) -> Plan:
    """A plan that fields ``lineup`` every week; ``credit`` is its banked-transfer value."""
    captain = lineup[0] if captain is None else captain
    expected_points = {gw: expected / len(GWS) for gw in GWS}
    return Plan(
        squads={gw: list(lineup) for gw in GWS},
        lineups={gw: list(lineup) for gw in GWS},
        captains={gw: captain for gw in GWS},
        transfers_in={GWS[0]: list(bought or [])},
        transfers_out={GWS[0]: list(sold or [])},
        hits={GWS[0]: hits},
        chips={},
        objective=expected - 4.0 * hits + credit,
        expected_points=expected_points,
        status="Optimal",
    )


def samples_where(scores: dict[int, float], draws: int = 400, seed: int = 0) -> np.ndarray:
    """``(draws, players, gameweeks)`` points centred on ``scores`` per player per week."""
    rng = np.random.default_rng(seed)
    base = np.array([scores.get(int(e), 0.0) for e in ELEMENTS])
    noise = rng.normal(0.0, 1.0, size=(draws, len(ELEMENTS), len(GWS)))
    return base[None, :, None] + noise


# ------------------------------------------------------------------ robustness


def test_a_plan_better_in_every_revision_is_fully_robust():
    strong, weak = make_plan([1]), make_plan([2])
    projections = np.zeros((50, len(ELEMENTS), len(GWS)))
    projections[:, 0, :] = 5.0
    projections[:, 1, :] = 1.0
    shares = robustness([strong, weak], projections, ELEMENTS)
    assert shares == pytest.approx([1.0, 0.0])


def test_tied_plans_split_the_revisions_and_shares_sum_to_one():
    a, b, c = make_plan([1]), make_plan([1]), make_plan([2])
    projections = samples_where({1: 3.0, 2: 3.0})
    shares = robustness([a, b, c], projections, ELEMENTS)
    assert shares.sum() == pytest.approx(1.0)
    assert shares[0] == pytest.approx(shares[1])


def test_robustness_counts_the_planners_price_on_a_banked_transfer():
    """A roll that banks a transfer can be the planner's pick against slightly more points."""
    move = make_plan([2], bought=[2], sold=[1], expected=10.4)
    roll = make_plan([1], credit=3.0, expected=10.0)
    assert economy_credit(roll) == pytest.approx(3.0)
    projections = np.zeros((20, len(ELEMENTS), len(GWS)))
    projections[:, 0, :] = 2.5  # 10 over two weeks, captained
    projections[:, 1, :] = 2.6  # 10.4
    shares = robustness([move, roll], projections, ELEMENTS)
    assert shares == pytest.approx([0.0, 1.0])
    # Against rivals the ranking counts points scored alone, and so does its robustness.
    shares = robustness([move, roll], projections, ELEMENTS, credit=False)
    assert shares == pytest.approx([1.0, 0.0])


# ------------------------------------------------------------------ ranking


def test_on_points_the_ranking_is_the_planners_value_and_roll_is_kept_as_reference():
    plans = [
        make_plan([1], bought=[1], sold=[4], credit=0.0, expected=12.0),
        make_plan([2], bought=[2], sold=[4], credit=0.0, expected=11.0),
        make_plan([3], bought=[3], sold=[4], credit=0.0, expected=10.5),
        make_plan([4], credit=0.0, expected=9.0),  # the roll, last
    ]
    samples = samples_where({1: 3.0, 2: 2.75, 3: 2.6, 4: 2.25})
    options = rank(plans, samples, ELEMENTS, GWS[0], objective="points", n_show=2)

    assert [o.plan for o in options[:2]] == plans[:2]
    assert [o.rank for o in options] == [1, 2, 4]
    assert options[-1].reference and is_roll(options[-1].plan, GWS[0])
    assert options[0].metrics["gap"] == 0.0
    assert options[1].metrics["gap"] < 0
    assert "points" in options[0].tags


def test_against_rivals_the_ranking_is_rivals_beaten():
    """Chasing a leader, the higher-variance plan beats more rivals despite fewer points."""
    steady = make_plan([1], bought=[1], sold=[4], expected=12.0)
    swing = make_plan([2], bought=[2], sold=[4], expected=11.0)
    draws = 4000
    rng = np.random.default_rng(3)
    samples = np.zeros((draws, len(ELEMENTS), len(GWS)))
    samples[:, 0, :] = 2.0 + rng.normal(0, 0.1, (draws, len(GWS)))
    samples[:, 1, :] = 1.8 + rng.normal(0, 6.0, (draws, len(GWS)))
    league = LeagueState(
        league_id=1,
        name="Test League",
        rivals=[Rival(entry=2, name="r", manager="m", total_points=40, rank=1)],
        my_entry=1,
        my_points=10,
    )
    rival_totals = np.full((draws, 1), 12.0)
    options = rank(
        [steady, swing], samples, ELEMENTS, GWS[0],
        league=league, rival_totals=rival_totals, objective="league",
    )
    assert options[0].plan is swing
    first, second = options[0].metrics, options[1].metrics
    assert first["expected_rivals_beaten"] > second["expected_rivals_beaten"]
    assert first["expected_points"] < second["expected_points"]
    assert "rivals" in options[0].tags and "points" in options[1].tags


def test_against_rivals_the_planners_own_pick_is_kept_as_a_reference():
    """When the league ranking and the planner's economy disagree, both answers are shown."""
    top = make_plan([1], bought=[1], sold=[4], expected=12.0)
    second = make_plan([3], bought=[3], sold=[4], expected=10.4)
    banker = make_plan([2], bought=[2], sold=[4], expected=10.0, credit=4.0)  # value 14
    roll = make_plan([4], expected=9.0)
    samples = samples_where({1: 3.0, 2: 2.5, 3: 2.6, 4: 2.25})
    league = LeagueState(
        league_id=1,
        name="Test League",
        rivals=[Rival(entry=2, name="r", manager="m", total_points=10, rank=1)],
        my_entry=1,
        my_points=10,
    )
    options = rank(
        [top, second, banker, roll], samples, ELEMENTS, GWS[0],
        league=league, rival_totals=np.zeros((samples.shape[0], 1)), objective="league",
        n_show=1,
    )
    assert [o.plan for o in options] == [top, banker, roll]
    assert [o.reference for o in options] == [False, True, True]
    assert "value" in options[1].tags


def test_tags_are_left_off_when_the_options_do_not_differ():
    plans = [make_plan([1], bought=[1], sold=[4]), make_plan([1], bought=[1], sold=[3])]
    samples = samples_where({1: 3.0})
    options = rank(plans, samples, ELEMENTS, GWS[0], objective="points")
    assert all("ceiling" not in o.tags and "safest" not in o.tags for o in options)


# ------------------------------------------------------------------ chips


def test_chip_calls_compare_this_week_with_the_floor_and_skip_used_chips():
    valuations = pd.DataFrame(
        [
            {"chip": "3xc", "gameweek": 11, "window": 0, "mean_gain": 5.0, "upside_gain": 14.0,
             "note": "captain 1"},
            {"chip": "3xc", "gameweek": 12, "window": 0, "mean_gain": 6.5, "upside_gain": 15.0,
             "note": "captain 2"},
            {"chip": "freehit", "gameweek": 12, "window": 0, "mean_gain": 4.0, "upside_gain": 9.0,
             "note": ""},
            {"chip": "bboost", "gameweek": 11, "window": 0, "mean_gain": 3.0, "upside_gain": 8.0,
             "note": ""},
        ]
    )
    calls = chip_calls(
        valuations, {"3xc": 7.0, "freehit": 12.0}, 11, chips_used={"bboost:0"}
    ).set_index("chip")
    assert "bboost" not in calls.index
    assert calls.loc["3xc", "short_by"] == pytest.approx(2.0)
    assert calls.loc["3xc", "best_gameweek"] == 12
    assert np.isnan(calls.loc["freehit", "now_mean"])  # not legal this week in this horizon

    floor = chip_calls(valuations, lambda chip, gw: 20.0 - gw, 11, chips_used=set())
    assert floor.set_index("chip").loc["3xc", "now_floor"] == pytest.approx(9.0)


# ------------------------------------------------------------------ the report


def test_move_summary_names_the_roll_and_the_lost_transfer():
    names = {1: "Saka", 2: "Isak", 4: "Wissa"}
    roll = make_plan([4])
    move = make_plan([2], bought=[2], sold=[4], hits=1)
    assert move_summary(roll, GWS[0], names, 2) == "Roll — 3 free transfers next week"
    assert "one is lost" in move_summary(roll, GWS[0], names, 5)
    assert move_summary(move, GWS[0], names, 1) == "Wissa -> Isak (-4)"


def test_the_report_leads_with_the_ranked_options_and_their_paths():
    players = pd.DataFrame(
        {
            "element": ELEMENTS,
            "web_name": ["Saka", "Isak", "Palmer", "Wissa"],
            "position": ["MID", "FWD", "MID", "FWD"],
        }
    )
    plans = [
        make_plan([1], bought=[1], sold=[4], expected=12.0),
        make_plan([2], bought=[2], sold=[4], expected=11.0),
        make_plan([4], expected=9.0),
    ]
    samples = samples_where({1: 3.0, 2: 2.75, 4: 2.25})
    projections = samples_where({1: 3.0, 2: 2.75, 4: 2.25}, draws=100, seed=1)
    options = rank(
        plans, samples, ELEMENTS, GWS[0], objective="points", projections=projections, n_show=2
    )
    roadmap = ChipRoadmap(schedule={}, valuations=pd.DataFrame(), notes=[])
    recommendation = Recommendation(
        gameweek=GWS[0],
        plan=options[0].plan,
        roadmap=roadmap,
        comparison=pd.DataFrame(),
        league_metrics={},
        price_advice=pd.DataFrame(),
        sell_alerts=pd.DataFrame(),
        buy_alerts=pd.DataFrame(),
        expected_points=pd.DataFrame(index=ELEMENTS, columns=GWS, data=1.0),
        players=players,
        squad_before=[4],
        deadline=dt.datetime.now(dt.UTC) + dt.timedelta(days=2),
        options=options,
        objective="points",
        free_transfers=1,
    )
    report = format_report(recommendation)
    lines = report.splitlines()
    assert lines[3].startswith("OPTIONS")
    assert "Robust" in lines[4]
    assert "Wissa -> Saka; (C) Saka" in report
    assert "Wissa -> Isak" in report
    assert "3*" in report and "Roll — 2 free transfers next week" in report
    assert "Option #1: Wissa -> Saka" in report
    assert "Option #3 (reference)" in report
    assert "TRANSFER" not in report
