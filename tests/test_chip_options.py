"""Tests for chip timing as an exercise problem, on synthetic seasons with known answers."""

from __future__ import annotations

import pandas as pd

from fplass.optimise import milp
from fplass.options import chips


def _gains() -> pd.DataFrame:
    rows = []
    for season, peak in (("A", 5), ("B", 8), ("C", 3)):
        for gw in range(1, 11):
            rows.append(
                {"season": season, "gameweek": gw, "bboost": 10.0 + (15.0 if gw == peak else 0.0),
                 "3xc": 5.0, "freehit": 6.0}
            )
    return pd.DataFrame(rows)


def test_continuation_values_are_the_best_later_gain_averaged_over_seasons():
    table = chips.continuation_values(_gains(), "bboost", (1, 10))
    by_left = dict(zip(table["weeks_left"], table["mean"], strict=True))
    # With nine weeks left every season still has its peak ahead.
    assert by_left[9] == 25.0
    # With one week left only the season whose peak is the last week... none: 10 everywhere.
    assert by_left[1] == 10.0


def test_the_continuation_rule_waits_for_the_peak_and_the_floor_rule_can_fire_early():
    gains = _gains()
    table = chips.evaluate_rules(gains, "bboost", (1, 10), floor=9.0)
    assert (table["floor_week"] == 1).all(), "a floor below every week fires immediately"
    assert (table["floor_gain"] == 10.0).all()
    # Holding out the season being decided, the others' peaks are at 3 and 8 (for A), so the
    # rule waits until a week beats the expected best later gain.
    a = table[table["season"] == "A"].iloc[0]
    assert a["continuation_week"] == 5 and a["continuation_gain"] == 25.0
    assert (table["hindsight_gain"] == 25.0).all()
    assert table["continuation_gain"].mean() >= table["floor_gain"].mean()


def test_best_eleven_is_legal():
    positions = {**{e: "GKP" for e in (1, 2)}, **{e: "DEF" for e in range(3, 8)},
                 **{e: "MID" for e in range(8, 13)}, **{e: "FWD" for e in (13, 14, 15)}}
    expected = pd.Series({e: float(e) for e in positions})
    eleven = chips.best_eleven(expected, list(positions), positions)
    assert len(eleven) == milp.LINEUP_SIZE
    counts = pd.Series([positions[e] for e in eleven]).value_counts()
    assert counts["GKP"] == 1 and counts["DEF"] >= 3 and counts["MID"] >= 2 and counts["FWD"] >= 1
    assert 2 in eleven and 1 not in eleven, "the better keeper starts"


def test_thresholds_fall_as_the_window_closes_and_leave_out_the_season():
    from fplass.optimise import chips as roadmap

    windows = milp.ChipWindows(windows={"bboost": [(1, 10)], "3xc": [(1, 10)], "freehit": [(1, 10)],
                                        "wildcard": [(1, 10)]})
    thresholds = chips.ContinuationThresholds.from_gains(_gains(), windows, exclude_season="A")
    assert thresholds.floor_for("bboost", 1) == 25.0, "both other seasons' peaks lie ahead"
    assert thresholds.floor_for("bboost", 9) == 10.0
    assert thresholds.floor_for("bboost", 10) == 0.0, "last week: play for any gain"
    assert thresholds.floor_for("wildcard", 5) == roadmap.DEFAULT_MIN_GAIN["wildcard"]
    assert thresholds.floor_for("bboost", 11) == roadmap.DEFAULT_MIN_GAIN["bboost"], "outside the window"


def test_roadmap_accepts_a_threshold_function(con):
    import numpy as np
    import pandas as pd

    from fplass.optimise import chips as roadmap
    from fplass.optimise.milp import ChipWindows, SquadState, solve

    rows = []
    element = 1
    rng = np.random.default_rng(2)
    for club in range(1, 9):
        for position, count in (("GKP", 2), ("DEF", 3), ("MID", 3), ("FWD", 2)):
            for _ in range(count):
                rows.append({"element": element, "position": position, "team_id": club,
                             "price": int(rng.integers(40, 90)), "web_name": f"p{element}"})
                element += 1
    universe = pd.DataFrame(rows)
    gameweeks = [1, 2, 3]
    base = universe.set_index("element")["price"] / 20.0
    points = pd.DataFrame({gw: base for gw in gameweeks}, index=base.index)
    windows = ChipWindows(windows={"wildcard": [(2, 19)], "freehit": [(2, 19)],
                                   "bboost": [(1, 19)], "3xc": [(1, 19)]})
    state = SquadState(players={}, bank=1000, free_transfers=15)
    plan = solve(points, universe, state, windows, gameweeks=gameweeks, allow_chips=False, time_limit=20)
    samples = points.to_numpy()[None, :, :]
    never = roadmap.build_roadmap(con, samples, points.index.to_numpy(), np.array(gameweeks), universe,
                                  state, plan, windows, "2026-27", min_gain=lambda chip, gw: 1e9,
                                  wildcard_candidates=0)
    assert never.schedule == {}
    always = roadmap.build_roadmap(con, samples, points.index.to_numpy(), np.array(gameweeks), universe,
                                   state, plan, windows, "2026-27", min_gain=lambda chip, gw: 0.0,
                                   wildcard_candidates=0)
    assert "3xc" in always.schedule.values()


def test_the_default_policy_reads_the_measured_thresholds(con):
    from fplass.optimise import chips as roadmap
    from fplass.optimise.policy import PolicyConfig

    windows = milp.ChipWindows.from_warehouse(con, "2026-27")
    config = PolicyConfig()
    assert config.chip_rule == "continuation"
    floor = config.thresholds(windows, "2026-27")
    assert callable(floor)
    assert floor("3xc", 38) == 0.0, "the last week of the window plays the chip for any gain"
    assert floor("3xc", 21) > roadmap.DEFAULT_MIN_GAIN["3xc"], "early in the second half it waits"
    assert floor("wildcard", 5) == roadmap.DEFAULT_MIN_GAIN["wildcard"], "the wildcard keeps its floor"
    assert "chips=continuation" in config.tag()

    missing = PolicyConfig.parse("chip_gains=/nowhere/gains.csv")
    assert missing.thresholds(windows, "2026-27") is None, "no file: flat floors"
