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
