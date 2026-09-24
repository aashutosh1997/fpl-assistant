"""Tests for the minutes model's availability adjustment.

The adjustment is small but load-bearing: it is the only place live injury news enters the
projection, because the historical dataset records no per-gameweek availability at all.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fplass.features.minutes import (
    FULL_APPEARANCES_PER_TEAM,
    apply_availability,
    availability_by_gameweek,
    calibrate_to_lineup,
    return_date,
)


def predictions(rows):
    return pd.DataFrame(rows, columns=["p_none", "p_cameo", "p_full"])


def test_no_news_leaves_the_prediction_alone():
    """A ceiling of 100% says nothing about whether a player starts.

    This is the bug that promoted a backup goalkeeper to captain: treating the ceiling as a
    target forced every unflagged player to a 100% chance of playing, erasing squad depth.
    """
    before = predictions([[0.70, 0.10, 0.20], [0.05, 0.10, 0.85]])
    after = apply_availability(
        before,
        status=pd.Series(["a", "a"]),
        chance_of_playing=pd.Series([None, None]),
    )
    pd.testing.assert_series_equal(after["p_full"], before["p_full"], check_names=False)
    pd.testing.assert_series_equal(after["p_cameo"], before["p_cameo"], check_names=False)


def test_a_flagged_player_is_scaled_down():
    before = predictions([[0.10, 0.10, 0.80]])
    after = apply_availability(
        before, status=pd.Series(["d"]), chance_of_playing=pd.Series([50])
    )
    assert after["p_cameo"].iloc[0] + after["p_full"].iloc[0] == pytest_approx(0.50)
    # Returning from a knock means a shorter outing, so weight shifts toward a cameo.
    assert after["p_full"].iloc[0] < before["p_full"].iloc[0] * 0.5 / 0.9 + 1e-9


def test_ceiling_never_raises_a_low_prediction():
    """FPL saying 75% must not lift a player the model rates at 40%."""
    before = predictions([[0.60, 0.10, 0.30]])
    after = apply_availability(
        before, status=pd.Series(["d"]), chance_of_playing=pd.Series([75])
    )
    assert after["p_cameo"].iloc[0] + after["p_full"].iloc[0] <= 0.40 + 1e-9


def test_ruled_out_players_cannot_play():
    before = predictions([[0.05, 0.10, 0.85]] * 4)
    after = apply_availability(
        before,
        status=pd.Series(["i", "s", "u", "n"]),
        chance_of_playing=pd.Series([None] * 4),
    )
    assert (after["p_none"] > 0.999).all()


def test_lineup_calibration_hits_the_measured_lineup():
    """Each club must field about ten full appearances and three substitutes, not 17 or 7.

    The target for the sixty-minute class is 10.3 rather than eleven: one starter in fourteen is
    withdrawn before the hour, measured at 10.28-10.33 per team-match in every season 2022-26.
    Forcing eleven predicted 220 full appearances in each of the first two 2026/27 gameweeks
    against 210 and 209 actual.
    """
    assert 10.2 <= FULL_APPEARANCES_PER_TEAM <= 10.4
    # Two clubs with very different squad sizes and confidence spreads. Both are large enough
    # to field eleven starters and three substitutes, as every real Premier League squad is.
    rows = [[0.2, 0.2, 0.6]] * 34 + [[0.7, 0.15, 0.15]] * 22
    groups = pd.Series(["a"] * 34 + ["b"] * 22)
    after = calibrate_to_lineup(predictions(rows), groups)

    for club in ("a", "b"):
        mask = groups == club
        assert abs(after.loc[mask.values, "p_full"].sum() - FULL_APPEARANCES_PER_TEAM) < 0.05
        assert abs(after.loc[mask.values, "p_cameo"].sum() - 3.0) < 0.4

    # Still a valid probability distribution.
    total = after["p_none"] + after["p_cameo"] + after["p_full"]
    assert (total.sub(1.0).abs() < 1e-9).all()


def pytest_approx(value, tol=1e-6):
    class _Approx:
        def __eq__(self, other):
            return abs(other - value) < tol

        def __repr__(self):
            return f"~{value}"

    return _Approx()


def _horizon_rows():
    return pd.DataFrame({"element": [1] * 3 + [2] * 3 + [3] * 3 + [4] * 3 + [5] * 3,
                         "event": [6, 7, 8] * 5})


DEADLINES = {6: pd.Timestamp("2026-10-10 10:00"), 7: pd.Timestamp("2026-10-17 10:00"),
             8: pd.Timestamp("2026-10-23 17:30")}


def test_news_dates_resolve_to_the_right_year():
    assert return_date("Hamstring injury - Expected back 11 Oct", "2026-27") == pd.Timestamp("2026-10-11")
    assert return_date("Suspended until 03 Jan", "2026-27") == pd.Timestamp("2027-01-03")
    assert return_date("Knee injury - 75% chance of playing", "2026-27") is None
    assert return_date("Unspecified injury - Unknown return date", "2026-27") is None


def test_flags_apply_to_the_gameweeks_they_describe():
    """Pinned against GW6: Palmer's 75% doubt held him at a reduced ceiling through GW13."""
    availability = pd.DataFrame({
        "element": [1, 2, 3, 4, 5],
        "status": ["d", "i", "s", "i", "u"],
        "chance_of_playing_next_round": [75.0, 0.0, 0.0, 0.0, 0.0],
        "news": ["Muscular injury - 75% chance of playing",
                 "Foot injury - Expected back 11 Oct",
                 "Suspended until 17 Oct",
                 "Unspecified injury - Unknown return date",
                 "Has joined Stoke City permanently"],
    })
    flags = availability_by_gameweek(availability, _horizon_rows(), DEADLINES, "2026-27")
    status = flags["status"].to_numpy().reshape(5, 3)
    chance = flags["chance_of_playing_next_round"].to_numpy().reshape(5, 3)
    # A doubt caps the next gameweek only.
    assert list(status[0]) == ["d", "a", "a"] and chance[0, 0] == 75.0 and np.isnan(chance[0, 1])
    # Dated absences lift on their date: back 11 Oct misses GW6 (10 Oct) and plays GW7.
    assert list(status[1]) == ["i", "a", "a"]
    assert list(status[2]) == ["s", "a", "a"]
    # No date, or gone: out for the whole horizon.
    assert list(status[3]) == ["i", "i", "i"]
    assert list(status[4]) == ["u", "u", "u"]
