"""Tests for the minutes model's availability adjustment.

The adjustment is small but load-bearing: it is the only place live injury news enters the
projection, because the historical dataset records no per-gameweek availability at all.
"""

from __future__ import annotations

import pandas as pd

from fplass.features.minutes import apply_availability, calibrate_to_lineup


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


def test_lineup_calibration_hits_eleven_and_three():
    """Each club must field eleven starters and three substitutes, not seventeen or seven."""
    # Two clubs with very different squad sizes and confidence spreads. Both are large enough
    # to field eleven starters and three substitutes, as every real Premier League squad is.
    rows = [[0.2, 0.2, 0.6]] * 34 + [[0.7, 0.15, 0.15]] * 22
    groups = pd.Series(["a"] * 34 + ["b"] * 22)
    after = calibrate_to_lineup(predictions(rows), groups)

    for club in ("a", "b"):
        mask = groups == club
        assert abs(after.loc[mask.values, "p_full"].sum() - 11.0) < 0.6
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
