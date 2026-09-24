"""Tests for the per-90 rate table.

The defect these pin: a rate's numerator only covers the seasons that published the stat, so its
denominator must too. DEFCON counts exist for 2025-26 (and reconstructed 2016-19) but not 2019-25;
dividing last season's counts by ten seasons of minutes halved every veteran's rate, put the
model's chance of a defender hitting the DEFCON threshold at 0.12 against an observed 0.27, and
cost defenders roughly half a point a match in the projections for the first two gameweeks of
2026/27.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fplass.features import rates


@pytest.fixture(scope="module")
def rate_table(con):
    table, _ = rates.build(con)
    return table


def _actual_per_90(con, season: str, position: str, min_minutes: int = 900) -> pd.Series:
    """Each player's observed DEFCON per 90 in one season, indexed by code."""
    return con.execute(
        """
        SELECT pl.code, sum(p.defcon_count) * 90.0 / sum(p.minutes) AS per90
        FROM player_gw_derived p
        JOIN players pl ON pl.season = p.season AND pl.element = p.element
        WHERE p.season = ? AND p.position = ?
        GROUP BY pl.code
        HAVING sum(p.minutes) >= ?
        """,
        [season, position, min_minutes],
    ).fetchdf().set_index("code")["per90"]


def test_exposure_only_counts_seasons_that_published_the_stat():
    """Synthetic totals: one season with DEFCON, one without, equal minutes."""
    frame = pd.DataFrame(
        {
            "code": [1, 1],
            "season": ["2024-25", "2025-26"],
            "gw_seq": [1, 1],
            "position": ["DEF", "DEF"],
            "minutes": [900.0, 900.0],
            "prev_value": [50, 50],
            "goals_scored": [1, 1],
            "assists": [0, 0],
            "defcon_count": [np.nan, 90.0],
            "saves": [0, 0],
            "yellow_cards": [0, 0],
            "bonus": [0, 0],
            "expected_goals": [0.5, 0.5],
            "expected_assists": [0.1, 0.1],
            "expected_goals_conceded": [10.0, 10.0],
        }
    )
    # Drive player_totals' aggregation through a stand-in connection returning this frame.

    class Stub:
        def execute(self, *_args, **_kwargs):
            return self

        def fetchdf(self):
            return frame.copy()

    totals = rates.player_totals(Stub())
    row = totals.iloc[0]
    exposure = row[rates.exposure_column("defcon_count")]
    # Only the 2025-26 minutes count toward DEFCON exposure: 900 at full weight.
    assert exposure == pytest.approx(900.0)
    # Goals were observed in both seasons, so their exposure is the recency-weighted total.
    assert row[rates.exposure_column("goals_scored")] > 900.0


def test_defender_defcon_rate_matches_last_season(con, rate_table):
    """League-wide, the modelled per-90 must sit near what defenders actually produced."""
    actual = _actual_per_90(con, "2025-26", "DEF")
    if actual.empty:
        pytest.skip("no 2025-26 DEFCON data")
    modelled = rate_table.set_index("code")["defcon_rate"].reindex(actual.index).dropna()
    assert abs(modelled.mean() / actual.mean() - 1) < 0.10, (
        f"modelled DEF DEFCON {modelled.mean():.2f}/90 vs actual {actual.mean():.2f}/90"
    )


def test_veteran_defcon_rate_is_not_diluted_by_seasons_without_the_stat(con, rate_table):
    """Virgil has ~27,000 Premier League minutes, most in seasons with no DEFCON published."""
    code = con.execute(
        "SELECT code FROM players WHERE season = '2026-27' AND web_name = 'Virgil'"
    ).fetchone()
    if code is None:
        pytest.skip("Virgil not in this season's player pool")
    actual = _actual_per_90(con, "2025-26", "DEF")
    if code[0] not in actual.index:
        pytest.skip("Virgil did not play 900 minutes in 2025-26")
    modelled = rate_table.set_index("code").at[code[0], "defcon_rate"]
    assert abs(modelled / actual[code[0]] - 1) < 0.15, (
        f"Virgil modelled {modelled:.2f}/90 vs 2025-26 actual {actual[code[0]]:.2f}/90"
    )


def _one_forward(**overrides) -> pd.DataFrame:
    row = {
        "code": 1, "position": "FWD", "price_tier": "premium", "minutes": 9000.0,
        "raw_minutes": 9000.0, "thin_sample": False, "goals_scored": 50.0, "assists": 30.0,
        "defcon_count": 0.0, "saves": 0.0, "yellow_cards": 0.0, "bonus": 0.0,
        "expected_goals": 40.0, "expected_assists": 10.0, "expected_goals_conceded": 0.0,
    }
    row.update(overrides)
    return pd.DataFrame([row])


FLAT = rates.RatePriors(
    fallback={r: (1e-6, 1e-6) for r in (*rates.COUNT_RATES, *rates.EXPECTED_RATES)}
)


def test_goal_and_assist_rates_blend_expected_stats_by_their_own_weights():
    """A hundred nineties and a negligible prior: each rate is its own count per 90."""
    out = rates.shrink(_one_forward(), FLAT).iloc[0]
    g, a = rates.XG_GOAL_WEIGHT, rates.XA_ASSIST_WEIGHT
    assert out["goal_rate"] == pytest.approx(g * 0.40 + (1 - g) * 0.50, rel=1e-4)
    assert out["assist_rate"] == pytest.approx(a * 0.10 + (1 - a) * 0.30, rel=1e-4)


def test_expected_stats_are_converted_into_fpl_units_by_position_before_blending():
    """Forwards earn about two FPL assists per unit of xA; the blend must see that, not raw xA."""
    conversion = {"xg": {"FWD": 1.0}, "xa": {"FWD": 2.0}}
    out = rates.shrink(_one_forward(), FLAT, conversion=conversion).iloc[0]
    a = rates.XA_ASSIST_WEIGHT
    assert out["assist_rate"] == pytest.approx(a * 0.20 + (1 - a) * 0.30, rel=1e-4)


def test_expected_stats_before_fpl_published_them_are_not_zeros(con):
    """2022-23 stored 0.0 for gameweeks 1-15; read as data, midfielders beat their xG by half."""
    conversion = rates.measure_expected_conversion(con, "2022-23")
    if conversion is None:
        pytest.skip("2022-23 not in the warehouse")
    assert 0.85 < conversion["xg"]["MID"] < 1.15
    assert conversion["xa"]["FWD"] > 1.7
    assert conversion["xg"]["DEF"] < 1.0
