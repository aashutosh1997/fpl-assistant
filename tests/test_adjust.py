"""Tests for the minutes recalibration layer.

The named cases here are the actual gameweek 1 failures. Each one cost real points, and each is
pinned so it cannot silently return: the layer exists because the base model rated Joao Pedro 0.69
to play an hour (he played ninety and scored eleven) and Dubravka 0.93 (he played none).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fplass.features import adjust
from fplass.ingest import preseason


# ------------------------------------------------------------------ unit behaviour


def _frame(n: int, **columns) -> pd.DataFrame:
    base = {
        "preseason_minutes_avg": np.zeros(n),
        "preseason_observed": np.zeros(n),
        "log_ownership": np.zeros(n),
    }
    base.update(columns)
    return pd.DataFrame(base)


def test_refuses_to_fit_on_too_little_data():
    """Four coefficients on a handful of rows would look confident and mean nothing."""
    n = 50
    rng = np.random.default_rng(0)
    p = rng.uniform(0.1, 0.9, n)
    assert adjust.fit(_frame(n), p, (rng.random(n) < p).astype(float)) is None


def test_passthrough_when_features_are_uninformative():
    """With no signal to add, the layer must leave the base model roughly where it was.

    This is the property that makes it safe to always enable: a layer that collapsed predictions
    toward the base rate whenever it learned nothing would be worse than no layer at all.
    """
    rng = np.random.default_rng(1)
    n = 4000
    p = rng.uniform(0.05, 0.95, n)
    outcome = (rng.random(n) < p).astype(float)

    layer = adjust.fit(_frame(n), p, outcome)
    assert layer is not None
    adjusted = layer.apply(_frame(n), p)
    # Should stay close to the (already well-calibrated) input rather than flattening it.
    assert np.corrcoef(adjusted, p)[0, 1] > 0.95
    assert abs(adjusted.mean() - p.mean()) < 0.05


def test_learns_a_feature_the_base_model_lacks():
    """When the outcome depends on preseason minutes and the base model is blind to them,
    the layer must recover that dependence."""
    rng = np.random.default_rng(2)
    n = 4000
    preseason_minutes = rng.uniform(0, 90, n)
    # Truth depends on preseason; the base model is a coin flip and knows nothing.
    truth = 1 / (1 + np.exp(-(preseason_minutes - 45) / 12))
    outcome = (rng.random(n) < truth).astype(float)
    base = np.full(n, 0.5)

    frame = _frame(n, preseason_minutes_avg=preseason_minutes, preseason_observed=np.ones(n))
    layer = adjust.fit(frame, base, outcome)
    assert layer is not None
    assert layer.coefficients["preseason_minutes_avg"] > 0
    assert layer.improvement > 0.3, "should substantially beat an uninformative base model"

    adjusted = layer.apply(frame, base)
    assert adjusted[preseason_minutes > 80].mean() > adjusted[preseason_minutes < 10].mean() + 0.3


def test_shift_is_bounded():
    """Fitted on one gameweek, the layer may correct a bias but must not overrule the base model."""
    rng = np.random.default_rng(3)
    n = 2000
    base = np.full(n, 0.5)
    frame = _frame(n, preseason_minutes_avg=np.full(n, 90.0), preseason_observed=np.ones(n))
    layer = adjust.fit(frame, base, np.ones(n))
    assert layer is not None
    adjusted = layer.apply(frame, base)
    ceiling = 1 / (1 + np.exp(-(0.0 + adjust.MAX_SHIFT)))
    assert adjusted.max() <= ceiling + 1e-6


def test_correction_fades_with_current_season_evidence():
    """The layer stands in for evidence the base model lacks in August, and must step aside once
    the base model has real minutes.

    Applied at full strength to gameweek 2, the gameweek-1 layer made the base model *worse*
    (Brier 0.092 -> 0.119); scaled by 1/(1+n)^2 it was a footnote (0.096).
    """
    weights = adjust.evidence_weight(np.array([0, 1, 2, 3, 10]))
    assert weights[0] == 1.0
    assert weights[1] == pytest.approx(0.25)
    assert weights[3] < 0.07
    assert np.all(np.diff(weights) < 0)

    n = 2000
    base = np.full(n, 0.5)
    frame = _frame(n, preseason_minutes_avg=np.full(n, 90.0), preseason_observed=np.ones(n))
    layer = adjust.fit(frame, base, np.ones(n))
    assert layer is not None

    fresh = layer.apply(frame.assign(n_current=0.0), base)
    one_match = layer.apply(frame.assign(n_current=1.0), base)
    settled = layer.apply(frame.assign(n_current=5.0), base)
    assert fresh.mean() > one_match.mean() > settled.mean()
    assert abs(settled.mean() - 0.5) < 0.05, "after five matches the base model should stand"
    # Without the column the layer applies in full, so existing callers are unchanged.
    assert layer.apply(frame, base).mean() == pytest.approx(fresh.mean())


def test_probabilities_stay_in_range():
    rng = np.random.default_rng(4)
    n = 3000
    p = rng.uniform(0.01, 0.99, n)
    frame = _frame(
        n,
        preseason_minutes_avg=rng.uniform(0, 90, n),
        preseason_observed=rng.integers(0, 2, n).astype(float),
        log_ownership=rng.uniform(0, 4, n),
    )
    layer = adjust.fit(frame, p, (rng.random(n) < p).astype(float))
    adjusted = layer.apply(frame, p)
    assert adjusted.min() >= 0.0 and adjusted.max() <= 1.0


# ------------------------------------------------- against the real gameweek 1 data


@pytest.fixture(scope="module")
def preseason_features(con):
    try:
        return preseason.player_features(con, "2026-27")
    except Exception:  # pragma: no cover
        pytest.skip("preseason tables not loaded")


def test_preseason_data_is_loaded(preseason_features):
    observed = preseason_features["preseason_observed"].sum()
    if observed == 0:
        pytest.skip("preseason friendlies not ingested; run `fpl ingest preseason`")
    assert observed > 300, "expected preseason minutes for most of the player pool"


def test_preseason_separates_the_spurs_goalkeepers(con, preseason_features):
    """The single clearest gameweek 1 failure.

    The base model had Dubravka at 0.93 to play an hour and Kinsky at 0.08, reading each from his
    *previous* club. Kinsky played ninety, Dubravka played none. Preseason had it plainly: Kinsky
    67.5 minutes a game, Dubravka 22.5.
    """
    if preseason_features["preseason_observed"].sum() == 0:
        pytest.skip("preseason friendlies not ingested")

    names = con.execute(
        "SELECT element, web_name FROM players WHERE season = '2026-27'"
    ).fetchdf()
    frame = preseason_features.merge(names, on="element")
    lookup = frame.set_index("web_name")["preseason_minutes_avg"]

    for keeper in ("Kinsky", "Dubravka"):
        if keeper not in lookup.index:
            pytest.skip(f"{keeper} not in this season's player pool")

    assert lookup["Kinsky"] > lookup["Dubravka"], (
        "preseason minutes should rank Kinsky ahead of Dubravka, which is what actually happened"
    )


def test_joao_pedro_preseason_marks_him_as_nailed(con, preseason_features):
    """He played 80 minutes across four friendlies and scored seven. The model said 0.69."""
    if preseason_features["preseason_observed"].sum() == 0:
        pytest.skip("preseason friendlies not ingested")

    names = con.execute(
        "SELECT element, web_name FROM players WHERE season = '2026-27'"
    ).fetchdf()
    frame = preseason_features.merge(names, on="element")
    row = frame[frame["web_name"] == "João Pedro"]
    if row.empty:
        pytest.skip("João Pedro not in this season's player pool")

    assert row["preseason_minutes_avg"].iloc[0] > 70
    assert row["preseason_matches"].iloc[0] >= 3


def test_layer_improves_gameweek_one(con):
    """End to end: the layer must materially beat the base model on the gameweek it was built for.

    In-sample by construction — gameweek 1 is the only data the layer has — so this asserts the
    machinery is wired up, not that the gain generalises. Gameweek 2 is the honest test.
    """
    from fplass.backtest import calibrate_live

    report = calibrate_live.run(con, upto_gw=99, fit_adjustment=True)
    if report is None or report.adjustment is None:
        pytest.skip("no stored projections to calibrate against")

    assert report.adjustment.improvement > 0.15, (
        f"layer only improved Brier by {report.adjustment.improvement:.1%}"
    )
    assert report.adjustment.coefficients["log_ownership"] > 0, (
        "higher ownership should raise the probability of playing"
    )
