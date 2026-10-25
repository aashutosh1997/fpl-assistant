"""Tests for the spread of the simulated defensive-action count.

Pinned against the GW7 review of 2026-27: replayed over the season's deadlines, the engine paid
midfielders 1.57 times the DEFCON points they earned, because a hand-set dispersion of 4 drew the
count nearly three times as spread as it is and the payoff sits at a threshold above the mean.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import nbinom

from fplass.sim import engine


def test_a_well_known_player_sits_near_the_match_dispersion():
    k = engine.defcon_dispersion(np.array(["MID"]), np.array([1e6]))[0]
    assert k == pytest.approx(engine.DEFCON_DISPERSION["MID"], rel=1e-3)


def test_an_uncertain_rate_widens_the_count():
    known, unknown = engine.defcon_dispersion(np.array(["MID", "MID"]), np.array([200.0, 10.0]))
    assert unknown < known < engine.DEFCON_DISPERSION["MID"]


def test_without_a_shape_only_the_match_dispersion_applies():
    k = engine.defcon_dispersion(np.array(["DEF", "MID"]), None)
    assert k.tolist() == [engine.DEFCON_DISPERSION["DEF"], engine.DEFCON_DISPERSION["MID"]]


def test_the_measured_spread_clears_the_bar_less_often_than_the_old_one():
    """A midfielder averaging nine actions: the threshold is twelve."""
    mean, threshold = 9.0, engine.DEFCON_THRESHOLDS["MID"]

    def p_hit(k: float) -> float:
        return float(1 - nbinom.cdf(threshold - 1, k, k / (k + mean)))

    k = engine.defcon_dispersion(np.array(["MID"]), np.array([150.0]))[0]
    assert p_hit(k) < p_hit(4.0)
    # And the sampler draws that distribution.
    rng = np.random.default_rng(7)
    draws = engine._negative_binomial(np.full(200_000, mean), rng, dispersion=np.full(200_000, k))
    assert np.mean(draws >= threshold) == pytest.approx(p_hit(k), abs=0.005)
