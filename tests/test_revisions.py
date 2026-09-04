"""Tests for the revision process: the volatility of the projection between deadlines."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fplass.backtest import manager
from fplass.options import revisions as rv


@pytest.fixture(scope="module")
def table(con):
    stored = con.execute("SELECT count(*) FROM projection_panel").fetchone()[0]
    if stored:
        frame = rv.revisions(con)
    else:
        sources = sorted(manager.PANEL.glob("*.parquet"))
        if not sources:
            pytest.skip("no projection panel to measure revisions on")
        frame = rv.revisions(con, sources=sources[:1])
    if frame.empty:
        pytest.skip("panel has no consecutive deadlines")
    return frame


def test_revisions_pair_consecutive_deadlines_on_common_targets(table):
    assert (table["weeks"] >= 1).all() and (table["weeks"] <= 7).all()
    assert (table["next_gw"] > table["as_of_gw"]).all()
    assert set(table["certainty"].unique()) <= set(rv.CERTAINTY_LABELS)
    # The projection is not systematically revised in one direction.
    assert abs(table["rel"].mean()) < 0.05
    # Nailed starters move less than fringe players.
    spread = table.groupby("certainty", observed=True)["rel"].std()
    assert spread["nailed"] < spread["fringe"]


def test_sampler_scales_each_player_by_a_drawn_revision(table):
    sampler = rv.RevisionSampler.fit(table)
    assert sampler.fallback, "position pools always exist"
    expected = pd.DataFrame({5: [4.0, 1.0, 0.0], 6: [4.0, 1.0, 0.0]}, index=[1, 2, 3])
    p_full = pd.Series({1: 0.95, 2: 0.4, 3: 0.05})
    players = pd.DataFrame(
        {"element": [1, 2, 3], "position": ["MID", "DEF", "GKP"], "price": [120, 45, 40]}
    )
    draws = sampler.sample(expected, p_full, players, draws=200, rng=np.random.default_rng(1))
    assert draws.shape == (200, 3, 2)
    assert (draws >= 0).all()
    assert np.allclose(draws[:, 2, :], 0.0), "nothing projected stays nothing"
    # Both weeks of a player move together: one revision per player, not per week.
    ratio = draws[:, 0, 0] / draws[:, 0, 1]
    assert np.allclose(ratio, 1.0)
    assert draws[:, 0, 0].std() > 0, "and it does move"
    assert abs(draws[:, 0, 0].mean() - 4.0) < 0.5


def test_summary_and_stability_tables_have_the_keys(table):
    summary = rv.summarise(table)
    assert {"position", "premium", "certainty", "sd_rel", "p_jump"} <= set(summary.columns)
    if table["season"].nunique() >= 2:
        stable = rv.stability(table)
        assert "ratio" in stable.columns
