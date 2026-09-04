"""Tests for the order-flow layer.

Two facts carry the whole module and both are pinned here on the real warehouse: that a
gameweek's transfer counts are the flow *before* its deadline, and that among players the model
would call certain starters, the ones their owners sold are the ones who then did not play.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fplass.features import flow, minutes


@pytest.fixture(scope="module")
def recent(con) -> list[str]:
    seasons = [
        r[0]
        for r in con.execute(
            "SELECT season FROM player_gw WHERE selected IS NOT NULL AND season >= '2019-20' "
            "GROUP BY season HAVING count(DISTINCT gw) >= 37 ORDER BY season"
        ).fetchall()
    ]
    if len(seasons) < 3:
        pytest.skip("not enough seasons with ownership data")
    return seasons


def test_transfer_counts_are_the_flow_before_the_deadline(con, recent):
    """Ownership change tracks the same gameweek's net transfers, not the previous week's."""
    table = con.execute(
        """
        WITH weekly AS (
            SELECT season, element, gw, any_value(selected) AS owners,
                   any_value(transfers_in) - any_value(transfers_out) AS net
            FROM player_gw WHERE season >= '2019-20' GROUP BY 1, 2, 3
        ),
        lagged AS (
            SELECT *, lag(owners) OVER w AS prev_owners, lag(net) OVER w AS prev_net
            FROM weekly WINDOW w AS (PARTITION BY season, element ORDER BY gw)
        )
        SELECT season, corr(owners - prev_owners, net) AS same_week,
               corr(owners - prev_owners, prev_net) AS previous_week
        FROM lagged WHERE prev_owners IS NOT NULL AND prev_net IS NOT NULL
        GROUP BY season ORDER BY season
        """
    ).fetchdf()
    for row in table.itertuples():
        assert row.same_week >= 0.85, f"{row.season}: {row.same_week:.2f}"
        assert row.same_week > row.previous_week + 0.4


def test_owners_selling_a_certain_starter_predicts_his_absence(con, recent):
    frame = con.execute(
        """
        WITH weekly AS (
            SELECT season, element, gw, any_value(transfers_out) AS tout,
                   any_value(selected) AS owners, sum(minutes) AS minutes
            FROM player_gw WHERE season >= '2019-20' GROUP BY 1, 2, 3
        ),
        lagged AS (
            SELECT *, lag(owners) OVER w AS prev_owners,
                   avg(minutes) OVER (w ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING) AS roll5,
                   lag(minutes) OVER w AS minutes_last
            FROM weekly WINDOW w AS (PARTITION BY season, element ORDER BY gw)
        )
        SELECT minutes, tout * 1.0 / prev_owners AS out_ratio
        FROM lagged WHERE roll5 >= 75 AND minutes_last >= 60 AND prev_owners >= 20000
        """
    ).fetchdf().dropna()
    frame["absent"] = (frame["minutes"] == 0).astype(float)
    top = frame[frame["out_ratio"] >= frame["out_ratio"].quantile(0.9)]
    assert top["absent"].mean() > 2.0 * frame["absent"].mean()


def test_flow_frame_is_null_where_there_is_no_information(con, recent):
    season = recent[-1]
    frame = flow.flow_frame(con, season)
    first = frame[frame["gw"] == frame["gw"].min()]
    assert first["flow_out"].isna().all(), "no previous owners before the first deadline"
    thin = frame[frame["previous_owners"] < flow.MIN_OWNERS]
    assert thin["flow_out"].isna().all()
    assert not frame.duplicated(["element", "gw"]).any(), "one row per player and gameweek"
    known = frame.dropna(subset=["flow_out"])
    assert (known["flow_out"] >= 0).all() and known["flow_out"].median() < 0.1


def _layer(coef_out: float, coef_in: float) -> flow.FlowLayer:
    return flow.FlowLayer(
        coefficients=pd.Series([1.0, coef_out, coef_in], index=["base_logit", "f_out", "f_in"]),
        intercept=0.0,
        n_train=1,
        seasons=("test",),
        brier_before=0.1,
        brier_after=0.1,
    )


def test_layer_moves_known_rows_only_and_keeps_classes_coherent():
    probabilities = pd.DataFrame(
        {"p_none": [0.05, 0.05, 0.6], "p_cameo": [0.05, 0.05, 0.2], "p_full": [0.9, 0.9, 0.2]}
    )
    flows = pd.DataFrame({"flow_out": [0.3, np.nan, 0.0], "flow_in": [0.0, np.nan, 0.5]})
    out = _layer(-1.0, 0.5).apply(probabilities, flows)
    assert out.at[0, "p_full"] < 0.9, "a sold starter is less certain"
    assert out.at[1, "p_full"] == 0.9, "no flow, no change"
    assert out.at[2, "p_full"] > 0.2, "a bought squad player is more likely"
    assert np.allclose(out[["p_none", "p_cameo", "p_full"]].sum(axis=1), 1.0)
    assert (out["p_cameo"] >= 0).all()
    capped = _layer(-10.0, 0.0).apply(probabilities, flows, max_shift=0.5)
    assert capped.at[0, "p_full"] > 0.8, "the cap bounds the move"


def test_layer_improves_a_held_out_season(con, recent):
    """Nested: the test season is unseen by the base fits and by the layer fit."""
    features = minutes.build_features(con)
    table = flow.evaluate(con, seasons=[recent[-1]], features=features)
    assert len(table) == 1
    row = table.iloc[0]
    assert row["brier_flow"] < row["brier_base"], table.to_string()
    assert row["starters_brier_flow"] < row["starters_brier_base"], table.to_string()
    assert row["coef_out"] < 0, "selling predicts absence"


def test_live_flow_extrapolates_a_partial_week():
    players = pd.DataFrame(
        {
            "element": [1, 2],
            "transfers_in_event": [1000, 10],
            "transfers_out_event": [20000, 1],
            "selected_by_percent": [10.0, 0.001],
            "total_players": [10_000_000, 10_000_000],
        }
    )
    half = flow.live_flow(players, elapsed=0.5)
    assert half.at[0, "flow_out"] == pytest.approx(0.04)
    assert np.isnan(half.at[1, "flow_out"]), "too few owners to read anything"
    early = flow.live_flow(players, elapsed=0.01)
    assert early.at[0, "flow_out"] == pytest.approx(0.02 / flow.MIN_ELAPSED)


def test_live_flow_is_aligned_to_the_deadline_week_only():
    matches = pd.DataFrame({"element": [1, 1, 2], "event": [4, 5, 4]})
    live = pd.DataFrame({"element": [1, 2], "flow_out": [0.1, 0.2], "flow_in": [0.0, 0.0]})
    aligned = flow.align_live(live, matches, 4)
    assert aligned["flow_out"].tolist()[0] == 0.1
    assert np.isnan(aligned["flow_out"].tolist()[1])
    assert aligned["flow_out"].tolist()[2] == 0.2
