"""Tests for minutes uncertainty from new signings.

Pinned against the 2026 summer deadline: Manchester City signed Enzo, Ndiaye and Allan in the
days after gameweek 2, and nothing in the data could say whether Foden, Cherki or Semenyo would
keep starting. The model must not pretend to know — it widens them, and the optimiser prices it.
"""

from __future__ import annotations

import pandas as pd
import pytest

from fplass.features import arrivals


def _players():
    return pd.DataFrame(
        {
            "element": [1, 2, 3, 4, 5],
            "team_id": [1, 1, 1, 1, 2],
            "position": ["MID", "MID", "MID", "FWD", "MID"],
            "price": [85, 70, 69, 155, 80],
            "web_name": ["incumbent_a", "incumbent_b", "arrival", "striker", "elsewhere"],
        }
    )


def test_only_the_competing_group_is_widened():
    signing = arrivals.Arrival(3, "arrival", 1, "MID", 69, "transfer from X")
    weight = arrivals.disruption(_players(), [signing])
    assert weight[0] > 0 and weight[1] > 0, "midfielders at the signing's club are disrupted"
    assert weight[2] > 0, "the arrival is uncertain too: his minutes were earned elsewhere"
    assert weight[3] == 0, "a forward is not threatened by a midfielder"
    assert weight[4] == 0, "another club is untouched"


def test_a_cheap_squad_filler_threatens_nobody():
    filler = arrivals.Arrival(3, "arrival", 1, "MID", 45, "new registration")
    weight = arrivals.disruption(_players(), [filler])
    assert weight[0] == 0 and weight[1] == 0


def test_price_compression_at_the_bottom_of_the_scale():
    """A 4.5m backup is three quarters of a 5.9m starter's price but not a threat to him."""
    assert not arrivals.competes(45, 59)
    assert arrivals.competes(45, 45), "he does compete with someone priced the same"
    assert arrivals.competes(50, 49)
    assert arrivals.competes(69, 85), "a 6.9m midfielder threatens an 8.5m one"
    assert not arrivals.competes(50, 85)


def test_widening_preserves_the_group_total_and_spreads_it():
    signing = arrivals.Arrival(3, "arrival", 1, "MID", 69, "transfer from X")
    players = _players()
    probabilities = pd.DataFrame(
        {
            "p_none": [0.02, 0.05, 0.6, 0.02, 0.1],
            "p_cameo": [0.03, 0.05, 0.2, 0.03, 0.1],
            "p_full": [0.95, 0.90, 0.20, 0.95, 0.8],
        }
    )
    weight = arrivals.disruption(players, [signing])
    widened = arrivals.widen(probabilities, players, weight)

    group = [0, 1, 2]
    assert widened.loc[group, "p_full"].sum() == pytest.approx(
        probabilities.loc[group, "p_full"].sum()
    )
    assert widened.at[0, "p_full"] < 0.95
    assert widened.at[2, "p_full"] > 0.20
    assert widened.at[0, "p_full"] > widened.at[2, "p_full"], "ordering is kept"
    total = widened[["p_none", "p_cameo", "p_full"]].sum(axis=1)
    assert (total.sub(1.0).abs() < 1e-9).all()
    assert widened.at[3, "p_full"] == 0.95 and widened.at[4, "p_full"] == 0.8


def test_city_deadline_signings_are_detected(con):
    """Enzo moved from Chelsea after gameweek 2; the warehouse must show it without being told."""
    played = con.execute(
        "SELECT count(DISTINCT gw) FROM player_gw WHERE season = '2026-27'"
    ).fetchone()[0]
    if played < 2:
        pytest.skip("gameweek 2 not ingested")
    found = {a.web_name: a for a in arrivals.detect_arrivals(con, "2026-27")}
    if "Enzo" not in found:
        pytest.skip("player pool predates the 2026 deadline day")
    assert found["Enzo"].origin == "transfer from CHE"
    assert found["Enzo"].position == "MID"

    from fplass.sim.project import current_players

    players = current_players(con, "2026-27")
    weight = arrivals.disruption(players, list(found.values()))
    by_name = pd.Series(weight.to_numpy(), index=players["web_name"].to_numpy())
    assert by_name["Foden"] > 0 and by_name["Cherki"] > 0 and by_name["Semenyo"] > 0
    assert by_name["Haaland"] == 0
