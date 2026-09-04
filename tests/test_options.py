"""Tests for the option values: swaps, the transfer and bank options, the terminal value."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fplass.optimise import milp
from fplass.options import value


def _pool() -> pd.DataFrame:
    rows = []
    element = 1
    for club in range(1, 7):
        for position, count in (("GKP", 2), ("DEF", 3), ("MID", 3), ("FWD", 2)):
            for _ in range(count):
                rows.append(
                    {"element": element, "position": position, "team_id": club, "price": 50}
                )
                element += 1
    return pd.DataFrame(rows)


def _squad(pool: pd.DataFrame) -> milp.SquadState:
    """Fifteen legal players: clubs 1-4 held to the limit of three, clubs 5 and 6 partly."""
    def pick(club: int, position: str, n: int) -> list[int]:
        rows = pool[(pool["team_id"] == club) & (pool["position"] == position)]
        return [int(e) for e in rows["element"].iloc[:n]]

    chosen = (
        pick(1, "GKP", 1) + pick(2, "GKP", 1)
        + pick(1, "DEF", 2) + pick(2, "DEF", 2) + pick(3, "DEF", 1)
        + pick(3, "MID", 2) + pick(4, "MID", 3)
        + pick(5, "FWD", 2) + pick(6, "FWD", 1)
    )
    assert len(chosen) == 15
    return milp.SquadState(players={e: 50 for e in chosen}, bank=0, free_transfers=1)


def test_best_swap_targets_a_starter_and_respects_limits():
    pool = _pool()
    state = _squad(pool)
    horizon = pd.Series(10.0, index=pool["element"])
    owned = list(state.players)
    # One outsider midfielder from club 5 (one owned) is far better than any owned one.
    star = int(pool[(pool["team_id"] == 5) & (pool["position"] == "MID")]["element"].iloc[0])
    horizon[star] = 30.0
    swaps = value.swap_gains(horizon, pool, state, top=2)
    assert swaps and swaps[0].into == star and swaps[0].gain == pytest.approx(20.0)
    assert pool.set_index("element").at[swaps[0].out, "position"] == "MID"
    assert swaps[0].out in owned
    # An outsider from a club already held three times cannot come in, however good.
    blocked = int(pool[(pool["team_id"] == 1) & (pool["position"] == "MID")]["element"].iloc[0])
    assert blocked not in owned
    horizon[blocked] = 50.0
    swaps = value.swap_gains(horizon, pool, state, top=2)
    assert all(s.into != blocked for s in swaps)
    # A bench-for-bench upgrade scores nothing: an outsider a little better than the squad's
    # eleventh-best would not start.
    horizon = pd.Series(10.0, index=pool["element"])
    horizon[star] = 9.0
    assert value.swap_gains(horizon, pool, state, top=1) == []


def test_an_unaffordable_swap_is_worth_nothing_until_the_bank_allows_it():
    pool = _pool().copy()
    state = _squad(pool)
    star = int(pool[(pool["team_id"] == 6) & (pool["position"] == "FWD")]["element"].iloc[1])
    pool.loc[pool["element"] == star, "price"] = 55
    horizon = pd.Series(10.0, index=pool["element"])
    horizon[star] = 25.0
    assert value.swap_gains(horizon, pool, state, top=1) == []
    assert value.bank_option_value(horizon, pool, state, extra=5) == pytest.approx(15.0)
    assert value.bank_option_value(horizon, pool, state, extra=1) == 0.0


def test_transfer_option_value_is_the_hit_saved_on_the_next_swap():
    assert value.transfer_option_value([12.0, 6.0, 1.0], held=1) == 4.0
    assert value.transfer_option_value([12.0, 2.5], held=1) == 2.5
    assert value.transfer_option_value([12.0], held=1) == 0.0
    assert value.transfer_option_value([12.0, 6.0, 1.0], held=2) == 1.0
    assert value.transfer_option_value([12.0, -3.0], held=1) == 0.0


def test_squads_are_rebuilt_from_a_trace():
    trace = pd.DataFrame(
        {
            "gameweek": [1, 2, 3, 4],
            "chip": ["", "", "freehit", ""],
            "transfers_in": ["1 2 3", "4", "9 9", "5"],
            "transfers_out": ["", "1", "9", "2"],
        }
    )
    squads = value.squads_from_trace(trace)
    assert set(squads[1]) == {1, 2, 3}
    assert set(squads[2]) == {2, 3, 4}
    assert set(squads[3]) == {2, 3, 4}, "a free-hit week leaves the squad alone"
    assert set(squads[4]) == {3, 4, 5}


def test_terminal_value_stops_the_solver_selling_the_future():
    """A player with everything to come just past the horizon is kept when the edge is valued."""
    rng = np.random.default_rng(5)
    rows = []
    element = 1
    for club in range(1, 9):
        for position, count in (("GKP", 2), ("DEF", 3), ("MID", 3), ("FWD", 2)):
            for _ in range(count):
                rows.append({"element": element, "position": position, "team_id": club,
                             "price": int(rng.integers(40, 90)), "web_name": f"p{element}"})
                element += 1
    universe = pd.DataFrame(rows)
    gameweeks = [1, 2]
    base = universe.set_index("element")["price"] / 20.0
    points = pd.DataFrame({gw: base for gw in gameweeks}, index=base.index)
    windows = milp.ChipWindows(windows={"wildcard": [(2, 19)], "freehit": [(2, 19)],
                                        "bboost": [(1, 19)], "3xc": [(1, 19)]})
    opening = milp.solve(points, universe, milp.SquadState(players={}, bank=1000, free_transfers=15),
                         windows, gameweeks=gameweeks, allow_chips=False, time_limit=30)
    squad = opening.squads[1]
    price = universe.set_index("element")["price"]
    held = milp.SquadState(players={e: int(price.loc[e]) for e in squad},
                           bank=1000 - int(price.loc[squad].sum()), free_transfers=1)
    # An owned forward is worthless inside the horizon but a monster after it; an affordable
    # outsider is slightly better inside it.
    forwards = [e for e in squad if universe.set_index("element").at[e, "position"] == "FWD"]
    sleeper = forwards[0]
    points.loc[sleeper] = 0.0
    outsiders = universe[(~universe["element"].isin(squad)) & (universe["position"] == "FWD")
                         & (universe["price"] <= price.loc[sleeper] + held.bank)]
    assert len(outsiders), "need an affordable forward outside the squad"
    rival = int(outsiders.iloc[0]["element"])
    points.loc[rival] = 1.0

    without = milp.solve(points, universe, held, windows, gameweeks=gameweeks,
                         allow_chips=False, time_limit=30)
    assert sleeper in without.transfers_out.get(1, []) + without.transfers_out.get(2, [])

    beyond = pd.Series(0.0, index=universe["element"])
    beyond[sleeper] = 40.0
    with_edge = milp.solve(points, universe, held, windows, gameweeks=gameweeks,
                           allow_chips=False, time_limit=30, terminal_value=beyond)
    assert sleeper in with_edge.squads[2], "kept for what comes after the horizon"
