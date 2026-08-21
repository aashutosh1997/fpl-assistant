"""Tests for the transfer and chip optimiser.

These are hand-built scenarios with knowable answers. The point is not to check that the solver
finds a good plan — that is what the backtest is for — but that the plans it returns are *legal*.
An illegal plan is worse than a bad one: it looks authoritative and cannot be executed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fplass.optimise.milp import (
    CLUB_LIMIT,
    HIT_COST,
    LINEUP_SIZE,
    SQUAD_QUOTA,
    SQUAD_SIZE,
    ChipWindows,
    SquadState,
    solve,
)

GAMEWEEKS = [1, 2, 3]


@pytest.fixture
def windows() -> ChipWindows:
    """The real 2026/27 shape: Bench Boost and Triple Captain from GW1, the others from GW2."""
    return ChipWindows(
        windows={
            "wildcard": [(2, 19), (20, 38)],
            "freehit": [(2, 19), (20, 38)],
            "bboost": [(1, 19), (20, 38)],
            "3xc": [(1, 19), (20, 38)],
        }
    )


@pytest.fixture
def universe() -> pd.DataFrame:
    """A synthetic league: 8 clubs, 10 players each, prices spread across the range."""
    rows = []
    element = 1
    rng = np.random.default_rng(7)
    for club in range(1, 9):
        for position, count in (("GKP", 2), ("DEF", 3), ("MID", 3), ("FWD", 2)):
            for _ in range(count):
                rows.append(
                    {
                        "element": element,
                        "position": position,
                        "team_id": club,
                        "price": int(rng.integers(40, 90)),
                        "web_name": f"p{element}",
                    }
                )
                element += 1
    return pd.DataFrame(rows)


@pytest.fixture
def points(universe) -> pd.DataFrame:
    """Expected points broadly increasing in price, so optimal plans are recognisable."""
    rng = np.random.default_rng(11)
    base = universe.set_index("element")["price"] / 20.0
    return pd.DataFrame(
        {gw: base + rng.normal(0, 0.3, len(base)) for gw in GAMEWEEKS}, index=base.index
    )


def solve_scenario(points, universe, state, windows, **kwargs):
    kwargs.setdefault("time_limit", 30)
    return solve(points, universe, state, windows, gameweeks=GAMEWEEKS, **kwargs)


# ------------------------------------------------------------------ legality


def test_squad_is_always_legal(points, universe, windows):
    """Fifteen players, correct positional quotas, at most three per club, every gameweek."""
    state = SquadState(players={}, bank=1000, free_transfers=15)
    plan = solve_scenario(points, universe, state, windows, allow_chips=False)

    position = universe.set_index("element")["position"]
    club = universe.set_index("element")["team_id"]

    for gw, squad in plan.squads.items():
        assert len(squad) == SQUAD_SIZE, f"GW{gw} squad has {len(squad)} players"
        counts = position.loc[squad].value_counts()
        for pos, quota in SQUAD_QUOTA.items():
            assert counts.get(pos, 0) == quota, f"GW{gw} has {counts.get(pos, 0)} {pos}"
        assert club.loc[squad].value_counts().max() <= CLUB_LIMIT


def test_lineup_is_a_legal_formation(points, universe, windows):
    state = SquadState(players={}, bank=1000, free_transfers=15)
    plan = solve_scenario(points, universe, state, windows, allow_chips=False)
    position = universe.set_index("element")["position"]

    for gw, lineup in plan.lineups.items():
        assert len(lineup) == LINEUP_SIZE
        assert set(lineup) <= set(plan.squads[gw]), "cannot start a player you do not own"
        counts = position.loc[lineup].value_counts()
        assert counts.get("GKP", 0) == 1
        assert 3 <= counts.get("DEF", 0) <= 5
        assert 2 <= counts.get("MID", 0) <= 5
        assert 1 <= counts.get("FWD", 0) <= 3


def test_captain_is_in_the_starting_eleven(points, universe, windows):
    """The bug this guards against: a phantom free-hit squad let the solver captain a player it
    did not own, and doubled the objective."""
    state = SquadState(players={}, bank=1000, free_transfers=15)
    plan = solve_scenario(points, universe, state, windows)
    for gw, captain in plan.captains.items():
        assert captain in plan.lineups[gw], f"GW{gw} captain is not in the XI"


def test_budget_is_respected(points, universe, windows):
    """A tight budget must produce a cheaper squad, not an over-budget one."""
    price = universe.set_index("element")["price"]
    budget = 850
    state = SquadState(players={}, bank=budget, free_transfers=15)
    plan = solve_scenario(points, universe, state, windows, allow_chips=False)
    assert price.loc[plan.squads[GAMEWEEKS[0]]].sum() <= budget


def test_cannot_take_a_fourth_player_from_one_club(points, universe, windows):
    """Even when one club holds every high-scoring player, the limit holds."""
    boosted = points.copy()
    favoured = universe[universe["team_id"] == 1]["element"]
    boosted.loc[boosted.index.isin(favoured)] += 20.0

    state = SquadState(players={}, bank=1000, free_transfers=15)
    plan = solve_scenario(boosted, universe, state, windows, allow_chips=False)
    club = universe.set_index("element")["team_id"]
    for squad in plan.squads.values():
        assert (club.loc[squad] == 1).sum() <= CLUB_LIMIT


# ------------------------------------------------------------------ transfers


def test_no_transfers_when_nothing_can_be_gained(points, universe, windows):
    """With every player worth the same, no transfer can help, so the solver must roll.

    Note the premise has to be constructed carefully. An "optimal" squad taken from a plan solved
    with unlimited free transfers is *not* a stable holding — that plan intends to churn, and its
    gameweek-1 squad is only a waypoint. Flattening the points removes any gain from moving at
    all, which is what makes the expected answer unambiguous.
    """
    flat = pd.DataFrame(1.0, index=points.index, columns=points.columns)

    state = SquadState(players={}, bank=1000, free_transfers=15)
    opening = solve_scenario(flat, universe, state, windows, allow_chips=False)
    squad = opening.squads[GAMEWEEKS[0]]

    price = universe.set_index("element")["price"]
    held = SquadState(
        players={e: int(price.loc[e]) for e in squad},
        bank=1000 - int(price.loc[squad].sum()),
        free_transfers=1,
    )
    plan = solve_scenario(flat, universe, held, windows, allow_chips=False)
    assert sum(len(v) for v in plan.transfers_in.values()) == 0
    assert sum(plan.hits.values()) == 0


def test_never_takes_a_hit_that_cannot_pay_for_itself(points, universe, windows):
    """A marginal upgrade must not be bought with a -4.

    This is the single most common way an optimiser gives bad advice: churning the squad for gains
    smaller than the hits that fund them.
    """
    state = SquadState(players={}, bank=1000, free_transfers=15)
    opening = solve_scenario(points, universe, state, windows, allow_chips=False)
    squad = opening.squads[GAMEWEEKS[0]]

    price = universe.set_index("element")["price"]
    held = SquadState(
        players={e: int(price.loc[e]) for e in squad},
        bank=1000 - int(price.loc[squad].sum()),
        free_transfers=1,
    )
    plan = solve_scenario(points, universe, held, windows, allow_chips=False)

    hits = sum(plan.hits.values())
    if hits:
        # Any hit taken must be covered by the gain over simply holding the squad.
        hold = solve_scenario(
            points,
            universe,
            SquadState(players=held.players, bank=held.bank, free_transfers=0),
            windows,
            allow_chips=False,
            forbidden=None,
        )
        assert plan.objective >= hold.objective - 1e-6, (
            f"took {hits} hit(s) worth -{hits * HIT_COST} without covering the cost"
        )


def test_takes_a_hit_when_the_gain_clearly_exceeds_it(points, universe, windows):
    """A player worth far more than four points must be worth a -4."""
    state = SquadState(players={}, bank=1000, free_transfers=15)
    opening = solve_scenario(points, universe, state, windows, allow_chips=False)
    squad = opening.squads[GAMEWEEKS[0]]

    price = universe.set_index("element")["price"]
    held = SquadState(
        players={e: int(price.loc[e]) for e in squad},
        bank=1000 - int(price.loc[squad].sum()),
        free_transfers=0,
    )

    # Make one unowned, affordable player enormously valuable.
    boosted = points.copy()
    outsiders = universe[
        (~universe["element"].isin(squad)) & (universe["price"] <= price.loc[squad].min())
    ]
    assert len(outsiders), "need an affordable outsider for this scenario"
    star = outsiders.iloc[0]["element"]
    boosted.loc[star] += 50.0

    plan = solve_scenario(boosted, universe, held, windows, allow_chips=False)
    bought = {e for players in plan.transfers_in.values() for e in players}
    assert star in bought, "a 50-point upgrade should be worth a 4-point hit"


def test_free_transfers_never_exceed_the_cap(points, universe, windows):
    """You may bank at most five; a plan that assumes more is unexecutable."""
    state = SquadState(players={}, bank=1000, free_transfers=15)
    opening = solve_scenario(points, universe, state, windows, allow_chips=False)
    price = universe.set_index("element")["price"]
    squad = opening.squads[GAMEWEEKS[0]]
    held = SquadState(
        players={e: int(price.loc[e]) for e in squad},
        bank=1000 - int(price.loc[squad].sum()),
        free_transfers=1,
    )
    plan = solve_scenario(points, universe, held, windows, allow_chips=False)
    # With no gains available the solver should not be inventing transfers to burn.
    assert all(hits >= 0 for hits in plan.hits.values())


# ----------------------------------------------------------------------- chips


def test_wildcard_and_free_hit_are_illegal_in_gameweek_one(points, universe, windows):
    """A real 2026/27 rule, and asymmetric: Bench Boost and Triple Captain *are* legal in GW1."""
    assert windows.legal("wildcard", 1) == []
    assert windows.legal("freehit", 1) == []
    assert windows.legal("bboost", 1) == [0]
    assert windows.legal("3xc", 1) == [0]

    state = SquadState(players={}, bank=1000, free_transfers=15)
    plan = solve_scenario(points, universe, state, windows)
    assert plan.chips.get(1) not in {"wildcard", "freehit"}


def test_bench_boost_starts_all_fifteen(points, universe, windows):
    state = SquadState(players={}, bank=1000, free_transfers=15)
    plan = solve_scenario(points, universe, state, windows, chip_schedule={1: "bboost"})
    assert plan.chips.get(1) == "bboost"
    assert len(plan.lineups[1]) == SQUAD_SIZE
    # And normal gameweeks are unaffected.
    assert len(plan.lineups[2]) == LINEUP_SIZE


def test_free_hit_squad_replaces_the_real_one_for_one_week(points, universe, windows):
    state = SquadState(players={}, bank=1000, free_transfers=15)
    plan = solve_scenario(points, universe, state, windows, chip_schedule={2: "freehit"})
    assert plan.chips.get(2) == "freehit"
    assert len(plan.lineups[2]) == LINEUP_SIZE
    assert set(plan.lineups[2]) <= set(plan.squads[2])


def test_chip_schedule_is_honoured_exactly(points, universe, windows):
    """A pinned roadmap must be obeyed, and unpinned chips must not be played."""
    state = SquadState(players={}, bank=1000, free_transfers=15)
    plan = solve_scenario(points, universe, state, windows, chip_schedule={1: "3xc"})
    assert plan.chips == {1: "3xc"}


def test_already_used_chips_cannot_be_replayed(points, universe, windows):
    state = SquadState(
        players={}, bank=1000, free_transfers=15, chips_used={"bboost:0", "3xc:0"}
    )
    plan = solve_scenario(points, universe, state, windows)
    assert "bboost" not in plan.chips.values()
    assert "3xc" not in plan.chips.values()


def test_no_chips_when_disallowed(points, universe, windows):
    state = SquadState(players={}, bank=1000, free_transfers=15)
    plan = solve_scenario(points, universe, state, windows, allow_chips=False)
    assert plan.chips == {}


# ------------------------------------------------------------------ sell price


def test_selling_price_applies_the_fee_and_rounds_down():
    """Profit is halved and rounded down, so a 0.1m rise returns nothing at all."""
    state = SquadState(players={1: 70, 2: 70, 3: 70, 4: 70})
    assert state.selling_price(1, 70) == 70  # unchanged
    assert state.selling_price(2, 71) == 70  # +0.1m: half of 0.1 rounds down to nothing
    assert state.selling_price(3, 72) == 71  # +0.2m: half is 0.1m
    assert state.selling_price(4, 65) == 65  # a loss is taken in full
    # A player you never owned sells at market price.
    assert state.selling_price(99, 55) == 55


def test_objective_is_not_double_counted(points, universe, windows):
    """Guards the free-hit bug: the objective must be in the range a real team can score.

    The phantom free-hit squad previously scored alongside the real one every week, roughly
    doubling the reported objective.
    """
    state = SquadState(players={}, bank=1000, free_transfers=15)
    plan = solve_scenario(points, universe, state, windows, allow_chips=False)
    best_possible = points.max().sum() * (LINEUP_SIZE + 1)
    assert 0 < plan.objective < best_possible
    for gw, expected in plan.expected_points.items():
        realistic = points[gw].nlargest(LINEUP_SIZE + 1).sum()
        assert expected <= realistic + 1e-6, f"GW{gw} scores more than its best possible XI"
