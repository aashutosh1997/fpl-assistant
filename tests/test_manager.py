"""Tests for the paper manager: the game's rules, applied the way the game applies them.

A replay that scores a squad wrongly is worse than none, because it would attribute the error
to the policy. So the substitution, captaincy and free-transfer rules are pinned on hand-worked
cases before any season is played.
"""

from __future__ import annotations

import pytest

from fplass.backtest import manager
from fplass.optimise import milp

# A squad: goalkeepers 1-2, defenders 3-7, midfielders 8-12, forwards 13-15.
POSITIONS = {
    **{e: "GKP" for e in (1, 2)},
    **{e: "DEF" for e in (3, 4, 5, 6, 7)},
    **{e: "MID" for e in (8, 9, 10, 11, 12)},
    **{e: "FWD" for e in (13, 14, 15)},
}
LINEUP = [1, 3, 4, 5, 8, 9, 10, 11, 13, 14, 15]  # 3-4-3
BENCH = [2, 6, 12, 7]  # keeper, then a defender, a midfielder, a defender


def _played(*absent: int) -> dict[int, bool]:
    return {e: e not in absent for e in POSITIONS}


def test_a_missing_defender_is_replaced_by_the_first_substitute():
    eleven, made = manager.auto_substitute(LINEUP, BENCH, POSITIONS, _played(4))
    assert made == 1
    assert 4 not in eleven and 6 in eleven


def test_a_substitute_who_would_break_the_formation_is_skipped():
    """3-4-3 losing a defender cannot take a midfielder: two defenders is illegal."""
    bench = [2, 12, 6, 7]  # midfielder first
    eleven, made = manager.auto_substitute(LINEUP, bench, POSITIONS, _played(4))
    assert made == 1
    assert 12 not in eleven and 6 in eleven


def test_a_forward_can_be_replaced_by_a_defender_when_the_formation_allows():
    """Losing a forward from 3-4-3, the first bench defender comes on for a legal 4-4-2."""
    eleven, made = manager.auto_substitute(LINEUP, BENCH, POSITIONS, _played(13))
    assert made == 1 and 6 in eleven and 13 not in eleven


def test_goalkeepers_only_swap_with_goalkeepers():
    eleven, made = manager.auto_substitute(LINEUP, BENCH, POSITIONS, _played(1))
    assert made == 1 and 2 in eleven and 1 not in eleven
    # Bench keeper absent too: no outfielder may replace a goalkeeper.
    eleven, made = manager.auto_substitute(LINEUP, BENCH, POSITIONS, _played(1, 2))
    assert made == 0 and 1 in eleven


def test_substitutes_who_did_not_play_are_passed_over():
    eleven, made = manager.auto_substitute(LINEUP, BENCH, POSITIONS, _played(4, 6))
    assert made == 1 and 7 in eleven and 6 not in eleven


def test_captaincy_passes_to_the_vice_when_the_captain_does_not_play():
    points = {e: 2 for e in POSITIONS}
    points[13] = 10
    points[8] = 6
    minutes = {e: 90 for e in POSITIONS}
    total, extra, benched, subs = manager.score_gameweek(
        LINEUP, BENCH, 13, 8, None, POSITIONS, points, minutes
    )
    assert extra == 10 and total == sum(points[e] for e in LINEUP) + 10 and subs == 0
    assert benched == sum(points[e] for e in BENCH)

    minutes[13] = 0
    points[13] = 0
    total, extra, benched, subs = manager.score_gameweek(
        LINEUP, BENCH, 13, 8, None, POSITIONS, points, minutes
    )
    assert extra == 6, "the vice-captain is doubled instead"
    assert subs == 1, "and the absent captain is substituted"

    minutes[8] = 0
    points[8] = 0
    _, extra, _, _ = manager.score_gameweek(LINEUP, BENCH, 13, 8, None, POSITIONS, points, minutes)
    assert extra == 0, "nobody is doubled when both are absent"


def test_triple_captain_and_bench_boost():
    points = {e: 1 for e in POSITIONS}
    points[13] = 8
    minutes = {e: 90 for e in POSITIONS}
    total, extra, _, _ = manager.score_gameweek(
        LINEUP, BENCH, 13, 8, "3xc", POSITIONS, points, minutes
    )
    assert extra == 16
    minutes[4] = 0
    total, _, benched, subs = manager.score_gameweek(
        LINEUP, BENCH, 13, 8, "bboost", POSITIONS, points, minutes
    )
    assert subs == 0 and benched == 0 and total == sum(points.values()) + 8


def test_free_transfer_arithmetic():
    assert manager.next_free_transfers(1, 0, None) == (2, 0)
    assert manager.next_free_transfers(5, 0, None) == (5, 0), "five is the cap"
    assert manager.next_free_transfers(2, 3, None) == (1, 1)
    assert manager.next_free_transfers(1, 4, None) == (1, 3)
    assert manager.next_free_transfers(2, 8, "wildcard") == (3, 0)
    assert manager.next_free_transfers(1, 0, "freehit") == (2, 0)
    assert manager.next_free_transfers(15, 15, None) == (1, 0), "the opening build"


def test_a_free_hit_week_leaves_the_permanent_squad_alone():
    """The solver once churned the real squad for free while the free-hit squad played."""
    import numpy as np
    import pandas as pd

    from fplass.optimise.milp import ChipWindows, SquadState, solve

    rows = []
    element = 1
    rng = np.random.default_rng(3)
    for club in range(1, 9):
        for position, count in (("GKP", 2), ("DEF", 3), ("MID", 3), ("FWD", 2)):
            for _ in range(count):
                rows.append(
                    {"element": element, "position": position, "team_id": club,
                     "price": int(rng.integers(40, 90)), "web_name": f"p{element}"}
                )
                element += 1
    universe = pd.DataFrame(rows)
    gameweeks = [2, 3, 4]
    base = universe.set_index("element")["price"] / 20.0
    points = pd.DataFrame({gw: base for gw in gameweeks}, index=base.index)
    windows = ChipWindows(
        windows={"wildcard": [(2, 19)], "freehit": [(2, 19)], "bboost": [(1, 19)], "3xc": [(1, 19)]}
    )
    opening = solve(points, universe, SquadState(players={}, bank=1000, free_transfers=15),
                    windows, gameweeks=gameweeks, allow_chips=False, time_limit=30)
    squad = opening.squads[2]
    price = universe.set_index("element")["price"]
    held = SquadState(players={e: int(price.loc[e]) for e in squad},
                      bank=1000 - int(price.loc[squad].sum()), free_transfers=1)
    # Outsiders are worth a fortune in the free-hit week only, and a little afterwards — the
    # temptation is to keep them for free.
    boosted = points.copy()
    outsiders = universe[~universe["element"].isin(squad)]["element"].tolist()
    boosted.loc[outsiders, 2] += 30.0
    boosted.loc[outsiders, [3, 4]] += 2.0
    plan = solve(boosted, universe, held, windows, gameweeks=gameweeks,
                 chip_schedule={2: "freehit"}, time_limit=30)
    assert plan.chips.get(2) == "freehit"
    assert plan.transfers_in.get(2, []) == [] and plan.transfers_out.get(2, []) == []
    assert plan.hits.get(2, 0) == 0


def test_bench_order_puts_the_keeper_first_then_by_expectation():
    import pandas as pd

    squad = list(POSITIONS)
    expected = pd.Series({e: float(e) for e in POSITIONS})
    order = manager.bench_order(squad, LINEUP, expected, POSITIONS)
    assert order[0] == 2
    assert order[1:] == [12, 7, 6]


@pytest.mark.slow
def test_two_gameweeks_of_a_real_season_replay(con):
    """An end-to-end pass over the opening two deadlines of a panelled season."""
    seasons = [r[0] for r in con.execute(
        "SELECT DISTINCT season FROM projection_panel ORDER BY season DESC").fetchall()]
    source = None
    if not seasons:
        candidates = sorted(manager.PANEL.glob("*.parquet"))
        if not candidates:
            pytest.skip("no panel rows or parquet files to replay")
        source = candidates[-1]
        seasons = [source.stem]
    replay = manager.replay_season(
        con, seasons[0], policy="hold", solver_time_limit=10, max_gameweek=2, source=source
    )
    assert len(replay.records) == 2
    first = replay.records[0]
    assert len(first.transfers_in) == milp.SQUAD_SIZE and first.hits == 0
    assert first.free_transfers == milp.SQUAD_SIZE and replay.records[1].free_transfers == 1
    assert replay.records[1].transfers_in == []
    assert first.bank >= 0 and first.squad_value <= manager.STARTING_BANK + 50
    assert replay.total == sum(r.points for r in replay.records)
