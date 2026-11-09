"""The learned policy's pieces: legal actions, the fast environment's rules, and the learner.

The environment is checked against the game, not against the planner: a hit is charged for the
transfer beyond the free ones, a Free Hit leaves the squad and the bank as they were, a Bench
Boost adds exactly the bench and a Triple Captain exactly one more copy of the captain. The
learner is checked on problems with a known answer, including one where the better first move
only pays a week later, which is the thing fitted Q-iteration exists to find.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
import pytest

from fplass.optimise import milp
from fplass.optimise.policy import PolicyConfig
from fplass.rl import elite, env as env_module, fqi, policies
from fplass.rl import season as season_module
from fplass.rl.features import (
    ACTIONS, FEATURES, N_ACTIONS, ROLL, action_index, best_eleven, chip_status, legal_actions,
)

CALENDAR = [1, 2, 3, 4]


@pytest.fixture(scope="module")
def data() -> season_module.SeasonData:
    rng = np.random.default_rng(5)
    rows, element = [], 1
    for club in range(1, 9):
        for position, count in (("GKP", 2), ("DEF", 4), ("MID", 4), ("FWD", 2)):
            for _ in range(count):
                rows.append({"element": element, "web_name": f"p{element}", "position": position,
                             "team_id": club, "price": int(rng.integers(40, 80))})
                element += 1
    universe = pd.DataFrame(rows)
    base = universe.set_index("element")["price"] / 15.0
    ep = {
        gw: pd.DataFrame({target: base + rng.normal(0, 0.5, len(base)) for target in CALENDAR if target >= gw})
        for gw in CALENDAR
    }
    points = {gw: {int(e): int(rng.integers(0, 10)) for e in universe["element"]} for gw in CALENDAR}
    minutes = {gw: {int(e): 90 for e in universe["element"]} for gw in CALENDAR}
    fixtures = {gw: {club: 1 for club in range(1, 9)} for gw in CALENDAR}
    fixtures[3][1], fixtures[3][2] = 2, 0
    windows = milp.ChipWindows(
        windows={"wildcard": [(2, 3)], "freehit": [(2, 3)], "bboost": [(1, 3)], "3xc": [(1, 4)]}
    )
    data = season_module.SeasonData(
        season="test", calendar=CALENDAR, ep=ep, players={gw: universe.copy() for gw in CALENDAR},
        points=points, minutes=minutes, fixtures=fixtures, windows=windows,
        position_of=dict(zip(universe["element"], universe["position"], strict=True)),
    )
    config = PolicyConfig(solver_time_limit=30)
    data.start, data.start_bank = season_module.opening_squad(data, config)
    return data


def test_the_first_deadline_only_chooses_a_chip_and_counts_need_swaps():
    chips = {"wildcard": True, "freehit": True, "bboost": True, "3xc": True}
    first = legal_actions(0, 5, chips)
    assert {ACTIONS[i] for i in np.flatnonzero(first)} == {(0, None), (0, "bboost"), (0, "3xc")}
    later = legal_actions(3, 1, {"bboost": True})
    assert {ACTIONS[i] for i in np.flatnonzero(later)} == {(0, None), (1, None), (0, "bboost"), (1, "bboost")}
    assert legal_actions(3, 0, {})[ROLL] and legal_actions(3, 0, {}).sum() == 1


def test_chip_windows_count_the_calendar_not_gameweek_labels():
    data = season_module.SeasonData(
        season="covid", calendar=[1, 2, 29, 39], ep={}, players={}, points={}, minutes={},
        fixtures={}, windows=milp.ChipWindows(windows={"bboost": [(1, 19), (20, 38)]}),
    )
    assert chip_status(data, 1, set(), "bboost") == (True, 1, True)
    assert chip_status(data, 1, {"bboost:0"}, "bboost") == (False, 0, True)
    # GW39 lies outside both windows: nothing is legal and nothing is left.
    assert chip_status(data, 3, set(), "bboost") == (False, 0, False)


def test_the_eleven_is_legal_and_the_best_there_is():
    rng = np.random.default_rng(1)
    positions = {}
    squad = list(range(1, 16))
    for e, pos in zip(squad, ["GKP"] * 2 + ["DEF"] * 5 + ["MID"] * 5 + ["FWD"] * 3, strict=True):
        positions[e] = pos
    week = {e: float(rng.uniform(0, 10)) for e in squad}
    xi = best_eleven(squad, week, positions)
    counts = pd.Series([positions[e] for e in xi]).value_counts()
    assert len(xi) == 11 and counts["GKP"] == 1 and counts["DEF"] >= 3 and counts["MID"] >= 2 and counts["FWD"] >= 1
    legal = (
        c for c in combinations(squad, 11)
        if sum(positions[e] == "GKP" for e in c) == 1 and sum(positions[e] == "DEF" for e in c) >= 3
        and sum(positions[e] == "MID" for e in c) >= 2 and sum(positions[e] == "FWD" for e in c) >= 1
    )
    assert sum(week[e] for e in xi) == pytest.approx(max(sum(week[e] for e in c) for c in legal))


def _after_first(data) -> env_module.SeasonEnv:
    env = env_module.SeasonEnv(data, PolicyConfig(solver_time_limit=30))
    env.reset()
    env.step(ROLL)
    return env


def test_a_transfer_beyond_the_free_one_costs_four_and_the_money_moves(data):
    env = _after_first(data)
    before = env.state
    assert before.free_transfers == 1
    obs = env.observe()
    if not obs.mask[action_index(2, None)]:
        pytest.skip("the synthetic squad has fewer than two useful swaps")
    step = env.step(action_index(2, None), obs)
    assert step.hits == 1 and len(step.transfers_in) == 2
    price = data.prices(2)
    expected_bank = before.bank + sum(before.selling_price(o, price[o]) for o in step.transfers_out) - sum(
        price[i] for i in step.transfers_in
    )
    assert env.state.bank == expected_bank
    assert len(env.state.players) == 15 and env.state.free_transfers == 1


def test_a_free_hit_leaves_the_squad_and_bank_as_they_were(data):
    env = _after_first(data)
    before = env.state
    step = env.step(action_index(0, "freehit"))
    assert step.chip == "freehit" and step.hits == 0
    assert env.state.players == before.players and env.state.bank == before.bank
    assert env.state.free_transfers == min(before.free_transfers + 1, milp.MAX_BANKED_TRANSFERS)
    assert "freehit:0" in env.state.chips_used


def test_bench_boost_adds_the_bench_and_triple_captain_one_more_captain(data):
    plain = env_module.SeasonEnv(data)
    plain.reset()
    base = plain.step(ROLL)
    boosted = env_module.SeasonEnv(data)
    boosted.reset()
    boost = boosted.step(action_index(0, "bboost"))
    tripled = env_module.SeasonEnv(data)
    tripled.reset()
    triple = tripled.step(action_index(0, "3xc"))
    week = data.week(1).to_dict()
    lineup, bench, captain, _ = env_module.team_sheet(list(data.start), week, data.position_of, None)
    gw_points = data.points[1]
    assert triple.points == base.points + gw_points[captain]
    # Every minute is 90, so nobody is substituted: the boost is exactly the bench's points
    # (and the captain is chosen from all fifteen).
    _, _, boost_captain, _ = env_module.team_sheet(list(data.start), week, data.position_of, "bboost")
    assert boost.points == sum(gw_points[e] for e in data.start) + gw_points[boost_captain]
    assert base.points == sum(gw_points[e] for e in lineup) + gw_points[captain]


def test_a_whole_season_plays_legally_with_any_policy(data):
    env = env_module.SeasonEnv(data, PolicyConfig(solver_time_limit=30))
    for policy in (policies.RulePolicy(), policies.RandomPolicy()):
        episode = env_module.play(env, policy, np.random.default_rng(3))
        assert len(episode.steps) == len(CALENDAR)
        played = [s.chip for s in episode.steps if s.chip]
        assert len(played) == len(set(played))  # one instance of each chip in these windows


def test_returns_to_go_sum_the_rest_of_each_season():
    rewards = np.array([1.0, 2.0, 3.0, 10.0, 20.0])
    episode = np.array([0, 0, 0, 1, 1])
    t = np.array([0, 1, 2, 0, 1])
    assert fqi.returns_to_go(rewards, episode, t).tolist() == [6.0, 5.0, 3.0, 30.0, 20.0]


def _transitions(n: int, rng) -> fqi.Transitions:
    """Two-deadline seasons: action 1 at the first scores nothing but unlocks 12 points at the
    second; action 2 scores 5 at once and unlocks nothing."""
    weeks = FEATURES.index("weeks_left")
    rows = []
    for episode in range(n):
        first = int(rng.integers(0, 3))
        second = int(rng.integers(0, 3))
        x0 = np.zeros(len(FEATURES))
        x0[weeks] = 2
        x1 = np.zeros(len(FEATURES))
        x1[weeks] = 1
        x1[0] = 1.0 if first == 1 else 0.0
        mask = np.zeros(N_ACTIONS, dtype=bool)
        mask[:3] = True
        r0 = 5.0 if first == 2 else 0.0
        r1 = 12.0 if (first == 1 and second == 0) else 0.0
        rows.append((episode, 0, x0, mask, first, r0, x1, mask, False))
        rows.append((episode, 1, x1, mask, second, r1, np.zeros(len(FEATURES)), np.zeros(N_ACTIONS, bool), True))
    episode, t, x, mask, action, reward, x_next, mask_next, done = zip(*rows, strict=True)
    return fqi.Transitions(
        season=np.array(["s"] * len(rows)), episode=np.array(episode), t=np.array(t),
        x=np.vstack(x), mask=np.vstack(mask), action=np.array(action), reward=np.array(reward),
        expected=np.array(reward), x_next=np.vstack(x_next), mask_next=np.vstack(mask_next),
        done=np.array(done), behaviour=np.array(["random"] * len(rows)),
    )


def test_fitted_q_finds_a_move_that_only_pays_a_week_later():
    data = _transitions(3000, np.random.default_rng(0))
    model = fqi.fit(data, iterations=3)
    policy = policies.ScoredPolicy(model, "q")
    weeks = FEATURES.index("weeks_left")
    first = np.zeros(len(FEATURES))
    first[weeks] = 2
    unlocked = first.copy()
    unlocked[weeks], unlocked[0] = 1, 1.0
    mask = np.zeros(N_ACTIONS, dtype=bool)
    mask[:3] = True

    class Obs:
        def __init__(self, vector):
            self.vector, self.mask = vector, mask

    assert policy.act(Obs(first)) == 1
    assert policy.act(Obs(unlocked)) == 0
    q = model.values(first[None, :])[0]
    assert q[1] == pytest.approx(12.0, abs=1.5) and q[2] == pytest.approx(5.0, abs=1.5)


def test_free_transfers_are_rebuilt_from_the_record():
    made = {2: 0, 3: 2, 4: 0, 5: 1}
    held = elite.free_transfers(made, {4: "wildcard"}, 6)
    assert held == {2: 1, 3: 2, 4: 1, 5: 2, 6: 2}
    windows = milp.ChipWindows(windows={"wildcard": [(2, 19), (20, 38)]})
    assert elite.chips_used_before({3: "wildcard", 25: "wildcard"}, 30, windows) == {"wildcard:0", "wildcard:1"}
    assert elite.chips_used_before({3: "wildcard", 25: "wildcard"}, 10, windows) == {"wildcard:0"}


def test_the_solver_makes_exactly_the_transfers_it_is_told(data):
    gw = 2
    state = milp.SquadState(players=dict(data.start), bank=data.start_bank, free_transfers=1)
    table = data.ep[gw]
    for count in (0, 2):
        plan = milp.solve(table, data.players[gw], state, data.windows, chip_schedule={},
                          transfers={gw: count}, time_limit=30)
        assert len(plan.transfers_in.get(gw, [])) == count
