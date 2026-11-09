"""The fast environment: a season played with the game's rules and a quick executor.

The learner picks a transfer count and a chip; this module turns the choice into a team:

* ``k`` transfers are the best single swap by the horizon projection, taken ``k`` times in turn
  (each after the last has moved the money), the same instrument :mod:`fplass.options.value`
  prices the planner's constants with;
* a Wildcard is a one-week solve over the horizon's total projection with every transfer free,
  and the squad it picks is kept; a Free Hit is the same solve over this week's projection,
  played once, with the squad and bank restored after;
* the eleven, captain and vice are the best by this week's projection, the bench in the
  replay's order.

Scoring, substitutions, the captain's fallback, hits, the free-transfer bank, selling prices and
the chip windows are the paper manager's own functions (:mod:`fplass.backtest.manager`), so a
season here differs from a replayed one only in how the moves are chosen.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..backtest import manager
from ..optimise import milp
from ..optimise.policy import PolicyConfig
from ..options.value import swap_gains
from .features import ACTIONS, Observation, best_eleven, observe
from .season import SeasonData


@dataclass(slots=True)
class Step:
    """One deadline played: what was chosen, what it cost and what it scored."""

    gw: int
    t: int
    action: int
    chip: str | None
    transfers_in: list[int]
    transfers_out: list[int]
    hits: int
    points: int  # net of hits
    expected: float  # what the week's projection said the fielded team would score, net of hits
    captain: int
    free_transfers: int  # entering the deadline
    bank: int  # after the deadline's transfers


@dataclass(slots=True)
class Episode:
    season: str
    steps: list[Step] = field(default_factory=list)
    observations: list[Observation] = field(default_factory=list)

    @property
    def points(self) -> int:
        return sum(s.points for s in self.steps)

    @property
    def expected(self) -> float:
        return float(sum(s.expected for s in self.steps))

    def trace(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "season": self.season, "gameweek": s.gw, "action": s.action,
                    "transfers": len(s.transfers_in) if s.chip not in ("wildcard", "freehit") else 0,
                    "chip": s.chip or "", "hits": s.hits, "points": s.points,
                    "expected": round(s.expected, 2), "captain": s.captain,
                    "free_transfers": s.free_transfers, "bank": s.bank,
                    "transfers_in": " ".join(map(str, s.transfers_in)),
                    "transfers_out": " ".join(map(str, s.transfers_out)),
                }
                for s in self.steps
            ]
        )


def apply_swaps(state: milp.SquadState, swaps, price: dict[int, int]) -> milp.SquadState:
    players = dict(state.players)
    bank = state.bank
    for out, into in swaps:
        bank += state.selling_price(out, price.get(out, players[out])) - price[into]
        del players[out]
        players[into] = price[into]
    return milp.SquadState(
        players=players, bank=bank, free_transfers=state.free_transfers, chips_used=set(state.chips_used)
    )


def greedy_transfers(
    horizon: pd.Series, players: pd.DataFrame, state: milp.SquadState, count: int, first=None
) -> list[tuple[int, int]]:
    """``count`` best single swaps, each chosen after the previous one is made."""
    price = dict(zip(players["element"].astype(int), players["price"].astype(int), strict=True))
    made: list[tuple[int, int]] = []
    current = state
    for i in range(count):
        options = [first] if i == 0 and first is not None else swap_gains(horizon, players, current, top=1)
        if not options:
            break
        move = (options[0].out, options[0].into)
        made.append(move)
        current = apply_swaps(current, [move], price)
    return made


def solve_squad(
    values: pd.Series, gw: int, players: pd.DataFrame, state: milp.SquadState,
    windows: milp.ChipWindows, config: PolicyConfig,
) -> list[int]:
    """The best fifteen for one column of values with every transfer free and the money the
    squad would sell for: a Wildcard over the horizon's total, a Free Hit over the week's."""
    table = pd.DataFrame({gw: values.reindex(players["element"]).fillna(0.0).to_numpy()},
                         index=players["element"].to_numpy())
    free = milp.SquadState(
        players=dict(state.players), bank=state.bank, free_transfers=milp.SQUAD_SIZE,
        chips_used=set(state.chips_used),
    )
    plan = milp.solve(
        table, players, free, windows, gameweeks=[gw], allow_chips=False,
        bench_weight=config.bench_weight, banked_transfer_value=0.0,
        time_limit=config.solver_time_limit,
    )
    squad = [int(e) for e in plan.squads.get(gw, [])]
    if len(squad) != milp.SQUAD_SIZE:
        return list(state.players)
    return squad


def team_sheet(squad: list[int], week: dict[int, float], positions: dict[int, str], chip: str | None):
    """The eleven, the bench in order, the captain and the vice."""
    lineup = best_eleven(squad, week, positions)
    pool = squad if chip == "bboost" else lineup
    ranked = sorted(pool, key=lambda e: -week.get(e, 0.0))
    captain = ranked[0]
    vice = ranked[1] if len(ranked) > 1 else None
    bench = manager.bench_order(squad, lineup, pd.Series(week, dtype="float64"), positions)
    return lineup, bench, captain, vice


class SeasonEnv:
    """One season, deadline by deadline, from the planner's opening squad."""

    def __init__(self, data: SeasonData, config: PolicyConfig | None = None):
        if not data.start:
            raise ValueError(f"{data.season} has no opening squad; build it with start=True")
        self.data = data
        self.config = config or PolicyConfig()
        self.t = 0
        self.state = self._opening()

    def _opening(self) -> milp.SquadState:
        return milp.SquadState(players=dict(self.data.start), bank=self.data.start_bank, free_transfers=1)

    def reset(self) -> Observation:
        self.t = 0
        self.state = self._opening()
        return self.observe()

    @property
    def done(self) -> bool:
        return self.t >= len(self.data.calendar)

    def observe(self) -> Observation:
        return observe(self.data, self.t, self.state)

    def step(self, action: int, obs: Observation | None = None) -> Step:
        data, state, t = self.data, self.state, self.t
        gw = data.calendar[t]
        obs = obs if obs is not None and obs.t == t else self.observe()
        if not obs.mask[action]:
            raise ValueError(f"action {ACTIONS[action]} is not legal at GW{gw}")
        count, chip = ACTIONS[action]
        players = data.players[gw]
        price = data.prices(gw)
        positions = data.position_of
        week = data.week(gw).to_dict()

        ins: list[int] = []
        outs: list[int] = []
        holdings = dict(state.players)
        bank = state.bank
        if chip == "freehit":
            fielded = solve_squad(data.week(gw), gw, players, state, data.windows, self.config)
            ins = [e for e in fielded if e not in holdings]
            outs = [e for e in holdings if e not in set(fielded)]
            made = 0
        elif chip == "wildcard":
            fielded = solve_squad(data.horizon_ep(gw), gw, players, state, data.windows, self.config)
            ins = [e for e in fielded if e not in holdings]
            outs = [e for e in holdings if e not in set(fielded)]
            moved = apply_swaps(state, list(zip(outs, ins, strict=True)), price)
            holdings, bank = moved.players, moved.bank
            made = 0
        else:
            moves = (
                greedy_transfers(data.horizon_ep(gw), players, state, count,
                                 first=obs.swaps[0] if obs.swaps else None)
                if count and t > 0 else []
            )
            if moves:
                moved = apply_swaps(state, moves, price)
                holdings, bank = moved.players, moved.bank
            outs = [o for o, _ in moves]
            ins = [i for _, i in moves]
            fielded = list(holdings)
            made = len(moves)

        if t == 0:
            next_ft, hits = 1, 0
        else:
            next_ft, hits = manager.next_free_transfers(state.free_transfers, made, chip)

        lineup, bench, captain, vice = team_sheet(fielded, week, positions, chip)
        raw, *_ = manager.score_gameweek(
            lineup, bench, captain, vice, chip, positions, data.points.get(gw, {}), data.minutes.get(gw, {})
        )
        counted = fielded if chip == "bboost" else lineup
        multiplier = 3 if chip == "3xc" else 2
        expected = (
            sum(week.get(e, 0.0) for e in counted) + (multiplier - 1) * week.get(captain, 0.0)
            - milp.HIT_COST * hits
        )

        chips_used = set(state.chips_used)
        if chip:
            for window in data.windows.legal(chip, gw):
                if f"{chip}:{window}" not in chips_used:
                    chips_used.add(f"{chip}:{window}")
                    break
        self.state = milp.SquadState(players=holdings, bank=bank, free_transfers=next_ft, chips_used=chips_used)
        self.t += 1
        return Step(
            gw=gw, t=t, action=action, chip=chip, transfers_in=ins, transfers_out=outs, hits=hits,
            points=int(raw - int(milp.HIT_COST) * hits), expected=float(expected), captain=int(captain),
            free_transfers=state.free_transfers, bank=bank,
        )


def play(env: SeasonEnv, policy, rng: np.random.Generator | None = None, *, keep: bool = True) -> Episode:
    """One season with a policy: ``policy.act(observation, rng) -> action``."""
    rng = rng or np.random.default_rng()
    episode = Episode(season=env.data.season)
    obs = env.reset()
    if hasattr(policy, "begin"):
        policy.begin(rng)
    while True:
        action = int(policy.act(obs, rng))
        step = env.step(action, obs)
        episode.steps.append(step)
        if keep:
            episode.observations.append(obs)
        if env.done:
            break
        obs = env.observe()
    return episode
