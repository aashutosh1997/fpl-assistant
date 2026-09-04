"""Pricing the planner's flexibilities from what next week's projection turns out to be.

The transfer solver carries a handful of constants where a value should be: a banked transfer is
worth ``banked_transfer_value`` a week, a bench place ``bench_weight`` of a starting one, money in
the bank nothing at all, and the squad at the end of the horizon nothing either. Each is the price
of an option — the right to act on next week's projection instead of this week's — and each is
measurable now that the panel records both.

The instrument is a single swap. Given a squad, a bank and a projection, the best swap is the
one-for-one change (same position, club limit, affordable at the selling price) that adds the most
expected points over the remaining horizon *and would start*; the second-best is the best swap
disjoint from it, and so on. Under next week's projection those gains are what the free transfers
are spent on, so:

* a **banked transfer** is worth the gain it lets you take without a hit: with one transfer the
  second-best swap costs four points, with two it is free, so the extra transfer is worth
  ``min(max(gain_2, 0), 4)``, averaged over what next week looks like;
* **money** is worth the improvement in the best swap when the bank is larger, per 0.1m;
* the **bench** is worth what substitutes actually delivered, as a share of what the plan
  expected of them — read straight from the paper manager's traces, no scenarios needed.

Two ways to average "over what next week looks like". The honest one is history: every deadline
of the panel has an actual next-week projection, and the paper manager's traces supply realistic
squads to hold at each of them, so the constants are measured as means over some three hundred
real weeks. The live one uses the revision sampler to draw next weeks for the squad in hand, so
the advisor can say what rolling is worth *this* week.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..optimise import milp
from .revisions import RevisionSampler

log = logging.getLogger(__name__)


@dataclass(slots=True)
class Swap:
    out: int
    into: int
    gain: float


def swap_gains(
    horizon: pd.Series,
    players: pd.DataFrame,
    state: milp.SquadState,
    *,
    top: int = 3,
    extra_bank: int = 0,
) -> list[Swap]:
    """The best ``top`` disjoint single swaps for a squad under a projection.

    Args:
        horizon: Expected points over the remaining horizon, indexed by element.
        players: ``element``, ``position``, ``team_id``, ``price`` for the pool.
        state: The squad, purchase prices and bank.
        top: How many disjoint swaps to return, best first.
        extra_bank: Extra money to allow, in tenths — for pricing the bank.

    A swap only counts if the incoming player would displace someone from the best eleven by
    horizon points; upgrading a bench player to a better bench player scores nothing.
    """
    info = players.set_index("element")
    owned = [e for e in state.players if e in info.index]
    if not owned:
        return []
    price = info["price"].astype(int)
    position = info["position"]
    club = info["team_id"]
    value = horizon.reindex(info.index).fillna(0.0)

    squad_values = value.reindex(owned).fillna(0.0).sort_values(ascending=False)
    eleventh = float(squad_values.iloc[min(len(squad_values), milp.LINEUP_SIZE) - 1])
    club_counts = club.reindex(owned).value_counts()
    bank = state.bank + extra_bank

    candidates: list[Swap] = []
    outsiders = info[~info.index.isin(owned)]
    for out in owned:
        budget = state.selling_price(out, int(price[out])) + bank
        pool = outsiders[
            (outsiders["position"] == position[out]) & (price.reindex(outsiders.index) <= budget)
        ]
        if pool.empty:
            continue
        # Club limit after the swap: the incoming club's count, less one if the outgoing
        # player is from that club.
        counts = club_counts.reindex(pool["team_id"]).fillna(0).to_numpy()
        counts = counts - (pool["team_id"].to_numpy() == club[out])
        pool = pool[counts < milp.CLUB_LIMIT]
        if pool.empty:
            continue
        gains = value.reindex(pool.index) - float(value[out])
        # The incoming player must be worth a starting place.
        gains = gains[value.reindex(pool.index) > eleventh]
        if gains.empty:
            continue
        best = gains.idxmax()
        candidates.append(Swap(out=int(out), into=int(best), gain=float(gains[best])))

    candidates.sort(key=lambda s: -s.gain)
    chosen: list[Swap] = []
    used_out: set[int] = set()
    used_in: set[int] = set()
    for swap in candidates:
        if swap.out in used_out or swap.into in used_in:
            continue
        chosen.append(swap)
        used_out.add(swap.out)
        used_in.add(swap.into)
        if len(chosen) >= top:
            break
    return chosen


def transfer_option_value(gains: list[float], held: int = 1) -> float:
    """What one more free transfer beyond ``held`` is worth, given the ranked swap gains.

    With ``held`` transfers the first ``held`` swaps are free and the next costs a hit, so the
    extra transfer saves ``min(gain, HIT_COST)`` on that swap, and nothing if it is not worth
    making at all.
    """
    if len(gains) <= held:
        return 0.0
    return float(min(max(gains[held], 0.0), milp.HIT_COST))


def bank_option_value(
    horizon: pd.Series, players: pd.DataFrame, state: milp.SquadState, *, extra: int
) -> float:
    """Points the best swap improves by when the bank is ``extra`` tenths larger."""
    base = swap_gains(horizon, players, state, top=1)
    more = swap_gains(horizon, players, state, top=1, extra_bank=extra)
    best = max((s.gain for s in base), default=0.0)
    better = max((s.gain for s in more), default=0.0)
    return float(max(better - best, 0.0))


# ------------------------------------------------------------------ live estimates


@dataclass(slots=True)
class OptionValues:
    """What flexibility is worth to this squad, this week."""

    banked_transfer: float  # points, one extra free transfer next week
    bank_per_tenth: float  # points per 0.1m
    bank_half_million: float  # points for 0.5m
    bank_million: float  # points for 1.0m
    best_swap_now: float  # the best swap under this week's projection, for context
    draws: int

    def summary(self) -> str:
        return (
            f"a banked transfer is worth {self.banked_transfer:.2f} pts next week; "
            f"0.1m in the bank {self.bank_per_tenth:.2f}, 0.5m {self.bank_half_million:.2f}, "
            f"1.0m {self.bank_million:.2f} (over {self.draws} sampled next weeks)"
        )


def live_option_values(
    expected: pd.DataFrame,
    p_full: pd.Series,
    players: pd.DataFrame,
    state: milp.SquadState,
    sampler: RevisionSampler,
    *,
    draws: int = 200,
    seed: int = 7,
) -> OptionValues:
    """Value the squad's flexibilities under sampled next-week projections.

    ``expected`` is this week's projection of the gameweeks that will remain *next* week (drop
    the coming one before calling).
    """
    rng = np.random.default_rng(seed)
    futures = sampler.sample(expected, p_full, players, draws=draws, rng=rng)
    held = max(state.free_transfers, 1)
    transfer, tenth, half, million = [], [], [], []
    for draw in range(draws):
        horizon = pd.Series(futures[draw].sum(axis=1), index=expected.index)
        swaps = swap_gains(horizon, players, state, top=held + 1)
        transfer.append(transfer_option_value([s.gain for s in swaps], held=held))
        tenth.append(bank_option_value(horizon, players, state, extra=1))
        half.append(bank_option_value(horizon, players, state, extra=5))
        million.append(bank_option_value(horizon, players, state, extra=10))
    now = swap_gains(pd.Series(expected.sum(axis=1)), players, state, top=1)
    return OptionValues(
        banked_transfer=float(np.mean(transfer)),
        bank_per_tenth=float(np.mean(tenth)),
        bank_half_million=float(np.mean(half)),
        bank_million=float(np.mean(million)),
        best_swap_now=max((s.gain for s in now), default=0.0),
        draws=draws,
    )


# ------------------------------------------------------------------ historical measurement


def _elements(cell: object) -> list[int]:
    """A space-separated list of element ids from a trace cell; empty cells read back as NaN."""
    if not isinstance(cell, str):
        return []
    return [int(e) for e in cell.split() if e]


def squads_from_trace(trace: pd.DataFrame) -> dict[int, dict[int, int]]:
    """Rebuild the squad held after each gameweek's transfers, from a paper-manager trace.

    Purchase prices are not in the trace, so the selling price is taken as the price at the
    deadline being valued: the 50% sell-on fee is ignored here, which flatters affordability by
    a tenth or two on risers and is the same for every constant being measured.
    """
    squads: dict[int, dict[int, int]] = {}
    held: dict[int, int] = {}
    for row in trace.sort_values("gameweek").itertuples():
        if row.chip == "freehit":
            squads[int(row.gameweek)] = dict(held)
            continue
        outs = _elements(row.transfers_out)
        ins = _elements(row.transfers_in)
        for e in outs:
            held.pop(e, None)
        for e in ins:
            held[e] = 0
        squads[int(row.gameweek)] = dict(held)
    return squads


def measure_from_traces(
    con,
    traces: list[Path],
    *,
    panel_sources: list[Path] | None = None,
    horizon_weeks: int = 7,
) -> pd.DataFrame:
    """Measure the transfer and bank option values over real weeks.

    For every deadline in a trace, hold the squad the paper manager held, look at the *next*
    deadline's actual projection, and record the swap gains it offered. One row per week; the
    means are the constants.
    """
    from ..sim import project
    from .revisions import _panel_sql

    rows: list[dict[str, object]] = []
    for path in traces:
        trace = pd.read_csv(path)
        if trace.empty:
            continue
        season = str(trace["season"].iloc[0])
        squads = squads_from_trace(trace)
        banks = dict(zip(trace["gameweek"].astype(int), trace["bank"].astype(int), strict=True))
        free = dict(
            zip(trace["gameweek"].astype(int), trace["free_transfers"].astype(int), strict=True)
        )
        gameweeks = sorted(squads)
        panel = con.execute(
            f"SELECT as_of_gw, target_gw, element, ep_mean FROM {_panel_sql(panel_sources)} "
            "WHERE season = ?",
            [season],
        ).fetchdf()
        for gw, next_gw in zip(gameweeks, gameweeks[1:], strict=False):
            block = panel[(panel["as_of_gw"] == next_gw)]
            targets = sorted(block["target_gw"].unique())[:horizon_weeks]
            horizon = (
                block[block["target_gw"].isin(targets)].groupby("element")["ep_mean"].sum()
            )
            players = project.current_players(con, season, as_of_gameweek=next_gw)
            price = dict(zip(players["element"].astype(int), players["price"].astype(int), strict=True))
            holdings = {e: price.get(e, 0) for e in squads[gw]}
            state = milp.SquadState(
                players=holdings, bank=int(banks[gw]), free_transfers=int(free.get(next_gw, 1))
            )
            swaps = swap_gains(horizon, players, state, top=3)
            gains = [s.gain for s in swaps]
            rows.append(
                {
                    "season": season,
                    "gameweek": next_gw,
                    "free_transfers": state.free_transfers,
                    "gain_1": gains[0] if gains else 0.0,
                    "gain_2": gains[1] if len(gains) > 1 else 0.0,
                    "gain_3": gains[2] if len(gains) > 2 else 0.0,
                    "transfer_value": transfer_option_value(gains, held=1),
                    "bank_tenth": bank_option_value(horizon, players, state, extra=1),
                    "bank_half": bank_option_value(horizon, players, state, extra=5),
                    "bank_million": bank_option_value(horizon, players, state, extra=10),
                }
            )
        log.info("%s: %d weeks measured from %s", season, len(gameweeks) - 1, path.name)
    return pd.DataFrame(rows)


def bench_weight_from_traces(traces: list[Path]) -> tuple[float, pd.DataFrame]:
    """Points the substitutes delivered as a share of what the plan expected of the bench."""
    frames = [pd.read_csv(p) for p in traces]
    if not frames:
        return 0.0, pd.DataFrame()
    table = pd.concat(frames, ignore_index=True)
    table = table[table["chip"] != "bboost"]
    per_season = table.groupby("season").agg(
        sub_points=("sub_points", "sum"),
        bench_expected=("bench_expected", "sum"),
        auto_subs=("auto_subs", "sum"),
        weeks=("gameweek", "size"),
    )
    per_season["weight"] = per_season["sub_points"] / per_season["bench_expected"]
    overall = float(per_season["sub_points"].sum() / max(per_season["bench_expected"].sum(), 1e-9))
    return overall, per_season.round(3).reset_index()
