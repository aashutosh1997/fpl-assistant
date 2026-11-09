"""What the learner sees at a deadline: a few dozen numbers about the squad, the money and time.

Every feature is something the planner already computes or could: the best swaps the squad has
(by the horizon projection and by this week's), what a Bench Boost or Triple Captain would add
this week, how far this week's best eleven falls short of the best eleven money cannot buy (the
Free Hit's raw material), the free transfers and bank, which chips are left and how many weeks
each window has to run, and the doubles and blanks now and just ahead. Nothing names a player,
so a policy fitted on eight seasons can be played on a ninth.

The time features count the season's calendar, not gameweek labels: 2019-20 runs from 29 to 39
across the pandemic and 2022-23 has 37 gameweeks.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..optimise import milp
from ..options.value import Swap, swap_gains
from .season import SeasonData

CHIPS = ("wildcard", "freehit", "bboost", "3xc")
# The learner's actions: a transfer count with or without a chip played alongside, or a squad
# chip instead of transfers.
MAX_TRANSFERS = 3
ACTIONS: tuple[tuple[int, str | None], ...] = (
    *((k, None) for k in range(MAX_TRANSFERS + 1)),
    *((k, "bboost") for k in range(MAX_TRANSFERS + 1)),
    *((k, "3xc") for k in range(MAX_TRANSFERS + 1)),
    (0, "wildcard"),
    (0, "freehit"),
)
N_ACTIONS = len(ACTIONS)
ROLL = ACTIONS.index((0, None))
SWAPS = 5
LOOKAHEAD = 3  # weeks of doubles and blanks ahead

FEATURES: tuple[str, ...] = (
    "progress", "weeks_left", "free_transfers", "bank", "value",
    *(f"{c}_{k}" for c in CHIPS for k in ("now", "window_left", "later")),
    "chips_now",
    *(f"gain_{i}" for i in range(1, SWAPS + 1)),
    "swaps",
    *(f"week_gain_{i}" for i in range(1, 4)),
    "xi_week", "bench_week", "captain_week", "vice_week", "xi_horizon", "best_xi_gap",
    "squad_blanks", "squad_doubles",
    "league_doubles", "league_blanks", "league_doubles_ahead", "league_blanks_ahead",
)


def action_index(transfers: int, chip: str | None) -> int:
    if chip in ("wildcard", "freehit"):
        return ACTIONS.index((0, chip))
    return ACTIONS.index((min(int(transfers), MAX_TRANSFERS), chip))


@dataclass(slots=True)
class Observation:
    """One deadline, as the learner sees it, with what was computed on the way."""

    gw: int
    t: int
    vector: np.ndarray
    mask: np.ndarray
    named: dict[str, float]
    swaps: list[Swap] = field(default_factory=list)


def chip_status(
    data: SeasonData, t: int, chips_used: set[str], chip: str
) -> tuple[bool, int, bool]:
    """Whether a chip can be played now, how many weeks its window has left, and whether an
    instance in a later window is still unplayed."""
    gw = data.calendar[t]
    ahead = data.calendar[t:]
    windows = data.windows.windows.get(chip, [])
    now = [w for w in data.windows.legal(chip, gw) if f"{chip}:{w}" not in chips_used]
    left = 0
    if now:
        start, stop = windows[now[0]]
        left = sum(1 for g in ahead if start <= g <= stop)
    later = any(
        f"{chip}:{w}" not in chips_used and start > gw and any(start <= g <= stop for g in ahead)
        for w, (start, stop) in enumerate(windows)
    )
    return bool(now), left, later


def best_eleven(elements, week: dict[int, float], positions: dict[int, str]) -> list[int]:
    """The best legal eleven by the week's projection: a keeper, the formation's minimum, the
    best of the rest within the maximum per position."""
    ranked = sorted(elements, key=lambda e: -week.get(e, 0.0))
    keepers = [e for e in ranked if positions.get(e) == "GKP"][:1]
    chosen: list[int] = []
    counts = {"DEF": 0, "MID": 0, "FWD": 0}
    for pos, need in milp.LINEUP_MIN.items():
        if pos == "GKP":
            continue
        picks = [e for e in ranked if positions.get(e) == pos][:need]
        chosen += picks
        counts[pos] += len(picks)
    for e in ranked:
        if len(chosen) >= milp.LINEUP_SIZE - 1:
            break
        pos = positions.get(e)
        if pos == "GKP" or e in chosen or pos not in counts:
            continue
        if counts[pos] < milp.LINEUP_MAX[pos]:
            chosen.append(e)
            counts[pos] += 1
    return keepers + chosen


def observe(data: SeasonData, t: int, state: milp.SquadState) -> Observation:
    """The features and the legal actions at the ``t``-th deadline of the season."""
    gw = data.calendar[t]
    n = len(data.calendar)
    players = data.players[gw]
    positions = data.position_of
    price = data.prices(gw)
    week_series = data.week(gw)
    week = week_series.to_dict()
    horizon = data.horizon_ep(gw)
    squad = list(state.players)

    named: dict[str, float] = {
        "progress": t / n,
        "weeks_left": float(n - t),
        "free_transfers": float(state.free_transfers),
        "bank": state.bank / 10,
        "value": (state.bank + sum(state.selling_price(e, price.get(e, p)) for e, p in state.players.items())) / 10,
    }
    status = {}
    for chip in CHIPS:
        now, left, later = chip_status(data, t, state.chips_used, chip)
        status[chip] = now
        named[f"{chip}_now"] = float(now)
        named[f"{chip}_window_left"] = float(left)
        named[f"{chip}_later"] = float(later)
    named["chips_now"] = float(sum(status.values()))

    swaps = swap_gains(horizon, players, state, top=SWAPS) if t > 0 else []
    for i in range(SWAPS):
        named[f"gain_{i + 1}"] = swaps[i].gain if i < len(swaps) else 0.0
    named["swaps"] = float(len(swaps))
    for i in range(3):
        named[f"week_gain_{i + 1}"] = (
            week.get(swaps[i].into, 0.0) - week.get(swaps[i].out, 0.0) if i < len(swaps) else 0.0
        )

    xi = best_eleven(squad, week, positions)
    bench = [e for e in squad if e not in set(xi)]
    ranked = sorted((week.get(e, 0.0) for e in xi), reverse=True)
    named["xi_week"] = float(sum(ranked))
    named["bench_week"] = float(sum(week.get(e, 0.0) for e in bench))
    named["captain_week"] = float(ranked[0]) if ranked else 0.0
    named["vice_week"] = float(ranked[1]) if len(ranked) > 1 else 0.0
    named["xi_horizon"] = float(horizon.reindex(squad).fillna(0.0).nlargest(milp.LINEUP_SIZE).sum())
    pool = week_series[week_series > 0].nlargest(120).index
    best = best_eleven(list(pool), week, positions)
    named["best_xi_gap"] = float(sum(week.get(e, 0.0) for e in best) - named["xi_week"])

    teams = dict(zip(players["element"].astype(int), players["team_id"].astype(int), strict=True))
    fixtures = data.fixtures.get(gw, {})
    counts = [fixtures.get(teams.get(e, -1), 0) for e in squad]
    named["squad_blanks"] = float(sum(c == 0 for c in counts))
    named["squad_doubles"] = float(sum(c >= 2 for c in counts))
    named["league_doubles"] = float(sum(c >= 2 for c in fixtures.values()))
    named["league_blanks"] = float(sum(c == 0 for c in fixtures.values()))
    ahead = data.calendar[t + 1 : t + 1 + LOOKAHEAD]
    named["league_doubles_ahead"] = float(
        sum(sum(c >= 2 for c in data.fixtures.get(g, {}).values()) for g in ahead)
    )
    named["league_blanks_ahead"] = float(
        sum(sum(c == 0 for c in data.fixtures.get(g, {}).values()) for g in ahead)
    )

    mask = legal_actions(t, len(swaps), status)
    vector = np.array([named[f] for f in FEATURES], dtype="float64")
    return Observation(gw=gw, t=t, vector=vector, mask=mask, named=named, swaps=swaps)


def legal_actions(t: int, swaps: int, chips_now: dict[str, bool]) -> np.ndarray:
    """Rolling is always legal; a transfer count needs that many candidate swaps; a chip needs
    an unplayed instance whose window is open. The first deadline builds the squad, so only a
    chip played alongside it is a choice."""
    mask = np.zeros(N_ACTIONS, dtype=bool)
    for i, (k, chip) in enumerate(ACTIONS):
        if t == 0 and (k > 0 or chip in ("wildcard", "freehit")):
            continue
        if k > min(swaps, MAX_TRANSFERS):
            continue
        if chip is not None and not chips_now.get(chip, False):
            continue
        mask[i] = True
    return mask


def frame(observations: list[Observation]) -> pd.DataFrame:
    """Observations as a table, for inspection."""
    return pd.DataFrame([o.named for o in observations]).assign(
        gw=[o.gw for o in observations], t=[o.t for o in observations]
    )
