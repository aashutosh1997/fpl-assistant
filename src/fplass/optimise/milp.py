"""Multi-gameweek squad, transfer and chip optimisation.

Single-gameweek greedy transfers are the classic FPL mistake. Buying the best player for next week
repeatedly walks you into a corner: no budget, three players from one club, a wildcard burned in
October, and a bench that cannot support a Bench Boost. The decisions are coupled across time
through money, free transfers and the chips, so they have to be solved together.

This is a mixed-integer program over a planning horizon (eight gameweeks by default). It decides,
jointly:

* which fifteen players to own in each gameweek;
* which eleven to start, in a legal formation, and who to captain;
* which transfers to make, when to take a points hit, and when to bank a free transfer;
* when to play each chip, respecting the real windows.

The objective is expected points net of transfer hits. That is deliberately only the *first* stage:
a mean-optimal plan is not necessarily the plan that wins a mini-league, so :mod:`fplass.optimise.
league` re-scores a pool of near-optimal plans from this solver against simulated rivals. The MILP
exists to propose strong candidates cheaply, not to have the final word.

Two things are modelled with more care than usual, because they are where plans usually go wrong:

**Selling prices.** FPL charges a 50% sell-on fee on profit, rounded down to 0.1m. So a player
bought at 7.0 and now worth 7.5 sells for 7.2, not 7.5. Ignoring this makes every plan look richer
than it is and produces recommendations that are simply infeasible when you try to execute them.

**Free transfer accrual.** You gain one per gameweek and may bank up to five. A plan that takes a
-4 hit in gameweek two to save a free transfer that expires unused in gameweek three is worse than
doing nothing, and only a multi-week formulation can see that.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import pulp

log = logging.getLogger(__name__)

POSITIONS = ("GKP", "DEF", "MID", "FWD")
SQUAD_QUOTA = {"GKP": 2, "DEF": 5, "MID": 5, "FWD": 3}
LINEUP_MIN = {"GKP": 1, "DEF": 3, "MID": 2, "FWD": 1}
LINEUP_MAX = {"GKP": 1, "DEF": 5, "MID": 5, "FWD": 3}

SQUAD_SIZE = 15
LINEUP_SIZE = 11
CLUB_LIMIT = 3
HIT_COST = 4.0
MAX_BANKED_TRANSFERS = 5
SELL_ON_FEE = 0.5

CHIPS = ("wildcard", "freehit", "bboost", "3xc")


@dataclass(slots=True)
class SquadState:
    """The squad you own right now, and the money and transfers available."""

    players: dict[int, int]  # element -> purchase price, in FPL tenths
    bank: int = 0  # tenths of a million
    free_transfers: int = 1
    chips_used: set[str] = field(default_factory=set)  # e.g. {"wildcard:1", "bboost:1"}

    @property
    def elements(self) -> set[int]:
        return set(self.players)

    def selling_price(self, element: int, current_price: int) -> int:
        """What FPL pays you for a player, after the 50% sell-on fee on profit.

        Profit is halved and rounded *down* to the nearest 0.1m, which is why a 0.1m rise returns
        nothing at all and a 0.3m rise returns only 0.1m.
        """
        bought = self.players.get(element)
        if bought is None:
            return current_price
        if current_price <= bought:
            return current_price
        profit = current_price - bought
        return bought + int(np.floor(profit * SELL_ON_FEE))


@dataclass(slots=True)
class ChipWindows:
    """Legal gameweek ranges per chip, as published by the API.

    Read from the warehouse rather than hardcoded. For 2026/27 the windows are asymmetric —
    Bench Boost and Triple Captain are legal from gameweek 1, Wildcard and Free Hit only from
    gameweek 2 — and each chip has a second window in the back half of the season.
    """

    windows: dict[str, list[tuple[int, int]]]

    @classmethod
    def from_warehouse(cls, con, season: str) -> ChipWindows:
        rows = con.execute(
            "SELECT name, start_event, stop_event FROM chips WHERE season = ? "
            "ORDER BY name, start_event",
            [season],
        ).fetchall()
        windows: dict[str, list[tuple[int, int]]] = {}
        for name, start, stop in rows:
            windows.setdefault(name, []).append((int(start), int(stop)))
        if not windows:
            raise ValueError(
                f"no chip windows stored for {season}; run `fpl ingest current` first"
            )
        return cls(windows=windows)

    def legal(self, chip: str, gameweek: int) -> list[int]:
        """Indices of the windows (chip instances) that permit playing ``chip`` in ``gameweek``."""
        return [
            i
            for i, (start, stop) in enumerate(self.windows.get(chip, []))
            if start <= gameweek <= stop
        ]


@dataclass(slots=True)
class Plan:
    """A solved multi-gameweek plan."""

    squads: dict[int, list[int]]  # gameweek -> 15 elements
    lineups: dict[int, list[int]]
    captains: dict[int, int]
    transfers_in: dict[int, list[int]]
    transfers_out: dict[int, list[int]]
    hits: dict[int, int]
    chips: dict[int, str]  # gameweek -> chip name
    objective: float
    expected_points: dict[int, float]
    status: str

    def summary(self, names: dict[int, str]) -> str:
        lines = []
        for gw in sorted(self.squads):
            chip = f"  [{self.chips[gw].upper()}]" if gw in self.chips else ""
            out = ", ".join(names.get(e, str(e)) for e in self.transfers_out.get(gw, []))
            into = ", ".join(names.get(e, str(e)) for e in self.transfers_in.get(gw, []))
            move = f"  {out} -> {into}" if into or out else "  (roll)"
            hit = f"  -{self.hits.get(gw, 0) * int(HIT_COST)}" if self.hits.get(gw) else ""
            captain = names.get(self.captains.get(gw, -1), "?")
            lines.append(
                f"GW{gw}: {self.expected_points.get(gw, 0):5.1f} pts"
                f"{chip}{move}{hit}  (C) {captain}"
            )
        return "\n".join(lines)

    def lineups_summary(
        self,
        names: dict[int, str],
        positions: dict[int, str],
        expected_points: "pd.DataFrame | None" = None,
    ) -> str:
        """Each gameweek's starting eleven by position, the bench, and the moves that got there.

        This is the view a manager actually checks a plan against: who starts, who sits, and
        whether the bench is rotating into the side or dead weight. Per-player expected points
        are shown when ``expected_points`` (players by gameweek) is given.
        """
        order = ("GKP", "DEF", "MID", "FWD")

        def label(e: int, gw: int) -> str:
            name = names.get(e, str(e))
            if expected_points is not None and e in expected_points.index and gw in expected_points.columns:
                name += f" {float(expected_points.at[e, gw]):.1f}"
            return name

        lines = []
        for gw in sorted(self.squads):
            lineup = [e for e in self.lineups.get(gw, [])]
            bench = [e for e in self.squads.get(gw, []) if e not in set(lineup)]
            captain = self.captains.get(gw)
            chip = f" [{self.chips[gw].upper()}]" if gw in self.chips else ""
            out = ", ".join(names.get(e, str(e)) for e in self.transfers_out.get(gw, []))
            into = ", ".join(names.get(e, str(e)) for e in self.transfers_in.get(gw, []))
            hit = f" (-{self.hits.get(gw, 0) * int(HIT_COST)})" if self.hits.get(gw) else ""
            move = f"{out} -> {into}{hit}" if into or out else "roll"
            lines.append(f"GW{gw}{chip}: {move}")
            for position in order:
                players = [e for e in lineup if positions.get(e) == position]
                if not players:
                    continue
                shown = ", ".join(
                    label(e, gw) + (" (C)" if e == captain else "") for e in players
                )
                lines.append(f"  {position}: {shown}")
            lines.append("  bench: " + ", ".join(label(e, gw) for e in bench))
        return "\n".join(lines)


def _best_solver(*, msg: bool, time_limit: int):
    """Pick the fastest available solver.

    HiGHS (via ``highspy``) is several times quicker than the bundled CBC on a problem this size,
    which matters because the league-aware stage solves this repeatedly to build a plan pool. CBC
    ships with PuLP, so it is the fallback rather than a hard dependency.
    """
    for factory in (
        lambda: pulp.HiGHS(msg=msg, timeLimit=time_limit),
        lambda: pulp.HiGHS_CMD(msg=msg, timeLimit=time_limit),
        lambda: pulp.PULP_CBC_CMD(msg=msg, timeLimit=time_limit),
    ):
        try:
            solver = factory()
            if solver.available():
                return solver
        except (AttributeError, pulp.PulpSolverError):
            continue
    raise RuntimeError("no MILP solver available; install highspy or pulp's CBC")


def shortlist(
    expected_points: pd.DataFrame,
    players: pd.DataFrame,
    current: SquadState,
    *,
    per_position: int = 40,
    must_keep: set[int] | None = None,
) -> pd.DataFrame:
    """Cut the player pool down to a tractable, near-lossless shortlist.

    Six hundred players over eight gameweeks is a large integer program and most of it is dead
    weight — fourth-choice goalkeepers cannot appear in an optimal plan. We keep the best players
    per position by total expected points, the best per position *per price bracket* so cheap
    enablers survive (an optimal plan often needs a 4.0m bench defender that no ranking by points
    would ever surface), and everyone currently owned, since selling requires them to be modelled.
    """
    totals = expected_points.sum(axis=1).rename("ep_total")
    pool = players.join(totals, on="element")
    pool["ep_total"] = pool["ep_total"].fillna(0.0)

    keep: set[int] = set(current.elements) | set(must_keep or ())
    for position, group in pool.groupby("position"):
        keep |= set(group.nlargest(per_position, "ep_total")["element"])
        # Cheap enablers: the best player at each price point, so the solver can always find a
        # legal bench inside budget.
        for _, bracket in group.groupby("price"):
            keep |= set(bracket.nlargest(1, "ep_total")["element"])

    shortlisted = pool[pool["element"].isin(keep)].copy()
    log.info(
        "shortlist: %d of %d players (%d owned)",
        len(shortlisted),
        len(pool),
        len(current.elements),
    )
    return shortlisted


def solve(
    expected_points: pd.DataFrame,
    players: pd.DataFrame,
    current: SquadState,
    chip_windows: ChipWindows,
    *,
    gameweeks: list[int] | None = None,
    bench_weight: float = 0.12,
    banked_transfer_value: float = 0.25,
    allow_chips: bool = True,
    chip_schedule: dict[int, str] | None = None,
    forbidden: list[dict] | None = None,
    lock: dict[int, list[int]] | None = None,
    time_limit: int = 120,
    msg: bool = False,
) -> Plan:
    """Solve the multi-gameweek plan.

    Args:
        expected_points: Players (index ``element``) by gameweek. Typically the mean of the Monte
            Carlo samples.
        players: Must carry ``element``, ``position``, ``team_id``, ``price`` (tenths).
        current: Your current squad, bank and free transfers.
        chip_windows: Legal chip windows from the API.
        gameweeks: Which gameweeks to plan. Defaults to the columns of ``expected_points``.
        bench_weight: How much a bench place is worth relative to a starting place. Bench points
            only materialise through automatic substitutions, so this is well below 1 — but it is
            not zero, and setting it to zero produces squads with unplayable benches that make
            Bench Boost worthless.
        banked_transfer_value: Points value of carrying a free transfer into the next gameweek.
            Without this the solver is *indifferent* between using a free transfer and banking it
            whenever the gain is zero, and ties break arbitrarily — so it cheerfully recommends
            pointless sideways moves. A banked transfer genuinely is worth something: it is the
            option to react to an injury next week. Pricing it also makes "roll your transfer" an
            answer the model can actually give, which is frequently the correct advice.
        allow_chips: Set ``False`` to plan transfers only.
        chip_schedule: Pin chips to specific gameweeks, as ``{gameweek: chip}``. Chips not
            mentioned are forbidden inside this horizon. This is how a whole-season chip roadmap
            (see :mod:`fplass.optimise.chips`) is enforced: left to itself over an eight-week
            horizon the solver will always burn every chip it can, because a chip saved past the
            horizon appears worthless to it.
        lock: Players that must be in the squad in given gameweeks, as ``{gameweek: [elements]}``.
            The honest way to ask "what does the best plan with X look like": everything else
            is re-optimised around the constraint, rather than X being bribed into the plan with
            inflated points, which also makes him captain and distorts the rest of the squad.
        forbidden: Previously found solutions to exclude, so the solver can be called repeatedly
            to build a pool of distinct near-optimal plans for the league-aware stage.
        time_limit: Solver time limit in seconds.

    Returns:
        The solved :class:`Plan`.
    """
    gameweeks = gameweeks or [int(c) for c in expected_points.columns]
    lock = {int(gw): [int(e) for e in elements] for gw, elements in (lock or {}).items()}
    pool = shortlist(
        expected_points, players, current, must_keep={e for es in lock.values() for e in es}
    )

    elements = pool["element"].tolist()
    position = dict(zip(pool["element"], pool["position"], strict=True))
    team = dict(zip(pool["element"], pool["team_id"], strict=True))
    price = dict(zip(pool["element"], pool["price"].astype(int), strict=True))
    sell = {e: current.selling_price(e, price[e]) for e in elements}

    points = {
        (e, gw): float(expected_points.at[e, gw]) if e in expected_points.index else 0.0
        for e in elements
        for gw in gameweeks
    }

    problem = pulp.LpProblem("fpl_plan", pulp.LpMaximize)

    def binaries(name: str) -> dict:
        return pulp.LpVariable.dicts(
            name, ((e, gw) for e in elements for gw in gameweeks), cat="Binary"
        )

    squad = binaries("squad")
    lineup = binaries("lineup")
    captain = binaries("captain")
    bought = binaries("in")
    sold = binaries("out")
    # Free-hit squads are separate: the chip changes your team for one gameweek only, then the
    # previous squad returns. Without distinct variables a free hit would corrupt every later week.
    fh_squad = binaries("fh_squad")
    fh_lineup = binaries("fh_lineup")
    # Bench places, tracked explicitly rather than as (squad - lineup), so that a free-hit
    # gameweek — where the real squad sits out entirely — does not credit bench points for
    # fifteen players who are not playing.
    bench = binaries("bench")

    chip_play: dict[tuple[str, int, int], pulp.LpVariable] = {}
    if allow_chips:
        for chip in CHIPS:
            for gw in gameweeks:
                for window in chip_windows.legal(chip, gw):
                    if f"{chip}:{window}" in current.chips_used:
                        continue
                    chip_play[chip, window, gw] = pulp.LpVariable(
                        f"chip_{chip}_{window}_{gw}", cat="Binary"
                    )

    def chip_active(chip: str, gw: int):
        parts = [v for (c, _, g), v in chip_play.items() if c == chip and g == gw]
        return pulp.lpSum(parts) if parts else 0

    free_transfers = pulp.LpVariable.dicts(
        "ft", gameweeks, lowBound=0, upBound=MAX_BANKED_TRANSFERS, cat="Integer"
    )
    hits = pulp.LpVariable.dicts("hits", gameweeks, lowBound=0, cat="Integer")
    # Free transfers spent in each gameweek (see the transfer economy below).
    # Bounded by the squad size, not the bank cap: the pre-season build starts with fifteen.
    used = pulp.LpVariable.dicts("used_ft", gameweeks, lowBound=0, upBound=SQUAD_SIZE, cat="Integer")
    # 1 when more transfers are made than there are free ones (so all free ones are spent).
    fewer_free = pulp.LpVariable.dicts("fewer_free", gameweeks, cat="Binary")

    # ---------------------------------------------------------------- squad rules

    for gw in gameweeks:
        bb = chip_active("bboost", gw)
        free_hit = chip_active("freehit", gw)

        problem += pulp.lpSum(squad[e, gw] for e in elements) == SQUAD_SIZE
        for locked in lock.get(gw, []):
            if locked in elements:
                problem += squad[locked, gw] == 1

        # The free-hit squad exists *only* in a gameweek where the chip is played. Gating it on
        # the chip is essential, not tidiness: left free, it acts as a second team that scores
        # every week, which doubled the objective and let the solver captain a player it did not
        # own.
        problem += pulp.lpSum(fh_squad[e, gw] for e in elements) == SQUAD_SIZE * free_hit

        for pos, quota in SQUAD_QUOTA.items():
            members = [e for e in elements if position[e] == pos]
            problem += pulp.lpSum(squad[e, gw] for e in members) == quota
            problem += pulp.lpSum(fh_squad[e, gw] for e in members) == quota * free_hit

        for club in set(team.values()):
            members = [e for e in elements if team[e] == club]
            problem += pulp.lpSum(squad[e, gw] for e in members) <= CLUB_LIMIT
            problem += pulp.lpSum(fh_squad[e, gw] for e in members) <= CLUB_LIMIT

        # Exactly one team is fielded each gameweek: the real squad, or the free-hit squad.
        # Bench Boost starts all fifteen, so the lineup grows by four when it is active. Only one
        # chip may be played per gameweek, so bb and free_hit are never both 1 and the products
        # below stay linear.
        problem += (
            pulp.lpSum(lineup[e, gw] for e in elements) == (LINEUP_SIZE + 4 * bb) - LINEUP_SIZE * free_hit
        )
        problem += pulp.lpSum(fh_lineup[e, gw] for e in elements) == LINEUP_SIZE * free_hit

        for pos in POSITIONS:
            members = [e for e in elements if position[e] == pos]
            started = pulp.lpSum(lineup[e, gw] for e in members)
            # Under Bench Boost every squad member plays, so the formation bounds widen to the
            # squad quota rather than the outfield limits.
            problem += started >= LINEUP_MIN[pos] * (1 - free_hit)
            problem += started <= LINEUP_MAX[pos] + (SQUAD_QUOTA[pos] - LINEUP_MAX[pos]) * bb
            fh_started = pulp.lpSum(fh_lineup[e, gw] for e in members)
            problem += fh_started >= LINEUP_MIN[pos] * free_hit
            problem += fh_started <= LINEUP_MAX[pos] * free_hit

        for e in elements:
            # You can only start a player you own, from whichever squad is in play.
            problem += lineup[e, gw] <= squad[e, gw]
            problem += fh_lineup[e, gw] <= fh_squad[e, gw]
            problem += captain[e, gw] <= lineup[e, gw] + fh_lineup[e, gw]
            # Upper bounds only: bench carries a positive objective weight, so maximisation
            # drives it to the tighter of the two limits without needing an equality.
            problem += bench[e, gw] <= squad[e, gw] - lineup[e, gw]
            problem += bench[e, gw] <= 1 - free_hit
        problem += pulp.lpSum(captain[e, gw] for e in elements) == 1

    # ------------------------------------------------------------ squad transitions

    previous = {e: 1 if e in current.elements else 0 for e in elements}
    for index, gw in enumerate(gameweeks):
        for e in elements:
            prior = previous[e] if index == 0 else squad[e, gameweeks[index - 1]]
            problem += squad[e, gw] == prior + bought[e, gw] - sold[e, gw]
            problem += bought[e, gw] + sold[e, gw] <= 1

    # --------------------------------------------------------------------- budget

    # Money is tracked as a running balance. Selling uses the fee-adjusted price; buying uses the
    # market price. A free hit is budget-constrained too, but against the squad you actually own.
    for index, gw in enumerate(gameweeks):
        spent = pulp.lpSum(price[e] * bought[e, gw] for e in elements)
        raised = pulp.lpSum(sell[e] * sold[e, gw] for e in elements)
        if index == 0:
            problem += spent - raised <= current.bank
            running = current.bank - spent + raised
        else:
            problem += spent - raised <= running
            running = running - spent + raised

        # The free-hit squad must be affordable from the value of the squad you hold plus the bank.
        holdings = pulp.lpSum(sell[e] * squad[e, gw] for e in elements)
        problem += pulp.lpSum(price[e] * fh_squad[e, gw] for e in elements) <= holdings + running

    # ------------------------------------------------------------- transfer economy

    for index, gw in enumerate(gameweeks):
        made = pulp.lpSum(bought[e, gw] for e in elements)
        wildcard = chip_active("wildcard", gw)
        free_hit = chip_active("freehit", gw)
        unlimited = wildcard + free_hit

        available = current.free_transfers if index == 0 else free_transfers[gameweeks[index - 1]]

        # Free transfers actually spent this week: as many of the transfers made as there were
        # free ones available, and none at all under a wildcard or free hit, whose transfers are
        # outside the economy. Modelling this explicitly matters. The earlier form let the solver
        # take *extra* hits to bank extra free transfers — buying a transfer for four points now
        # rather than later, which costs the same but reported a three-transfer week as -12 and,
        # with the small bonus on banked transfers, nudged plans toward paying hits early.
        problem += used[gw] <= made
        problem += used[gw] <= available
        problem += used[gw] <= SQUAD_SIZE * (1 - unlimited)
        # ...and no fewer than that: free transfers are always consumed before a hit is paid.
        # Without the lower bound the solver could decline to spend a free transfer, pay a hit
        # instead and bank the free one for later — legal in the algebra, not in the game.
        # `min(made, available)` is linearised with one binary per gameweek.
        problem += used[gw] >= made - SQUAD_SIZE * fewer_free[gw] - SQUAD_SIZE * unlimited
        problem += (
            used[gw] >= available - SQUAD_SIZE * (1 - fewer_free[gw]) - SQUAD_SIZE * unlimited
        )

        # Hits are exactly the transfers beyond the free ones spent (never under a chip).
        # SQUAD_SIZE is a safe big-M: you cannot make more than fifteen transfers.
        problem += hits[gw] >= made - used[gw] - SQUAD_SIZE * unlimited
        problem += hits[gw] <= made - used[gw]

        # Free transfers accrue one per week on whatever was not spent, capped by the variable's
        # bound; a wildcard leaves the count untouched because its transfers are free.
        problem += free_transfers[gw] <= available - used[gw] + 1

        # One chip per gameweek.
        if allow_chips:
            active = [v for (_, _, g), v in chip_play.items() if g == gw]
            if active:
                problem += pulp.lpSum(active) <= 1

    # Each chip instance can be played at most once across the horizon.
    if allow_chips:
        for chip in CHIPS:
            for window in range(len(chip_windows.windows.get(chip, []))):
                instances = [
                    v for (c, w, _), v in chip_play.items() if c == chip and w == window
                ]
                if instances:
                    problem += pulp.lpSum(instances) <= 1

    # A pinned roadmap: force the scheduled chips and forbid everything else. Where a gameweek
    # falls inside more than one window for the same chip, exactly one instance is forced —
    # forcing both would contradict the once-per-instance constraint and make the model infeasible.
    if allow_chips and chip_schedule is not None:
        forced: set[tuple[str, int]] = set()
        for key in sorted(chip_play, key=lambda k: (k[2], k[0], k[1])):
            chip, _, gw = key
            wanted = chip_schedule.get(gw) == chip and (chip, gw) not in forced
            problem += chip_play[key] == (1 if wanted else 0)
            if wanted:
                forced.add((chip, gw))
        missing = {
            gw: chip
            for gw, chip in chip_schedule.items()
            if gw in gameweeks and (chip, gw) not in forced
        }
        if missing:
            log.warning("chip schedule not satisfiable inside this horizon: %s", missing)

    # ------------------------------------------------------------------- objective

    contributions = []
    for gw in gameweeks:
        triple = chip_active("3xc", gw)
        for e in elements:
            ep = points[e, gw]
            if ep == 0:
                continue
            # Started players score once; the captain doubles, or triples under the chip. The
            # triple-captain term is linearised as an extra captain-equivalent, valid because
            # only one captain exists per gameweek.
            contributions.append(ep * lineup[e, gw])
            contributions.append(ep * fh_lineup[e, gw])
            contributions.append(ep * captain[e, gw])
            contributions.append(ep * bench_weight * bench[e, gw])
        if isinstance(triple, pulp.LpAffineExpression) or triple != 0:
            # Value the triple captain at the best available captain's points. Using a fixed
            # premium keeps the model linear; the pool re-scoring stage evaluates the real
            # distribution, which is what a triple captain should actually be chosen on.
            best = max((points[e, gw] for e in elements), default=0.0)
            contributions.append(best * triple)

    problem += (
        pulp.lpSum(contributions)
        - HIT_COST * pulp.lpSum(hits[gw] for gw in gameweeks)
        + banked_transfer_value * pulp.lpSum(free_transfers[gw] for gw in gameweeks)
    )

    # No-good cuts, so repeated calls yield genuinely different plans rather than the same one.
    for previous_plan in forbidden or []:
        for gw, chosen in previous_plan.items():
            in_plan = [squad[e, gw] for e in chosen if (e, gw) in squad]
            if in_plan:
                problem += pulp.lpSum(in_plan) <= len(in_plan) - 1

    problem.solve(_best_solver(msg=msg, time_limit=time_limit))

    status = pulp.LpStatus[problem.status]
    if status not in {"Optimal", "Not Solved"}:
        log.warning("solver returned status %s", status)

    return _extract(
        problem,
        status,
        gameweeks,
        elements,
        squad,
        lineup,
        captain,
        bought,
        sold,
        hits,
        fh_squad,
        fh_lineup,
        chip_play,
        points,
    )


def _value(variable) -> float:
    if isinstance(variable, (int, float)):
        return float(variable)
    value = variable.value()
    return 0.0 if value is None else float(value)


def _extract(
    problem,
    status,
    gameweeks,
    elements,
    squad,
    lineup,
    captain,
    bought,
    sold,
    hits,
    fh_squad,
    fh_lineup,
    chip_play,
    points,
) -> Plan:
    """Read the solution back out into a :class:`Plan`."""
    chips: dict[int, str] = {}
    for (chip, _, gw), variable in chip_play.items():
        if _value(variable) > 0.5:
            chips[gw] = chip

    squads, lineups, captains, ins, outs, hit_counts, expected = {}, {}, {}, {}, {}, {}, {}
    for gw in gameweeks:
        free_hit = chips.get(gw) == "freehit"
        squad_vars = fh_squad if free_hit else squad
        lineup_vars = fh_lineup if free_hit else lineup

        squads[gw] = [e for e in elements if _value(squad_vars[e, gw]) > 0.5]
        lineups[gw] = [e for e in elements if _value(lineup_vars[e, gw]) > 0.5]
        picked = [e for e in elements if _value(captain[e, gw]) > 0.5]
        captains[gw] = picked[0] if picked else -1
        ins[gw] = [e for e in elements if _value(bought[e, gw]) > 0.5]
        outs[gw] = [e for e in elements if _value(sold[e, gw]) > 0.5]
        hit_counts[gw] = int(round(_value(hits[gw])))

        multiplier = 3 if chips.get(gw) == "3xc" else 2
        expected[gw] = sum(points[e, gw] for e in lineups[gw]) + (multiplier - 1) * points.get(
            (captains[gw], gw), 0.0
        )

    return Plan(
        squads=squads,
        lineups=lineups,
        captains=captains,
        transfers_in=ins,
        transfers_out=outs,
        hits=hit_counts,
        chips=chips,
        objective=float(pulp.value(problem.objective) or 0.0),
        expected_points=expected,
        status=status,
    )
