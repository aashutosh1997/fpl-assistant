"""Whole-season chip strategy.

Chips are where the largest single-decision swings in FPL live, and where a short planning horizon
fails most badly. Asked to plan eight weeks, the transfer solver will always play every chip it is
allowed to, because a chip saved past the horizon looks worthless to it. Left unchecked it produced
a Bench Boost in gameweek 1 followed by a Free Hit in gameweek 2 — each locally defensible, jointly
absurd.

So chip timing is decided here, across the whole remaining season, and handed to the solver as a
fixed roadmap.

Each chip is valued in the currency that actually matters — points gained *over not playing it*:

**Bench Boost** — the points your four bench players would score. Wants a gameweek where all
fifteen have fixtures, which in practice means a double gameweek.

**Triple Captain** — one extra copy of your captain's score. Valued on the *upside* rather than the
mean, because a triple captain is a bet on a ceiling: doubling a player's median is worth far less
than the chance of tripling a hat-trick. We report both.

**Free Hit** — the best legal eleven available in that single gameweek, minus what your own squad
would have scored. Wants a blank gameweek, where your own squad has the fewest players available.

**Wildcard** — unlike the others this is not a one-week payoff; it resets your squad for the rest of
the season, so it is valued by how far your squad has drifted from the best available team over the
following weeks.

**On the 2026/27 season specifically.** Every gameweek currently has exactly ten fixtures — no
blanks, no doubles. They appear later, as cup progress forces postponements. That has a concrete
strategic consequence, and the roadmap states it rather than hiding it: first-half chips (which
expire in early January) mostly have to be spent on fixture quality, while second-half chips should
be held for the doubles and blanks that will emerge. Everything here is recomputed from the live
fixture list on every run, so the roadmap updates itself as those materialise.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .milp import (
    CLUB_LIMIT,
    HIT_COST,
    LINEUP_MAX,
    LINEUP_MIN,
    LINEUP_SIZE,
    ChipWindows,
    Plan,
    SquadState,
    solve,
)

log = logging.getLogger(__name__)

# The value floors: a chip is only scheduled inside the horizon when it clears its floor, the
# rest of the time it is held. These are the planner's guess at each chip's option value; the
# paper manager is where they get measured.
DEFAULT_MIN_GAIN = {"bboost": 12.0, "3xc": 6.0, "freehit": 14.0, "wildcard": 12.0}

CHIP_LABELS = {
    "bboost": "Bench Boost",
    "3xc": "Triple Captain",
    "freehit": "Free Hit",
    "wildcard": "Wildcard",
}


@dataclass(slots=True)
class ChipValuation:
    """What one chip is worth in one gameweek."""

    chip: str
    gameweek: int
    window: int
    mean_gain: float
    upside_gain: float  # 90th percentile, which is what a Triple Captain is really bought for
    note: str = ""


@dataclass(slots=True)
class ChipRoadmap:
    """A recommended chip schedule, with the valuations behind it."""

    schedule: dict[int, str]  # gameweek -> chip
    valuations: pd.DataFrame
    notes: list[str]

    def summary(self) -> str:
        lines = ["Chip roadmap:"]
        if not self.schedule:
            lines.append("  (hold everything for now)")
        for gw in sorted(self.schedule):
            chip = self.schedule[gw]
            row = self.valuations[
                (self.valuations["chip"] == chip) & (self.valuations["gameweek"] == gw)
            ]
            gain = row["mean_gain"].iloc[0] if len(row) else float("nan")
            lines.append(f"  GW{gw:<3} {CHIP_LABELS.get(chip, chip):<15} +{gain:.1f} pts")
        lines += [f"  note: {n}" for n in self.notes]
        return "\n".join(lines)


def fixture_counts(con, season: str, gameweeks: list[int]) -> pd.DataFrame:
    """Fixtures per club per gameweek — the raw material for chip timing."""
    rows = con.execute(
        """
        WITH appearances AS (
            SELECT event, team_h AS team FROM fixtures WHERE season = ? AND event IS NOT NULL
            UNION ALL
            SELECT event, team_a AS team FROM fixtures WHERE season = ? AND event IS NOT NULL
        )
        SELECT event, team, count(*) AS fixtures
        FROM appearances GROUP BY event, team
        """,
        [season, season],
    ).fetchdf()

    clubs = con.execute(
        "SELECT team_id FROM teams WHERE season = ?", [season]
    ).fetchdf()["team_id"]
    grid = pd.MultiIndex.from_product(
        [gameweeks, clubs], names=["event", "team"]
    ).to_frame(index=False)
    return grid.merge(rows, on=["event", "team"], how="left").fillna({"fixtures": 0})


def double_blank_summary(con, season: str, gameweeks: list[int]) -> pd.DataFrame:
    """Count of doubling and blanking clubs per gameweek."""
    counts = fixture_counts(con, season, gameweeks)
    return (
        counts.assign(
            doubles=(counts["fixtures"] >= 2).astype(int),
            blanks=(counts["fixtures"] == 0).astype(int),
        )
        .groupby("event", as_index=False)[["doubles", "blanks"]]
        .sum()
    )


def value_bench_boost(
    samples: np.ndarray,
    elements: np.ndarray,
    gameweeks: np.ndarray,
    squad: list[int],
    lineup_by_gw: dict[int, list[int]],
) -> list[tuple[int, float, float]]:
    """Points the bench would add, per gameweek.

    Computed on the joint samples rather than on summed expectations, so the reported upside
    accounts for the fact that bench returns are correlated — a double gameweek lifts several
    bench players at once, which is exactly what makes the chip worth saving for.
    """
    index = {e: i for i, e in enumerate(elements)}
    results = []
    for slot, gw in enumerate(gameweeks):
        starters = set(lineup_by_gw.get(int(gw), []))
        bench = [e for e in squad if e not in starters]
        columns = [index[e] for e in bench if e in index]
        if not columns:
            results.append((int(gw), 0.0, 0.0))
            continue
        totals = samples[:, columns, slot].sum(axis=1)
        results.append((int(gw), float(totals.mean()), float(np.quantile(totals, 0.90))))
    return results


def value_triple_captain(
    samples: np.ndarray,
    elements: np.ndarray,
    gameweeks: np.ndarray,
    squad: list[int],
) -> list[tuple[int, float, float, int]]:
    """The extra copy of your captain's score, per gameweek.

    The captain is chosen per gameweek as the squad member with the best mean, then valued on the
    distribution. Mean and 90th percentile are both returned because they can disagree sharply:
    a reliable midfielder may have the better mean while a striker has the ceiling, and a Triple
    Captain is a bet on the ceiling.
    """
    index = {e: i for i, e in enumerate(elements)}
    columns = [index[e] for e in squad if e in index]
    results = []
    for slot, gw in enumerate(gameweeks):
        if not columns:
            results.append((int(gw), 0.0, 0.0, -1))
            continue
        block = samples[:, columns, slot]
        means = block.mean(axis=0)
        best = int(np.argmax(means))
        picked = block[:, best]
        results.append(
            (int(gw), float(picked.mean()), float(np.quantile(picked, 0.90)), squad[best])
        )
    return results


def value_free_hit(
    samples: np.ndarray,
    elements: np.ndarray,
    gameweeks: np.ndarray,
    players: pd.DataFrame,
    squad: list[int],
    lineup_by_gw: dict[int, list[int]],
    budget: int,
) -> list[tuple[int, float, float]]:
    """Best affordable eleven for a single gameweek, minus what your own eleven would score.

    The replacement eleven is chosen greedily under the formation, club and budget limits rather
    than by a second integer program. Greedy is close to optimal for a single gameweek — there is
    no cross-week coupling to get wrong — and keeps this cheap enough to evaluate at every
    gameweek of the season.
    """
    index = {e: i for i, e in enumerate(elements)}
    reference = players.set_index("element")
    results = []

    for slot, gw in enumerate(gameweeks):
        means = pd.Series(samples[:, :, slot].mean(axis=0), index=elements)
        own = [index[e] for e in lineup_by_gw.get(int(gw), squad[:LINEUP_SIZE]) if e in index]
        own_points = samples[:, own, slot].sum(axis=1) if own else np.zeros(samples.shape[0])

        candidates = reference.join(means.rename("ep")).dropna(subset=["ep"])
        candidates = candidates.sort_values("ep", ascending=False)

        chosen: list[int] = []
        by_position: dict[str, int] = dict.fromkeys(LINEUP_MIN, 0)
        by_club: dict[int, int] = {}
        spend = 0

        # Two passes: satisfy the formation minimums first, then fill to eleven with the best
        # remaining players. Without the first pass a greedy fill can leave no affordable
        # goalkeeper.
        for enforce_minimum in (True, False):
            for element, row in candidates.iterrows():
                if len(chosen) >= LINEUP_SIZE or element in chosen:
                    continue
                position = row["position"]
                if enforce_minimum and by_position[position] >= LINEUP_MIN[position]:
                    continue
                if by_position[position] >= LINEUP_MAX[position]:
                    continue
                if by_club.get(row["team_id"], 0) >= CLUB_LIMIT:
                    continue
                if spend + int(row["price"]) > budget:
                    continue
                chosen.append(element)
                by_position[position] += 1
                by_club[row["team_id"]] = by_club.get(row["team_id"], 0) + 1
                spend += int(row["price"])

        columns = [index[e] for e in chosen if e in index]
        replacement = samples[:, columns, slot].sum(axis=1) if columns else np.zeros(len(own_points))
        gain = replacement - own_points
        results.append((int(gw), float(gain.mean()), float(np.quantile(gain, 0.90))))
    return results


def plan_value(plan: Plan) -> float:
    """Expected points of a plan over its horizon, net of hits."""
    return float(sum(plan.expected_points.values()) - HIT_COST * sum(plan.hits.values()))


def value_wildcard(
    expected_points: pd.DataFrame,
    players: pd.DataFrame,
    state: SquadState,
    chip_windows: ChipWindows,
    baseline: Plan,
    *,
    candidates: int = 3,
    time_limit: int = 60,
    solve_options: dict | None = None,
) -> list[tuple[int, float, float]]:
    """Value a wildcard in each of the next few gameweeks against the chip-free plan.

    A wildcard is not a one-week payoff, so it cannot be read off the samples like the other
    chips: its worth is the gap between the best plan that rebuilds the squad in that gameweek
    and the best plan that does not, over the whole horizon. That means one solve per candidate
    gameweek, which is why only the next few are tried — the gain from rebuilding later is
    always available to re-measure next week, and the first-half wildcard's option value
    beyond the horizon is what the floor in :func:`build_roadmap` stands for.

    Until this existed the roadmap never scheduled a wildcard at all, and the advisor could not
    say whether a rebuild beat a string of hits. On gameweek 3 of 2026/27 it did, by fifteen
    points over eight gameweeks.

    Returns:
        ``(gameweek, mean_gain, upside)`` per candidate; upside is the gain itself, since the
        rebuilt squad's distribution is not sampled here.
    """
    reference = plan_value(baseline)
    horizon = [int(g) for g in expected_points.columns]
    rows: list[tuple[int, float, float]] = []
    tried = 0
    for gw in horizon:
        if tried >= candidates:
            break
        windows = chip_windows.legal("wildcard", gw)
        if not windows or all(f"wildcard:{w}" in state.chips_used for w in windows):
            continue
        tried += 1
        plan = solve(
            expected_points,
            players,
            state,
            chip_windows,
            chip_schedule={gw: "wildcard"},
            time_limit=time_limit,
            **(solve_options or {}),
        )
        if plan.status not in {"Optimal", "Not Solved"}:
            continue
        gain = plan_value(plan) - reference
        rows.append((gw, gain, gain))
    return rows


def build_roadmap(
    con,
    samples: np.ndarray,
    elements: np.ndarray,
    gameweeks: np.ndarray,
    players: pd.DataFrame,
    state: SquadState,
    baseline: Plan,
    chip_windows: ChipWindows,
    season: str,
    *,
    min_gain: dict[str, float] | None = None,
    wildcard_candidates: int = 3,
    solver_time_limit: int = 60,
    solve_options: dict | None = None,
) -> ChipRoadmap:
    """Value every chip in every legal gameweek and pick a schedule.

    Args:
        samples: Joint point samples over the horizon, ``(draws, players, gameweeks)``.
        baseline: A chip-free plan, supplying the squad and lineups each chip is measured against.
        min_gain: Minimum points gain before a chip is worth playing at all. This is the chip's
            option value: playing a Bench Boost for two points now forfeits the chance of a much
            better one later, so a floor prevents the solver frittering chips away.
        wildcard_candidates: How many of the next legal gameweeks to try a wildcard in; each
            costs a solve. Zero skips the wildcard entirely.
        solver_time_limit: Seconds per wildcard solve.
        solve_options: Extra keyword arguments for the wildcard solves (bench weight, banked
            transfer value, terminal value), so a chip is valued under the same economy the
            plan is.

    Returns:
        A :class:`ChipRoadmap`.
    """
    # Floors reflect what a genuinely good week for each chip looks like. A Bench Boost in a
    # double gameweek is routinely worth 20+; taking one worth 6 wastes the chip.
    min_gain = min_gain or dict(DEFAULT_MIN_GAIN)

    horizon = [int(g) for g in gameweeks]
    squad = baseline.squads.get(horizon[0], [])
    budget = sum(players.set_index("element")["price"].reindex(squad).fillna(0).astype(int))
    budget += state.bank

    rows: list[ChipValuation] = []

    for gw, mean_gain, upside in value_bench_boost(
        samples, elements, gameweeks, squad, baseline.lineups
    ):
        for window in chip_windows.legal("bboost", gw):
            rows.append(ChipValuation("bboost", gw, window, mean_gain, upside))

    for gw, mean_gain, upside, captain in value_triple_captain(
        samples, elements, gameweeks, squad
    ):
        for window in chip_windows.legal("3xc", gw):
            rows.append(
                ChipValuation("3xc", gw, window, mean_gain, upside, note=f"captain {captain}")
            )

    for gw, mean_gain, upside in value_free_hit(
        samples, elements, gameweeks, players, squad, baseline.lineups, budget
    ):
        for window in chip_windows.legal("freehit", gw):
            rows.append(ChipValuation("freehit", gw, window, mean_gain, upside))

    if wildcard_candidates > 0:
        expected = pd.DataFrame(samples.mean(axis=0), index=elements, columns=horizon)
        for gw, mean_gain, upside in value_wildcard(
            expected,
            players,
            state,
            chip_windows,
            baseline,
            candidates=wildcard_candidates,
            time_limit=solver_time_limit,
            solve_options=solve_options,
        ):
            for window in chip_windows.legal("wildcard", gw):
                rows.append(ChipValuation("wildcard", gw, window, mean_gain, upside))

    # dataclasses.asdict, not vars(): these are slots dataclasses and have no __dict__.
    valuations = pd.DataFrame([asdict(r) for r in rows])
    if valuations.empty:
        # No chip is legal in any gameweek of the horizon (all windows used, or the horizon
        # lies outside every window, as 2019-20's post-suspension gameweeks 39-47 do under
        # the live windows). Nothing to schedule, and nothing to merge fixtures onto.
        return ChipRoadmap(
            schedule={},
            valuations=pd.DataFrame(
                columns=["chip", "gameweek", "window", "mean_gain", "upside_gain", "note"]
            ),
            notes=["No chip is playable in this horizon."],
        )

    # Fixture context, which is what actually drives chip timing.
    fixtures = double_blank_summary(con, season, horizon)
    valuations = valuations.merge(
        fixtures, left_on="gameweek", right_on="event", how="left"
    ).drop(columns=["event"])

    schedule: dict[int, str] = {}
    notes: list[str] = []

    # Greedy assignment, best chip-gameweek first, honouring one chip per gameweek and one use
    # per chip instance.
    used_instances: set[tuple[str, int]] = set()
    for _, row in valuations.sort_values("mean_gain", ascending=False).iterrows():
        chip, gw, window = row["chip"], int(row["gameweek"]), int(row["window"])
        if row["mean_gain"] < min_gain.get(chip, 0.0):
            continue
        if gw in schedule or (chip, window) in used_instances:
            continue
        if f"{chip}:{window}" in state.chips_used:
            continue
        schedule[gw] = chip
        used_instances.add((chip, window))

    total_doubles = int(fixtures["doubles"].sum())
    total_blanks = int(fixtures["blanks"].sum())
    if total_doubles == 0 and total_blanks == 0:
        notes.append(
            f"No double or blank gameweeks exist yet in GW{horizon[0]}-{horizon[-1]}. They appear "
            "as cup postponements land, and Bench Boost and Free Hit are worth far more once they "
            "do — so second-half chips are best held rather than spent on fixture quality."
        )
    skipped = [c for c in ("bboost", "3xc", "freehit", "wildcard") if c not in schedule.values()]
    if skipped:
        notes.append(
            "Holding "
            + ", ".join(CHIP_LABELS[c] for c in skipped)
            + ": nothing in this horizon clears the value floor."
        )

    return ChipRoadmap(schedule=schedule, valuations=valuations, notes=notes)
