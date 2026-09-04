"""Chip timing as an exercise problem: play now, or hold for a better week.

A chip is an American option. Each week inside its window it can be exercised for that week's
gain — the bench's points, an extra copy of the captain, the best eleven minus your own — or
held for a later week that might be better. The roadmap in :mod:`fplass.optimise.chips` looks
across the planning horizon and plays the best week that clears a hand-set floor, and the floor
stands in for everything the horizon cannot see: the double gameweeks that appear in March, the
captain's run of fixtures in April.

The measured replacement is the **expected best later opportunity**. From the paper manager's
replayed seasons we have, for every deadline, what each chip would have been worth to the squad
actually held, so for a week with ``L`` weeks left in the window we can read off, season by
season, the best gain that was still to come. Its average across seasons is the continuation
value: play now if this week's gain beats it, hold otherwise. It is estimated with the season
being decided held out, so the rule never sees its own future.

This is least-squares Monte Carlo reduced to its essentials — the regression is on one state
variable, weeks left, and the paths are the nine seasons that happened. Nine paths is thin; the
point is that the thresholds come from the same replay that judges them, and the table reports
how much of the hindsight-best each rule captured so the trade-off is visible.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ..optimise import chips as roadmap_module
from ..optimise import milp
from .value import squads_from_trace

log = logging.getLogger(__name__)

CHIPS = ("bboost", "3xc", "freehit")


def best_eleven(expected: pd.Series, squad: list[int], positions: dict[int, str]) -> list[int]:
    """The formation-legal eleven with the most expected points, greedily."""
    order = sorted(squad, key=lambda e: -float(expected.get(e, 0.0)))
    chosen: list[int] = []
    counts = dict.fromkeys(milp.LINEUP_MIN, 0)
    for enforce_minimum in (True, False):
        for e in order:
            if len(chosen) >= milp.LINEUP_SIZE or e in chosen:
                continue
            pos = positions.get(e, "MID")
            if enforce_minimum and counts[pos] >= milp.LINEUP_MIN[pos]:
                continue
            if counts[pos] >= milp.LINEUP_MAX[pos]:
                continue
            chosen.append(e)
            counts[pos] += 1
    return chosen


def chip_gains_from_traces(
    con, traces: list[Path], *, panel_sources: list[Path] | None = None
) -> pd.DataFrame:
    """What each chip was worth, every week, to the squad the paper manager held.

    Bench Boost is the bench's expected points that week, Triple Captain the best squad
    member's, Free Hit the best affordable eleven's minus the held eleven's (the roadmap's own
    greedy valuation, handed the means).
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
        panel = con.execute(
            f"SELECT as_of_gw, target_gw, element, ep_mean FROM {_panel_sql(panel_sources)} "
            "WHERE season = ? AND as_of_gw = target_gw",
            [season],
        ).fetchdf()
        for gw, squad in squads.items():
            week = panel[panel["as_of_gw"] == gw].set_index("element")["ep_mean"]
            if week.empty or not squad:
                continue
            players = project.current_players(con, season, as_of_gameweek=gw)
            positions = dict(zip(players["element"].astype(int), players["position"], strict=True))
            price = dict(zip(players["element"].astype(int), players["price"].astype(int), strict=True))
            held = [e for e in squad if e in positions]
            eleven = best_eleven(week, held, positions)
            bench = [e for e in held if e not in eleven]
            samples = week.reindex(players["element"]).fillna(0.0).to_numpy()[None, :, None]
            elements = players["element"].to_numpy()
            budget = sum(price.get(e, 0) for e in held) + int(banks.get(gw, 0))
            free_hit = roadmap_module.value_free_hit(
                samples, elements, np.array([gw]), players, held, {gw: eleven}, budget
            )
            rows.append(
                {
                    "season": season,
                    "gameweek": gw,
                    "bboost": float(sum(week.get(e, 0.0) for e in bench)),
                    "3xc": float(max((week.get(e, 0.0) for e in held), default=0.0)),
                    "freehit": float(free_hit[0][1]) if free_hit else 0.0,
                }
            )
        log.info("%s: chip gains for %d weeks", season, len(squads))
    return pd.DataFrame(rows)


def continuation_values(
    gains: pd.DataFrame, chip: str, window: tuple[int, int]
) -> pd.DataFrame:
    """Expected best later gain by weeks left in the window, one row per weeks-left.

    ``mean`` is the continuation value; ``p25``/``p75`` show how much it varies by season.
    """
    start, stop = window
    inside = gains[(gains["gameweek"] >= start) & (gains["gameweek"] <= stop)]
    records = []
    for season, group in inside.groupby("season"):
        series = group.set_index("gameweek")[chip].sort_index()
        weeks = list(series.index)
        for i, gw in enumerate(weeks):
            later = series.iloc[i + 1 :]
            if later.empty:
                continue
            records.append(
                {"season": season, "weeks_left": len(later), "best_later": float(later.max())}
            )
    table = pd.DataFrame(records)
    if table.empty:
        return table
    return (
        table.groupby("weeks_left")["best_later"]
        .agg(mean="mean", p25=lambda s: s.quantile(0.25), p75=lambda s: s.quantile(0.75), n="size")
        .round(2)
        .reset_index()
    )


def evaluate_rules(
    gains: pd.DataFrame, chip: str, window: tuple[int, int], *, floor: float
) -> pd.DataFrame:
    """Realised gain per season for three ways of timing one chip in one window.

    * ``floor``: the roadmap's rule — play the first week that clears the floor, else the last.
    * ``continuation``: play the first week whose gain beats the expected best later gain,
      estimated on the other seasons; else the last week.
    * ``hindsight``: the best week of the window, which no rule can beat.
    """
    start, stop = window
    inside = gains[(gains["gameweek"] >= start) & (gains["gameweek"] <= stop)]
    seasons = sorted(inside["season"].unique())
    rows = []
    for season in seasons:
        others = inside[inside["season"] != season]
        continuation = continuation_values(others, chip, window)
        threshold = (
            dict(zip(continuation["weeks_left"], continuation["mean"], strict=True))
            if not continuation.empty
            else {}
        )
        series = inside[inside["season"] == season].set_index("gameweek")[chip].sort_index()
        weeks = list(series.index)

        def first_week(rule) -> int:
            for i, gw in enumerate(weeks):
                if rule(float(series[gw]), len(weeks) - i - 1):
                    return gw
            return weeks[-1]

        by_floor = first_week(lambda gain, left: gain >= floor)
        by_continuation = first_week(
            lambda gain, left: left == 0 or gain >= threshold.get(left, float("inf"))
        )
        rows.append(
            {
                "season": season,
                "floor_week": by_floor,
                "floor_gain": round(float(series[by_floor]), 2),
                "continuation_week": by_continuation,
                "continuation_gain": round(float(series[by_continuation]), 2),
                "hindsight_week": int(series.idxmax()),
                "hindsight_gain": round(float(series.max()), 2),
            }
        )
    return pd.DataFrame(rows)


def report(gains: pd.DataFrame, windows: milp.ChipWindows) -> str:
    """The chip-timing tables for every chip and window, as text."""
    lines: list[str] = []
    for chip in CHIPS:
        floor = roadmap_module.DEFAULT_MIN_GAIN.get(chip, 0.0)
        for start, stop in windows.windows.get(chip, []):
            table = evaluate_rules(gains, chip, (start, stop), floor=floor)
            if table.empty:
                continue
            lines.append(f"{roadmap_module.CHIP_LABELS[chip]} GW{start}-{stop}  (floor {floor:g})")
            lines.append(table.to_string(index=False))
            lines.append(
                "  mean gain: floor rule {:.1f}, continuation rule {:.1f}, hindsight {:.1f}".format(
                    table["floor_gain"].mean(),
                    table["continuation_gain"].mean(),
                    table["hindsight_gain"].mean(),
                )
            )
            continuation = continuation_values(gains, chip, (start, stop))
            if not continuation.empty:
                lines.append("  expected best later gain by weeks left:")
                lines.append(
                    "  "
                    + " ".join(
                        f"{int(r.weeks_left)}w:{r.mean:.1f}"
                        for r in continuation.sort_values("weeks_left").itertuples()
                    )
                )
            lines.append("")
    return "\n".join(lines)
