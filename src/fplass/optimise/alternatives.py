"""The best few distinct decisions for the coming week, ranked, and how sure each one is.

One plan hides how close the call was. A transfer that beats the next-best by half a point reads
exactly like one that beats it by ten, and the manager — who knows things the model does not, and
has to live with the result — never sees the alternatives the planner weighed.

So the planner proposes a pool and this module ranks it:

* **Distinct now, free later.** Each candidate fields a different squad in the coming gameweek
  (a no-good cut on that week alone) and is re-optimised freely afterwards, so the candidates are
  different *decisions* rather than one decision with cosmetic changes in week seven. Rolling
  the transfer is always among them, as the reference every move has to beat.
* **Ranked on the objective the plan was chosen on** — rivals beaten in a mini-league, the
  planner's own value otherwise (expected points plus what it prices a banked transfer at) —
  with the spread of each candidate's points from the same simulated draws.
* **Robustness.** The share of plausible projection revisions in which each candidate is the best
  of the pool. The revisions are the ones measured between consecutive deadlines over the replayed
  seasons (:mod:`fplass.options.revisions`), injuries and lost places included, so it reads as
  "how often would next week's model still pick this". The P10–P90 range is the other kind of
  uncertainty: the luck of the matches themselves.

Chips are reported beside the ranking rather than inside it (:func:`chip_calls`): a chip played
inside an eight-week horizon looks free next to a plan that holds it, because its season-long
option value is invisible to the horizon. What matters for the decision is how close each chip is
to the value floor the roadmap holds it to.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import league as league_module
from .chips import CHIP_LABELS, DEFAULT_MIN_GAIN
from .milp import HIT_COST, Plan, SquadState

# Solver statuses that carry a usable plan ("Not Solved" is HiGHS stopping on its clock with an
# incumbent in hand).
USABLE = {"Optimal", "Not Solved"}

# Chips that replace the squad for the week, under which "roll" has no meaning.
SQUAD_CHIPS = {"wildcard", "freehit"}

TAG_LABELS = {
    "rivals": "most rivals beaten",
    "points": "most points",
    "ceiling": "highest ceiling (P90)",
    "safest": "safest (P10)",
    "robust": "most robust",
    "value": "the planner's pick on points plus banked transfers",
}
_TAG_METRICS = {
    "rivals": "expected_rivals_beaten",
    "points": "expected_points",
    "ceiling": "points_p90",
    "safest": "points_p10",
    "robust": "robustness",
    "value": "planner_value",
}


@dataclass(slots=True)
class PlanOption:
    """One ranked candidate plan."""

    rank: int  # 1-based position among every candidate in the pool
    plan: Plan
    metrics: dict[str, float]
    tags: list[str] = field(default_factory=list)
    # Shown only as the do-nothing reference, outside the top of the ranking.
    reference: bool = False


def is_roll(plan: Plan, gameweek: int) -> bool:
    """Whether the plan makes no change to the squad in ``gameweek``."""
    return not plan.transfers_in.get(gameweek) and plan.chips.get(gameweek) not in SQUAD_CHIPS


def distinct_plans(
    solve: Callable[..., Plan],
    gameweek: int,
    current: SquadState,
    *,
    pool_size: int = 8,
    chip: str | None = None,
) -> list[Plan]:
    """Up to ``pool_size`` plans that each field a different squad in ``gameweek``, plus roll.

    Args:
        solve: Solves the plan with every shared argument already bound; called with
            ``forbidden=`` for the pool and ``lock=`` for the roll reference.
        gameweek: The coming gameweek, the decision being made now.
        current: The squad held, which rolling keeps.
        chip: The chip the roadmap plays in ``gameweek``, if any. Under a wildcard or free hit
            every candidate rebuilds the squad, so there is no roll to add.
    """
    plans: list[Plan] = []
    forbidden: list[dict[int, list[int]]] = []
    for _ in range(pool_size):
        plan = solve(forbidden=list(forbidden))
        if plan.status not in USABLE or gameweek not in plan.squads:
            break
        plans.append(plan)
        forbidden.append({gameweek: list(plan.squads[gameweek])})

    if chip not in SQUAD_CHIPS and current.players and not any(is_roll(p, gameweek) for p in plans):
        roll = solve(lock={gameweek: sorted(current.elements)})
        if roll.status in USABLE:
            plans.append(roll)
    return plans


def economy_credit(plan: Plan) -> float:
    """What the planner values in a plan beyond its starting elevens' points.

    The solver's objective adds a price on every free transfer carried into the next week and a
    small weight on bench points to the points the elevens score; those are what make rolling a
    transfer a real answer, and the ten-season replay is where their size was measured. Simulated
    points alone would rank every sideways move above a roll.
    """
    hits = HIT_COST * sum(plan.hits.values())
    return float(plan.objective) - (float(sum(plan.expected_points.values())) - hits)


def planner_value(plan: Plan) -> float:
    """The solver's objective: expected points net of hits, plus the economy credit."""
    return float(plan.objective)


def robustness(
    plans: list[Plan],
    projections: np.ndarray,
    elements: np.ndarray,
    *,
    credit: bool = True,
) -> np.ndarray:
    """Each plan's share of ``projections`` in which it is the best of the pool.

    A plan is valued under each projection on its elevens' points, net of hits, plus — with
    ``credit``, when the ranking is on the planner's value — its :func:`economy_credit`, held
    fixed. Against rivals the ranking counts points scored alone, and so does this.

    Args:
        projections: ``(revisions, players, gameweeks)`` expected points, aligned with
            ``elements`` and the plans' gameweeks in order — perturbed versions of the projection
            the plans were solved on.

    Plans tied in a revision (identical starting elevens, differing only on the bench) split it,
    so the shares always sum to one.
    """
    if not plans:
        return np.zeros(0)
    scores = np.column_stack(
        [
            league_module.score_plan(plan, projections, elements)
            + (economy_credit(plan) if credit else 0.0)
            for plan in plans
        ]
    )
    best = scores.max(axis=1, keepdims=True)
    winners = scores >= best - 1e-9
    return (winners / winners.sum(axis=1, keepdims=True)).mean(axis=0)


def rank(
    plans: list[Plan],
    samples: np.ndarray,
    elements: np.ndarray,
    gameweek: int,
    *,
    league: league_module.LeagueState | None = None,
    rival_totals: np.ndarray | None = None,
    objective: str = "league",
    projections: np.ndarray | None = None,
    projection_elements: np.ndarray | None = None,
    n_show: int = 5,
) -> list[PlanOption]:
    """Rank the pool and return the top ``n_show``, plus references that fell outside it.

    Against rivals the ranking is the league stage's own (expected rivals beaten, or the blend),
    which counts points scored and nothing else. On points it is the planner's value
    (:func:`planner_value`), so the top option is the solver's own optimum and a move that
    scores a little more by spending a transfer the planner would bank ranks below the roll.

    References, shown below the top when they fall outside it: the roll, always, and against
    rivals the planner's own pick, so a disagreement between the two economies is visible.

    Args:
        samples: ``(draws, players, gameweeks)`` simulated points, the draws the rivals were
            simulated on.
        league: Rivals to rank against; without them every objective ranks on points.
        projections: Revised projections for :func:`robustness`; omitted, it is not reported.
        projection_elements: The players ``projections`` covers, in order; ``elements`` if
            omitted.
    """
    if not plans:
        return []
    against_rivals = league is not None and bool(league.rivals) and rival_totals is not None
    on_value = not against_rivals or objective == "points"
    shares = (
        robustness(
            plans,
            projections,
            elements if projection_elements is None else projection_elements,
            credit=on_value,
        )
        if projections is not None
        else None
    )

    rows = []
    for i, plan in enumerate(plans):
        if against_rivals:
            metrics = league_module.evaluate(plan, league, rival_totals, samples, elements)
        else:
            mine = league_module.score_plan(plan, samples, elements)
            metrics = {
                "expected_points": float(mine.mean()),
                "points_p10": float(np.quantile(mine, 0.10)),
                "points_p90": float(np.quantile(mine, 0.90)),
            }
        metrics["planner_value"] = planner_value(plan)
        if shares is not None:
            metrics["robustness"] = float(shares[i])
        rows.append(metrics)

    table = pd.DataFrame(rows)
    table["plan"] = range(len(plans))
    if against_rivals and objective != "points":
        table["score"] = league_module.objective_score(table, objective)
    else:
        table["score"] = table["planner_value"]
    order = table.sort_values("score", ascending=False, kind="stable")["plan"].tolist()
    best_points = rows[order[0]]["expected_points"]

    options = []
    for position, i in enumerate(order, start=1):
        metrics = dict(rows[i])
        metrics["gap"] = metrics["expected_points"] - best_points
        options.append(PlanOption(rank=position, plan=plans[i], metrics=metrics))

    shown = options[: max(n_show, 1)]
    references = []
    if not any(is_roll(o.plan, gameweek) for o in shown):
        references.append(next((o for o in options if is_roll(o.plan, gameweek)), None))
    if not on_value:
        references.append(max(options, key=lambda o: o.metrics["planner_value"]))
    unique = {id(o): o for o in references if o is not None}
    for option in sorted(unique.values(), key=lambda o: o.rank):
        if all(option is not o for o in shown):
            option.reference = True
            shown.append(option)
    _tag(shown, value=not on_value)
    return shown


def _tag(options: list[PlanOption], *, value: bool = False) -> None:
    """Mark what each option leads the shown options on, where the options differ at all.

    ``value`` marks the planner's own pick too, which is worth saying only when the ranking is
    on something else.
    """
    if len(options) < 2:
        return
    for tag, metric in _TAG_METRICS.items():
        if tag == "value" and not value:
            continue
        values = [o.metrics.get(metric) for o in options]
        if any(v is None for v in values):
            continue
        top = max(values)
        leaders = [o for o, v in zip(options, values, strict=True) if v >= top - 1e-9]
        if len(leaders) < len(options):
            for option in leaders:
                option.tags.append(tag)


def chip_calls(
    valuations: pd.DataFrame,
    floor_for: dict[str, float] | Callable[[str, int], float] | None,
    gameweek: int,
    chips_used: set[str],
    schedule: dict[int, str] | None = None,
) -> pd.DataFrame:
    """How close each chip still held is to being worth playing, from the roadmap's valuations.

    One row per chip instance legal somewhere in the horizon: its gain in the coming gameweek
    (mean and 90th percentile, when it is legal then), the floor it has to clear there — the
    expected value of holding it for a better week — and its best week in the horizon.
    """
    columns = [
        "chip", "label", "window", "now_mean", "now_upside", "now_floor", "short_by",
        "best_gameweek", "best_mean", "best_floor", "scheduled", "note",
    ]
    if valuations is None or valuations.empty:
        return pd.DataFrame(columns=columns)
    if callable(floor_for):
        floor = floor_for
    else:
        floors = floor_for if floor_for is not None else DEFAULT_MIN_GAIN

        def floor(chip: str, gw: int) -> float:
            return float(floors.get(chip, DEFAULT_MIN_GAIN.get(chip, 0.0)))

    scheduled = {chip: gw for gw, chip in (schedule or {}).items()}
    rows = []
    for (chip, window), group in valuations.groupby(["chip", "window"], sort=False):
        if f"{chip}:{int(window)}" in chips_used:
            continue
        now = group[group["gameweek"] == gameweek]
        best = group.loc[group["mean_gain"].idxmax()]
        now_mean = float(now["mean_gain"].iloc[0]) if len(now) else float("nan")
        now_floor = float(floor(chip, gameweek)) if len(now) else float("nan")
        rows.append(
            {
                "chip": chip,
                "label": CHIP_LABELS.get(chip, chip),
                "window": int(window),
                "now_mean": now_mean,
                "now_upside": float(now["upside_gain"].iloc[0]) if len(now) else float("nan"),
                "now_floor": now_floor,
                "short_by": now_floor - now_mean,
                "best_gameweek": int(best["gameweek"]),
                "best_mean": float(best["mean_gain"]),
                "best_floor": float(floor(chip, int(best["gameweek"]))),
                "scheduled": scheduled.get(chip),
                "note": str(now["note"].iloc[0]) if len(now) and "note" in now else "",
            }
        )
    return pd.DataFrame(rows, columns=columns)
