"""Optimising to win a mini-league, rather than to score the most points.

These are not the same objective, and the difference is the whole point of this module.

Maximising expected points is the right objective if you are playing against the whole world and
care about your overall rank. In a mini-league of a dozen people it is often wrong. If you are 80
points behind with eight gameweeks left, the plan that maximises your expected score is very likely
the template — and the template loses, slowly and predictably, because your rivals own it too. What
you need is variance: differentials that might not come off, because the ones that do are the only
path to catching up. Conversely, if you lead comfortably, the correct move is to *converge* on your
rivals' squads, neutralising their chances to swing the gap regardless of how many points anyone
scores.

Both behaviours fall out automatically from optimising the right thing. We simulate every rival on
the *same* Monte Carlo draws as ourselves — so when Haaland blanks, he blanks for all of us — and
maximise the expected number of rivals we finish above. No risk-appetite parameter to guess: being
behind makes differentials optimal on their own, and being ahead makes the template optimal.

The rivals' future squads are modelled as template drift: managers gradually migrate toward highly
owned, in-form players. This is crude, and deliberately so — modelling a specific opponent's
transfer psychology would be false precision. What matters for the objective is roughly *how
correlated* their squads are with each other and with ours, and template drift captures that.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..api import FPLAPIError, FPLClient
from .milp import Plan

log = logging.getLogger(__name__)

CAPTAIN_MULTIPLIER = 2
TRIPLE_CAPTAIN_MULTIPLIER = 3


@dataclass(slots=True)
class Rival:
    """One opponent in a mini-league."""

    entry: int
    name: str
    manager: str
    total_points: int
    rank: int
    squad: list[int] = field(default_factory=list)
    lineup: list[int] = field(default_factory=list)
    captain: int | None = None


@dataclass(slots=True)
class LeagueState:
    """A mini-league, its rivals, and where you stand in it."""

    league_id: int
    name: str
    rivals: list[Rival]
    my_entry: int
    my_points: int

    def deficits(self) -> np.ndarray:
        """Points each rival is ahead of you by. Negative means you lead."""
        return np.array([r.total_points - self.my_points for r in self.rivals], dtype="float64")


def load_league(
    client: FPLClient, league_id: int, my_entry: int, *, gameweek: int, max_rivals: int = 30
) -> LeagueState:
    """Fetch a classic league's standings and each rival's most recent squad.

    Squads come from ``entry/{id}/event/{gw}/picks/``, which is public only *after* that
    gameweek's deadline. So we read the most recently completed gameweek — you can never see what
    a rival is doing in the gameweek you are both about to play, which is a property of the game
    rather than a limitation here.
    """
    standings = client.league_standings(league_id)
    league_name = standings.get("league", {}).get("name", str(league_id))
    results = standings.get("standings", {}).get("results", [])

    my_points = 0
    rivals: list[Rival] = []
    for row in results:
        if row["entry"] == my_entry:
            my_points = row["total"]
            continue
        rivals.append(
            Rival(
                entry=row["entry"],
                name=row.get("entry_name", ""),
                manager=row.get("player_name", ""),
                total_points=row.get("total", 0),
                rank=row.get("rank", 0),
            )
        )

    rivals = sorted(rivals, key=lambda r: r.rank)[:max_rivals]

    if gameweek >= 1:
        for rival in rivals:
            try:
                picks = client.entry_picks(rival.entry, gameweek)
            except FPLAPIError:
                # Before gameweek 1, or for a manager who joined late, picks simply do not exist.
                log.debug("no picks for entry %d in GW%d", rival.entry, gameweek)
                continue
            rival.squad = [p["element"] for p in picks.get("picks", [])]
            rival.lineup = [p["element"] for p in picks.get("picks", []) if p["position"] <= 11]
            captains = [p["element"] for p in picks.get("picks", []) if p.get("is_captain")]
            rival.captain = captains[0] if captains else None

    log.info("league %s: %d rivals loaded", league_name, len(rivals))
    return LeagueState(
        league_id=league_id,
        name=league_name,
        rivals=rivals,
        my_entry=my_entry,
        my_points=my_points,
    )


def template_squad(
    ownership: pd.Series, players: pd.DataFrame, expected_points: pd.Series
) -> list[int]:
    """The squad the field converges on: highly owned, well-performing, legally shaped.

    Used to stand in for a rival whose real picks we cannot see — before gameweek 1, or for a
    manager who joined the league late.
    """
    from .milp import CLUB_LIMIT, SQUAD_QUOTA

    frame = players.set_index("element").join(
        [ownership.rename("ownership"), expected_points.rename("ep")]
    )
    frame["ownership"] = frame["ownership"].fillna(0.0)
    frame["ep"] = frame["ep"].fillna(0.0)
    # Ownership dominates: the template is defined by what people own, not by what is optimal.
    frame["score"] = frame["ownership"] + 2.0 * frame["ep"]

    chosen: list[int] = []
    by_club: dict[int, int] = {}
    for position, quota in SQUAD_QUOTA.items():
        pool = frame[frame["position"] == position].sort_values("score", ascending=False)
        taken = 0
        for element, row in pool.iterrows():
            if taken >= quota:
                break
            if by_club.get(row["team_id"], 0) >= CLUB_LIMIT:
                continue
            chosen.append(element)
            by_club[row["team_id"]] = by_club.get(row["team_id"], 0) + 1
            taken += 1
    return chosen


def simulate_rivals(
    league: LeagueState,
    samples: np.ndarray,
    elements: np.ndarray,
    players: pd.DataFrame,
    ownership: pd.Series,
    *,
    drift: float = 0.15,
) -> np.ndarray:
    """Score every rival on the same draws we score ourselves on.

    Sharing the draws is the essential part. A rival simulated on independent randomness looks far
    more beatable than they are, because the correlation that actually dominates a mini-league —
    everyone owning the same captain — disappears. On common draws, a blank from the template
    captain hurts you and your rivals together, which is exactly the situation where a differential
    is worth owning.

    Args:
        drift: Fraction of a rival's squad assumed to migrate toward the template over the
            horizon. Applied by blending each rival's scores toward the template's, which
            represents the squads converging without pretending to predict individual transfers.

    Returns:
        ``(n_draws, n_rivals)`` total points over the horizon.
    """
    index = {e: i for i, e in enumerate(elements)}
    n_draws = samples.shape[0]

    mean_points = pd.Series(samples.mean(axis=(0, 2)), index=elements)
    template = template_squad(ownership, players, mean_points)
    template_columns = [index[e] for e in template if e in index]
    template_scores = _score_squad(samples, template_columns, players, elements, template)

    totals = np.zeros((n_draws, len(league.rivals)), dtype="float64")
    for i, rival in enumerate(league.rivals):
        squad = rival.squad or template
        columns = [index[e] for e in squad if e in index]
        if not columns:
            totals[:, i] = template_scores
            continue
        own = _score_squad(samples, columns, players, elements, squad, captain=rival.captain)
        totals[:, i] = (1 - drift) * own + drift * template_scores

    return totals


def _score_squad(
    samples: np.ndarray,
    columns: list[int],
    players: pd.DataFrame,
    elements: np.ndarray,
    squad: list[int],
    *,
    captain: int | None = None,
) -> np.ndarray:
    """Total horizon points for a squad, picking a starting eleven each gameweek.

    Rivals are assumed to field their best legal eleven and captain their best player, which is
    generous but keeps us from overestimating our own chances.
    """
    from .milp import LINEUP_MAX, LINEUP_MIN, LINEUP_SIZE

    if not columns:
        return np.zeros(samples.shape[0])

    position = players.set_index("element")["position"]
    positions = [position.get(e, "MID") for e in squad if e in set(elements)]
    n_draws, _, n_gws = samples.shape
    totals = np.zeros(n_draws)

    for slot in range(n_gws):
        block = samples[:, columns, slot]
        means = block.mean(axis=0)
        order = np.argsort(-means)

        selected: list[int] = []
        counts = dict.fromkeys(LINEUP_MIN, 0)
        for enforce_minimum in (True, False):
            for j in order:
                if len(selected) >= LINEUP_SIZE or j in selected:
                    continue
                pos = positions[j] if j < len(positions) else "MID"
                if enforce_minimum and counts[pos] >= LINEUP_MIN[pos]:
                    continue
                if counts[pos] >= LINEUP_MAX[pos]:
                    continue
                selected.append(j)
                counts[pos] += 1

        gameweek_points = block[:, selected].sum(axis=1)
        if selected:
            captain_column = (
                columns.index(captain) if captain in columns else selected[int(np.argmax(means[selected]))]
            )
            gameweek_points = gameweek_points + block[:, captain_column]
        totals += gameweek_points
    return totals


def score_plan(
    plan: Plan,
    samples: np.ndarray,
    elements: np.ndarray,
    *,
    chips: dict[int, str] | None = None,
) -> np.ndarray:
    """Our own total points per draw, under a given plan."""
    index = {e: i for i, e in enumerate(elements)}
    gameweeks = sorted(plan.lineups)
    totals = np.zeros(samples.shape[0])
    chips = chips or plan.chips

    for slot, gw in enumerate(gameweeks):
        columns = [index[e] for e in plan.lineups[gw] if e in index]
        if not columns:
            continue
        totals += samples[:, columns, slot].sum(axis=1)
        captain = plan.captains.get(gw)
        if captain in index:
            multiplier = (
                TRIPLE_CAPTAIN_MULTIPLIER if chips.get(gw) == "3xc" else CAPTAIN_MULTIPLIER
            )
            totals += (multiplier - 1) * samples[:, index[captain], slot]

    totals -= sum(plan.hits.values()) * 4.0
    return totals


def evaluate(
    plan: Plan,
    league: LeagueState,
    rival_totals: np.ndarray,
    samples: np.ndarray,
    elements: np.ndarray,
) -> dict[str, float]:
    """Score a plan by what it does to your league position, not by its point total.

    Returns both, so the trade-off is visible: a plan that wins more often while scoring slightly
    fewer expected points is usually the right choice in a small league, and you should be able to
    see that is what you are choosing.
    """
    mine = score_plan(plan, samples, elements)
    deficits = league.deficits()

    # A rival is beaten over the horizon if our gain exceeds their current lead.
    beaten = (mine[:, None] - rival_totals) > deficits[None, :]
    expected_beaten = float(beaten.sum(axis=1).mean())

    return {
        "expected_points": float(mine.mean()),
        "points_p10": float(np.quantile(mine, 0.10)),
        "points_p90": float(np.quantile(mine, 0.90)),
        "expected_rivals_beaten": expected_beaten,
        "n_rivals": float(len(league.rivals)),
        "win_probability": float(np.mean(beaten.all(axis=1))) if len(league.rivals) else 1.0,
        "expected_rank": float(len(league.rivals) + 1 - expected_beaten),
    }


def choose(
    plans: list[Plan],
    league: LeagueState,
    rival_totals: np.ndarray,
    samples: np.ndarray,
    elements: np.ndarray,
    *,
    objective: str = "league",
) -> tuple[Plan, pd.DataFrame]:
    """Pick the best plan from a candidate pool under the chosen objective.

    Args:
        objective: ``league`` maximises expected rivals beaten; ``points`` maximises expected
            points; ``blend`` averages the two ranks, which is a reasonable default when you care
            about overall rank as well as your mini-leagues.

    Returns:
        The winning plan and the full comparison table.
    """
    rows = []
    for i, plan in enumerate(plans):
        metrics = evaluate(plan, league, rival_totals, samples, elements)
        metrics["plan"] = i
        rows.append(metrics)

    table = pd.DataFrame(rows)
    if objective == "points":
        table["score"] = table["expected_points"]
    elif objective == "blend":
        table["score"] = (
            table["expected_rivals_beaten"].rank() + table["expected_points"].rank()
        ) / 2
    else:
        # Expected rivals beaten, with expected points as a tiebreak — among plans that win the
        # league equally often, prefer the one that also scores more.
        table["score"] = table["expected_rivals_beaten"] + 1e-6 * table["expected_points"]

    table = table.sort_values("score", ascending=False, ignore_index=True)
    return plans[int(table.iloc[0]["plan"])], table
