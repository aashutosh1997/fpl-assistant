"""The engine against history, one scoring component at a time.

The projection panel keeps each player's total only, and totals hide offsetting errors: charging
goals conceded to keepers who never played cancelled out uniform minutes and a hand-set assist
share until the first was fixed. This replays sampled deadlines of the completed seasons exactly
as the panel does, keeps each player's expected points per component, and sets them against what
FPL actually scored, by position.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

from ..features import minutes as minutes_module
from ..ingest.warehouse import connect
from ..scoring import normalise_positions, points_from_events
from ..sim import project
from ..sim.engine import simulate
from . import panel as panel_module

log = logging.getLogger(__name__)

COMPONENTS = (
    "appearance",
    "goals",
    "assists",
    "clean_sheets",
    "goals_conceded",
    "saves",
    "yellow_cards",
    "defensive_contribution",
    "bonus",
)
POSITIONS = ("GKP", "DEF", "MID", "FWD")
# FPL published expected goals and assists from 2022-23; before it the rates are goals only.
XG_ERA = "2022-23"


def sampled_gameweeks(gameweeks: list[int], every: int) -> list[int]:
    """Every ``every``-th deadline, from the second, so each sample has a season behind it."""
    return gameweeks[1::every]


def season_components(
    season: str, *, every: int = 4, n_draws: int = 2000, seed: int = 20262027
) -> pd.DataFrame:
    """Expected and actual points per component for each player in the sampled gameweeks.

    One row per (element, gameweek), outer-joined: ``sim_*`` columns are the as-of projection
    for the deadline's own gameweek, ``act_*`` what FPL scored. A player the projection never
    saw has no ``sim_*`` values, so points from outside the pool stay visible.
    """
    con = connect(read_only=True)
    con.execute("SET threads TO 1")
    try:
        context = panel_module.season_context(con, season)
        chosen = sampled_gameweeks(context.gameweeks, every)
        frames: list[pd.DataFrame] = []
        for gw in chosen:
            tick = time.time()
            models = panel_module.models_as_of(con, context, gw)
            matches, probabilities = project.build_projection_inputs(
                con, models, [gw], as_of_gameweek=gw, historical=True
            )
            result = simulate(
                matches,
                probabilities,
                models.rules,
                models.bps,
                rho=models.strength.rho,
                n_draws=n_draws,
                seed=seed + gw,
                minutes_profile=models.minutes_profile,
                team_returns=models.team_returns,
                components=True,
            )
            rows = matches.reset_index(drop=True)[
                ["element", "event", "position", "goal_rate", "assist_rate", "code"]
            ].copy()
            # Players with no league record at all: their rates are stand-ins, not their own.
            rows["newcomer"] = ~rows["code"].isin(models.rates["code"])
            rows = rows.drop(columns="code")
            rows["p_full"] = probabilities[:, minutes_module.CLASS_FULL]
            rows["p_cameo"] = probabilities[:, minutes_module.CLASS_CAMEO]
            parts = result.components.reindex(columns=list(COMPONENTS), fill_value=0.0)
            frame = pd.concat([rows, parts], axis=1)
            # Mean simulated minutes per player, for weighing who was on the pitch to share goals.
            mean_minutes = pd.Series(
                result.minutes_played.mean(axis=0)[:, 0], index=result.elements
            )
            frame["sim_minutes"] = frame["element"].map(mean_minutes)
            frames.append(frame)
            log.info("%s GW%-2d components in %.1fs", season, gw, time.time() - tick)

        simulated = pd.concat(frames, ignore_index=True)
        simulated = simulated.groupby(["element", "event"], as_index=False).agg(
            position=("position", "first"),
            p_full=("p_full", "max"),
            p_cameo=("p_cameo", "max"),
            newcomer=("newcomer", "first"),
            goal_rate=("goal_rate", "first"),
            assist_rate=("assist_rate", "first"),
            sim_minutes=("sim_minutes", "first"),
            **{c: (c, "sum") for c in COMPONENTS},
        )

        played = con.execute(
            """
            SELECT element, gw AS event, position, minutes, goals_scored, assists, clean_sheets,
                   goals_conceded, own_goals, penalties_saved, penalties_missed, yellow_cards,
                   red_cards, saves, bonus, defcon_count
            FROM player_gw_derived
            WHERE season = ? AND list_contains(?, gw) AND position <> 'AM'
            """,
            [season, chosen],
        ).fetchdf()
        scored = points_from_events(played, context.rules)
        actual = pd.concat(
            [
                played[["element", "event"]],
                normalise_positions(played["position"]).rename("position"),
                scored[list(COMPONENTS)],
            ],
            axis=1,
        )
        actual["act_minutes"] = played["minutes"].to_numpy()
        actual = actual.groupby(["element", "event"], as_index=False).agg(
            position=("position", "first"),
            act_minutes=("act_minutes", "sum"),
            **{c: (c, "sum") for c in COMPONENTS},
        )
    finally:
        con.close()

    merged = simulated.merge(
        actual, on=["element", "event"], how="outer", suffixes=("_sim", "_act")
    )
    merged["position"] = merged["position_sim"].fillna(merged["position_act"])
    merged["projected"] = merged["position_sim"].notna()
    merged["act_minutes"] = merged["act_minutes"].fillna(0.0)
    merged = merged.drop(columns=["position_sim", "position_act"])
    for c in COMPONENTS:
        merged = merged.rename(columns={c + "_sim": f"sim_{c}", c + "_act": f"act_{c}"})
        merged[f"sim_{c}"] = merged[f"sim_{c}"].fillna(0.0)
        merged[f"act_{c}"] = merged[f"act_{c}"].fillna(0.0)
    merged.insert(0, "season", season)
    return merged


def _season_task(args: tuple) -> pd.DataFrame:
    season, every, n_draws = args
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s"
    )
    logging.getLogger("fplass").setLevel(logging.WARNING)
    log.setLevel(logging.INFO)
    return season_components(season, every=every, n_draws=n_draws)


def measure(
    seasons: list[str], *, every: int = 4, n_draws: int = 2000, workers: int = 1
) -> pd.DataFrame:
    """Component rows for several seasons, in parallel processes when asked.

    The calling process must not hold the warehouse open read-write while this runs.
    """
    panel_module._single_threaded_environment()
    tasks = [(s, every, n_draws) for s in seasons]
    if workers <= 1 or len(tasks) == 1:
        return pd.concat([_season_task(t) for t in tasks], ignore_index=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return pd.concat(list(pool.map(_season_task, tasks)), ignore_index=True)


def _per_week(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
    weeks = frame.groupby("season")["event"].nunique().sum()
    columns = [f"{prefix}{c}" for c in COMPONENTS]
    table = frame.groupby("position")[columns].sum() / max(weeks, 1)
    table.columns = list(COMPONENTS)
    return table.reindex(list(POSITIONS))


def position_table(frame: pd.DataFrame) -> pd.DataFrame:
    """League-wide points a gameweek by position and component: simulated, actual, and ratio.

    Only players the projection covered, so the comparison is like for like.
    """
    covered = frame[frame["projected"]]
    simulated = _per_week(covered, "sim_")
    actual = _per_week(covered, "act_")
    rows = []
    for position in POSITIONS:
        for component in COMPONENTS:
            sim = float(simulated.at[position, component])
            act = float(actual.at[position, component])
            if abs(sim) < 0.05 and abs(act) < 0.05:
                continue
            rows.append(
                {
                    "position": position,
                    "component": component,
                    "simulated": round(sim, 1),
                    "actual": round(act, 1),
                    "ratio": round(sim / act, 2) if abs(act) > 1e-9 else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def starter_bias(frame: pd.DataFrame) -> pd.DataFrame:
    """Actual minus expected points a week for likely starters (p_full > 0.5), by component."""
    starters = frame[frame["projected"] & (frame["p_full"] > 0.5)]
    out = {}
    for position, group in starters.groupby("position"):
        out[position] = {
            c: round(float((group[f"act_{c}"] - group[f"sim_{c}"]).mean()), 3) for c in COMPONENTS
        }
        out[position]["total"] = round(sum(out[position].values()), 3)
        out[position]["n"] = len(group)
    return pd.DataFrame(out).T.reindex(list(POSITIONS))


def report(frame: pd.DataFrame) -> str:
    """The position-by-component comparison, overall and split at the xG era."""
    lines = []
    eras = [
        ("all seasons", frame),
        (f"before {XG_ERA} (goals-only rates)", frame[frame["season"] < XG_ERA]),
        (f"{XG_ERA} on (xG-blended rates)", frame[frame["season"] >= XG_ERA]),
    ]
    for label, subset in eras:
        if subset.empty:
            continue
        weeks = subset.groupby("season")["event"].nunique().sum()
        lines.append(f"=== {label}: {weeks} sampled gameweeks")
        lines.append("League points a gameweek, players the projection covered:")
        lines.append(position_table(subset).to_string(index=False))
        lines.append("Starters (p_full > 0.5), actual minus expected a week:")
        lines.append(starter_bias(subset).to_string())
        uncovered = subset[~subset["projected"]]
        act_total = uncovered[[f"act_{c}" for c in COMPONENTS]].sum(axis=1)
        lines.append(
            f"Points scored by players outside the projection: {act_total.sum() / weeks:.1f} "
            f"a gameweek ({(act_total > 0).sum()} player-weeks)"
        )
        lines.append("")
    return "\n".join(lines)
