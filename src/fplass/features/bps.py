"""Modelling the Bonus Points System.

Bonus is worth up to 3 points a match and is awarded to the top three BPS scorers in each fixture.
That makes it both material and awkward: it is not a property of a player in isolation but of his
ranking against the other twenty-one players on the pitch, so it can only be resolved inside a
simulated match.

Two things make 2026/27 harder than previous seasons:

1. **The weights were reworked and never published.** FPL announced the direction of travel —
   tackles no longer penalised, clearances/blocks/interceptions cut from one point per two actions
   to one per three, goalkeeper saves restructured with new categories for saves outside the box
   and for big chances — but not the numbers. So the weights have to be fitted from observed BPS,
   and refitted as the season produces data.

2. **BPS depends on inputs FPL does not expose.** Passes attempted and completed, key passes, big
   chances created and missed, errors leading to a goal, fouls, offsides and dribbles all feed BPS,
   and none appear in the public API for 2026/27. Any model built on the available columns is
   therefore incomplete *by construction*.

The response to (2) is not to pretend otherwise. We fit what we can observe and then measure the
residual spread, and the simulator samples that residual rather than treating the fit as exact. A
model that claimed to know BPS to the point would produce falsely confident bonus projections, and
bonus is exactly where overconfidence is most expensive — it is the difference between a triple
captain paying off and not.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Observable BPS drivers. Interactions with position are handled by fitting per position, since
# the same action is worth different amounts to a defender and a forward.
PREDICTORS = (
    "played_60",
    "played_any",
    "goals_scored",
    "assists",
    "clean_sheet",
    "saves",
    "defcon_count",
    "goals_conceded",
    "yellow_cards",
    "red_cards",
    "own_goals",
    "penalties_saved",
    "penalties_missed",
)

BONUS_AWARDS = (3, 2, 1)


@dataclass(slots=True)
class BPSModel:
    """Per-position linear BPS models plus the residual spread each leaves behind."""

    coefficients: dict[str, pd.Series] = field(default_factory=dict)
    intercepts: dict[str, float] = field(default_factory=dict)
    residual_sd: dict[str, float] = field(default_factory=dict)
    r_squared: dict[str, float] = field(default_factory=dict)
    n_train: dict[str, int] = field(default_factory=dict)
    season: str = ""

    def predict(self, events: pd.DataFrame) -> np.ndarray:
        """Expected BPS for each row. Rows with no minutes score zero."""
        design = build_design(events)
        prediction = np.zeros(len(design), dtype="float64")
        positions = design["position"].to_numpy()

        for position, coefficients in self.coefficients.items():
            mask = positions == position
            if not mask.any():
                continue
            matrix = design.loc[mask, list(PREDICTORS)].to_numpy(dtype="float64")
            prediction[mask] = matrix @ coefficients.to_numpy() + self.intercepts[position]

        return np.where(design["played_any"].to_numpy() > 0, prediction, 0.0)

    def residual_scale(self, events: pd.DataFrame) -> np.ndarray:
        """Per-row residual standard deviation, for sampling BPS in the simulator."""
        design = build_design(events)
        default = float(np.mean(list(self.residual_sd.values()))) if self.residual_sd else 8.0
        scale = design["position"].map(self.residual_sd).fillna(default).to_numpy(dtype="float64")
        return np.where(design["played_any"].to_numpy() > 0, scale, 0.0)


def build_design(events: pd.DataFrame) -> pd.DataFrame:
    """Derive the BPS predictor columns from raw event counts."""
    frame = pd.DataFrame(index=events.index)
    position = events["position"].replace({"GK": "GKP"})
    frame["position"] = position

    minutes = pd.to_numeric(events["minutes"], errors="coerce").fillna(0)
    frame["played_any"] = (minutes > 0).astype("float64")
    frame["played_60"] = (minutes >= 60).astype("float64")

    for column in PREDICTORS:
        if column in ("played_any", "played_60", "clean_sheet"):
            continue
        frame[column] = (
            pd.to_numeric(events[column], errors="coerce").fillna(0.0)
            if column in events.columns
            else 0.0
        )

    clean_sheet = (
        pd.to_numeric(events["clean_sheets"], errors="coerce").fillna(0.0)
        if "clean_sheets" in events.columns
        else 0.0
    )
    frame["clean_sheet"] = np.where(minutes >= 60, clean_sheet, 0.0)
    return frame


def fit(con, *, seasons: list[str], min_rows: int = 200) -> BPSModel:
    """Fit BPS weights per position by least squares on observed match BPS.

    Fitted separately per position because the same action carries different BPS weight depending
    on where a player plays, and a single pooled fit would average those away.

    Args:
        seasons: Which seasons to fit on. For in-season use pass only the current season, since
            the 2026/27 weights differ from 2025-26's; earlier seasons are useful as a prior
            before enough current data exists.
    """
    placeholders = ", ".join("?" for _ in seasons)
    frame = con.execute(
        f"""
        SELECT season, position, minutes, bps, goals_scored, assists, clean_sheets,
               goals_conceded, saves, yellow_cards, red_cards, own_goals,
               penalties_saved, penalties_missed, defcon_count
        FROM player_gw_derived
        WHERE season IN ({placeholders}) AND position NOT IN ('AM') AND minutes > 0
        """,
        list(seasons),
    ).fetchdf()

    if frame.empty:
        raise ValueError(f"no BPS training rows for seasons {seasons}")

    design = build_design(frame)
    design["bps"] = pd.to_numeric(frame["bps"], errors="coerce")
    design = design.dropna(subset=["bps"])

    model = BPSModel(season=",".join(seasons))
    for position, group in design.groupby("position", observed=True):
        if len(group) < min_rows:
            log.warning("skipping BPS fit for %s: only %d rows", position, len(group))
            continue

        matrix = group[list(PREDICTORS)].to_numpy(dtype="float64")
        target = group["bps"].to_numpy(dtype="float64")
        # Append an intercept column and solve by least squares.
        augmented = np.column_stack([matrix, np.ones(len(matrix))])
        solution, *_ = np.linalg.lstsq(augmented, target, rcond=None)

        fitted = augmented @ solution
        residuals = target - fitted
        total_variance = np.var(target)

        model.coefficients[str(position)] = pd.Series(solution[:-1], index=list(PREDICTORS))
        model.intercepts[str(position)] = float(solution[-1])
        model.residual_sd[str(position)] = float(np.std(residuals))
        model.r_squared[str(position)] = float(
            1 - np.var(residuals) / total_variance if total_variance > 0 else 0.0
        )
        model.n_train[str(position)] = int(len(group))

    log.info(
        "BPS fitted on %s: R2 %s, residual SD %s",
        model.season,
        {k: round(v, 3) for k, v in model.r_squared.items()},
        {k: round(v, 1) for k, v in model.residual_sd.items()},
    )
    return model


def allocate_bonus(bps: np.ndarray, match_ids: np.ndarray) -> np.ndarray:
    """Award 3/2/1 bonus points to the top BPS scorers within each match.

    FPL's tie handling is specific and matters more than it sounds, because ties are common in a
    discrete metric: players tied for first all receive 3 and the next distinct score receives 1
    (second place is consumed); players tied for second all receive 2. Getting this wrong biases
    bonus downward for exactly the high-scoring players a triple captain would target.

    Args:
        bps: Simulated or observed BPS, one entry per player-match.
        match_ids: Which match each entry belongs to.

    Returns:
        Bonus points, same shape as ``bps``.
    """
    bonus = np.zeros(len(bps), dtype="float64")
    order = np.argsort(match_ids, kind="stable")
    sorted_matches = match_ids[order]
    boundaries = np.flatnonzero(np.diff(sorted_matches)) + 1

    for block in np.split(order, boundaries):
        if block.size == 0:
            continue
        values = bps[block]
        # Only players who registered any BPS can place.
        eligible = values > 0
        if not eligible.any():
            continue

        distinct = np.unique(values[eligible])[::-1]
        awarded = 0
        for rank, score in enumerate(distinct):
            if awarded >= 3 or rank >= 3:
                break
            tied = block[values == score]
            bonus[tied] = BONUS_AWARDS[awarded]
            # A tie consumes as many placings as there are tied players.
            awarded += len(tied)

    return bonus


def verify_allocation(con, season: str) -> pd.DataFrame:
    """Check :func:`allocate_bonus` against FPL's actual bonus awards.

    Uses observed BPS, so any mismatch is a fault in the tie-handling rules rather than in the BPS
    prediction. Returns the disagreeing rows; empty means the allocation logic is exact.
    """
    frame = con.execute(
        """
        SELECT season, fixture_id, element, web_name, position, bps, bonus
        FROM player_gw_derived
        WHERE season = ? AND position NOT IN ('AM')
        """,
        [season],
    ).fetchdf()

    frame["computed_bonus"] = allocate_bonus(
        frame["bps"].fillna(0).to_numpy(dtype="float64"),
        frame["fixture_id"].to_numpy(),
    )
    mismatched = frame[frame["computed_bonus"] != frame["bonus"].fillna(0)]
    log.info(
        "%s bonus allocation: %d/%d exact (%.3f%%)",
        season,
        len(frame) - len(mismatched),
        len(frame),
        100 * (1 - len(mismatched) / max(len(frame), 1)),
    )
    return mismatched
