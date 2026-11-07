"""Do the top managers know something the model doesn't? A test, held out a gameweek at a time.

The question is narrow on purpose: given the model's projection at a deadline, does what a
cohort of strong managers did at that deadline predict the points (or the minutes) the
projection missed? If it does, out of sample, it is a candidate layer — the same shape as the
order-flow layer in :mod:`fplass.features.flow`, which already reads the whole crowd's
transfers for minutes. If it does not, the cohort's edge (if any) lies in decisions the model
already makes, such as timing and chips, not in information.

Each variant predicts a player's points in the deadline's gameweek as the projection plus a
ridge regression on its features, fitted on every other gameweek and scored on the held-out
one:

``calibrated``   the projection rescaled (``a + b * ep``): the baseline every variant must beat
``crowd``        + the whole game's buy and sell rates and ownership
``elite``        + the cohort's net buying and ownership
``both``         + all of them

Minutes are tested the same way with a logistic on the base log-odds, as the flow layer is.
With one season's deadlines there are few folds, so the verdict asks for a gain in most of
them and twice its standard error before anything is called a signal.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import cohorts as cohorts_module
from .missed import Inputs

log = logging.getLogger(__name__)

CROWD_FEATURES = ("crowd_in", "crowd_out", "crowd_own")
VARIANTS = {
    "calibrated": (),
    "crowd": CROWD_FEATURES,
    "elite": ("elite_net", "elite_own"),
    "both": (*CROWD_FEATURES, "elite_net", "elite_own"),
}
TOP_N = 30
# Rows the test is about: players the model gave a real chance of playing, or the cohort held.
MIN_P_FULL = 0.1
MIN_ELITE_OWN = 0.01


def build_rows(inputs: Inputs, panel: pd.DataFrame, cohort: str) -> pd.DataFrame:
    """One row per (deadline, player) with the projection, the flows and the outcome."""
    week = panel[panel["target_gw"] == panel["as_of_gw"]].groupby(["as_of_gw", "element"]).agg(
        ep=("ep_mean", "sum"), p_full=("p_full", "max")
    )
    week.index.names = ["gw", "element"]
    rows = week.reset_index()
    rows = rows[rows["gw"].between(2, inputs.last_played)]


    flows = cohorts_module.cohort_flows(
        inputs.members[inputs.members["cohort"] == cohort], inputs.squads, inputs.chips
    )
    flows = flows.rename(columns={"net": "elite_net", "own_before": "elite_own"})
    rows = rows.merge(flows[["gw", "element", "elite_net", "elite_own"]], on=["gw", "element"], how="left")
    rows[["elite_net", "elite_own"]] = rows[["elite_net", "elite_own"]].fillna(0.0)

    crowd = inputs.crowd.sort_values(["element", "gw"]).copy()
    crowd["previous"] = crowd.groupby("element")["selected"].shift()
    owners = crowd["previous"].where(crowd["previous"] >= 500)
    crowd["crowd_in"] = np.log1p(20 * (crowd["transfers_in"] / owners).clip(lower=0))
    crowd["crowd_out"] = np.log1p(20 * (crowd["transfers_out"] / owners).clip(lower=0))
    crowd["crowd_own"] = np.log1p(crowd["previous"].fillna(0) / 1_000)
    rows = rows.merge(crowd[["gw", "element", *CROWD_FEATURES]], on=["gw", "element"], how="left")
    rows[list(CROWD_FEATURES)] = rows[list(CROWD_FEATURES)].fillna(0.0)
    return rows


def attach_outcomes(con, rows: pd.DataFrame, season: str) -> pd.DataFrame:
    outcomes = con.execute(
        "SELECT gw, element, sum(total_points) AS points, sum(minutes) AS minutes, "
        "max(minutes) AS longest FROM player_gw WHERE season = ? GROUP BY 1, 2",
        [season],
    ).fetchdf()
    rows = rows.merge(outcomes, on=["gw", "element"], how="left")
    rows[["points", "minutes", "longest"]] = rows[["points", "minutes", "longest"]].fillna(0.0)
    rows["hour"] = (rows["longest"] >= 60).astype(int)
    keep = (rows["p_full"] >= MIN_P_FULL) | (rows["elite_own"] >= MIN_ELITE_OWN)
    return rows[keep].reset_index(drop=True)


def _design(frame: pd.DataFrame, features: tuple[str, ...]) -> np.ndarray:
    return frame[["ep", *features]].to_numpy(dtype="float64")


def _minutes_design(frame: pd.DataFrame, features: tuple[str, ...]) -> np.ndarray:
    p = np.clip(frame["p_full"].to_numpy(dtype="float64"), 1e-4, 1 - 1e-4)
    logit = np.log(p / (1 - p))
    return np.column_stack([logit, frame[list(features)].to_numpy(dtype="float64")])


def _top_lift(pred: np.ndarray, points: np.ndarray) -> float:
    order = np.argsort(-pred)[:TOP_N]
    mean = points.mean()
    return float(points[order].mean() / mean) if mean > 0 else float("nan")


@dataclass(slots=True)
class FoldScore:
    variant: str
    gw: int
    rmse: float
    spearman: float
    lift: float
    brier: float


def held_out(rows: pd.DataFrame, *, alpha: float = 10.0) -> pd.DataFrame:
    """Leave-one-gameweek-out scores for every variant, points and minutes."""
    scores: list[FoldScore] = []
    weeks = sorted(rows["gw"].unique())
    for gw in weeks:
        train, test = rows[rows["gw"] != gw], rows[rows["gw"] == gw]
        if train.empty or test.empty:
            continue
        y = test["points"].to_numpy(dtype="float64")
        for variant, features in VARIANTS.items():
            # Standardised so the ridge penalty treats a share and a points projection alike.
            model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
            model.fit(_design(train, features), train["points"])
            pred = model.predict(_design(test, features))
            minutes = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000))
            minutes.fit(_minutes_design(train, features), train["hour"])
            p = minutes.predict_proba(_minutes_design(test, features))[:, 1]
            scores.append(
                FoldScore(
                    variant=variant,
                    gw=int(gw),
                    rmse=float(np.sqrt(np.mean((pred - y) ** 2))),
                    spearman=float(spearmanr(pred, y).statistic),
                    lift=_top_lift(pred, y),
                    brier=float(np.mean((p - test["hour"].to_numpy()) ** 2)),
                )
            )
        # The raw projection, unfitted, for reference.
        scores.append(
            FoldScore(
                variant="projection",
                gw=int(gw),
                rmse=float(np.sqrt(np.mean((test["ep"].to_numpy() - y) ** 2))),
                spearman=float(spearmanr(test["ep"], y).statistic),
                lift=_top_lift(test["ep"].to_numpy(), y),
                brier=float(np.mean((test["p_full"].to_numpy() - test["hour"].to_numpy()) ** 2)),
            )
        )
    return pd.DataFrame([asdict(s) for s in scores])


def summarise(folds: pd.DataFrame, baseline: str = "calibrated") -> pd.DataFrame:
    """Mean score per variant, and its paired gain over the baseline with a fold-level SE.

    Lower is better for RMSE and Brier, higher for Spearman and lift; every ``gain_*`` column is
    signed so that positive means better than the baseline.
    """
    base = folds[folds["variant"] == baseline].set_index("gw")
    rows = []
    for variant, group in folds.groupby("variant"):
        group = group.set_index("gw")
        row = {"variant": variant, "folds": len(group)}
        for metric, sign in (("rmse", -1), ("spearman", 1), ("lift", 1), ("brier", -1)):
            row[metric] = group[metric].mean()
            diff = sign * (group[metric] - base[metric].reindex(group.index))
            row[f"gain_{metric}"] = diff.mean()
            row[f"gain_{metric}_se"] = diff.std(ddof=1) / np.sqrt(len(diff)) if len(diff) > 1 else np.nan
            row[f"gain_{metric}_up"] = int((diff > 0).sum())
        rows.append(row)
    return pd.DataFrame(rows).set_index("variant")


def verdict(summary: pd.DataFrame, variant: str, metric: str) -> str:
    """Whether a variant's gain on a metric clears the bar: two SEs and most folds."""
    if variant not in summary.index:
        return "not tested"
    row = summary.loc[variant]
    gain, se, up, folds = (
        row[f"gain_{metric}"], row[f"gain_{metric}_se"], row[f"gain_{metric}_up"], row["folds"]
    )
    if np.isnan(se) or folds < 4:
        return f"too few gameweeks ({int(folds)})"
    if gain > 2 * se and up >= 0.75 * folds:
        return f"signal: +{gain:.4f} (SE {se:.4f}), better in {int(up)} of {int(folds)} weeks"
    if gain < -2 * se:
        return f"worse: {gain:.4f} (SE {se:.4f}), better in {int(up)} of {int(folds)} weeks"
    return f"no clear signal: {gain:+.4f} (SE {se:.4f}), better in {int(up)} of {int(folds)} weeks"
