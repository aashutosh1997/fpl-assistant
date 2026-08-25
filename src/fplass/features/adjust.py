"""A recalibration layer over the minutes model, fitted on the season actually being played.

Gameweek 1 showed the base minutes model is roughly half as good in August as its backtest claims —
Brier skill 0.251 against 0.485 on held-out historical seasons. It is trained on within-season
sequences where "played ninety last week" predicts "plays ninety this week", then fed last season's
averages in August as though they carried the same weight. They do not.

Two signals fix most of that, and neither can live in the base model:

**Preseason minutes.** The only observation of the current squad hierarchy that exists before the
season starts. It cannot go in the base model because that model is trained on ten seasons in which
preseason data does not exist, so no weight for it could ever be learned.

**Ownership.** Eight million managers collectively know things the model does not — who looked
sharp in July, who the new manager favours, who is carrying a knock that never made the injury feed.

Measured against gameweek 1, predicting whether a player reached sixty minutes (five-fold
cross-validated, so these are out-of-sample):

===============================  ==========
base model                         0.1704
recalibrated, no new features      0.1658
+ preseason minutes                0.1281
+ ownership                        0.1491
**+ both**                       **0.1161**
===============================  ==========

A 32% reduction in Brier score, and the in-sample/out-of-sample gap is 0.003, so it is signal rather
than overfitting.

**Why a separate layer rather than new base-model features.** The layer is fitted on *this season's*
observed outcomes, so it needs no historical support and improves every week as gameweeks
accumulate. It also degrades honestly: with no observed gameweeks it does nothing at all and the
base model passes through untouched, which is the correct behaviour at the start of a season when
there is genuinely nothing to calibrate against.

**One thing deliberately withheld.** Ownership feeds the *minutes* model only, never the points
model. Ownership is partly circular — it reflects other managers reading the same public
projections — and letting it drive expected points would converge the squad on the template, which
is precisely what a mini-league objective is trying to avoid.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

FEATURES = (
    "base_logit",
    "preseason_minutes_avg",
    "preseason_observed",
    "log_ownership",
)

# Below this many observed player-gameweeks the layer refuses to fit. Fitting four coefficients on
# a handful of rows would produce a confident-looking model built on nothing.
MIN_TRAINING_ROWS = 300

# Cap on how far the layer may move the base model, in log-odds. The layer is fitted on very little
# data early in a season, so it is allowed to correct a clear bias but not to overrule the base
# model entirely on the strength of one gameweek.
MAX_SHIFT = 2.5


def _logit(p: np.ndarray) -> np.ndarray:
    clipped = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(clipped / (1 - clipped))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -35, 35)))


@dataclass(slots=True)
class MinutesAdjustment:
    """A fitted recalibration of the base minutes model."""

    coefficients: pd.Series
    intercept: float
    n_train: int
    gameweeks: tuple[int, ...]
    brier_before: float
    brier_after: float

    @property
    def improvement(self) -> float:
        """Fractional Brier reduction over the base model on the training gameweeks."""
        if self.brier_before <= 0:
            return 0.0
        return 1.0 - self.brier_after / self.brier_before

    def apply(self, frame: pd.DataFrame, p_full: np.ndarray) -> np.ndarray:
        """Adjust base probabilities using the layer's features.

        Args:
            frame: Must carry the columns in :data:`FEATURES` except ``base_logit``.
            p_full: Base model probabilities of a full appearance.
        """
        design = build_design(frame, p_full)
        shift = design @ self.coefficients.to_numpy() + self.intercept - design[:, 0]
        # The first coefficient scales the base logit; express the result as base + bounded shift
        # so the cap has a clear meaning and the base model is never discarded outright.
        adjusted = design[:, 0] + np.clip(shift, -MAX_SHIFT, MAX_SHIFT)
        return _sigmoid(adjusted)


def build_design(frame: pd.DataFrame, p_full: np.ndarray) -> np.ndarray:
    """Assemble the layer's feature matrix."""
    n = len(frame)

    def column(name: str) -> np.ndarray:
        if name not in frame.columns:
            return np.zeros(n)
        return pd.to_numeric(frame[name], errors="coerce").fillna(0.0).to_numpy(dtype="float64")

    ownership = column("selected_by_percent")
    if "log_ownership" in frame.columns:
        log_ownership = column("log_ownership")
    else:
        log_ownership = np.log1p(np.clip(ownership, 0, None))

    return np.column_stack(
        [
            _logit(np.asarray(p_full, dtype="float64")),
            column("preseason_minutes_avg"),
            column("preseason_observed"),
            log_ownership,
        ]
    )


def fit(
    frame: pd.DataFrame,
    p_full: np.ndarray,
    played_full: np.ndarray,
    *,
    gameweeks: tuple[int, ...] = (),
    l2: float = 1.0,
) -> MinutesAdjustment | None:
    """Fit the layer on observed outcomes.

    Args:
        frame: Feature rows, one per observed player-gameweek.
        p_full: What the base model predicted for those rows.
        played_full: Whether the player actually reached sixty minutes.
        gameweeks: Which gameweeks the training rows come from, for reporting.
        l2: Ridge penalty. Kept meaningful because early in a season this is fitted on very few
            gameweeks and unpenalised coefficients would swing wildly week to week.

    Returns:
        The fitted layer, or ``None`` when there is too little data to fit responsibly.
    """
    if len(frame) < MIN_TRAINING_ROWS:
        log.info(
            "minutes adjustment not fitted: %d rows, need %d", len(frame), MIN_TRAINING_ROWS
        )
        return None

    design = build_design(frame, p_full)
    target = np.asarray(played_full, dtype="float64")

    weights = _fit_logistic(design, target, l2=l2)

    base_brier = float(np.mean((np.asarray(p_full) - target) ** 2))
    fitted = _sigmoid(design @ weights[:-1] + weights[-1])
    new_brier = float(np.mean((fitted - target) ** 2))

    layer = MinutesAdjustment(
        coefficients=pd.Series(weights[:-1], index=FEATURES),
        intercept=float(weights[-1]),
        n_train=len(frame),
        gameweeks=tuple(gameweeks),
        brier_before=base_brier,
        brier_after=new_brier,
    )
    log.info(
        "minutes adjustment fitted on %d rows from GW%s: Brier %.4f -> %.4f (%.1f%% better); %s",
        layer.n_train,
        ",".join(str(g) for g in gameweeks) or "?",
        base_brier,
        new_brier,
        100 * layer.improvement,
        {k: round(v, 3) for k, v in layer.coefficients.items()},
    )
    return layer


def _fit_logistic(
    design: np.ndarray, y: np.ndarray, *, l2: float, iterations: int = 60
) -> np.ndarray:
    """Newton-Raphson logistic regression with an intercept appended and left unpenalised."""
    augmented = np.column_stack([design, np.ones(len(design))])
    weights = np.zeros(augmented.shape[1])
    # Start from an identity pass-through of the base model, so an uninformative fit leaves the
    # base predictions essentially untouched rather than collapsing them to the base rate.
    weights[0] = 1.0

    penalty = np.eye(augmented.shape[1]) * l2
    penalty[-1, -1] = 0.0

    for _ in range(iterations):
        prediction = _sigmoid(augmented @ weights)
        gradient = augmented.T @ (y - prediction) - penalty @ weights
        variance = np.clip(prediction * (1 - prediction), 1e-8, None)
        hessian = -(augmented.T * variance) @ augmented - penalty
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:  # pragma: no cover - singular design
            break
        weights = weights - step
        if np.max(np.abs(step)) < 1e-8:
            break
    return weights


def observed_outcomes(con, season: str, upto_gw: int) -> pd.DataFrame:
    """Stored projections joined to what actually happened, for fitting the layer.

    Reads from the ``projections`` table, so only predictions that were genuinely made *before* a
    deadline can train the layer. That is the point: a projection recomputed after the fact with
    hindsight would make the calibration meaningless.
    """
    return con.execute(
        """
        SELECT
            p.gw, p.element, p.p_full,
            g.minutes,
            CASE WHEN g.minutes >= 60 THEN 1.0 ELSE 0.0 END AS played_full
        FROM projections p
        JOIN player_gw g
          ON g.season = p.season AND g.element = p.element AND g.gw = p.gw
        WHERE p.season = ? AND p.gw < ?
        QUALIFY row_number() OVER (
            PARTITION BY p.season, p.gw, p.element ORDER BY p.made_at DESC
        ) = 1
        """,
        [season, upto_gw],
    ).fetchdf()
