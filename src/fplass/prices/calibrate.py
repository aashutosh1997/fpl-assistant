"""Learning what FPL's new price-change fields actually mean.

For 2026/27 FPL began publishing price-change data directly on each player:

    price_change_percent       progress toward the next move
    price_change_hourly_rate   how fast that progress is currently moving
    price_change_projections   [{offset, projected_percent, likelihood}] for three days ahead
    price_change_locked_until  a timestamp before which the price cannot move
    price_change_calibrating   true while FPL's own model is still warming up

None of it is documented, and every field reads zero before the season starts. So rather than
assume a reading, this module learns one: it lines up each snapshot against what the price
actually did at the next daily change deadline and measures the relationship.

The value of a calibration is that it produces *probabilities* rather than guesses. "Saka is at
94% and rising 3 points an hour" is only actionable once you know how often a player in that state
actually rose — and that is what :func:`fit` estimates and :func:`reliability` checks.

Until enough moves have been observed, :func:`classical_model` provides the fallback that predated
the official fields: net transfers measured against an ownership-scaled threshold. It is less
precise but needs no calibration period, which matters in August when the official model is itself
flagged as calibrating.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Snapshots this close to a change deadline are the ones worth learning from: a reading twelve
# hours out says much less than one taken an hour before the price moves.
PRE_DEADLINE_HOURS = 3.0


@dataclass(slots=True)
class PriceModel:
    """A fitted mapping from the published fields to the probability of a price move."""

    rise_coefficients: pd.Series
    fall_coefficients: pd.Series
    n_train: int
    n_rises: int
    n_falls: int
    accuracy: dict[str, float]
    calibrating: bool

    def predict(self, snapshot: pd.DataFrame) -> pd.DataFrame:
        """Probability of a rise and of a fall at the next change deadline."""
        design = _design_matrix(snapshot)
        return pd.DataFrame(
            {
                "element": snapshot["element"].to_numpy(),
                "p_rise": _sigmoid(design @ self.rise_coefficients.to_numpy()),
                "p_fall": _sigmoid(design @ self.fall_coefficients.to_numpy()),
            }
        )


FEATURES = (
    "intercept",
    "percent",
    "hourly_rate",
    "proj0_percent",
    "proj0_likelihood",
    "net_transfer_rate",
    "log_ownership",
)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -35, 35)))


def _design_matrix(frame: pd.DataFrame) -> np.ndarray:
    """Assemble the predictor matrix, tolerating fields that are absent or still zero."""
    n = len(frame)

    def column(name: str, default: float = 0.0) -> np.ndarray:
        if name not in frame.columns:
            return np.full(n, default)
        return pd.to_numeric(frame[name], errors="coerce").fillna(default).to_numpy(dtype="float64")

    ownership = np.clip(column("selected_by_percent", 1.0), 0.01, 100.0)
    transfers_in = column("transfers_in_event")
    transfers_out = column("transfers_out_event")
    # Net transfers scaled by ownership: the threshold FPL applies is ownership-dependent, so the
    # raw net count is not comparable between a 60%-owned premium and a 2%-owned enabler.
    net_rate = (transfers_in - transfers_out) / (ownership * 10_000.0)

    return np.column_stack(
        [
            np.ones(n),
            column("price_change_percent") / 100.0,
            column("price_change_hourly_rate"),
            column("proj0_percent") / 100.0,
            column("proj0_likelihood"),
            net_rate,
            np.log(ownership),
        ]
    )


def load_snapshots(path_or_frame) -> pd.DataFrame:
    """Read the committed price snapshots into a single frame."""
    from pathlib import Path

    if isinstance(path_or_frame, pd.DataFrame):
        frame = path_or_frame
    else:
        root = Path(path_or_frame)
        files = sorted(root.glob("*.csv")) if root.is_dir() else [root]
        if not files:
            raise FileNotFoundError(f"no price snapshots under {root}")
        frame = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

    frame["ts"] = pd.to_datetime(frame["ts"], errors="coerce", utc=True)
    return frame.dropna(subset=["ts"]).sort_values(["element", "ts"], ignore_index=True)


def label_moves(snapshots: pd.DataFrame) -> pd.DataFrame:
    """Label each snapshot with what the price did next.

    A price move is detected as a change in ``now_cost`` between consecutive snapshots for the same
    player. Only the last snapshot before each move is kept as a training row — that is the state a
    decision would actually have been made from, and including the stale readings from twelve hours
    earlier would teach the model that high readings usually mean nothing.
    """
    frame = snapshots.copy()
    frame["next_cost"] = frame.groupby("element")["now_cost"].shift(-1)
    frame["next_ts"] = frame.groupby("element")["ts"].shift(-1)
    frame = frame.dropna(subset=["next_cost"])

    delta = frame["next_cost"] - frame["now_cost"]
    frame["rose"] = (delta > 0).astype(int)
    frame["fell"] = (delta < 0).astype(int)

    gap_hours = (frame["next_ts"] - frame["ts"]).dt.total_seconds() / 3600.0
    frame["gap_hours"] = gap_hours
    # Only pairs of snapshots close together in time tell us about an imminent move.
    return frame[gap_hours <= PRE_DEADLINE_HOURS + 1.0]


def fit(snapshots: pd.DataFrame, *, min_events: int = 40) -> PriceModel | None:
    """Fit rise and fall models from observed price moves.

    Returns ``None`` until enough moves have been seen — roughly the first few days of a season.
    A model fitted on a handful of events would be worse than the classical fallback while looking
    more authoritative, which is the wrong trade.
    """
    labelled = label_moves(snapshots)
    if labelled.empty:
        log.info("no labelled price moves yet")
        return None

    n_rises = int(labelled["rose"].sum())
    n_falls = int(labelled["fell"].sum())
    if min(n_rises, n_falls) < min_events:
        log.info(
            "only %d rises and %d falls observed; need %d of each before fitting "
            "(using the classical fallback until then)",
            n_rises,
            n_falls,
            min_events,
        )
        return None

    design = _design_matrix(labelled)
    coefficients = {}
    accuracy = {}
    for target in ("rose", "fell"):
        y = labelled[target].to_numpy(dtype="float64")
        weights = _fit_logistic(design, y)
        coefficients[target] = pd.Series(weights, index=FEATURES)
        predicted = _sigmoid(design @ weights)
        accuracy[f"{target}_auc"] = _auc(y, predicted)
        accuracy[f"{target}_base_rate"] = float(y.mean())

    calibrating = bool(labelled.get("price_change_calibrating", pd.Series([False])).astype(bool).any())

    model = PriceModel(
        rise_coefficients=coefficients["rose"],
        fall_coefficients=coefficients["fell"],
        n_train=len(labelled),
        n_rises=n_rises,
        n_falls=n_falls,
        accuracy=accuracy,
        calibrating=calibrating,
    )
    log.info(
        "price model fitted on %d snapshots (%d rises, %d falls): AUC rise %.3f, fall %.3f",
        model.n_train,
        n_rises,
        n_falls,
        accuracy["rose_auc"],
        accuracy["fell_auc"],
    )
    return model


def _fit_logistic(
    design: np.ndarray, y: np.ndarray, *, l2: float = 1.0, iterations: int = 60
) -> np.ndarray:
    """Newton-Raphson logistic regression with an L2 penalty.

    Written out rather than pulled from scikit-learn so the ridge term and the intercept handling
    are explicit — the intercept is already a column of the design matrix and must not be
    penalised, or the fitted base rate is biased toward one half.
    """
    weights = np.zeros(design.shape[1])
    penalty = np.eye(design.shape[1]) * l2
    penalty[0, 0] = 0.0

    for _ in range(iterations):
        prediction = _sigmoid(design @ weights)
        gradient = design.T @ (y - prediction) - penalty @ weights
        variance = np.clip(prediction * (1 - prediction), 1e-8, None)
        hessian = -(design.T * variance) @ design - penalty
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:  # pragma: no cover - singular design
            break
        weights = weights - step
        if np.max(np.abs(step)) < 1e-8:
            break
    return weights


def _auc(y: np.ndarray, scores: np.ndarray) -> float:
    """Area under the ROC curve, via the rank-sum identity."""
    positives, negatives = y == 1, y == 0
    if not positives.any() or not negatives.any():
        return float("nan")
    ranks = pd.Series(scores).rank().to_numpy()
    n_pos, n_neg = positives.sum(), negatives.sum()
    return float((ranks[positives].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def official_projection(
    snapshot: pd.DataFrame, *, hours_to_deadline: float | None = None
) -> pd.DataFrame:
    """Read FPL's own price fields directly, now that their semantics are known.

    Four days of hourly snapshots decoded them, and they are simpler than a fitted model:

    ``price_change_percent``
        **Signed** progress toward the next move, observed spanning -86 to +102. Positive counts
        toward a rise, negative toward a fall, and 100 is the threshold. The player observed at
        101.7 was the only one due to change at the next deadline.

    ``price_change_projections[].likelihood``
        **Not a probability.** An ordinal severity bucket running -3 to +5 whose sign always agrees
        with the percent — a discretisation of the same quantity, not independent evidence. Treating
        it as a 0-1 probability, as this module originally did, silently multiplied every estimate
        by a number up to five.

    ``price_change_hourly_rate``
        How fast the percent is moving, which is what lets an in-progress reading be extrapolated
        to the deadline rather than read as if it were final.

    Because the threshold is explicit there is nothing to fit: the probability of a move is a
    function of how far the projected percent lands past 100. The logistic model in :func:`fit`
    remains useful for learning how *reliable* these fields turn out to be, but it is no longer
    required to interpret them.

    Args:
        hours_to_deadline: Hours until the next daily price change. When supplied, the current
            percent is extrapolated forward at the hourly rate.
    """
    frame = snapshot.copy()

    def column(name: str, default: float = 0.0) -> np.ndarray:
        if name not in frame.columns:
            return np.full(len(frame), default)
        return pd.to_numeric(frame[name], errors="coerce").fillna(default).to_numpy(dtype="float64")

    percent = column("price_change_percent")
    rate = column("price_change_hourly_rate")

    projected = percent.copy()
    if hours_to_deadline is not None and hours_to_deadline > 0:
        projected = percent + rate * hours_to_deadline
    else:
        # Fall back to FPL's own next-day projection where it is present.
        published = column("proj0_percent")
        projected = np.where(published != 0, published, percent)

    # A soft threshold rather than a step: the percent is an estimate and players sitting just
    # short of 100 do sometimes tip over before the deadline.
    margin = 12.0
    p_rise = np.clip((projected - 100.0) / margin + 0.5, 0.0, 1.0)
    p_fall = np.clip((-projected - 100.0) / margin + 0.5, 0.0, 1.0)

    return pd.DataFrame(
        {
            "element": frame["element"].to_numpy(),
            "p_rise": p_rise,
            "p_fall": p_fall,
            "percent": percent,
            "projected_percent": projected,
            "source": "official",
        }
    )


def classical_model(snapshot: pd.DataFrame) -> pd.DataFrame:
    """The pre-2026/27 heuristic: net transfers against an ownership-scaled threshold.

    FPL's historical mechanic was that a player rises once net transfers in cross a threshold
    proportional to their ownership, and falls once net transfers out cross a similar one. The
    constant below is the community's long-standing estimate rather than anything official, so
    this is a fallback: useful before the official fields have been observed long enough to
    calibrate, and honest about being approximate.
    """
    frame = snapshot.copy()
    ownership = pd.to_numeric(frame.get("selected_by_percent"), errors="coerce").fillna(1.0)
    ownership = ownership.clip(lower=0.01)
    net = pd.to_numeric(frame.get("transfers_in_event"), errors="coerce").fillna(0) - pd.to_numeric(
        frame.get("transfers_out_event"), errors="coerce"
    ).fillna(0)

    total_managers = 8_200_000
    owners = ownership / 100.0 * total_managers
    threshold = np.maximum(owners * 0.0075, 8_000)

    progress = net / threshold
    return pd.DataFrame(
        {
            "element": frame["element"].to_numpy(),
            "p_rise": np.clip(progress, 0, 1.5).clip(0, 1),
            "p_fall": np.clip(-progress, 0, 1.5).clip(0, 1),
            "source": "classical",
        }
    )


def reliability(model: PriceModel, snapshots: pd.DataFrame, bins: int = 10) -> pd.DataFrame:
    """Predicted versus observed rise rate, in probability bins.

    Calibration is what matters here rather than ranking. The decision layer trades a probability
    of a 0.1m move against the option value of waiting for team news, so a model that is
    systematically overconfident will make you transfer early and repeatedly.
    """
    labelled = label_moves(snapshots)
    if labelled.empty:
        return pd.DataFrame()

    predicted = model.predict(labelled)["p_rise"].to_numpy()
    observed = labelled["rose"].to_numpy(dtype="float64")

    edges = np.linspace(0, 1, bins + 1)
    which = np.clip(np.digitize(predicted, edges) - 1, 0, bins - 1)
    rows = []
    for b in range(bins):
        mask = which == b
        if not mask.any():
            continue
        rows.append(
            {
                "bin": f"{edges[b]:.1f}-{edges[b + 1]:.1f}",
                "n": int(mask.sum()),
                "predicted": float(predicted[mask].mean()),
                "observed": float(observed[mask].mean()),
            }
        )
    return pd.DataFrame(rows)
