"""Scoring the model against the season it is actually predicting.

A backtest tells you how a model would have done on history it was built from. This tells you how
it is doing right now, on a season nobody has seen, and it is the difference between a model that
degrades quietly and one that corrects itself.

Gameweek 1 made the case. The minutes model's held-out historical Brier skill is 0.485; its actual
gameweek 1 skill was 0.251. Nothing in the backtest could have revealed that, because the failure
mode — treating last season's form as though it were current-season form — only exists in August,
and every historical fold had a current season to draw on.

The harness does three things:

* **Scores stored projections** against what happened, producing calibration tables per component.
* **Fits the recalibration layer** in :mod:`fplass.features.adjust` on those outcomes.
* **Reports honestly**, including when the model is beaten by something simpler.

It reads from the ``projections`` table rather than recomputing, so it can only score predictions
that were genuinely recorded before a deadline. A projection regenerated after the fact would know
the team news and the calibration would be meaningless.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..features import adjust as adjust_module
from ..ingest import preseason as preseason_module
from ..ingest.sources import CURRENT_SEASON

log = logging.getLogger(__name__)


@dataclass(slots=True)
class CalibrationReport:
    """What the model got right and wrong over the gameweeks scored so far."""

    season: str
    gameweeks: tuple[int, ...]
    n_rows: int
    minutes_bins: pd.DataFrame
    minutes_brier: float
    minutes_brier_baseline: float
    points_by_position: pd.DataFrame
    points_bias: float
    points_correlation: float
    rank_correlation: float
    top30_lift: float
    adjustment: adjust_module.MinutesAdjustment | None

    @property
    def minutes_skill(self) -> float:
        if self.minutes_brier_baseline <= 0:
            return 0.0
        return 1.0 - self.minutes_brier / self.minutes_brier_baseline

    def summary(self) -> str:
        lines = [
            f"Calibration for {self.season} GW{','.join(str(g) for g in self.gameweeks)} "
            f"({self.n_rows} player-gameweeks)",
            "",
            "MINUTES  P(60+) predicted vs actual",
            self.minutes_bins.to_string(index=False),
            f"  Brier {self.minutes_brier:.4f}  baseline {self.minutes_brier_baseline:.4f}  "
            f"skill {self.minutes_skill:.3f}",
            "",
            "POINTS",
            self.points_by_position.to_string(),
            f"  bias {self.points_bias:+.2f}  corr {self.points_correlation:.3f}  "
            f"spearman {self.rank_correlation:.3f}  top-30 lift {self.top30_lift:.2f}x",
        ]
        if self.adjustment is not None:
            a = self.adjustment
            lines += [
                "",
                f"RECALIBRATION LAYER  Brier {a.brier_before:.4f} -> {a.brier_after:.4f} "
                f"({100 * a.improvement:.1f}% better) on {a.n_train} rows",
                "  " + "  ".join(f"{k}={v:+.3f}" for k, v in a.coefficients.items()),
            ]
        else:
            lines += ["", "RECALIBRATION LAYER  not fitted (too few observed gameweeks)"]
        return "\n".join(lines)


def scored_frame(con, season: str = CURRENT_SEASON, upto_gw: int | None = None) -> pd.DataFrame:
    """Stored projections joined to actuals, with the recalibration features attached."""
    limit = upto_gw if upto_gw is not None else 999
    frame = con.execute(
        """
        WITH latest AS (
            SELECT p.*,
                   row_number() OVER (
                       PARTITION BY p.season, p.gw, p.element ORDER BY p.made_at DESC
                   ) AS recency
            FROM projections p
            WHERE p.season = ? AND p.gw < ?
        )
        SELECT
            l.gw, l.element, l.p_full, l.p_cameo, l.p_none, l.expected_points,
            pl.web_name,
            CASE pl.element_type WHEN 1 THEN 'GKP' WHEN 2 THEN 'DEF'
                                 WHEN 3 THEN 'MID' ELSE 'FWD' END AS position,
            g.minutes, g.total_points,
            CASE WHEN g.minutes >= 60 THEN 1.0 ELSE 0.0 END AS played_full
        FROM latest l
        JOIN players pl ON pl.season = l.season AND pl.element = l.element
        JOIN (
            SELECT season, element, gw, sum(minutes) AS minutes, sum(total_points) AS total_points
            FROM player_gw GROUP BY season, element, gw
        ) g ON g.season = l.season AND g.element = l.element AND g.gw = l.gw
        WHERE l.recency = 1
        """,
        [season, limit],
    ).fetchdf()

    if frame.empty:
        return frame

    features = preseason_module.player_features(con, season)
    frame = frame.merge(features, on="element", how="left")
    for column in features.columns:
        if column != "element":
            frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
    return frame


def load_price_snapshots() -> pd.DataFrame | None:
    """The logged ownership history, if the price logger has produced any."""
    from ..paths import PRICE_SNAPSHOTS

    try:
        files = sorted(PRICE_SNAPSHOTS.glob("*.csv"))
        if not files:
            return None
        frame = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
        frame["ts"] = pd.to_datetime(frame["ts"], errors="coerce", utc=True)
        return frame.dropna(subset=["ts"])
    except (OSError, ValueError) as exc:  # pragma: no cover
        log.debug("could not read price snapshots: %s", exc)
        return None


def attach_ownership(frame: pd.DataFrame, snapshots: pd.DataFrame | None = None) -> pd.DataFrame:
    """Add the ownership recorded closest before each gameweek's deadline.

    Loads the snapshot history itself when none is supplied. The previous behaviour — silently
    filling the column with zeros — meant a caller who forgot to pass snapshots got a layer fitted
    with ownership permanently switched off, and nothing anywhere said so. A feature that
    disappears without complaint is worse than one that errors.
    """
    out = frame.copy()
    if snapshots is None:
        snapshots = load_price_snapshots()
    if snapshots is None or snapshots.empty:
        log.warning(
            "no price snapshots found; fitting without the ownership signal, "
            "which was worth ~12%% of Brier on gameweek 1"
        )
        out["log_ownership"] = 0.0
        return out

    latest = (
        snapshots.sort_values("ts").groupby("element", as_index=False).tail(1)[
            ["element", "selected_by_percent"]
        ]
    )
    out = out.merge(latest, on="element", how="left")
    ownership = pd.to_numeric(out["selected_by_percent"], errors="coerce").fillna(0.0)
    out["log_ownership"] = np.log1p(ownership.clip(lower=0))
    return out


def minutes_calibration(frame: pd.DataFrame, bins: int = 10) -> pd.DataFrame:
    """Predicted vs observed rate of a full appearance, in probability bins."""
    edges = np.linspace(0, 1, bins + 1)
    which = np.clip(np.digitize(frame["p_full"], edges) - 1, 0, bins - 1)
    rows = []
    for b in range(bins):
        mask = which == b
        if not mask.any():
            continue
        rows.append(
            {
                "bin": f"{edges[b]:.1f}-{edges[b + 1]:.1f}",
                "n": int(mask.sum()),
                "predicted": round(float(frame.loc[mask, "p_full"].mean()), 3),
                "actual": round(float(frame.loc[mask, "played_full"].mean()), 3),
            }
        )
    table = pd.DataFrame(rows)
    table["error"] = (table["actual"] - table["predicted"]).round(3)
    return table


def run(
    con,
    *,
    season: str = CURRENT_SEASON,
    upto_gw: int | None = None,
    snapshots: pd.DataFrame | None = None,
    fit_adjustment: bool = True,
) -> CalibrationReport | None:
    """Score every stored projection and, if there is enough data, fit the recalibration layer."""
    frame = scored_frame(con, season=season, upto_gw=upto_gw)
    if frame.empty:
        log.info("no stored projections to score for %s", season)
        return None

    frame = attach_ownership(frame, snapshots)

    predicted = frame["p_full"].to_numpy(dtype="float64")
    actual = frame["played_full"].to_numpy(dtype="float64")
    base_rate = actual.mean()

    points = frame.dropna(subset=["expected_points"])
    by_position = (
        points.groupby("position")
        .agg(n=("expected_points", "size"), predicted=("expected_points", "mean"),
             actual=("total_points", "mean"))
        .round(2)
    )
    by_position["bias"] = (by_position["actual"] - by_position["predicted"]).round(2)

    top30 = points.nlargest(30, "expected_points")
    lift = (
        float(top30["total_points"].mean() / points["total_points"].mean())
        if points["total_points"].mean() > 0
        else float("nan")
    )

    layer = None
    if fit_adjustment:
        layer = adjust_module.fit(
            frame,
            predicted,
            actual,
            gameweeks=tuple(sorted(frame["gw"].unique().tolist())),
        )

    return CalibrationReport(
        season=season,
        gameweeks=tuple(sorted(int(g) for g in frame["gw"].unique())),
        n_rows=len(frame),
        minutes_bins=minutes_calibration(frame),
        minutes_brier=float(np.mean((predicted - actual) ** 2)),
        minutes_brier_baseline=float(np.mean((base_rate - actual) ** 2)),
        points_by_position=by_position,
        points_bias=float(points["total_points"].mean() - points["expected_points"].mean()),
        points_correlation=float(points["expected_points"].corr(points["total_points"])),
        rank_correlation=float(
            points["expected_points"].corr(points["total_points"], method="spearman")
        ),
        top30_lift=lift,
        adjustment=layer,
    )
