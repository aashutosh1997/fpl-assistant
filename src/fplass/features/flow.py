"""Order flow: what eleven million managers did before the deadline, read as a signal.

The historical dataset records, for every player and gameweek, how many managers owned him and
how many transferred him in and out. Those columns were quarantined as post-hoc by the ``as_of``
view, on the reasonable worry that a number describing the market's reaction to a gameweek must
not be used to predict it. The timing turns out to be the other way round. The change in
ownership between two gameweeks tracks that gameweek's own net transfers at a correlation of
0.89–0.97 in every season, against 0.15–0.34 for the previous gameweek's, so ``transfers_in`` and
``transfers_out`` for gameweek N are the flows between deadline N−1 and deadline N. They are
known at the deadline — they are the order book.

And the order book carries news the model cannot otherwise see. History has no availability
flags: a player ruled out on Friday looks, to the minutes model, exactly like one who is fit.
Managers know. Among established starters (five-match average of 75+ minutes, an hour in their
last match, at least 20,000 owners, 2019–26, 26,000 player-weeks), the share who then failed to
appear rises with the fraction of owners who sold them that week:

    decile of out-flow    1     2     3     4     5     6     7     8     9    10
    did not play        .037  .044  .045  .051  .044  .049  .053  .062  .092  .195

against a base rate of 0.067, and 0.345 in the top two percent. That is ``chance_of_playing``,
reconstructed from behaviour, for ten seasons in which it was never recorded.

So this module fits a small layer on top of the base minutes model — the same shape as the live
recalibration in :mod:`fplass.features.adjust`: a logistic on the base log-odds and two flow
features — and applies it to the gameweek whose flow is known, which is the deadline's own. Later
gameweeks in a projection have no flow yet and are left alone. The layer is fitted with every
season's base predictions out of sample and evaluated by holding whole seasons out, so the Brier
gain it reports is one the live model can expect.

Live, the same two numbers come from the price snapshots (``transfers_out_event`` over the
number of owners), scaled up for how much of the week has elapsed when the plan is run.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..ingest.sources import CURRENT_SEASON
from . import adjust as adjust_module
from . import minutes as minutes_module

log = logging.getLogger(__name__)

FEATURES = ("f_out", "f_in")

# log1p(FLOW_SCALE * fraction): the median out-flow (4% of owners) lands near 0.6 and the 99th
# percentile (34%) near 2, so the logistic sees a well-spread feature rather than a spike at zero.
FLOW_SCALE = 20.0

# Below this many previous owners a ratio of transfers to owners is noise, and the row is
# treated as carrying no flow information at all.
MIN_OWNERS = 500

# Live flows are the week's transfers so far; dividing by the elapsed share of the week
# extrapolates them, floored so a plan run an hour after the previous deadline is not
# multiplied into absurdity.
MIN_ELAPSED = 0.25


def flow_frame(con, season: str, *, gameweek: int | None = None) -> pd.DataFrame:
    """Per player and gameweek: the fractions of previous owners who sold and who bought.

    ``transfers_*`` are per gameweek and repeated on each fixture row of a double gameweek, so
    one value is taken per (element, gameweek). Gameweek 1 has no previous ownership and no
    flow, and rows with too few owners are left null: null means "no information", which the
    layer treats as "leave the base model alone".
    """
    where = "" if gameweek is None else "WHERE gw = ?"
    params: list[object] = [season, MIN_OWNERS, MIN_OWNERS]
    if gameweek is not None:
        params.append(gameweek)
    return con.execute(
        f"""
        WITH weekly AS (
            SELECT season, element, gw,
                   any_value(transfers_in) AS tin, any_value(transfers_out) AS tout,
                   any_value(selected) AS owners
            FROM player_gw WHERE season = ? GROUP BY season, element, gw
        ),
        lagged AS (
            SELECT *, lag(owners) OVER (PARTITION BY season, element ORDER BY gw) AS previous_owners
            FROM weekly
        )
        SELECT season, element, gw, previous_owners,
               CASE WHEN previous_owners >= ? THEN tout * 1.0 / previous_owners END AS flow_out,
               CASE WHEN previous_owners >= ? THEN tin * 1.0 / previous_owners END AS flow_in
        FROM lagged {where}
        """,
        params,
    ).fetchdf()


def transform(flow_out: np.ndarray, flow_in: np.ndarray) -> np.ndarray:
    """Feature matrix ``(n, 2)``; rows with a missing flow come back as NaN."""
    out = np.log1p(FLOW_SCALE * np.clip(np.asarray(flow_out, dtype="float64"), 0, None))
    inflow = np.log1p(FLOW_SCALE * np.clip(np.asarray(flow_in, dtype="float64"), 0, None))
    return np.column_stack([out, inflow])


@dataclass(slots=True)
class FlowLayer:
    """A fitted order-flow correction to the base minutes model."""

    coefficients: pd.Series  # over ("base_logit", *FEATURES)
    intercept: float
    n_train: int
    seasons: tuple[str, ...]
    brier_before: float
    brier_after: float

    @property
    def improvement(self) -> float:
        return 0.0 if self.brier_before <= 0 else 1.0 - self.brier_after / self.brier_before

    def shift(self, p_full: np.ndarray, features: np.ndarray, *, max_shift: float | None = None):
        """Log-odds shift per row; NaN where the flow is unknown."""
        logit = adjust_module._logit(np.asarray(p_full, dtype="float64"))
        weights = self.coefficients.to_numpy()
        raw = (weights[0] - 1.0) * logit + features @ weights[1:] + self.intercept
        cap = adjust_module.MAX_SHIFT if max_shift is None else max_shift
        return np.clip(raw, -cap, cap)

    def apply(
        self,
        probabilities: pd.DataFrame,
        flow: pd.DataFrame,
        *,
        max_shift: float | None = None,
    ) -> pd.DataFrame:
        """Shift ``p_full`` where flow is known; keep the three classes coherent.

        Args:
            probabilities: ``p_none``/``p_cameo``/``p_full`` rows.
            flow: Aligned rows with ``flow_out`` and ``flow_in``; NaN leaves the row untouched.
            max_shift: Cap on the log-odds move. The live path passes a tighter one than the
                fitted default, since its flows are extrapolated from a partial week.
        """
        out = probabilities.copy()
        flow_out = pd.to_numeric(flow["flow_out"], errors="coerce").to_numpy(dtype="float64")
        flow_in = pd.to_numeric(flow["flow_in"], errors="coerce").to_numpy(dtype="float64")
        known = ~(np.isnan(flow_out) | np.isnan(flow_in))
        if not known.any():
            return out

        p_full = out["p_full"].to_numpy(dtype="float64")
        features = transform(np.where(known, flow_out, 0.0), np.where(known, flow_in, 0.0))
        shift = self.shift(p_full, features, max_shift=max_shift)
        logit = adjust_module._logit(p_full)
        new_full = np.where(known, adjust_module._sigmoid(logit + shift), p_full)

        old_remaining = np.clip(1.0 - p_full, 1e-9, None)
        cameo = out["p_cameo"].to_numpy(dtype="float64") / old_remaining * (1.0 - new_full)
        out["p_full"] = new_full
        out["p_cameo"] = cameo
        out["p_none"] = 1.0 - out["p_full"] - out["p_cameo"]
        if "expected_minutes" in out.columns:
            out["expected_minutes"] = out["p_cameo"] * 30.0 + out["p_full"] * 82.0
        return out


def _history_seasons(con, exclude: tuple[str, ...] | list[str]) -> list[str]:
    rows = con.execute(
        "SELECT DISTINCT season FROM player_gw WHERE season <> ? ORDER BY season", [CURRENT_SEASON]
    ).fetchall()
    return [r[0] for r in rows if r[0] not in set(exclude)]


def training_table(
    con,
    features: pd.DataFrame,
    seasons: list[str],
    *,
    exclude: tuple[str, ...] | list[str] = (),
) -> pd.DataFrame:
    """Base predictions and flow for every row of ``seasons``, each season's base out of sample.

    For each season in ``seasons`` the base model is refitted with that season *and* every
    season in ``exclude`` held out, so a layer fitted on the table never learns to correct
    in-sample predictions, and a layer evaluated on an excluded season never saw it at all.
    """
    tables = []
    for season in seasons:
        model, _ = minutes_module.fit(features, holdout_seasons=[season, *exclude])
        rows = features[features["season"] == season].dropna(subset=["minutes_class"])
        predicted = model.predict(rows)
        table = rows[["season", "element", "gw", "fixture_id", "minutes_class"]].copy()
        table["p_base"] = predicted["p_full"].to_numpy()
        table["played_full"] = (table["minutes_class"] == minutes_module.CLASS_FULL).astype(float)
        flow = flow_frame(con, season)[["season", "element", "gw", "flow_out", "flow_in"]]
        table = table.merge(flow, on=["season", "element", "gw"], how="left")
        tables.append(table)
    out = pd.concat(tables, ignore_index=True)
    return out.dropna(subset=["flow_out", "flow_in"])


def fit(table: pd.DataFrame, *, l2: float = 1.0) -> FlowLayer:
    """Fit the layer on a :func:`training_table`."""
    p_base = table["p_base"].to_numpy(dtype="float64")
    design = np.column_stack(
        [
            adjust_module._logit(p_base),
            transform(table["flow_out"].to_numpy(), table["flow_in"].to_numpy()),
        ]
    )
    target = table["played_full"].to_numpy(dtype="float64")
    weights = adjust_module._fit_logistic(design, target, l2=l2)

    fitted = adjust_module._sigmoid(design @ weights[:-1] + weights[-1])
    layer = FlowLayer(
        coefficients=pd.Series(weights[:-1], index=["base_logit", *FEATURES]),
        intercept=float(weights[-1]),
        n_train=len(table),
        seasons=tuple(sorted(table["season"].unique())),
        brier_before=float(np.mean((p_base - target) ** 2)),
        brier_after=float(np.mean((fitted - target) ** 2)),
    )
    log.info(
        "flow layer fitted on %d rows from %s: Brier %.4f -> %.4f; %s intercept %.3f",
        layer.n_train,
        ",".join(layer.seasons),
        layer.brier_before,
        layer.brier_after,
        {k: round(v, 3) for k, v in layer.coefficients.items()},
        layer.intercept,
    )
    return layer


def fit_layer(
    con, features: pd.DataFrame | None = None, *, exclude: tuple[str, ...] | list[str] = ()
) -> FlowLayer | None:
    """Fit on every completed season not excluded. ``None`` when there is nothing to fit on."""
    seasons = _history_seasons(con, exclude)
    if not seasons:
        return None
    if features is None:
        features = minutes_module.build_features(con)
    table = training_table(con, features, seasons, exclude=exclude)
    if table.empty:
        return None
    return fit(table)


def evaluate(con, seasons: list[str] | None = None, features: pd.DataFrame | None = None) -> pd.DataFrame:
    """Held-out Brier per season: the base model alone against base plus flow.

    Each test season is excluded from both the base fits and the layer fit, so the gain is the
    one a season the model has never seen would show.
    """
    features = features if features is not None else minutes_module.build_features(con)
    seasons = seasons or _history_seasons(con, ())
    rows = []
    for season in seasons:
        others = [s for s in _history_seasons(con, ()) if s != season]
        layer = fit(training_table(con, features, others, exclude=[season]))
        test = training_table(con, features, [season])
        if test.empty:
            continue
        probabilities = pd.DataFrame({"p_none": 0.0, "p_cameo": 0.0, "p_full": test["p_base"]})
        adjusted = layer.apply(probabilities, test[["flow_out", "flow_in"]])
        target = test["played_full"].to_numpy()
        base = float(np.mean((test["p_base"].to_numpy() - target) ** 2))
        after = float(np.mean((adjusted["p_full"].to_numpy() - target) ** 2))
        starters = test["p_base"] >= 0.75
        rows.append(
            {
                "season": season,
                "n": len(test),
                "brier_base": round(base, 4),
                "brier_flow": round(after, 4),
                "gain": round(1 - after / base, 3),
                "starters_brier_base": round(
                    float(np.mean((test.loc[starters, "p_base"] - target[starters]) ** 2)), 4
                ),
                "starters_brier_flow": round(
                    float(np.mean((adjusted.loc[starters, "p_full"] - target[starters]) ** 2)), 4
                ),
                "coef_out": round(float(layer.coefficients["f_out"]), 3),
                "coef_in": round(float(layer.coefficients["f_in"]), 3),
            }
        )
    return pd.DataFrame(rows)


def historical_flow(con, season: str, gameweek: int, player_matches: pd.DataFrame) -> pd.DataFrame:
    """Flow aligned to a player-match frame: known for the deadline's gameweek, NaN elsewhere."""
    weekly = flow_frame(con, season, gameweek=gameweek)[["element", "flow_out", "flow_in"]]
    aligned = player_matches[["element", "event"]].merge(weekly, on="element", how="left")
    later = aligned["event"].to_numpy() != gameweek
    aligned.loc[later, ["flow_out", "flow_in"]] = np.nan
    return aligned[["flow_out", "flow_in"]].reset_index(drop=True)


def live_flow(players: pd.DataFrame, *, elapsed: float) -> pd.DataFrame:
    """The week's flow so far from the live player table, extrapolated to the deadline.

    Args:
        players: Must carry ``element``, ``transfers_in_event``, ``transfers_out_event``,
            ``selected_by_percent`` and ``total_players``.
        elapsed: Share of the gameweek's transfer window that has passed when the plan is run.
    """
    owners = (
        pd.to_numeric(players["selected_by_percent"], errors="coerce").fillna(0.0)
        / 100.0
        * pd.to_numeric(players["total_players"], errors="coerce").fillna(0.0)
    )
    scale = 1.0 / max(float(elapsed), MIN_ELAPSED)
    out = pd.DataFrame({"element": players["element"].to_numpy()})
    known = owners >= MIN_OWNERS
    out["flow_out"] = np.where(
        known,
        pd.to_numeric(players["transfers_out_event"], errors="coerce").fillna(0.0)
        / owners.replace(0, np.nan)
        * scale,
        np.nan,
    )
    out["flow_in"] = np.where(
        known,
        pd.to_numeric(players["transfers_in_event"], errors="coerce").fillna(0.0)
        / owners.replace(0, np.nan)
        * scale,
        np.nan,
    )
    return out


def align_live(flow: pd.DataFrame, player_matches: pd.DataFrame, gameweek: int) -> pd.DataFrame:
    """Live flow aligned to player-match rows, known only for the deadline's gameweek."""
    aligned = player_matches[["element", "event"]].merge(flow, on="element", how="left")
    later = aligned["event"].to_numpy() != gameweek
    aligned.loc[later, ["flow_out", "flow_in"]] = np.nan
    return aligned[["flow_out", "flow_in"]].reset_index(drop=True)
