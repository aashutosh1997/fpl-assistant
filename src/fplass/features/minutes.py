"""Predicting how long a player will actually be on the pitch.

This is the most important model in the system and the one most often skimped on. A midfielder
projected at 5.5 points is worth 5.5 points only if he starts; if there is a 30% chance he is
benched, his real expectation is closer to 4, and the difference between those two numbers decides
most transfers. Worse, the error is not symmetric — a benched premium costs you the captaincy too.

The target is deliberately not "expected minutes". FPL pays in steps: nothing for not playing, one
point up to 59 minutes, two points and eligibility for a clean sheet from 60. So we predict the
full three-way distribution and let the simulator sample from it:

    0        did not play
    1-59     came on, or was withdrawn early
    60+      full appearance, clean sheet eligible

**A limitation worth stating plainly.** The historical dataset carries no per-gameweek availability
flags — no ``status``, no ``chance_of_playing``, no injury news as it stood before each deadline.
So the model can only learn from *observed playing patterns*: rolling minutes, start rates, how
recently a player featured. At prediction time we then apply the live availability fields from the
API as an explicit adjustment on top (see :func:`apply_availability`). That split is honest about
what is learned from history versus what is known only today, and it avoids the trap of training on
a feature that will not exist when it matters.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

log = logging.getLogger(__name__)

# Minutes outcome classes, in the order the model's probability columns follow.
CLASS_NONE, CLASS_CAMEO, CLASS_FULL = 0, 1, 2
CLASS_LABELS = ("p_none", "p_cameo", "p_full")

FULL_APPEARANCE_MINUTES = 60

FEATURES = (
    "roll_minutes_3",
    "roll_minutes_5",
    "roll_minutes_10",
    "roll_start_rate_5",
    "roll_played_rate_5",
    "minutes_last",
    "minutes_prev",
    "started_last",
    "gap_days",
    "team_congestion_7d",
    "season_progress",
    "log_price",
    "is_gkp",
    "is_def",
    "is_mid",
    "is_fwd",
    "debut_window",
)


def build_features(con, *, seasons: list[str] | None = None) -> pd.DataFrame:
    """Assemble the per-player-per-match feature frame for the minutes model.

    Everything here is computed from the ``player_gw_as_of`` view and lagged window functions, so
    no row can see its own gameweek's outcome or the market's reaction to it.

    Rolling windows are ordered by ``gw_seq`` rather than ``gw``: 2019-20's gameweeks jump from 29
    to 39 across the COVID suspension, and a window keyed on the raw label would silently average
    across a four-month break.
    """
    season_filter = ""
    params: list[object] = []
    if seasons:
        placeholders = ", ".join("?" for _ in seasons)
        season_filter = f"AND p.season IN ({placeholders})"
        params = list(seasons)

    frame = con.execute(
        f"""
        WITH base AS (
            SELECT
                p.season, p.element, p.fixture_id, p.gw, p.gw_seq, p.position,
                p.kickoff_time, p.minutes, p.starts, p.prev_value,
                pl.team_id,
                CASE WHEN p.minutes >= {FULL_APPEARANCE_MINUTES} THEN {CLASS_FULL}
                     WHEN p.minutes > 0 THEN {CLASS_CAMEO}
                     ELSE {CLASS_NONE} END AS minutes_class
            FROM player_gw_as_of p
            JOIN players pl ON pl.season = p.season AND pl.element = p.element
            WHERE p.position NOT IN ('AM') {season_filter}
        ),
        windowed AS (
            SELECT
                *,
                -- All windows exclude the current row: "3 PRECEDING AND 1 PRECEDING".
                avg(minutes) OVER (w ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING)
                    AS roll_minutes_3,
                avg(minutes) OVER (w ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING)
                    AS roll_minutes_5,
                avg(minutes) OVER (w ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING)
                    AS roll_minutes_10,
                avg(CASE WHEN starts > 0 THEN 1.0 ELSE 0.0 END)
                    OVER (w ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING) AS roll_start_rate_5,
                avg(CASE WHEN minutes > 0 THEN 1.0 ELSE 0.0 END)
                    OVER (w ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING) AS roll_played_rate_5,
                lag(minutes, 1) OVER w AS minutes_last,
                lag(minutes, 2) OVER w AS minutes_prev,
                lag(CASE WHEN starts > 0 THEN 1.0 ELSE 0.0 END, 1) OVER w AS started_last,
                lag(kickoff_time, 1) OVER w AS previous_kickoff,
                count(*) OVER (w ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
                    AS matches_so_far
            FROM base
            WINDOW w AS (PARTITION BY season, element ORDER BY gw_seq, fixture_id)
        )
        SELECT
            w.*,
            -- How many matches the player's club plays in the surrounding week. Congestion is
            -- what drives rotation, and it is a property of the club's calendar, not the player.
            (
                SELECT count(*) FROM fixtures f
                WHERE f.season = w.season
                  AND (f.team_h = w.team_id OR f.team_a = w.team_id)
                  AND f.kickoff_time BETWEEN w.kickoff_time - INTERVAL 7 DAY
                                         AND w.kickoff_time + INTERVAL 7 DAY
            ) AS team_congestion_7d
        FROM windowed w
        ORDER BY w.season, w.element, w.gw_seq
        """,
        params,
    ).fetchdf()

    return _finalise_features(frame)


def _finalise_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Derive the remaining columns and fill the unavoidable gaps."""
    frame = frame.copy()

    gap = (frame["kickoff_time"] - frame["previous_kickoff"]).dt.total_seconds() / 86400.0
    # A player's first match of a season has no previous kickoff. 14 days stands in for a normal
    # pre-season or international break rather than pretending the gap is unknown.
    frame["gap_days"] = gap.fillna(14.0).clip(0, 60)

    frame["season_progress"] = frame["gw_seq"] / 38.0
    price = pd.to_numeric(frame["prev_value"], errors="coerce")
    # Price is our only proxy for "how good does FPL think this player is", which correlates
    # strongly with being picked. Missing at a player's first appearance.
    frame["log_price"] = np.log(price.fillna(price.median()) / 10.0)

    position = frame["position"].replace({"GK": "GKP"})
    for label, code in (("is_gkp", "GKP"), ("is_def", "DEF"), ("is_mid", "MID"), ("is_fwd", "FWD")):
        frame[label] = (position == code).astype("float64")

    # Early in a player's record the rolling features are mostly missing, and the model should
    # know that rather than seeing imputed values as though they were observed.
    frame["debut_window"] = (frame["matches_so_far"].fillna(0) < 3).astype("float64")

    for column in (
        "roll_minutes_3",
        "roll_minutes_5",
        "roll_minutes_10",
        "minutes_last",
        "minutes_prev",
    ):
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
    for column in ("roll_start_rate_5", "roll_played_rate_5", "started_last"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)

    frame["team_congestion_7d"] = (
        pd.to_numeric(frame["team_congestion_7d"], errors="coerce").fillna(1.0).clip(1, 5)
    )
    return frame


@dataclass(slots=True)
class MinutesModel:
    """A fitted three-class minutes model."""

    model: LogisticRegression
    scaler: StandardScaler
    features: tuple[str, ...]
    classes: np.ndarray
    n_train: int

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Probabilities of not playing / a cameo / a full appearance.

        Also returns ``expected_minutes``, but only as a diagnostic — the simulator samples the
        class and then the minutes within it, because expected minutes cannot express the step
        at 60 that FPL actually pays on.
        """
        missing = [f for f in self.features if f not in frame.columns]
        if missing:
            raise ValueError(f"missing features: {missing}")

        matrix = self.scaler.transform(frame[list(self.features)].to_numpy(dtype="float64"))
        probabilities = self.model.predict_proba(matrix)

        out = pd.DataFrame(index=frame.index)
        for class_index, label in enumerate(CLASS_LABELS):
            column = np.where(self.classes == class_index)[0]
            out[label] = probabilities[:, column[0]] if len(column) else 0.0

        # Mean minutes within each class, from the training distribution: a cameo averages around
        # half an hour, a full appearance a little over 80 minutes once substitutions are counted.
        out["expected_minutes"] = out["p_cameo"] * 30.0 + out["p_full"] * 82.0
        return out


def fit(
    features: pd.DataFrame,
    *,
    holdout_seasons: list[str] | None = None,
    regularisation: float = 1.0,
) -> tuple[MinutesModel, pd.DataFrame]:
    """Fit the minutes model, returning it alongside the held-out evaluation set.

    Args:
        features: Output of :func:`build_features`.
        holdout_seasons: Seasons to exclude from training and return for evaluation. Splitting by
            season rather than at random is the only honest option: rows from the same player in
            the same season are heavily correlated, so a random split would leak.
        regularisation: Inverse L2 strength passed to the logistic regression.
    """
    holdout_seasons = holdout_seasons or []
    train = features[~features["season"].isin(holdout_seasons)]
    holdout = features[features["season"].isin(holdout_seasons)]

    usable = train.dropna(subset=["minutes_class"])
    matrix = usable[list(FEATURES)].to_numpy(dtype="float64")
    target = usable["minutes_class"].to_numpy(dtype="int64")

    scaler = StandardScaler().fit(matrix)
    # lbfgs is multinomial by default in current scikit-learn; the explicit multi_class argument
    # was removed, so passing it would break on 1.7+.
    model = LogisticRegression(C=regularisation, max_iter=2000, solver="lbfgs").fit(
        scaler.transform(matrix), target
    )

    log.info(
        "minutes model fitted on %d rows (%d seasons), holdout %d rows",
        len(usable),
        usable["season"].nunique(),
        len(holdout),
    )
    return (
        MinutesModel(
            model=model,
            scaler=scaler,
            features=FEATURES,
            classes=model.classes_,
            n_train=len(usable),
        ),
        holdout,
    )


def apply_availability(
    predictions: pd.DataFrame,
    *,
    status: pd.Series | None = None,
    chance_of_playing: pd.Series | None = None,
) -> pd.DataFrame:
    """Fold today's injury and availability news into a historical-pattern prediction.

    The model cannot learn this from history because the dataset has no per-gameweek availability
    flags, so it is applied here instead.

    ``chance_of_playing_next_round`` is a **ceiling on availability, not a prediction of starting**,
    and the distinction is the whole subtlety here. A player with no news has a ceiling of 100%,
    which says nothing at all about whether he is first choice — a third-choice goalkeeper is
    perfectly available and still will not play.

    So the adjustment only ever *reduces* the model's probability, never raises it toward the
    ceiling. Treating the ceiling as a target instead forces every unflagged player to a 100%
    chance of playing, which flattens the entire squad-depth signal the minutes model exists to
    provide. In practice that promoted a backup goalkeeper to captain.

    Statuses: ``a`` available, ``d`` doubtful, ``i`` injured, ``s`` suspended, ``u`` unavailable,
    ``n`` on loan or otherwise not in the squad.
    """
    out = predictions.copy()
    play_probability = out["p_cameo"] + out["p_full"]

    target = pd.Series(1.0, index=out.index)
    if chance_of_playing is not None:
        chance = pd.to_numeric(chance_of_playing, errors="coerce") / 100.0
        target = target.where(chance.isna(), chance)
    if status is not None:
        # Hard zero for players who cannot feature at all, whatever the percentage says.
        ruled_out = status.isin(["i", "s", "u", "n"])
        target = target.mask(ruled_out, 0.0)
        # A doubt with no percentage attached: FPL uses 'd' for 75% and below.
        if chance_of_playing is not None:
            unquantified_doubt = (status == "d") & pd.to_numeric(
                chance_of_playing, errors="coerce"
            ).isna()
            target = target.mask(unquantified_doubt, 0.5)

    # Capped at 1.0 so the ceiling can only ever pull a probability down. Without the cap, a
    # target of 1.0 (no news) divided by a modest predicted play probability produces a scale
    # above 1 and inflates every unflagged player toward certainty.
    scale = np.where(
        play_probability > 0,
        np.minimum(1.0, target / play_probability.replace(0, np.nan)),
        0.0,
    )
    scale = np.nan_to_num(scale, nan=0.0)

    # Returning from a knock usually means a shorter outing, so shift a fifth of the retained
    # full-appearance mass into the cameo class when a player is flagged at all.
    flagged = target < 1.0
    new_full = out["p_full"] * scale
    new_cameo = out["p_cameo"] * scale
    shift = np.where(flagged, new_full * 0.2, 0.0)
    out["p_full"] = new_full - shift
    out["p_cameo"] = new_cameo + shift
    out["p_none"] = 1.0 - out["p_full"] - out["p_cameo"]
    out["expected_minutes"] = out["p_cameo"] * 30.0 + out["p_full"] * 82.0
    return out


def _logit(p: np.ndarray) -> np.ndarray:
    clipped = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(clipped / (1 - clipped))


def _solve_tilt(probabilities: np.ndarray, target: float, iterations: int = 80) -> float:
    """Find the shared log-odds shift that makes probabilities sum to ``target``.

    Bisection on lambda in ``sum(sigmoid(logit(p) + lambda)) == target``. Monotone in lambda, so
    bisection is reliable and needs no derivatives.
    """
    if len(probabilities) == 0:
        return 0.0
    logits = _logit(probabilities)
    low, high = -14.0, 14.0
    for _ in range(iterations):
        mid = (low + high) / 2
        total = 1.0 / (1.0 + np.exp(-(logits + mid)))
        if total.sum() < target:
            low = mid
        else:
            high = mid
    return (low + high) / 2


# Expected players per team-match who reach sixty minutes. Eleven start, but ``p_full`` is the
# probability of a *full* appearance and roughly one starter in fourteen is withdrawn before the
# hour: P(60+ | started) is 0.93 in every season 2022-26 and the count of sixty-minute appearances
# per team-match runs 10.28-10.33. Forcing the sum to eleven predicted 220 such appearances in each
# of the first two 2026/27 gameweeks against 210 and 209 actual, and dilutes every player's share
# of his team's goals by the same seven percent.
FULL_APPEARANCES_PER_TEAM = 10.3


def calibrate_to_lineup(
    probabilities: pd.DataFrame,
    team_match: pd.Series,
    *,
    starters: float = FULL_APPEARANCES_PER_TEAM,
    substitutes: float = 3.0,
) -> pd.DataFrame:
    """Rescale minutes probabilities so each team fields a legal number of players.

    The model predicts each player independently, which means nothing stops it from expecting
    seventeen players on the pitch for one club and seven for another. Measured across a real
    gameweek the spread was 7.2 to 17.7 expected players per team, against a true value of eleven
    starters plus three substitutes (see :data:`FULL_APPEARANCES_PER_TEAM` for why the target
    for the sixty-minute class is a little under eleven).

    That is not a cosmetic problem. The simulator distributes a team's goals among its players in
    proportion to rate times minutes, so an inflated squad-wide minutes total dilutes every
    individual's share. It penalises clubs whose threat is concentrated in one player — precisely
    the Haalands the optimiser most needs to price correctly — and flatters clubs with large,
    uncertain squads where the model spreads probability thinly.

    The adjustment is a shared shift in **log-odds**, solved per club so the probabilities sum to
    eleven starters and three substitutes.

    Working in log-odds rather than scaling the probabilities directly is what makes this safe.
    A multiplicative rescale has to be capped or it pushes nailed starters above 1.0 — and once
    capped, the entire shortfall is dumped on the players who still have headroom, inflating
    mid-probability ones absurdly. That is not hypothetical: it took Sessegnon, whose record is
    twenty full appearances in thirty-eight and whose raw probability was 0.61, and reported him at
    0.985 — a near-certain starter — which in turn made a 4.5m defender the third-highest scoring
    player in the league and the model's preferred captain.

    A log-odds shift has no such failure mode. It is monotone, so ordering is preserved exactly;
    it saturates naturally at 0 and 1, so no cap is needed; and it moves each player in proportion
    to how uncertain they already were, which is the correct distribution of the adjustment. A
    near-certain starter barely moves while a genuine squad player moves a lot.

    The shift can be large for a promoted club whose players have no Premier League record —
    Coventry's raw starters summed to 4.3 — and that is the honest answer: eleven of them will
    start, and we do not know which, so the probability spreads across the squad.
    """
    out = probabilities.copy()
    groups = pd.Series(team_match).to_numpy()

    for column, target in (("p_full", starters), ("p_cameo", substitutes)):
        values = out[column].to_numpy(dtype="float64").copy()
        for _, indices in pd.Series(np.arange(len(values))).groupby(groups):
            index = indices.to_numpy()
            block = values[index]
            if block.sum() <= 1e-9 or len(block) == 0:
                continue
            shift = _solve_tilt(block, min(target, len(block) * 0.98))
            values[index] = 1.0 / (1.0 + np.exp(-(_logit(block) + shift)))
        out[column] = values

    # Keep the three classes a valid distribution. Where the two targets cannot both be met —
    # a club with an unusually short listed squad — starters take priority over substitutes,
    # since eleven players must take the field and the bench is what gets cut short.
    out["p_full"] = out["p_full"].clip(0.0, 1.0)
    out["p_cameo"] = out["p_cameo"].clip(lower=0.0).clip(upper=1.0 - out["p_full"])
    out["p_none"] = 1.0 - out["p_full"] - out["p_cameo"]
    out["expected_minutes"] = out["p_cameo"] * 30.0 + out["p_full"] * 82.0
    return out


def evaluate(model: MinutesModel, holdout: pd.DataFrame) -> dict[str, float]:
    """Calibration and sharpness on held-out seasons.

    Log loss and the Brier score are both reported against a base-rate baseline. Accuracy alone
    would be misleading — always predicting "full appearance" scores well on accuracy and is
    useless for deciding a captaincy.
    """
    usable = holdout.dropna(subset=["minutes_class"])
    if usable.empty:
        return {}

    predictions = model.predict(usable)
    probabilities = predictions[list(CLASS_LABELS)].to_numpy()
    actual = usable["minutes_class"].to_numpy(dtype="int64")
    onehot = np.zeros_like(probabilities)
    onehot[np.arange(len(actual)), actual] = 1.0

    clipped = np.clip(probabilities, 1e-9, 1.0)
    log_loss = float(-np.mean(np.log(clipped[np.arange(len(actual)), actual])))
    brier = float(np.mean(np.sum((probabilities - onehot) ** 2, axis=1)))

    base_rates = np.bincount(actual, minlength=3) / len(actual)
    baseline_log_loss = float(-np.mean(np.log(np.clip(base_rates[actual], 1e-9, 1.0))))
    baseline_brier = float(np.mean(np.sum((base_rates[None, :] - onehot) ** 2, axis=1)))

    return {
        "n": int(len(usable)),
        "log_loss": log_loss,
        "baseline_log_loss": baseline_log_loss,
        "brier": brier,
        "baseline_brier": baseline_brier,
        "skill_vs_baseline": 1.0 - brier / baseline_brier,
        "accuracy": float(np.mean(np.argmax(probabilities, axis=1) == actual)),
    }


def reliability(model: MinutesModel, holdout: pd.DataFrame, bins: int = 10) -> pd.DataFrame:
    """Predicted vs observed rate of a full appearance, in probability bins.

    Calibration matters more than ranking here. Bench Boost value is a sum of eleven-to-fifteen
    such probabilities, so a model that is systematically 10% optimistic about players starting
    will systematically overvalue the chip.
    """
    usable = holdout.dropna(subset=["minutes_class"])
    predicted = model.predict(usable)["p_full"].to_numpy()
    observed = (usable["minutes_class"].to_numpy() == CLASS_FULL).astype(float)

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
    out = pd.DataFrame(rows)
    out["error"] = out["observed"] - out["predicted"]
    return out
