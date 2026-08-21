"""Team attack and defence ratings via a time-decayed Dixon-Coles model.

This replaces FPL's own Fixture Difficulty Rating, and it has to, for two reasons. FDR is a coarse
integer from 1 to 5, which cannot distinguish a hard fixture from a nearly-impossible one. And at
the start of a season FPL's ``strength_*`` fields are all zero, so there is nothing to read even if
we wanted to.

What we produce instead is a pair of expected goal rates for any fixture, which is what the
simulator actually needs:

    lambda_home = exp(mu + attack[home] - defence[away] + home_advantage)
    lambda_away = exp(mu + attack[away] - defence[home])

Three details matter more than the choice of model:

**Time decay.** A match from three years ago says less about a team than one from last month.
Observations are weighted ``exp(-xi * days_ago)``; the default half-life is about a year, which is
the range the football-modelling literature converges on and which we re-check empirically in
:func:`tune_decay`.

**Club identity across seasons.** Teams are keyed on the stable club ``code``, not the per-season
``team_id`` that FPL reassigns every year. Otherwise Arsenal's rating would reset each August.

**Promoted teams, and an identifiability trap.** Promoted sides are worse, and a shared "promoted"
effect fitted across all seasons captures that well — dropping it costs real out-of-sample accuracy
(cross-validated log-likelihood -2.9787 with it, -2.9883 without).

But the effect and a club's own rating are *not separately identified* when the club's entire
observed history is its promoted season. The fit can read the same results as "average club,
promoted penalty" or "good club, larger penalty", and it splits the difference arbitrarily. The
following season the penalty no longer applies and whatever quality it was offsetting appears as
genuine. Concretely, this rated Sunderland the third-best defence in the league — above Liverpool —
on twenty-nine effective matches, and would have loaded the squad with newly-established clubs'
defenders every August.

Both facts are true at once, so the effect is kept and the identifiability is fixed directly. Each
club carries ``promoted_share``: the time-weighted fraction of its observed matches played as a
promoted side. A club's rating for any future match is

    alpha + promoted_effect * max(promoted_share, is_promoted_in_that_match)

For an established club (share 0, not promoted) this is just its rating. For a club whose whole
history is one promoted season, it collapses to the level they were actually observed at, rather
than to the inflated residual. And a club promoted again is not penalised twice for it.

A club with no top-flight history at all has no rating to correct, so it falls back to a prior
estimated directly from how debutant clubs score and concede.

The Dixon-Coles low-score correction is included because independent Poisson scores under-predict
draws, and draws are exactly what clean-sheet points hinge on.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln

log = logging.getLogger(__name__)

# Both defaults come from the rolling-origin cross-validation in `tune`, not from taste.
#
# Decay: exp(-0.0019 * 365) ~= 0.50, a one-year half-life. Reassuringly this is where the
# football-modelling literature also lands, so two independent routes agree.
DEFAULT_DECAY = 0.0019

# Ridge penalty on team effects, i.e. a Normal prior pulling every team toward the league mean.
# The cross-validated optimum is far stronger than intuition suggests: at a light penalty the
# fitted attack ratings have a standard deviation of 0.26, and out-of-sample fit is clearly worse
# than at 0.14. Untuned, the model rated Luton the fourth-best attack in the league off seven
# effective matches. The curve is genuinely non-monotonic — collapsing ratings entirely
# (infinite ridge) is worse again — so team strength does carry real signal, just less of it than
# an unpenalised fit claims.
DEFAULT_RIDGE = 10.0


@dataclass(slots=True)
class TeamStrength:
    """Fitted attack and defence ratings, and the fixture rates they imply."""

    attack: dict[int, float]  # club code -> attack rating (higher scores more)
    defence: dict[int, float]  # club code -> defence rating (higher concedes less)
    home_advantage: float
    intercept: float
    rho: float  # Dixon-Coles low-score dependence
    # Shared promoted-side handicap, and the time-weighted share of each club's observed matches
    # that were played as a promoted side. The share is what makes the handicap safe to use;
    # see the module docstring.
    promoted_attack: float
    promoted_defence: float
    promoted_share: dict[int, float] = field(default_factory=dict)
    # Fallback for a club with no top-flight history at all, estimated directly from how debutant
    # clubs score and concede.
    debutant_attack: float = 0.0
    debutant_defence: float = 0.0
    decay: float = DEFAULT_DECAY
    n_matches: int = 0
    log_likelihood: float = 0.0
    match_counts: dict[int, float] = field(default_factory=dict)

    def _rating(self, code: int, *, promoted: bool) -> tuple[float, float]:
        """A club's attack and defence for a match, corrected for promotion history."""
        if code not in self.attack:
            # No rating at all: fall back to the debutant prior. The handicap is already baked
            # into that prior, so it must not be applied again on top.
            return self.debutant_attack, self.debutant_defence

        weight = max(self.promoted_share.get(code, 0.0), 1.0 if promoted else 0.0)
        return (
            self.attack[code] + self.promoted_attack * weight,
            self.defence.get(code, 0.0) + self.promoted_defence * weight,
        )

    def rates(
        self,
        home_code: int,
        away_code: int,
        *,
        home_promoted: bool = False,
        away_promoted: bool = False,
    ) -> tuple[float, float]:
        """Expected goals for (home, away) in a single fixture."""
        att_h, def_h = self._rating(home_code, promoted=home_promoted)
        att_a, def_a = self._rating(away_code, promoted=away_promoted)
        lam_h = np.exp(self.intercept + att_h - def_a + self.home_advantage)
        lam_a = np.exp(self.intercept + att_a - def_h)
        return float(lam_h), float(lam_a)

    def effective_table(self) -> pd.DataFrame:
        """Ratings as they are actually used for prediction, i.e. promotion-corrected.

        This is the table to read when sanity-checking the model. The raw ``table`` shows the
        fitted parameters, which for a mostly-promoted club are not a meaningful ability estimate
        on their own.
        """
        codes = sorted(set(self.attack) | set(self.defence))
        rows = []
        for code in codes:
            attack, defence = self._rating(code, promoted=False)
            rows.append(
                {
                    "code": code,
                    "attack": attack,
                    "defence": defence,
                    "promoted_share": self.promoted_share.get(code, 0.0),
                    "matches": self.match_counts.get(code, 0.0),
                }
            )
        return pd.DataFrame(rows).sort_values("attack", ascending=False, ignore_index=True)

    def table(self) -> pd.DataFrame:
        """Ratings as a frame, strongest attack first — for eyeballing plausibility."""
        codes = sorted(set(self.attack) | set(self.defence))
        return pd.DataFrame(
            {
                "code": codes,
                "attack": [self.attack.get(c, 0.0) for c in codes],
                "defence": [self.defence.get(c, 0.0) for c in codes],
                "matches": [self.match_counts.get(c, 0.0) for c in codes],
            }
        ).sort_values("attack", ascending=False, ignore_index=True)


def load_results(con, *, up_to: pd.Timestamp | None = None) -> pd.DataFrame:
    """Every played fixture across all seasons, keyed on stable club codes.

    Args:
        con: Warehouse connection.
        up_to: Only include matches kicking off strictly before this instant. This is what makes
            the model usable inside a backtest: fitting on the full history and then "predicting"
            2021 would be meaningless.
    """
    results = con.execute(
        """
        SELECT
            f.season, f.fixture_id, f.event, f.kickoff_time,
            th.code AS home_code, ta.code AS away_code,
            f.team_h_score AS home_goals, f.team_a_score AS away_goals
        FROM fixtures f
        JOIN teams th ON th.season = f.season AND th.team_id = f.team_h
        JOIN teams ta ON ta.season = f.season AND ta.team_id = f.team_a
        WHERE f.team_h_score IS NOT NULL
          AND f.team_a_score IS NOT NULL
          AND th.code IS NOT NULL AND ta.code IS NOT NULL
        ORDER BY f.kickoff_time
        """
    ).fetchdf()

    # 2016-17 has no teams.csv upstream, so its fixtures cannot be resolved to club codes and
    # drop out here. Everything from 2017-18 on is available.
    if up_to is not None:
        results = results[results["kickoff_time"] < pd.Timestamp(up_to)]
    return results.reset_index(drop=True)


def flag_promoted(results: pd.DataFrame) -> pd.DataFrame:
    """Mark each match according to whether either side is in its first season of a spell.

    "Promoted" here means the club did not appear in the immediately preceding season of our
    data. That is the honest definition given what we can observe: we have no second-tier data, so
    a club's first Premier League season back is exactly the case where its own rating is
    uninformative.
    """
    seasons = sorted(results["season"].unique())
    previous = {season: seasons[i - 1] if i else None for i, season in enumerate(seasons)}

    present: dict[str, set[int]] = {}
    for season, group in results.groupby("season"):
        present[season] = set(group["home_code"]) | set(group["away_code"])

    # Which earlier seasons each club has appeared in, so we can tell a club with no top-flight
    # history from one that was relegated last year.
    seen_before: dict[str, set[int]] = {}
    accumulated: set[int] = set()
    for season in seasons:
        seen_before[season] = set(accumulated)
        accumulated |= present[season]

    def promotion_state(season: str, code: int) -> int:
        """0 = established, 1 = returning after relegation, 2 = no top-flight history."""
        prior = previous[season]
        if prior is None:
            # No preceding season to compare against; treat as established rather than
            # labelling the entire first season of data as promoted.
            return 0
        if code in present.get(prior, set()):
            return 0
        return 1 if code in seen_before[season] else 2

    out = results.copy()
    for side in ("home", "away"):
        state = [
            promotion_state(s, c) for s, c in zip(out["season"], out[f"{side}_code"], strict=True)
        ]
        out[f"{side}_promoted"] = [s > 0 for s in state]
        out[f"{side}_returning"] = [s == 1 for s in state]
        out[f"{side}_debutant"] = [s == 2 for s in state]
    return out


def estimate_debutant_prior(results: pd.DataFrame) -> tuple[float, float]:
    """How good a club with no top-flight history is, in attack and defence rating units.

    Estimated directly rather than fitted alongside the team effects, precisely so that it cannot
    trade off against them. We compare the goals scored and conceded by clubs in their first
    observed season against the league average over the same matches, and convert the ratios to
    the model's log scale.

    Interpretation: a returned ``(-0.25, -0.20)`` means a debutant club is expected to score about
    22% fewer goals and concede about 22% more than an average side.
    """
    debut_rows = results[results["home_debutant"] | results["away_debutant"]]
    if debut_rows.empty:
        return 0.0, 0.0

    scored: list[float] = []
    conceded: list[float] = []
    for _, row in debut_rows.iterrows():
        if row["home_debutant"]:
            scored.append(row["home_goals"])
            conceded.append(row["away_goals"])
        if row["away_debutant"]:
            scored.append(row["away_goals"])
            conceded.append(row["home_goals"])

    league_mean = float(
        np.mean(np.concatenate([results["home_goals"].to_numpy(), results["away_goals"].to_numpy()]))
    )
    if league_mean <= 0:
        return 0.0, 0.0

    attack = float(np.log(max(np.mean(scored), 0.05) / league_mean))
    # Defence is oriented so that higher means conceding fewer.
    defence = -float(np.log(max(np.mean(conceded), 0.05) / league_mean))
    return attack, defence


def _dixon_coles_tau(
    home_goals: np.ndarray, away_goals: np.ndarray, lam_h: np.ndarray, lam_a: np.ndarray, rho: float
) -> np.ndarray:
    """Dixon-Coles correction for the four low-scoring results.

    Independent Poisson margins systematically under-predict 0-0 and 1-1 and over-predict 1-0 and
    0-1. Since clean sheets are worth 4 points to half the squad, getting the 0-0 and 1-1 mass
    right is not a cosmetic improvement.
    """
    tau = np.ones_like(lam_h)
    both_zero = (home_goals == 0) & (away_goals == 0)
    home_one = (home_goals == 1) & (away_goals == 0)
    away_one = (home_goals == 0) & (away_goals == 1)
    both_one = (home_goals == 1) & (away_goals == 1)

    tau[both_zero] = 1 - lam_h[both_zero] * lam_a[both_zero] * rho
    tau[home_one] = 1 + lam_a[home_one] * rho
    tau[away_one] = 1 + lam_h[away_one] * rho
    tau[both_one] = 1 - rho
    # tau must stay positive to take a log; rho is bounded in the optimiser but clip anyway.
    return np.clip(tau, 1e-10, None)


def fit(
    results: pd.DataFrame,
    *,
    decay: float = DEFAULT_DECAY,
    ridge: float = DEFAULT_RIDGE,
    reference_time: pd.Timestamp | None = None,
) -> TeamStrength:
    """Fit attack/defence ratings by weighted maximum likelihood.

    Args:
        results: Output of :func:`load_results`, optionally through :func:`flag_promoted`.
        decay: Exponential decay per day of match age.
        ridge: L2 penalty on team effects, shrinking sparse teams toward the league mean.
        reference_time: "Now" for the purposes of decay; defaults to the last kickoff.
    """
    if "home_promoted" not in results.columns:
        results = flag_promoted(results)
    if results.empty:
        raise ValueError("no results to fit on")

    codes = sorted(set(results["home_code"]) | set(results["away_code"]))
    index = {code: i for i, code in enumerate(codes)}
    n_teams = len(codes)

    home_idx = results["home_code"].map(index).to_numpy()
    away_idx = results["away_code"].map(index).to_numpy()
    home_goals = results["home_goals"].to_numpy(dtype="float64")
    away_goals = results["away_goals"].to_numpy(dtype="float64")
    debutant_attack, debutant_defence = estimate_debutant_prior(results)
    home_promoted = results["home_promoted"].to_numpy(dtype="float64")
    away_promoted = results["away_promoted"].to_numpy(dtype="float64")

    reference = pd.Timestamp(reference_time) if reference_time else results["kickoff_time"].max()
    age_days = (reference - results["kickoff_time"]).dt.total_seconds().to_numpy() / 86400.0
    weights = np.exp(-decay * np.clip(age_days, 0, None))

    # Log-factorial terms are constant in the parameters but keep the reported log-likelihood
    # comparable across different datasets.
    const = -(gammaln(home_goals + 1) + gammaln(away_goals + 1)) @ weights

    # Parameters: attack (n_teams), defence (n_teams), home advantage, intercept, rho, and the
    # shared promoted-side handicap in attack and defence.
    n_params = 2 * n_teams + 5

    def unpack(theta: np.ndarray):
        attack = theta[:n_teams]
        defence = theta[n_teams : 2 * n_teams]
        home_adv, intercept, rho, prom_att, prom_def = theta[2 * n_teams :]
        # Sum-to-zero identifiability: without it attack and the intercept trade off freely.
        attack = attack - attack.mean()
        defence = defence - defence.mean()
        return attack, defence, home_adv, intercept, rho, prom_att, prom_def

    def negative_log_likelihood(theta: np.ndarray) -> float:
        attack, defence, home_adv, intercept, rho, prom_att, prom_def = unpack(theta)

        att_h = attack[home_idx] + prom_att * home_promoted
        att_a = attack[away_idx] + prom_att * away_promoted
        def_h = defence[home_idx] + prom_def * home_promoted
        def_a = defence[away_idx] + prom_def * away_promoted

        log_lam_h = intercept + att_h - def_a + home_adv
        log_lam_a = intercept + att_a - def_h
        # Guard the exponential: an unbounded step can otherwise overflow to inf and stall.
        log_lam_h = np.clip(log_lam_h, -10, 3)
        log_lam_a = np.clip(log_lam_a, -10, 3)
        lam_h = np.exp(log_lam_h)
        lam_a = np.exp(log_lam_a)

        tau = _dixon_coles_tau(home_goals, away_goals, lam_h, lam_a, rho)
        per_match = (
            home_goals * log_lam_h - lam_h + away_goals * log_lam_a - lam_a + np.log(tau)
        )
        penalty = ridge * (np.sum(attack**2) + np.sum(defence**2))
        return -(per_match @ weights + const) + penalty

    initial = np.zeros(n_params)
    initial[2 * n_teams] = 0.25  # home advantage
    initial[2 * n_teams + 1] = np.log(max(np.mean(home_goals + away_goals) / 2, 0.1))
    initial[2 * n_teams + 2] = -0.05  # rho
    initial[2 * n_teams + 3] = -0.15  # promoted attack
    initial[2 * n_teams + 4] = -0.15  # promoted defence

    bounds = (
        [(-3, 3)] * (2 * n_teams)
        + [(-1, 1), (-3, 3), (-0.35, 0.35), (-2, 2), (-2, 2)]
    )

    result = minimize(
        negative_log_likelihood, initial, method="L-BFGS-B", bounds=bounds,
        options={"maxiter": 5000, "maxfun": 200_000},
    )
    if not result.success:
        log.warning("Dixon-Coles optimiser did not fully converge: %s", result.message)

    attack, defence, home_adv, intercept, rho, prom_att, prom_def = unpack(result.x)

    # Effective sample size per team, for reporting how much each rating can be trusted, and the
    # promoted share that makes the handicap safe to reuse across seasons.
    counts: dict[int, float] = dict.fromkeys(codes, 0.0)
    promoted_weight: dict[int, float] = dict.fromkeys(codes, 0.0)
    for i, w, promoted in zip(home_idx, weights, home_promoted, strict=True):
        counts[codes[i]] += w
        promoted_weight[codes[i]] += w * promoted
    for i, w, promoted in zip(away_idx, weights, away_promoted, strict=True):
        counts[codes[i]] += w
        promoted_weight[codes[i]] += w * promoted
    shares = {
        code: (promoted_weight[code] / counts[code] if counts[code] > 0 else 0.0)
        for code in codes
    }

    return TeamStrength(
        attack={code: float(attack[i]) for code, i in index.items()},
        defence={code: float(defence[i]) for code, i in index.items()},
        home_advantage=float(home_adv),
        intercept=float(intercept),
        rho=float(rho),
        promoted_attack=float(prom_att),
        promoted_defence=float(prom_def),
        promoted_share=shares,
        debutant_attack=debutant_attack,
        debutant_defence=debutant_defence,
        decay=decay,
        n_matches=len(results),
        log_likelihood=float(-result.fun),
        match_counts=counts,
    )


def score_predictions(strength: TeamStrength, holdout: pd.DataFrame) -> dict[str, float]:
    """Out-of-sample fit of a fitted model against unseen matches.

    Reports mean predictive log-likelihood (the quantity the fit maximises, so the honest
    selection criterion) alongside RMSE on goals, which is easier to sanity-check by eye.
    """
    if "home_promoted" not in holdout.columns:
        holdout = flag_promoted(holdout)

    rates = np.array(
        [
            strength.rates(
                int(h), int(a), home_promoted=bool(hp), away_promoted=bool(ap)
            )
            for h, a, hp, ap in zip(
                holdout["home_code"],
                holdout["away_code"],
                holdout["home_promoted"],
                holdout["away_promoted"],
                strict=True,
            )
        ]
    )
    lam_h, lam_a = rates[:, 0], rates[:, 1]
    goals_h = holdout["home_goals"].to_numpy(dtype="float64")
    goals_a = holdout["away_goals"].to_numpy(dtype="float64")

    tau = _dixon_coles_tau(goals_h, goals_a, lam_h, lam_a, strength.rho)
    log_likelihood = (
        goals_h * np.log(lam_h)
        - lam_h
        - gammaln(goals_h + 1)
        + goals_a * np.log(lam_a)
        - lam_a
        - gammaln(goals_a + 1)
        + np.log(tau)
    )
    return {
        "mean_log_likelihood": float(np.mean(log_likelihood)),
        "rmse": float(
            np.sqrt(np.mean((lam_h - goals_h) ** 2 + (lam_a - goals_a) ** 2) / 2)
        ),
        "n": int(len(holdout)),
    }


def tune(
    results: pd.DataFrame,
    *,
    decays: tuple[float, ...] = (0.0008, 0.0013, 0.0019, 0.0026, 0.0035),
    ridges: tuple[float, ...] = (0.02, 0.05, 0.12, 0.30, 0.75),
    folds: int = 6,
    holdout_matches: int = 200,
) -> pd.DataFrame:
    """Select ``decay`` and ``ridge`` by rolling-origin cross-validation.

    Both hyperparameters trade off in the same direction — how much to trust old or thin evidence
    — and guessing them is how a team with seven matches played ends up rated fourth best in the
    league. So they are chosen on held-out matches instead.

    Folds are strictly chronological: fit on everything before a cutoff, score the next
    ``holdout_matches`` fixtures, walk the cutoff forward. Never shuffled, because a model that
    has seen next season cannot tell us anything about predicting it.

    Returns:
        One row per (decay, ridge) with mean held-out log-likelihood, best first.
    """
    if "home_promoted" not in results.columns:
        results = flag_promoted(results)
    results = results.sort_values("kickoff_time").reset_index(drop=True)

    n = len(results)
    if n < holdout_matches * 2:
        raise ValueError(f"need at least {holdout_matches * 2} matches to tune, have {n}")

    # Cutoffs spread over the back half of the data, so every fold has a substantial training set.
    first_cut = max(n // 2, holdout_matches)
    cutoffs = np.linspace(first_cut, n - holdout_matches, folds, dtype=int)

    records = []
    for decay in decays:
        for ridge in ridges:
            scores = []
            for cut in cutoffs:
                train = results.iloc[:cut]
                holdout = results.iloc[cut : cut + holdout_matches]
                # Decay is measured relative to the cutoff, not to today, or later folds would
                # see their training data as artificially stale.
                reference = train["kickoff_time"].max()
                try:
                    strength = fit(
                        train, decay=decay, ridge=ridge, reference_time=reference
                    )
                except (ValueError, RuntimeError) as exc:  # pragma: no cover
                    log.warning("fold failed (decay=%s ridge=%s): %s", decay, ridge, exc)
                    continue
                # Holdout teams unseen in training get no rating; score_predictions falls back to
                # the league mean for them, which is the correct treatment of a genuine unknown.
                scores.append(score_predictions(strength, holdout)["mean_log_likelihood"])
            if scores:
                records.append(
                    {
                        "decay": decay,
                        "ridge": ridge,
                        "mean_log_likelihood": float(np.mean(scores)),
                        "folds": len(scores),
                        "half_life_days": float(np.log(2) / decay),
                    }
                )

    return pd.DataFrame(records).sort_values(
        "mean_log_likelihood", ascending=False, ignore_index=True
    )


def fixture_rates(
    con, strength: TeamStrength, season: str, *, promoted: set[int] | None = None
) -> pd.DataFrame:
    """Expected goals for every fixture of a season, played or not.

    This is the fixture-difficulty table the rest of the system reads: continuous, symmetric, and
    directly interpretable as "how many goals do we expect each side to score".
    """
    fixtures = con.execute(
        """
        SELECT f.fixture_id, f.event, f.kickoff_time,
               f.team_h, f.team_a, th.code AS home_code, ta.code AS away_code,
               th.short_name AS home, ta.short_name AS away,
               f.team_h_difficulty, f.team_a_difficulty
        FROM fixtures f
        JOIN teams th ON th.season = f.season AND th.team_id = f.team_h
        JOIN teams ta ON ta.season = f.season AND ta.team_id = f.team_a
        WHERE f.season = ?
        ORDER BY f.event, f.kickoff_time
        """,
        [season],
    ).fetchdf()

    if promoted is None:
        states = promotion_states(con, season)
        promoted = states["returning"] | states["debutant"]
    rates = [
        strength.rates(
            int(h), int(a), home_promoted=int(h) in promoted, away_promoted=int(a) in promoted
        )
        for h, a in zip(fixtures["home_code"], fixtures["away_code"], strict=True)
    ]
    fixtures["xg_home"] = [r[0] for r in rates]
    fixtures["xg_away"] = [r[1] for r in rates]
    return fixtures


def promotion_states(con, season: str) -> dict[str, set[int]]:
    """Split ``season``'s promoted clubs into returning and debutant.

    ``returning`` clubs appeared in some earlier season of our data (relegated and back); their
    own rating already carries evidence of how they fare in this division. ``debutant`` clubs have
    no top-flight history here at all, so their rating is uninformative and the model leans
    entirely on the shared promoted-team effect.
    """
    seasons = [
        r[0] for r in con.execute("SELECT DISTINCT season FROM teams ORDER BY season").fetchall()
    ]
    if season not in seasons:
        raise ValueError(f"unknown season {season}")
    position = seasons.index(season)
    if position == 0:
        return {"returning": set(), "debutant": set()}

    def codes_in(target: str) -> set[int]:
        return {
            r[0]
            for r in con.execute(
                "SELECT code FROM teams WHERE season = ? AND code IS NOT NULL", [target]
            ).fetchall()
        }

    current = codes_in(season)
    previous = codes_in(seasons[position - 1])
    earlier: set[int] = set()
    for prior in seasons[: position - 1]:
        earlier |= codes_in(prior)

    promoted = current - previous
    return {
        "returning": promoted & earlier,
        "debutant": promoted - earlier,
    }
