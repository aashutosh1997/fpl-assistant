"""Per-90 scoring rates for individual players, with empirical-Bayes shrinkage.

The problem this solves: a striker who has scored 3 goals in 180 minutes has an observed rate of
1.5 goals per 90. Taken at face value that makes him the best forward in the world. Taken at face
value in August, it makes the optimiser buy him.

So every rate here is shrunk toward a prior formed from comparable players — same position, same
price bracket — with the strength of the shrinkage set by how many minutes we have actually seen.
The Gamma-Poisson conjugate pair gives this exactly and cheaply:

    posterior_rate = (alpha + events) / (beta + minutes / 90)

where ``alpha`` and ``beta`` are fitted per position-and-price group by matching the observed mean
and variance of rates within that group. A player with 2000 minutes behind him is dominated by his
own record; a player with 180 minutes is dominated by his peers. Nothing has to be hand-tuned, and
the transition between the two is smooth rather than a minutes cutoff.

**Why expected goals rather than goals.** Where xG is available (2022-23 onward) it is the primary
signal, because for a given number of minutes it predicts future scoring better than goals do —
finishing fluctuates far more than shot volume and quality. Goals are retained as a secondary
signal so that genuinely exceptional finishing is not entirely discarded, and as the only signal
for the earlier seasons where FPL published no xG.

Rates here are *context-free*: what a player does per 90 minutes against an average opponent. The
simulator scales them by the specific fixture using the team-strength model, so a rate must not
already have fixture difficulty baked into it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Price brackets in FPL's internal units (tenths of a million). Chosen because FPL prices players
# by expected involvement, which makes price the single best available proxy for the role a manager
# is expected to play — and role drives rates far more than raw ability does.
PRICE_TIER_EDGES = (0, 45, 55, 70, 90, 1000)
PRICE_TIER_LABELS = ("budget", "rotation", "mid", "premium", "elite")

# Minutes below which a player's own record is treated as essentially uninformative on its own.
# Not a hard cutoff: it only sets the reporting flag, the shrinkage itself is continuous.
THIN_SAMPLE_MINUTES = 450

# The rates we model. Each maps to the count column it is estimated from.
COUNT_RATES = {
    "goals": "goals_scored",
    "assists": "assists",
    "defcon": "defcon_count",
    "saves": "saves",
    "yellow_cards": "yellow_cards",
    "bonus": "bonus",
}

# Continuous (already-expected) quantities, averaged per 90 rather than counted.
EXPECTED_RATES = {
    "xg": "expected_goals",
    "xa": "expected_assists",
    "xgc": "expected_goals_conceded",
}

# Per-stat exposure columns on the totals frame: ``min_<count column>`` holds the (recency-weighted)
# minutes played in seasons that actually published that stat, which is the only correct
# denominator for a rate whose numerator skips the seasons that did not.
MINUTES_PREFIX = "min_"


def exposure_column(count_column: str) -> str:
    """Name of the per-stat minutes column for ``count_column``."""
    return f"{MINUTES_PREFIX}{count_column}"


def _exposure(frame: pd.DataFrame, count_column: str) -> pd.Series:
    """Minutes over which ``count_column`` was observed, falling back to all minutes."""
    column = exposure_column(count_column)
    if column in frame.columns:
        return pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
    return pd.to_numeric(frame["minutes"], errors="coerce").fillna(0.0)


@dataclass(slots=True)
class RatePriors:
    """Fitted Gamma priors, one per (position, price tier, rate)."""

    priors: dict[tuple[str, str, str], tuple[float, float]] = field(default_factory=dict)
    fallback: dict[str, tuple[float, float]] = field(default_factory=dict)

    def get(self, position: str, tier: str, rate: str) -> tuple[float, float]:
        found = self.priors.get((position, tier, rate))
        if found is not None:
            return found
        # A position/tier combination we have never seen (a £13m goalkeeper, say) falls back to
        # the position-wide prior rather than to nothing.
        return self.fallback.get(f"{position}:{rate}", self.fallback.get(rate, (1.0, 1.0)))


def price_tier(value: pd.Series) -> pd.Series:
    """Bucket FPL prices into role-shaped brackets."""
    return pd.cut(
        pd.to_numeric(value, errors="coerce"),
        bins=list(PRICE_TIER_EDGES),
        labels=list(PRICE_TIER_LABELS),
        right=False,
    ).astype("object")


def player_totals(
    con,
    *,
    seasons: list[str] | None = None,
    up_to_season: str | None = None,
    up_to_gw_seq: int | None = None,
    decay_seasons: float = 1.5,
) -> pd.DataFrame:
    """Aggregate each player's event counts and minutes, weighted toward recent seasons.

    Args:
        seasons: Restrict to these seasons; defaults to all.
        up_to_season: With ``up_to_gw_seq``, stop at a point mid-season. This is what makes the
            rates usable inside a backtest.
        up_to_gw_seq: Exclusive upper bound on gameweek sequence within ``up_to_season``.
        decay_seasons: Exponential decay constant in seasons. A player's form two years ago is
            weighted ``exp(-2 / decay_seasons)`` relative to now.

    Returns:
        One row per (code, position) with weighted minutes and weighted event totals. Keyed on
        ``code`` so a player's record follows him across transfers and seasons.
    """
    filters = ["p.position NOT IN ('AM')"]
    params: list[object] = []
    if seasons:
        filters.append(f"p.season IN ({', '.join('?' for _ in seasons)})")
        params += list(seasons)
    if up_to_season is not None:
        # Everything from earlier seasons, plus the part of the current season already played.
        filters.append("(p.season < ? OR (p.season = ? AND p.gw_seq < ?))")
        params += [up_to_season, up_to_season, up_to_gw_seq or 1]

    # FPL began publishing expected stats in gameweek 16 of 2022-23 and stored 0.0, not null, for
    # the fifteen gameweeks before, which read as a season-long drought of chances for everyone
    # who played then. A gameweek whose league-wide expected goals sum to nothing published none.
    frame = con.execute(
        f"""
        WITH published AS (
            SELECT season, gw, sum(expected_goals) > 0 AS has_expected
            FROM player_gw
            GROUP BY season, gw
        )
        SELECT
            pl.code, p.season, p.gw_seq, p.position, p.minutes, p.prev_value,
            p.goals_scored, p.assists, p.defcon_count, p.saves, p.yellow_cards, p.bonus,
            CASE WHEN pub.has_expected THEN p.expected_goals END AS expected_goals,
            CASE WHEN pub.has_expected THEN p.expected_assists END AS expected_assists,
            CASE WHEN pub.has_expected THEN p.expected_goals_conceded END
                AS expected_goals_conceded
        FROM player_gw_as_of p
        JOIN players pl ON pl.season = p.season AND pl.element = p.element
        LEFT JOIN published pub ON pub.season = p.season AND pub.gw = p.gw
        WHERE {" AND ".join(filters)} AND pl.code IS NOT NULL
        """,
        params,
    ).fetchdf()

    if frame.empty:
        return frame

    # Season recency weight. Ordering seasons lexically works because they are "YYYY-YY".
    season_order = {s: i for i, s in enumerate(sorted(frame["season"].unique()))}
    age = frame["season"].map(season_order).max() - frame["season"].map(season_order)
    frame["weight"] = np.exp(-age / decay_seasons)

    frame["position"] = frame["position"].replace({"GK": "GKP"})
    weighted = frame.assign(w_minutes=frame["minutes"] * frame["weight"])
    for column in list(COUNT_RATES.values()) + list(EXPECTED_RATES.values()):
        values = pd.to_numeric(weighted[column], errors="coerce")
        weighted[f"w_{column}"] = values * weighted["weight"]
        # Exposure for *this* stat: only the minutes in seasons that published it. Summing the
        # counts (NaN seasons drop out) while dividing by every season's minutes halved the DEFCON
        # rate of anyone with pre-2025 history — Virgil came out at 4.9 per 90 against a true 9.1
        # — and quietly cost defenders half a point a match in the projections. xG, published from
        # 2022-23, was diluted the same way for veterans.
        weighted[f"{MINUTES_PREFIX}{column}"] = np.where(values.isna(), 0.0, weighted["w_minutes"])

    aggregations = {
        "minutes": ("w_minutes", "sum"),
        "raw_minutes": ("minutes", "sum"),
        "weight": ("weight", "sum"),
        "last_value": ("prev_value", "last"),
    }
    for column in list(COUNT_RATES.values()) + list(EXPECTED_RATES.values()):
        aggregations[column] = (f"w_{column}", "sum")
        aggregations[f"{MINUTES_PREFIX}{column}"] = (f"{MINUTES_PREFIX}{column}", "sum")

    # A player's position is whichever one he has played most of his recent minutes in; FPL
    # reclassifies players between seasons and occasionally mid-season.
    dominant = (
        weighted.groupby(["code", "position"])["w_minutes"]
        .sum()
        .reset_index()
        .sort_values("w_minutes")
        .groupby("code")
        .tail(1)
        .set_index("code")["position"]
    )

    totals = weighted.groupby("code").agg(**aggregations).reset_index()
    totals["position"] = totals["code"].map(dominant)
    totals["price_tier"] = price_tier(totals["last_value"])
    totals["thin_sample"] = totals["raw_minutes"] < THIN_SAMPLE_MINUTES
    return totals


def fit_priors(totals: pd.DataFrame, *, min_group: int = 15) -> RatePriors:
    """Fit Gamma priors per (position, price tier) by method of moments.

    For a Gamma(alpha, beta) prior on a Poisson rate, the marginal mean is ``alpha / beta`` and
    the variance across players exceeds the Poisson-only variance by ``alpha / beta**2``. Matching
    both to the observed spread of per-90 rates within a group recovers the pair directly, with no
    optimisation and no tuning.

    Groups with too few players fall back to a position-wide prior, and then to a global one.
    """
    priors: dict[tuple[str, str, str], tuple[float, float]] = {}
    fallback: dict[str, tuple[float, float]] = {}

    def moments(rates: np.ndarray, weights: np.ndarray) -> tuple[float, float] | None:
        """Recover (alpha, beta) from the weighted mean and variance of observed rates."""
        if len(rates) < 3:
            return None
        mean = float(np.average(rates, weights=weights))
        variance = float(np.average((rates - mean) ** 2, weights=weights))
        if mean <= 1e-9 or variance <= 1e-12:
            # No spread to explain: fall back to a weak prior centred on the mean, which behaves
            # like "worth about one match of evidence".
            return (max(mean, 1e-6), 1.0)
        beta = mean / variance
        alpha = mean * beta
        # Keep the prior weak enough that a full season of minutes can override it. Without this
        # cap, a tight group (goalkeeper saves, say) produces a prior so strong that no individual
        # ever escapes it.
        beta = float(np.clip(beta, 0.05, 12.0))
        alpha = float(max(mean * beta, 1e-6))
        return (alpha, beta)

    def observed(group: pd.DataFrame, count_column: str) -> tuple[np.ndarray, np.ndarray] | None:
        # Exposure is the minutes in seasons that published this stat, not all minutes: a prior
        # fitted on diluted rates is diluted itself, and then shrinks everyone toward the error.
        nineties = _exposure(group, count_column) / 90.0
        usable = (nineties > 2) & group[count_column].notna()
        if usable.sum() < 3:
            return None
        rates = (group.loc[usable, count_column] / nineties[usable]).to_numpy(dtype="float64")
        return rates, nineties[usable].to_numpy(dtype="float64")

    all_rates = {**COUNT_RATES, **EXPECTED_RATES}
    for rate, column in all_rates.items():
        for position, by_position in totals.groupby("position", observed=True):
            sampled = observed(by_position, column)
            if sampled:
                fitted = moments(*sampled)
                if fitted:
                    fallback[f"{position}:{rate}"] = fitted

            for tier, group in by_position.groupby("price_tier", observed=True):
                if len(group) < min_group:
                    continue
                sampled = observed(group, column)
                if not sampled:
                    continue
                fitted = moments(*sampled)
                if fitted:
                    priors[(str(position), str(tier), rate)] = fitted

        sampled = observed(totals, column)
        if sampled:
            fitted = moments(*sampled)
            if fitted:
                fallback[rate] = fitted

    log.info("fitted %d group priors across %d rates", len(priors), len(all_rates))
    return RatePriors(priors=priors, fallback=fallback)


def measure_expected_conversion(con, season: str) -> dict[str, dict[str, float]] | None:
    """FPL goals per unit of xG and FPL assists per unit of xA, by position, in one season.

    Opta's expected stats and FPL's counts are on different scales, and differently by position:
    in 2023-26 defenders scored 0.76-0.93 FPL goals per unit of xG (set-piece headers under-convert)
    while midfielders and forwards scored one for one, and forwards earned 2.1 FPL assists per
    unit of xA (rebounds, penalties won) against 1.1-1.4 for everyone else. Blending the raw
    stats into FPL rates handed defenders goals and took assists from forwards. Returns None for
    a season that published no expected stats.
    """
    rows = con.execute(
        """
        WITH published AS (
            SELECT season, gw FROM player_gw GROUP BY season, gw HAVING sum(expected_goals) > 0
        )
        SELECT p.position, sum(p.goals_scored), sum(p.expected_goals),
               sum(p.assists), sum(p.expected_assists)
        FROM player_gw p JOIN published USING (season, gw)
        WHERE p.season = ? AND p.minutes > 0 AND p.position NOT IN ('AM', 'GK', 'GKP')
        GROUP BY p.position
        """,
        [season],
    ).fetchall()
    conversion: dict[str, dict[str, float]] = {"xg": {}, "xa": {}}
    for position, goals, xg, assists, xa in rows:
        if xg and xg > 0:
            conversion["xg"][str(position)] = float(goals) / float(xg)
        if xa and xa > 0:
            conversion["xa"][str(position)] = float(assists) / float(xa)
    return conversion if conversion["xg"] or conversion["xa"] else None


# How much of a goal or assist rate comes from the expected stats (converted into FPL's units by
# :func:`measure_expected_conversion`) rather than FPL's own counts. Chosen by out-of-sample Poisson
# deviance of the next six gameweeks' FPL goals and assists over 2023-26, the seasons with at least
# a year of expected stats behind them. Once converted, both blends are calibrated by position at
# any weight, so the weight only decides how much the steadier expected stats are trusted.
XG_GOAL_WEIGHT = 0.8
XA_ASSIST_WEIGHT = 0.6


def shrink(
    totals: pd.DataFrame,
    priors: RatePriors,
    *,
    xg_weight: float = XG_GOAL_WEIGHT,
    xa_weight: float = XA_ASSIST_WEIGHT,
    conversion: dict[str, dict[str, float]] | None = None,
) -> pd.DataFrame:
    """Apply the priors, returning one shrunk per-90 rate per player.

    Args:
        xg_weight: How much of the goal rate comes from expected goals rather than actual ones,
            where xG is available. Shot volume and quality persist while finishing largely does
            not, so expected goals carry most of it.
        xa_weight: The same for assists.
        conversion: FPL goals per unit of xG and assists per unit of xA by position, from
            :func:`measure_expected_conversion`, applied before blending so that both halves of
            each blend are in FPL's units. None blends the raw stats.
    """
    out = totals[["code", "position", "price_tier", "minutes", "raw_minutes", "thin_sample"]].copy()

    for rate, column in {**COUNT_RATES, **EXPECTED_RATES}.items():
        events = pd.to_numeric(totals[column], errors="coerce").to_numpy(dtype="float64")
        alpha = np.empty(len(totals))
        beta = np.empty(len(totals))
        for i, (position, tier) in enumerate(
            zip(totals["position"], totals["price_tier"], strict=True)
        ):
            alpha[i], beta[i] = priors.get(str(position), str(tier), rate)

        # The exposure is the minutes in seasons that published this stat. Where none did, the
        # count is NaN and the exposure zero, so the player sits at the prior mean — the events
        # are unobserved rather than absent.
        observed_events = np.nan_to_num(events, nan=0.0)
        observed_nineties = (_exposure(totals, column) / 90.0).to_numpy(dtype="float64")
        observed_nineties = np.where(np.isnan(events), 0.0, observed_nineties)
        out[rate] = (alpha + observed_events) / (beta + observed_nineties)

    if conversion:
        positions = totals["position"].astype(str)
        out["xg"] = out["xg"] * positions.map(conversion.get("xg", {})).fillna(1.0).to_numpy()
        out["xa"] = out["xa"] * positions.map(conversion.get("xa", {})).fillna(1.0).to_numpy()

    # Blend expected and actual for the two rates where both exist. Where xG was never published
    # for a player, the prior-driven xg estimate carries no information about him specifically, so
    # lean on the goal rate instead.
    has_xg = (
        pd.to_numeric(totals["expected_goals"], errors="coerce").notna()
        & (_exposure(totals, "expected_goals") > 0)
    ).to_numpy()
    goal_weight = np.where(has_xg, xg_weight, 0.0)
    assist_weight = np.where(has_xg, xa_weight, 0.0)
    out["goal_rate"] = goal_weight * out["xg"] + (1 - goal_weight) * out["goals"]
    out["assist_rate"] = assist_weight * out["xa"] + (1 - assist_weight) * out["assists"]

    out["defcon_rate"] = out["defcon"]
    out["save_rate"] = out["saves"]
    out["card_rate"] = out["yellow_cards"]
    out["xgc_rate"] = out["xgc"]
    return out


def duties(con, season: str) -> pd.DataFrame:
    """Penalty and set-piece responsibility, from the live API's ordering fields.

    Worth its own lookup because these are step changes in value that historical rates pick up far
    too slowly. A player promoted to penalties gains roughly 0.1 goals a game overnight, and the
    rate model would take half a season to notice.
    """
    return con.execute(
        """
        SELECT pl.code, pl.element, pl.web_name, pl.element_type
        FROM players pl
        WHERE pl.season = ?
        """,
        [season],
    ).fetchdf()


def build(
    con,
    *,
    up_to_season: str | None = None,
    up_to_gw_seq: int | None = None,
    decay_seasons: float = 1.5,
) -> tuple[pd.DataFrame, RatePriors]:
    """Convenience path: aggregate, fit priors, and shrink in one call."""
    totals = player_totals(
        con,
        up_to_season=up_to_season,
        up_to_gw_seq=up_to_gw_seq,
        decay_seasons=decay_seasons,
    )
    if totals.empty:
        raise ValueError("no player history available for the requested cutoff")
    priors = fit_priors(totals)
    # The conversion comes from the last completed season before the one being projected.
    if up_to_season is not None:
        previous = con.execute(
            "SELECT max(season) FROM player_gw WHERE season < ?", [up_to_season]
        ).fetchone()[0]
    else:
        row = con.execute(
            "SELECT season FROM player_gw GROUP BY season "
            "HAVING count(DISTINCT gw) >= 38 ORDER BY season DESC LIMIT 1"
        ).fetchone()
        previous = row[0] if row else None
    conversion = measure_expected_conversion(con, previous) if previous else None
    return shrink(totals, priors, conversion=conversion), priors


@dataclass(slots=True)
class TeamReturns:
    """How a team's scoreline becomes its players' FPL returns, measured from a season.

    ``scored`` is the share of team goals credited to one of its players (the rest are own goals
    by the opposition) and ``assisted`` the share that also earns an FPL assist. FPL's assists are
    more generous than Opta's (rebounds, penalties won, deflected passes): in 2016-26 an assist
    went with 86-90% of team goals every season, not the two thirds a hand-set 35% unassisted
    share gave, which left every attacker's assists a quarter short.

    Saves respond to goals conceded far less than the hand-set ``0.6 + 0.4 * conceded`` had it:
    among ninety-minute keepers, saves ~ 2.6 + 0.1-0.3 per goal conceded in every season. The
    line is kept here and applied normalised to its mean, so a keeper's own saves-per-90 rate
    stays his average and the scoreline only tilts it.
    """

    scored: float
    assisted: float
    save_intercept: float
    save_slope: float
    mean_conceded: float
    seasons: tuple[str, ...] = ()

    def save_response(self, conceded: np.ndarray) -> np.ndarray:
        """Multiplier on a keeper's saves rate given goals conceded, averaging one."""
        typical = self.save_intercept + self.save_slope * self.mean_conceded
        return (self.save_intercept + self.save_slope * conceded) / typical


def measure_team_returns(con, seasons: list[str]) -> TeamReturns | None:
    """Goal, assist and save responses to the scoreline in the given seasons, or None."""
    team_goals, player_goals, assists = con.execute(
        """
        WITH team AS (
            SELECT sum(team_h_score + team_a_score) AS goals
            FROM fixtures
            WHERE list_contains(?, season) AND finished AND team_h_score IS NOT NULL
        ), credited AS (
            SELECT sum(goals_scored) AS goals, sum(assists) AS assists
            FROM player_gw
            WHERE list_contains(?, season)
        )
        SELECT team.goals, credited.goals, credited.assists FROM team, credited
        """,
        [list(seasons), list(seasons)],
    ).fetchone()
    intercept, slope, conceded = con.execute(
        """
        SELECT regr_intercept(saves, goals_conceded), regr_slope(saves, goals_conceded),
               avg(goals_conceded)
        FROM player_gw_derived
        WHERE list_contains(?, season) AND position IN ('GK', 'GKP') AND minutes = 90
        """,
        [list(seasons)],
    ).fetchone()
    if not team_goals or player_goals is None or assists is None or intercept is None:
        return None
    return TeamReturns(
        scored=min(float(player_goals) / float(team_goals), 1.0),
        assisted=min(float(assists) / float(team_goals), 1.0),
        save_intercept=float(intercept),
        save_slope=max(float(slope), 0.0),
        mean_conceded=float(conceded),
        seasons=tuple(seasons),
    )
