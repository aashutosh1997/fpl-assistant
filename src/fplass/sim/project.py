"""Assembling everything into a forward projection.

This is the seam between "models fitted on history" and "what happens over the next eight
gameweeks". It fits or accepts the component models, builds the feature rows for fixtures that have
not been played, and hands them to the simulator.

One approximation is made deliberately and is worth naming. The minutes model's features are
rolling form — recent minutes, recent starts — which are only known up to today. For the gameweek
immediately ahead that is exactly right. For gameweek six it cannot be, because what happens in
gameweeks two through five is unknown. Rather than pretend to simulate feature evolution, current
form is held constant across the horizon and only genuinely fixture-dependent inputs (congestion,
rest days, season progress) vary. This is the same approximation a human makes when they look at a
run of fixtures, and it degrades gracefully: near gameweeks are sharp, far ones regress toward the
player's established level, which is the honest shape of the uncertainty.

The other seam that matters is the season boundary. In August a player has no current-season
matches at all, so form is carried across from last season by joining on the stable club and player
codes rather than the per-season ids FPL reassigns.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..features import adjust as adjust_module
from ..features import bps as bps_module
from ..features import minutes as minutes_module
from ..features import rates as rates_module
from ..features import teams as teams_module
from ..ingest.sources import CURRENT_SEASON
from ..scoring import ScoringRules, rules_for_season
from .engine import SimulationResult, build_player_matches, simulate

log = logging.getLogger(__name__)

DEFAULT_HORIZON = 8


@dataclass(slots=True)
class ProjectionModels:
    """The fitted components a projection needs. Fit once, reuse across many plans."""

    strength: teams_module.TeamStrength
    minutes: minutes_module.MinutesModel
    rates: pd.DataFrame
    bps: bps_module.BPSModel
    rules: ScoringRules
    season: str
    # Recalibration of the minutes model on the season actually being played, using signals the
    # base model cannot carry (preseason friendlies, ownership). None until enough gameweeks have
    # been observed to fit it — at which point the base model passes through untouched, which is
    # the honest behaviour when there is nothing to calibrate against.
    minutes_adjustment: adjust_module.MinutesAdjustment | None = None


def fit_models(
    con,
    *,
    season: str = CURRENT_SEASON,
    bps_seasons: list[str] | None = None,
) -> ProjectionModels:
    """Fit every component model from the warehouse.

    Args:
        bps_seasons: Seasons to fit the bonus model on. Defaults to the most recent completed
            season as a prior for the reworked 2026/27 weights; once the current season has
            enough matches, pass it here instead so the model learns the actual new weights.
    """
    results = teams_module.flag_promoted(teams_module.load_results(con))
    strength = teams_module.fit(results)

    features = minutes_module.build_features(con)
    minutes_model, _ = minutes_module.fit(features)

    rate_table, _ = rates_module.build(con)

    if bps_seasons is None:
        played = con.execute(
            "SELECT season FROM player_gw GROUP BY season "
            "HAVING count(DISTINCT gw) >= 38 ORDER BY season DESC LIMIT 1"
        ).fetchone()
        bps_seasons = [played[0]] if played else [season]
    bps_model = bps_module.fit(con, seasons=bps_seasons)

    return ProjectionModels(
        strength=strength,
        minutes=minutes_model,
        rates=rate_table,
        bps=bps_model,
        rules=rules_for_season(con, season),
        season=season,
        minutes_adjustment=fit_minutes_adjustment(con, season),
    )


def fit_minutes_adjustment(
    con, season: str = CURRENT_SEASON
) -> adjust_module.MinutesAdjustment | None:
    """Fit the recalibration layer on whatever of this season has already been played.

    Returns ``None`` before there is enough data, in which case the base model is used unchanged.
    """
    from ..backtest import calibrate_live

    try:
        frame = calibrate_live.scored_frame(con, season=season)
    except Exception as exc:  # pragma: no cover - missing tables on a fresh warehouse
        log.debug("no scored projections available: %s", exc)
        return None
    if frame.empty:
        return None

    frame = calibrate_live.attach_ownership(frame)
    return adjust_module.fit(
        frame,
        frame["p_full"].to_numpy(dtype="float64"),
        frame["played_full"].to_numpy(dtype="float64"),
        gameweeks=tuple(sorted(int(g) for g in frame["gw"].unique())),
    )


def _price_snapshots() -> pd.DataFrame | None:
    """The logged ownership history. Single implementation lives in the calibration harness."""
    from ..backtest.calibrate_live import load_price_snapshots

    return load_price_snapshots()


def current_players(con, season: str = CURRENT_SEASON) -> pd.DataFrame:
    """Every selectable player with price, position and club.

    ``position`` needs the explicit ``AS`` keyword — it is a reserved word in DuckDB and the
    query is a parse error without it.
    """
    return con.execute(
        """
        SELECT
            pl.element, pl.code, pl.web_name, pl.team_id, pl.element_type,
            CASE pl.element_type WHEN 1 THEN 'GKP' WHEN 2 THEN 'DEF'
                                 WHEN 3 THEN 'MID' ELSE 'FWD' END AS position,
            pl.start_cost,
            pl.start_cost AS price,
            t.short_name AS team
        FROM players pl
        LEFT JOIN teams t ON t.season = pl.season AND t.team_id = pl.team_id
        WHERE pl.season = ?
        ORDER BY pl.element
        """,
        [season],
    ).fetchdf()


def live_player_table(con, client=None, season: str = CURRENT_SEASON) -> pd.DataFrame:
    """The player reference table with *live* prices and availability from the API.

    The warehouse stores each player's starting cost; prices drift daily from the moment the
    season opens, and a plan priced off stale costs may simply be unaffordable when you go to
    execute it. So anything that spends money reads prices from here.
    """
    from ..api import FPLClient

    owns_client = client is None
    client = client or FPLClient()
    try:
        bootstrap = client.bootstrap()
    finally:
        if owns_client:
            client.close()

    live = pd.DataFrame(
        [
            {
                "element": e["id"],
                "price": e["now_cost"],
                "status": e.get("status"),
                "chance_of_playing_next_round": e.get("chance_of_playing_next_round"),
                "news": e.get("news"),
                "selected_by_percent": pd.to_numeric(
                    e.get("selected_by_percent"), errors="coerce"
                ),
            }
            for e in bootstrap.get("elements", [])
        ]
    )
    reference = current_players(con, season).drop(columns=["price"])
    return reference.merge(live, on="element", how="left").assign(
        price=lambda f: f["price"].fillna(f["start_cost"]).astype(int)
    )


def player_form(con, season: str = CURRENT_SEASON, *, lookback: int = 10) -> pd.DataFrame:
    """Each current player's rolling-form features, handling the season boundary properly.

    Form has to be carried across the summer on the stable player ``code``, or every projection
    made in August would treat the whole league as debutants. But carrying it *literally* — taking
    the last five matches played — is actively harmful, because the end of a season is the least
    representative part of it: titles are decided, managers rotate, players carry knocks into the
    break, and some have already been sold.

    That is not a hypothetical. Taking Haaland's last five matches of 2025-26 gave him an average
    of 54 minutes, a 60% start rate and zero minutes in his most recent match, so the model rated
    the best striker in the league a rotation risk at 54% to play an hour. The minutes model is
    trained on within-season sequences, where "did not play last week" genuinely predicts being
    benched next week; across a three-month break that inference does not hold.

    So the window depends on where in the season we are:

    * **Matches already played this season** — use them. Recent form is the real signal, and the
      more of it there is, the less last season matters.
    * **No matches yet** — use the *whole* of the player's last season, excluding its final two
      gameweeks, and use averages rather than the literal most recent match. A season-long
      involvement rate is a far better predictor of next season's role than any single match.

    The two are blended on how much current-season evidence exists, so the projection transitions
    smoothly from "last season's role" in August to "this season's form" by autumn.
    """
    frame = con.execute(
        """
        WITH history AS (
            SELECT
                pl_now.element,
                p.season,
                p.gw_seq,
                p.minutes,
                p.starts,
                p.kickoff_time,
                p.prev_value,
                max(p.gw_seq) OVER (PARTITION BY pl_now.element, p.season) AS season_last_gw,
                row_number() OVER (
                    PARTITION BY pl_now.element ORDER BY p.kickoff_time DESC
                ) AS recency
            FROM players pl_now
            JOIN players pl_hist ON pl_hist.code = pl_now.code
            JOIN player_gw_as_of p
                ON p.season = pl_hist.season AND p.element = pl_hist.element
            WHERE pl_now.season = ?
        ),
        current AS (
            SELECT
                element,
                count(*)                                            AS n_current,
                avg(CASE WHEN recency <= 3 THEN minutes END)         AS cur_minutes_3,
                avg(CASE WHEN recency <= 5 THEN minutes END)         AS cur_minutes_5,
                avg(CASE WHEN recency <= 10 THEN minutes END)        AS cur_minutes_10,
                avg(CASE WHEN recency <= 5 THEN
                    CASE WHEN starts > 0 THEN 1.0 ELSE 0.0 END END)  AS cur_start_rate,
                avg(CASE WHEN recency <= 5 THEN
                    CASE WHEN minutes > 0 THEN 1.0 ELSE 0.0 END END) AS cur_played_rate,
                max(CASE WHEN recency = 1 THEN minutes END)          AS cur_minutes_last,
                max(CASE WHEN recency = 2 THEN minutes END)          AS cur_minutes_prev,
                max(CASE WHEN recency = 1 THEN
                    CASE WHEN starts > 0 THEN 1.0 ELSE 0.0 END END)  AS cur_started_last,
                max(CASE WHEN recency = 1 THEN kickoff_time END)     AS cur_last_kickoff,
                max(CASE WHEN recency = 1 THEN prev_value END)       AS cur_last_value
            FROM history
            WHERE season = ?
            GROUP BY element
        ),
        prior AS (
            -- The player's most recent *completed* season, excluding its last two gameweeks:
            -- end-of-season minutes reflect rotation and knocks, not next season's role.
            SELECT
                element,
                count(*)                                            AS n_prior,
                avg(minutes)                                        AS prior_minutes,
                avg(CASE WHEN starts > 0 THEN 1.0 ELSE 0.0 END)     AS prior_start_rate,
                avg(CASE WHEN minutes > 0 THEN 1.0 ELSE 0.0 END)    AS prior_played_rate,
                max(kickoff_time)                                   AS prior_last_kickoff,
                max(prev_value)                                     AS prior_last_value
            FROM (
                SELECT h.*,
                       -- dense_rank over *seasons*, not row_number over rows: row_number would
                       -- rank every individual match, so `= 1` would keep a single game and
                       -- `IS NOT NULL` (the original bug) keeps every season ever played. That
                       -- averaged a player's whole career instead of last season, dragging
                       -- established starters toward their earlier, more-rotated years — it put
                       -- Joao Pedro at a 50% chance of playing an hour when he had started 31 of
                       -- 38 and gone 60+ in 30 of them.
                       dense_rank() OVER (PARTITION BY element ORDER BY season DESC) AS season_rank
                FROM history h
                WHERE h.season <> ? AND h.gw_seq <= h.season_last_gw - 2
            )
            WHERE season_rank = 1
            GROUP BY element
        )
        SELECT
            COALESCE(c.element, p.element)                AS element,
            COALESCE(c.n_current, 0)                      AS n_current,
            COALESCE(p.n_prior, 0)                        AS n_prior,
            c.cur_minutes_3, c.cur_minutes_5, c.cur_minutes_10,
            c.cur_start_rate, c.cur_played_rate,
            c.cur_minutes_last, c.cur_minutes_prev, c.cur_started_last,
            c.cur_last_kickoff, c.cur_last_value,
            p.prior_minutes, p.prior_start_rate, p.prior_played_rate,
            p.prior_last_kickoff, p.prior_last_value
        FROM current c
        FULL OUTER JOIN prior p ON p.element = c.element
        """,
        [season, season, season],
    ).fetchdf()

    return _blend_form(frame, lookback=lookback)


def _blend_form(frame: pd.DataFrame, *, lookback: int) -> pd.DataFrame:
    """Blend current-season form with last season's role, weighted by evidence.

    The weight on current-season data rises with the number of matches played, reaching 1 once a
    player has ``lookback`` matches behind him. Before that, last season's season-long involvement
    fills the gap — as an average, never as a single most-recent match.
    """
    out = pd.DataFrame({"element": frame["element"]})
    n_current = frame["n_current"].fillna(0).to_numpy(dtype="float64")
    weight = np.clip(n_current / max(lookback, 1), 0.0, 1.0)

    def blend(current_column: str, prior_column: str) -> np.ndarray:
        current = pd.to_numeric(frame[current_column], errors="coerce").to_numpy(dtype="float64")
        prior = pd.to_numeric(frame[prior_column], errors="coerce").to_numpy(dtype="float64")
        # Where one side is missing, defer entirely to the other rather than to zero.
        current_ok, prior_ok = ~np.isnan(current), ~np.isnan(prior)
        effective = np.where(current_ok & prior_ok, weight, np.where(current_ok, 1.0, 0.0))
        return np.nan_to_num(current, nan=0.0) * effective + np.nan_to_num(
            prior, nan=0.0
        ) * (1 - effective)

    out["roll_minutes_3"] = blend("cur_minutes_3", "prior_minutes")
    out["roll_minutes_5"] = blend("cur_minutes_5", "prior_minutes")
    out["roll_minutes_10"] = blend("cur_minutes_10", "prior_minutes")
    out["roll_start_rate_5"] = blend("cur_start_rate", "prior_start_rate")
    out["roll_played_rate_5"] = blend("cur_played_rate", "prior_played_rate")
    # Across a season break the "last match" features fall back to the prior season *average*,
    # not its final match, which is the correction that stopped rating Haaland a rotation risk.
    out["minutes_last"] = blend("cur_minutes_last", "prior_minutes")
    out["minutes_prev"] = blend("cur_minutes_prev", "prior_minutes")
    out["started_last"] = blend("cur_started_last", "prior_start_rate")

    out["last_kickoff"] = frame["cur_last_kickoff"].fillna(frame["prior_last_kickoff"])
    out["last_value"] = pd.to_numeric(
        frame["cur_last_value"].fillna(frame["prior_last_value"]), errors="coerce"
    )
    out["matches_so_far"] = n_current + frame["n_prior"].fillna(0).to_numpy()
    return out


def build_projection_inputs(
    con,
    models: ProjectionModels,
    gameweeks: list[int],
    *,
    availability: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Build the player-match frame and minutes probabilities for the requested gameweeks.

    Args:
        availability: Optional live ``element``/``status``/``chance_of_playing_next_round`` frame.
            Applied on top of the model's historical-pattern prediction, since per-gameweek
            availability was never recorded historically and so cannot be learned.

    Returns:
        The player-match frame and its aligned ``(n_rows, 3)`` minutes probabilities.
    """
    season = models.season
    players = current_players(con, season)
    fixtures = teams_module.fixture_rates(con, models.strength, season)

    form = player_form(con, season)
    players = players.merge(form, on="element", how="left")

    # Player scoring rates join on the stable code, so a summer transfer keeps his history.
    rate_columns = [
        "code",
        "goal_rate",
        "assist_rate",
        "defcon_rate",
        "save_rate",
        "card_rate",
        "raw_minutes",
        "thin_sample",
    ]
    players = players.merge(models.rates[rate_columns], on="code", how="left")

    # A player with no history at all (a genuine newcomer to the league) gets his position's
    # median rate rather than a zero, which would make the optimiser treat him as worthless.
    for column in ("goal_rate", "assist_rate", "defcon_rate", "save_rate", "card_rate"):
        by_position = players.groupby("position")[column].transform("median")
        players[column] = players[column].fillna(by_position).fillna(0.0)

    player_matches = build_player_matches(players, fixtures, gameweeks)
    if player_matches.empty:
        raise ValueError(f"no fixtures found for gameweeks {gameweeks}")

    features = _projection_features(con, player_matches, season)
    probabilities = models.minutes.predict(features)

    if availability is not None:
        merged = player_matches[["element"]].merge(availability, on="element", how="left")
        probabilities = minutes_module.apply_availability(
            probabilities,
            status=merged.get("status"),
            chance_of_playing=merged.get("chance_of_playing_next_round"),
        )

    # Recalibrate on this season's evidence — preseason minutes and ownership — before the lineup
    # constraint, so the constraint is enforced on the corrected probabilities rather than being
    # undone by them.
    if models.minutes_adjustment is not None:
        features = _adjustment_features(con, player_matches, season)
        probabilities = probabilities.reset_index(drop=True)
        adjusted = models.minutes_adjustment.apply(
            features, probabilities["p_full"].to_numpy(dtype="float64")
        )
        # Keep the three classes coherent: the cameo mass rescales with the room left over.
        remaining = 1.0 - adjusted
        old_remaining = (1.0 - probabilities["p_full"]).clip(lower=1e-9)
        probabilities["p_cameo"] = (probabilities["p_cameo"] / old_remaining) * remaining
        probabilities["p_full"] = adjusted
        probabilities["p_none"] = 1.0 - probabilities["p_full"] - probabilities["p_cameo"]

    # Enforce eleven starters and three substitutes per club per fixture. Applied after the
    # availability adjustment so that a club missing players through injury redistributes those
    # minutes to its remaining squad, which is what actually happens.
    team_match = (
        player_matches["fixture_id"].astype(str) + ":" + player_matches["team_id"].astype(str)
    )
    probabilities = minutes_module.calibrate_to_lineup(
        probabilities.reset_index(drop=True), team_match.reset_index(drop=True)
    )

    matrix = probabilities[list(minutes_module.CLASS_LABELS)].to_numpy(dtype="float64")
    # Guard against any drift from the availability adjustment.
    matrix = np.clip(matrix, 1e-6, 1.0)
    matrix /= matrix.sum(axis=1, keepdims=True)
    return player_matches, matrix


def _adjustment_features(con, player_matches: pd.DataFrame, season: str) -> pd.DataFrame:
    """Preseason and ownership features aligned to the player-match rows."""
    from ..ingest import preseason as preseason_module

    base = player_matches[["element"]].reset_index(drop=True)
    try:
        pre = preseason_module.player_features(con, season)
        base = base.merge(pre, on="element", how="left")
    except Exception as exc:  # pragma: no cover - preseason tables absent
        log.debug("no preseason features available: %s", exc)

    snapshots = _price_snapshots()
    if snapshots is not None and not snapshots.empty:
        latest = (
            snapshots.sort_values("ts")
            .groupby("element", as_index=False)
            .tail(1)[["element", "selected_by_percent"]]
        )
        base = base.merge(latest, on="element", how="left")
        ownership = pd.to_numeric(base["selected_by_percent"], errors="coerce").fillna(0.0)
        base["log_ownership"] = np.log1p(ownership.clip(lower=0))
    else:
        base["log_ownership"] = 0.0

    for column in ("preseason_minutes_avg", "preseason_observed", "preseason_matches"):
        if column not in base.columns:
            base[column] = 0.0
        base[column] = pd.to_numeric(base[column], errors="coerce").fillna(0.0)
    return base


def _projection_features(con, player_matches: pd.DataFrame, season: str) -> pd.DataFrame:
    """Recreate the minutes model's feature columns for unplayed fixtures."""
    frame = player_matches.copy()

    congestion = con.execute(
        """
        SELECT f.fixture_id, t.team_id, (
            SELECT count(*) FROM fixtures g
            WHERE g.season = f.season
              AND (g.team_h = t.team_id OR g.team_a = t.team_id)
              AND g.kickoff_time BETWEEN f.kickoff_time - INTERVAL 7 DAY
                                     AND f.kickoff_time + INTERVAL 7 DAY
        ) AS team_congestion_7d
        FROM fixtures f
        JOIN teams t ON t.season = f.season
                    AND (t.team_id = f.team_h OR t.team_id = f.team_a)
        WHERE f.season = ?
        """,
        [season],
    ).fetchdf()
    frame = frame.merge(congestion, on=["fixture_id", "team_id"], how="left")

    gap = (frame["kickoff_time"] - frame["last_kickoff"]).dt.total_seconds() / 86400.0
    # Beyond the first projected gameweek the true gap is the spacing between fixtures, not the
    # distance back to a match played months ago, which would otherwise look like a long absence.
    frame["gap_days"] = gap.where(gap.between(0, 60), 7.0).fillna(14.0)

    frame["season_progress"] = frame["event"] / 38.0
    price = pd.to_numeric(frame["last_value"], errors="coerce")
    price = price.fillna(pd.to_numeric(frame["start_cost"], errors="coerce"))
    frame["log_price"] = np.log(price.fillna(price.median()) / 10.0)

    position = frame["position"]
    for label, code in (("is_gkp", "GKP"), ("is_def", "DEF"), ("is_mid", "MID"), ("is_fwd", "FWD")):
        frame[label] = (position == code).astype("float64")
    frame["debut_window"] = (frame["matches_so_far"].fillna(0) < 3).astype("float64")

    for column in (
        "roll_minutes_3",
        "roll_minutes_5",
        "roll_minutes_10",
        "minutes_last",
        "minutes_prev",
        "roll_start_rate_5",
        "roll_played_rate_5",
        "started_last",
    ):
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
    frame["team_congestion_7d"] = (
        pd.to_numeric(frame["team_congestion_7d"], errors="coerce").fillna(1.0).clip(1, 5)
    )
    return frame


def project(
    con,
    *,
    models: ProjectionModels | None = None,
    start_gameweek: int | None = None,
    horizon: int = DEFAULT_HORIZON,
    n_draws: int = 10_000,
    availability: pd.DataFrame | None = None,
    seed: int = 20262027,
    store: bool = False,
) -> tuple[SimulationResult, pd.DataFrame, ProjectionModels]:
    """Project the next ``horizon`` gameweeks.

    Args:
        store: Persist the projection so it can be scored once the gameweek is played. The weekly
            advisory path sets this; exploratory calls do not, to avoid filling the table with
            variations that were never acted on.

    Returns:
        The simulation result, the player-match frame it was built from, and the fitted models
        (so callers can reuse them rather than refitting for every scenario).
    """
    models = models or fit_models(con, season=models.season if models else CURRENT_SEASON)
    start = start_gameweek or next_gameweek(con, models.season)
    gameweeks = list(range(start, min(start + horizon, 39)))
    if not gameweeks:
        raise ValueError(f"no gameweeks remain from {start}")

    player_matches, probabilities = build_projection_inputs(
        con, models, gameweeks, availability=availability
    )
    log.info(
        "simulating gameweeks %d-%d: %d player-matches, %d draws",
        gameweeks[0],
        gameweeks[-1],
        len(player_matches),
        n_draws,
    )

    result = simulate(
        player_matches,
        probabilities,
        models.rules,
        models.bps,
        rho=models.strength.rho,
        n_draws=n_draws,
        seed=seed,
    )

    if store:
        try:
            store_projection(
                con,
                result,
                pd.DataFrame(probabilities, columns=list(minutes_module.CLASS_LABELS)),
                player_matches,
                season=models.season,
            )
        except Exception as exc:  # pragma: no cover - never lose a plan over bookkeeping
            log.warning("could not store projection: %s", exc)

    return result, player_matches, models


MODEL_VERSION = "2026-27.2"


def store_projection(
    con,
    result: SimulationResult,
    minutes_probabilities: pd.DataFrame,
    player_matches: pd.DataFrame,
    *,
    season: str = CURRENT_SEASON,
    made_at: dt.datetime | None = None,
    model_version: str = MODEL_VERSION,
) -> int:
    """Persist a projection so it can be scored after the gameweek is played.

    This is what makes the model self-correcting, and its absence was the reason gameweek 1 could
    only be reviewed by reconstructing the prediction by hand. A projection is only evidence about
    the model if it was recorded *before* the deadline, so ``made_at`` is stored and the
    calibration layer reads the latest projection at or before each deadline.
    """
    from ..ingest.warehouse import upsert

    made_at = made_at or dt.datetime.now(dt.UTC)
    # Collapse to one row per (element, gameweek): a double gameweek produces two player-match rows
    # but a single prediction per gameweek is what gets scored.
    probs = minutes_probabilities.reset_index(drop=True)
    frame = player_matches.reset_index(drop=True)[["element", "event"]].copy()
    frame[["p_none", "p_cameo", "p_full"]] = probs[list(minutes_module.CLASS_LABELS)].to_numpy()
    per_gw = frame.groupby(["element", "event"], as_index=False).agg(
        p_none=("p_none", "mean"), p_cameo=("p_cameo", "mean"), p_full=("p_full", "max")
    )

    expected = result.expected_points
    p10, p90 = result.quantile(0.10), result.quantile(0.90)
    per_gw["expected_points"] = [
        float(expected.at[e, g]) if e in expected.index and g in expected.columns else None
        for e, g in zip(per_gw["element"], per_gw["event"], strict=True)
    ]
    per_gw["ep_p10"] = [
        float(p10.at[e, g]) if e in p10.index and g in p10.columns else None
        for e, g in zip(per_gw["element"], per_gw["event"], strict=True)
    ]
    per_gw["ep_p90"] = [
        float(p90.at[e, g]) if e in p90.index and g in p90.columns else None
        for e, g in zip(per_gw["element"], per_gw["event"], strict=True)
    ]

    per_gw = per_gw.rename(columns={"event": "gw"})
    per_gw.insert(0, "season", season)
    per_gw["made_at"] = pd.Timestamp(made_at).tz_localize(None)
    per_gw["model_version"] = model_version

    written = upsert(con, "projections", per_gw, ("season", "gw", "element", "made_at"))
    log.info("stored %d projections for %s GW%s", written, season, sorted(per_gw["gw"].unique()))
    return written


def next_gameweek(con, season: str = CURRENT_SEASON) -> int:
    """The gameweek to plan for: the first whose deadline has not passed."""
    row = con.execute(
        """
        SELECT min(event) FROM events
        WHERE season = ? AND (finished IS NULL OR finished = FALSE)
        """,
        [season],
    ).fetchone()
    if row and row[0] is not None:
        return int(row[0])
    row = con.execute("SELECT min(event) FROM fixtures WHERE season = ?", [season]).fetchone()
    return int(row[0]) if row and row[0] is not None else 1
