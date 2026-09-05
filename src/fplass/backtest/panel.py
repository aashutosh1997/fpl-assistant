"""The projection panel: what the model would have said before every deadline it never saw.

The live ``projections`` table records each week's projection before the deadline so it can be
scored afterwards. That is the right discipline, and it produces one honest data point a week.
Ten completed seasons are sitting in the warehouse, and every one of their deadlines can be
replayed the same way: fit the models on what was known, project the next eight gameweeks, store
the result, move on. The panel is that replay — roughly 350 deadlines and two million
player-gameweek projections — and it is the foundation for two things the live table cannot give:

* **A ten-season backtest of the whole pipeline**, expected points against points, not just the
  minutes model against minutes. Until now the end-to-end accuracy was known for two gameweeks.
* **The price history behind every option value.** How far a projection moves between one
  deadline and the next is the volatility that a banked transfer, a bench place or a held chip is
  worth; it cannot be measured from a single season in progress.

Leakage is the whole difficulty, and every input is cut at the deadline rather than at the season:

=================  ================================================================
team strength      fitted on results that kicked off before the gameweek's first kickoff
minutes model      fitted with the season held out entirely (player-agnostic features)
per-90 rates       aggregated up to the gameweek sequence number, exclusive
bonus weights      fitted on the previous season, the same prior the live model starts from
form features      ``player_form(before_gw)``: current-season rows strictly before the gameweek
player pool        players registered by the gameweek; club and price as of its deadline
order flow         the flow layer fitted with the season excluded, applied to the deadline week
=================  ================================================================

Two things history cannot supply, and the panel says so rather than faking them: there are no
availability flags (``chance_of_playing`` is not recorded), and there is no recalibration layer
(it reads preseason friendlies and hourly ownership snapshots that exist for 2026/27 only). The
order-flow layer in :mod:`fplass.features.flow` is the historical stand-in for the first, and
from ``panel.2`` on it is applied. A
fixture postponed from an earlier gameweek and played later keeps its original label, so its
minutes enter the form features one gameweek early; that is a handful of matches a season and is
left as a known, small optimism.

Workers open the warehouse read-only and write one parquet file per season, because DuckDB has a
single writer and refuses read-only opens while another process holds the file read-write. The
files are loaded into ``projection_panel`` in one pass at the end.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..features import bps as bps_module
from ..features import flow as flow_module
from ..features import minutes as minutes_module
from ..features import rates as rates_module
from ..features import teams as teams_module
from ..ingest.sources import CURRENT_SEASON
from ..ingest.warehouse import connect, upsert
from ..paths import PANEL
from ..scoring import ScoringRules, rules_for_season
from ..sim import project
from ..sim.engine import SimulationResult, simulate

log = logging.getLogger(__name__)

PANEL_VERSION = "panel.2"
DEFAULT_DRAWS = 2_000
DEFAULT_HORIZON = 8
PANEL_KEY = ("season", "as_of_gw", "target_gw", "element")


@dataclass(slots=True)
class SeasonContext:
    """The per-season fits that do not change between deadlines."""

    season: str
    minutes: minutes_module.MinutesModel
    bps: bps_module.BPSModel
    rules: ScoringRules
    gameweeks: list[int]  # gameweek labels with fixtures, in order
    sequence: dict[int, int]  # gameweek label -> dense sequence number
    flow: flow_module.FlowLayer | None = None


def panel_seasons(con) -> list[str]:
    """Completed seasons that can be replayed: everything but the live one and the earliest.

    The earliest season has no results before its first deadline to fit team strength on.
    """
    rows = con.execute(
        "SELECT DISTINCT season FROM player_gw WHERE season <> ? ORDER BY season",
        [CURRENT_SEASON],
    ).fetchall()
    return [r[0] for r in rows][1:]


def season_gameweeks(con, season: str) -> list[int]:
    """Gameweek labels with fixtures, in order. 2019-20 jumps from 29 to 39 across COVID."""
    rows = con.execute(
        "SELECT DISTINCT event FROM fixtures WHERE season = ? AND event IS NOT NULL ORDER BY event",
        [season],
    ).fetchall()
    return [int(r[0]) for r in rows]


def gameweek_sequence(con, season: str) -> dict[int, int]:
    """Label to dense sequence number, the unit the per-90 rates are cut on."""
    rows = con.execute(
        "SELECT DISTINCT gw, gw_seq FROM player_gw_derived WHERE season = ?", [season]
    ).fetchall()
    return {int(g): int(s) for g, s in rows}


def deadline_cutoff(con, season: str, gameweek: int) -> pd.Timestamp | None:
    """The instant before which everything is known: the gameweek's first kickoff.

    Historical seasons have no ``events`` rows, so the deadline is taken as the first kickoff.
    Results are cut strictly before it, which excludes the gameweek itself and anything
    postponed into a later date.
    """
    row = con.execute(
        "SELECT min(kickoff_time) FROM fixtures WHERE season = ? AND event = ?",
        [season, gameweek],
    ).fetchone()
    if not row or row[0] is None:
        return None
    return pd.Timestamp(row[0])


def season_context(con, season: str) -> SeasonContext:
    """Fit the season-level pieces once: minutes model, bonus prior, rules, calendar."""
    features = minutes_module.build_features(con)
    minutes_model, _ = minutes_module.fit(features, holdout_seasons=[season])
    # The order-flow layer stands in for the availability news history never recorded. Fitted
    # with this season excluded, like everything else.
    flow_layer = flow_module.fit_layer(con, features, exclude=[season])

    previous = con.execute(
        "SELECT max(season) FROM player_gw WHERE season < ?", [season]
    ).fetchone()[0]
    bonus = bps_module.fit(con, seasons=[previous or season])

    return SeasonContext(
        season=season,
        minutes=minutes_model,
        bps=bonus,
        rules=rules_for_season(con, season),
        gameweeks=season_gameweeks(con, season),
        sequence=gameweek_sequence(con, season),
        flow=flow_layer,
    )


def models_as_of(con, context: SeasonContext, gameweek: int) -> project.ProjectionModels:
    """Team strength and rates as they could have been fitted before ``gameweek``'s deadline."""
    cutoff = deadline_cutoff(con, context.season, gameweek)
    if cutoff is None:
        raise ValueError(f"{context.season} has no fixtures in gameweek {gameweek}")

    results = teams_module.flag_promoted(teams_module.load_results(con, up_to=cutoff))
    strength = teams_module.fit(results, reference_time=cutoff)

    # The sequence number of a label that has no player rows (never the case for a completed
    # season) would be one past the last played one.
    sequence = context.sequence.get(
        gameweek, 1 + max((s for g, s in context.sequence.items() if g < gameweek), default=0)
    )
    rate_table, _ = rates_module.build(
        con, up_to_season=context.season, up_to_gw_seq=sequence
    )
    return project.ProjectionModels(
        strength=strength,
        minutes=context.minutes,
        rates=rate_table,
        bps=context.bps,
        rules=context.rules,
        season=context.season,
        flow=context.flow,
    )


def summarise(
    result: SimulationResult,
    probabilities: np.ndarray,
    player_matches: pd.DataFrame,
    *,
    season: str,
    as_of_gw: int,
) -> pd.DataFrame:
    """One panel row per (element, target gameweek) from a simulation.

    A double gameweek gives a player two player-match rows; the full-appearance probability
    kept is the larger (the chance he plays an hour in at least the better of them), and the
    fixture count is stored so the row can be read correctly.
    """
    matches = player_matches.reset_index(drop=True)
    frame = matches[["element", "event"]].copy()
    frame["p_full"] = probabilities[:, minutes_module.CLASS_FULL]
    frame["p_cameo"] = probabilities[:, minutes_module.CLASS_CAMEO]
    per_gw = frame.groupby(["element", "event"], as_index=False).agg(
        p_full=("p_full", "max"), p_cameo=("p_cameo", "mean"), n_fixtures=("p_full", "size")
    )

    key = pd.MultiIndex.from_arrays([per_gw["element"], per_gw["event"]])
    per_gw["ep_mean"] = result.expected_points.stack().reindex(key).to_numpy()
    per_gw["ep_p10"] = result.quantile(0.10).stack().reindex(key).to_numpy()
    per_gw["ep_p90"] = result.quantile(0.90).stack().reindex(key).to_numpy()

    per_gw = per_gw.rename(columns={"event": "target_gw"})
    per_gw.insert(0, "as_of_gw", int(as_of_gw))
    per_gw.insert(0, "season", season)
    per_gw["model_version"] = f"{project.MODEL_VERSION}+{PANEL_VERSION}"
    return per_gw


def project_deadline(
    con,
    context: SeasonContext,
    gameweek: int,
    *,
    horizon: int = DEFAULT_HORIZON,
    n_draws: int = DEFAULT_DRAWS,
    seed: int = 20262027,
) -> pd.DataFrame:
    """Project the next ``horizon`` gameweeks as they looked before ``gameweek``'s deadline."""
    targets = [g for g in context.gameweeks if g >= gameweek][:horizon]
    if not targets:
        raise ValueError(f"no gameweeks from {gameweek} in {context.season}")

    models = models_as_of(con, context, gameweek)
    player_matches, probabilities = project.build_projection_inputs(
        con, models, targets, as_of_gameweek=gameweek, historical=True
    )
    result = simulate(
        player_matches,
        probabilities,
        models.rules,
        models.bps,
        rho=models.strength.rho,
        n_draws=n_draws,
        seed=seed + gameweek,
    )
    return summarise(
        result, probabilities, player_matches, season=context.season, as_of_gw=gameweek
    )


def build_season(
    season: str,
    *,
    horizon: int = DEFAULT_HORIZON,
    n_draws: int = DEFAULT_DRAWS,
    gameweeks: list[int] | None = None,
    out_dir: Path = PANEL,
) -> Path:
    """Replay every deadline of one season and write the rows to a parquet file.

    Opens its own read-only connection, so it can run in a worker process alongside others.
    """
    con = connect(read_only=True)
    con.execute("SET threads TO 1")
    try:
        context = season_context(con, season)
        chosen = [g for g in context.gameweeks if gameweeks is None or g in gameweeks]
        frames: list[pd.DataFrame] = []
        started = time.time()
        for gw in chosen:
            tick = time.time()
            frames.append(project_deadline(con, context, gw, horizon=horizon, n_draws=n_draws))
            log.info(
                "%s GW%-2d projected in %4.1fs (%d rows, %.0fs elapsed)",
                season,
                gw,
                time.time() - tick,
                len(frames[-1]),
                time.time() - started,
            )
        panel = pd.concat(frames, ignore_index=True)

        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{season}.{PANEL_VERSION}.parquet"
        # DuckDB writes the file itself, so no parquet library is needed and a read-only
        # database connection is enough.
        con.register("_panel_rows", panel)
        con.execute(f"COPY _panel_rows TO '{path}' (FORMAT PARQUET)")
        con.unregister("_panel_rows")
        log.info("%s: %d panel rows -> %s", season, len(panel), path)
        return path
    finally:
        con.close()



# Thread limits for worker processes. DuckDB and the BLAS behind numpy each open a pool the size
# of the machine by default, so eight workers became well over a hundred runnable threads and
# a replay that should take an hour took a night. Set before the pool spawns; spawned children
# inherit the environment.
WORKER_THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}


def _single_threaded_environment() -> None:
    import os

    for key, value in WORKER_THREAD_ENV.items():
        os.environ.setdefault(key, value)


def _build_season_task(args: tuple) -> str:
    season, horizon, n_draws, gameweeks, out_dir = args
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s"
    )
    logging.getLogger("fplass").setLevel(logging.WARNING)
    log.setLevel(logging.INFO)
    return str(
        build_season(
            season, horizon=horizon, n_draws=n_draws, gameweeks=gameweeks, out_dir=Path(out_dir)
        )
    )


def build_panel(
    seasons: list[str],
    *,
    horizon: int = DEFAULT_HORIZON,
    n_draws: int = DEFAULT_DRAWS,
    workers: int = 1,
    gameweeks: list[int] | None = None,
    out_dir: Path = PANEL,
) -> list[Path]:
    """Replay several seasons, in parallel processes when asked, returning the parquet paths.

    The calling process must not hold the warehouse open read-write while this runs.
    """
    _single_threaded_environment()
    tasks = [(s, horizon, n_draws, gameweeks, str(out_dir)) for s in seasons]
    if workers <= 1 or len(tasks) == 1:
        return [Path(_build_season_task(t)) for t in tasks]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return [Path(p) for p in pool.map(_build_season_task, tasks)]


def load_into_warehouse(con, paths: list[Path]) -> int:
    """Upsert the per-season parquet files into ``projection_panel``."""
    total = 0
    for path in paths:
        frame = con.execute("SELECT * FROM read_parquet(?)", [str(path)]).fetchdf()
        total += upsert(con, "projection_panel", frame, PANEL_KEY)
        log.info("loaded %d rows from %s", len(frame), path)
    return total


def panel_files(version: str, out_dir: Path = PANEL) -> list[Path]:
    """The parquet files of one panel version, oldest season first."""
    return sorted(out_dir.glob(f"*.{version}.parquet"))


def _panel_relation(sources: list[Path] | None) -> str:
    if sources:
        files = ", ".join(f"'{p}'" for p in sources)
        return f"read_parquet([{files}])"
    return "projection_panel"


def scored_panel(
    con, seasons: list[str] | None = None, *, sources: list[Path] | None = None
) -> pd.DataFrame:
    """Panel rows joined to what actually happened, with ``weeks_ahead`` attached.

    Reads the warehouse table, or a panel version's parquet files when ``sources`` is given.
    """
    season_filter = ""
    params: list[object] = []
    if seasons:
        season_filter = f"AND p.season IN ({', '.join('?' for _ in seasons)})"
        params = list(seasons)
    return con.execute(
        f"""
        WITH actual AS (
            SELECT season, element, gw, sum(minutes) AS minutes, sum(total_points) AS total_points
            FROM player_gw GROUP BY season, element, gw
        ),
        seq AS (SELECT DISTINCT season, gw, gw_seq FROM player_gw_derived)
        SELECT p.season, p.as_of_gw, p.target_gw, p.element, p.p_full, p.ep_mean,
               p.ep_p10, p.ep_p90, p.n_fixtures,
               st.gw_seq - sa.gw_seq AS weeks_ahead,
               a.minutes, a.total_points,
               CASE WHEN a.minutes >= 60 THEN 1.0 ELSE 0.0 END AS played_full
        FROM {_panel_relation(sources)} p
        JOIN actual a ON a.season = p.season AND a.element = p.element AND a.gw = p.target_gw
        JOIN seq sa ON sa.season = p.season AND sa.gw = p.as_of_gw
        JOIN seq st ON st.season = p.season AND st.gw = p.target_gw
        WHERE 1 = 1 {season_filter}
        """,
        params,
    ).fetchdf()


def score_panel(
    con, seasons: list[str] | None = None, *, sources: list[Path] | None = None
) -> pd.DataFrame:
    """Accuracy per season and weeks ahead: the ten-season backtest of the whole pipeline.

    ``spearman`` and ``top30_lift`` are computed within each deadline and averaged, so a season
    with more gameweeks does not weigh more, and lift is against the gameweek's own mean.
    """
    frame = scored_panel(con, seasons, sources=sources)
    if frame.empty:
        return frame

    def per_deadline(group: pd.DataFrame) -> pd.Series:
        top = group.nlargest(30, "ep_mean")["total_points"].mean()
        mean = group["total_points"].mean()
        return pd.Series(
            {
                "spearman": group["ep_mean"].corr(group["total_points"], method="spearman"),
                "top30_lift": top / mean if mean > 0 else np.nan,
            }
        )

    by_deadline = (
        frame.groupby(["season", "weeks_ahead", "as_of_gw"])
        .apply(per_deadline, include_groups=False)
        .reset_index()
    )
    ranking = by_deadline.groupby(["season", "weeks_ahead"])[["spearman", "top30_lift"]].mean()

    frame["sq_error"] = (frame["p_full"] - frame["played_full"]) ** 2
    frame["base_sq_error"] = (
        frame.groupby(["season", "weeks_ahead"])["played_full"].transform("mean")
        - frame["played_full"]
    ) ** 2
    calibration = frame.groupby(["season", "weeks_ahead"]).agg(
        n=("element", "size"),
        ep_mean=("ep_mean", "mean"),
        points_mean=("total_points", "mean"),
        minutes_brier=("sq_error", "mean"),
        minutes_brier_base=("base_sq_error", "mean"),
    )
    calibration["minutes_skill"] = 1 - calibration["minutes_brier"] / calibration["minutes_brier_base"]
    calibration["ep_bias"] = calibration["points_mean"] - calibration["ep_mean"]

    out = calibration.join(ranking).reset_index()
    return out[
        [
            "season",
            "weeks_ahead",
            "n",
            "spearman",
            "top30_lift",
            "minutes_brier",
            "minutes_skill",
            "ep_mean",
            "points_mean",
            "ep_bias",
        ]
    ].round(3)
