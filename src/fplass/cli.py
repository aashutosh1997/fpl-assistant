"""Command line interface.

    fpl prices snapshot        one price/availability snapshot -> monthly CSV
    fpl prices status          what the price fields currently say
    fpl ingest history         load historical seasons into the warehouse
    fpl ingest current         refresh this season from the live API
"""

from __future__ import annotations

import logging

import typer

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Fantasy Premier League 2026/27 expert.",
)
prices_app = typer.Typer(no_args_is_help=True, help="Price tracking and price-rise optimisation.")
ingest_app = typer.Typer(no_args_is_help=True, help="Load data into the warehouse.")
backtest_app = typer.Typer(
    no_args_is_help=True, help="Ten-season replays: the projection panel and the paper manager."
)
app.add_typer(prices_app, name="prices")
app.add_typer(ingest_app, name="ingest")
study_app = typer.Typer(
    no_args_is_help=True, help="The top managers' decisions against the model's."
)
rl_app = typer.Typer(
    no_args_is_help=True,
    help="A learned policy for transfer counts and chips, judged by the nine-season replay.",
)
app.add_typer(backtest_app, name="backtest")
app.add_typer(study_app, name="study")
app.add_typer(rl_app, name="rl")


@app.callback()
def main(verbose: bool = typer.Option(False, "--verbose", "-v", help="Debug logging.")) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # httpx logs every request at INFO, which drowns our own output in cron logs.
    logging.getLogger("httpx").setLevel(logging.WARNING)


@prices_app.command("snapshot")
def prices_snapshot() -> None:
    """Append one row per player to this month's price snapshot CSV."""
    from .prices.snapshot import take_snapshot

    path, n = take_snapshot()
    typer.echo(f"appended {n} rows -> {path}")


@prices_app.command("status")
def prices_status(
    limit: int = typer.Option(20, help="How many players to show."),
    owned_only: bool = typer.Option(False, help="Restrict to your squad (needs a configured team)."),
) -> None:
    """Show what FPL's price-change fields currently predict."""
    from .api import FPLClient
    from .prices.snapshot import snapshot_metadata

    with FPLClient() as client:
        bootstrap = client.bootstrap(ttl=0)

    meta = snapshot_metadata(bootstrap)
    typer.echo(f"next price changes at: {', '.join(meta['price_change_deadlines']) or 'unknown'}")

    def move_score(e: dict) -> float:
        projections = e.get("price_change_projections") or []
        if not projections:
            return 0.0
        first = projections[0]
        try:
            return float(first.get("projected_percent") or 0) * float(first.get("likelihood") or 0)
        except (TypeError, ValueError):
            return 0.0

    elements = sorted(bootstrap["elements"], key=move_score)
    interesting = elements[:limit] + elements[-limit:]
    if all(move_score(e) == 0 for e in interesting):
        typer.echo(
            "All price-change projections are currently zero "
            "(pre-season, or FPL's model is still calibrating)."
        )
        return

    if owned_only:
        typer.echo("--owned-only needs a configured team; not yet wired up.")

    for e in interesting:
        typer.echo(
            f"{e['web_name']:<20} {e['now_cost'] / 10:>5.1f}  "
            f"pct={e.get('price_change_percent'):>7}  "
            f"score={move_score(e):+.3f}"
        )


@app.command("plan")
def plan_command(
    entry: int = typer.Option(None, help="Your FPL team (entry) id."),
    league: int = typer.Option(None, help="Mini-league id to optimise for."),
    gameweek: int = typer.Option(None, help="Gameweek to plan (default: the next one)."),
    horizon: int = typer.Option(8, help="Gameweeks to plan ahead."),
    draws: int = typer.Option(10_000, help="Monte Carlo draws."),
    objective: str = typer.Option(
        "league", help="league | points | blend — what to optimise for."
    ),
    solver_seconds: int = typer.Option(120, help="Per-solve time limit."),
    config: str = typer.Option(
        None, help="Planner knobs to override, e.g. 'banked_transfer_value=2.0,chip_rule=continuation'."
    ),
) -> None:
    """Recommend this week's transfers, captain and chip strategy."""
    from .advise import advise, format_report
    from .ingest.warehouse import connect
    from .optimise.policy import PolicyConfig

    con = connect()
    try:
        recommendation = advise(
            con,
            entry_id=entry,
            league_id=league,
            gameweek=gameweek,
            horizon=horizon,
            n_draws=draws,
            objective=objective,
            solver_time_limit=solver_seconds,
            config=PolicyConfig.parse(config),
        )
        typer.echo(format_report(recommendation))
    finally:
        con.close()


@app.command("squad")
def squad_command(
    budget: float = typer.Option(100.0, help="Budget in millions."),
    horizon: int = typer.Option(8, help="Gameweeks to optimise over."),
    draws: int = typer.Option(10_000, help="Monte Carlo draws."),
    solver_seconds: int = typer.Option(180, help="Solver time limit."),
) -> None:
    """Build an optimal squad from scratch — for a wildcard, or the start of a season."""
    from .ingest.warehouse import connect
    from .optimise import milp
    from .sim import project

    con = connect()
    try:
        models = project.fit_models(con)
        result, _, _ = project.project(con, models=models, horizon=horizon, n_draws=draws)
        players = project.live_player_table(con)
        state = milp.SquadState(players={}, bank=int(round(budget * 10)), free_transfers=15)
        windows = milp.ChipWindows.from_warehouse(con, "2026-27")
        plan = milp.solve(
            result.expected_points,
            players,
            state,
            windows,
            allow_chips=False,
            time_limit=solver_seconds,
        )

        gw = min(plan.squads)
        info = players.set_index("element")
        totals = result.expected_points.sum(axis=1)
        last = max(plan.squads)
        typer.echo(f"Optimal squad for GW{gw}-{last} (status {plan.status})")
        # The XI marker is for GW1 only, while the total spans the horizon, so both are labelled.
        # A player can rightly be benched in GW1 and start every week after.
        typer.echo(f"{'':<12}{'':<20}{'':<6}{'':<6}  GW{gw}   GW{gw}-{last}   starts\n")
        for position in ("GKP", "DEF", "MID", "FWD"):
            for element in plan.squads[gw]:
                row = info.loc[element]
                if row["position"] != position:
                    continue
                marker = "XI   " if element in plan.lineups[gw] else "bench"
                starts = sum(1 for g in plan.lineups if element in plan.lineups[g])
                typer.echo(
                    f"  {marker} {position} {row['web_name']:<18} {row['team']:<4} "
                    f"{row['price'] / 10:>4.1f}  {result.expected_points.at[element, gw]:>5.2f}  "
                    f"{totals.get(element, 0):>7.1f}  {starts:>5}/{len(plan.lineups)}"
                )
        cost = sum(int(info.loc[e, "price"]) for e in plan.squads[gw])
        typer.echo(f"\n  cost {cost / 10:.1f}m of {budget:.1f}m")
        typer.echo(f"  captain {info.loc[plan.captains[gw], 'web_name']}")
    finally:
        con.close()


@ingest_app.command("history")
def ingest_history(
    seasons: str = typer.Option("all", help="Comma-separated seasons, or 'all'."),
    rebuild: bool = typer.Option(False, help="Drop and recreate tables first."),
) -> None:
    """Load historical seasons from the vaastav dataset into DuckDB."""
    from .ingest.history import load_seasons

    which = None if seasons == "all" else [s.strip() for s in seasons.split(",")]
    summary = load_seasons(seasons=which, rebuild=rebuild)
    for table, count in sorted(summary.items()):
        typer.echo(f"{table:<24} {count:>9,} rows")


@ingest_app.command("current")
def ingest_current() -> None:
    """Refresh the current season's players, teams, fixtures and events from the live API."""
    from .ingest.current import refresh_current

    summary = refresh_current()
    for table, count in sorted(summary.items()):
        typer.echo(f"{table:<24} {count:>9,} rows")


@ingest_app.command("preseason")
def ingest_preseason(
    season: str = typer.Option("2026-27", help="Season to load friendlies for."),
    refresh: bool = typer.Option(False, help="Re-download rather than using the cache."),
) -> None:
    """Load preseason friendlies — the strongest available predictor of gameweek 1 minutes."""
    from .ingest import preseason
    from .ingest.warehouse import connect

    con = connect()
    try:
        summary = preseason.load(con, season, refresh=refresh)
        if not summary:
            typer.echo(f"no preseason friendlies published for {season}")
            return
        for table, count in sorted(summary.items()):
            typer.echo(f"{table:<24} {count:>9,} rows")
    finally:
        con.close()


@ingest_app.command("gameweek")
def ingest_gameweek(
    gameweek: int = typer.Argument(..., help="Gameweek to load results for."),
    season: str = typer.Option("2026-27", help="Season."),
) -> None:
    """Load a played gameweek's results from the live API. Safe to re-run as bonus settles."""
    from .ingest.current import ingest_live_gameweek
    from .ingest.warehouse import connect

    con = connect()
    try:
        typer.echo(f"{ingest_live_gameweek(con, gameweek, season=season):,} player rows")
    finally:
        con.close()


@ingest_app.command("market")
def ingest_market(season: str = typer.Option("2026-27", help="Season.")) -> None:
    """Fill this season's prices, ownership and transfers per gameweek (one call a player)."""
    from .ingest.current import ingest_market_history
    from .ingest.warehouse import connect

    con = connect()
    try:
        typer.echo(f"{ingest_market_history(con, season=season):,} player rows updated")
    finally:
        con.close()


@app.command("calibrate")
def calibrate_command(
    season: str = typer.Option("2026-27", help="Season to score."),
    upto: int = typer.Option(99, help="Score gameweeks before this one."),
) -> None:
    """Score stored projections against what actually happened.

    This is what makes the model self-correcting: it reports where the projections were wrong and
    refits the recalibration layer on the season being played.
    """
    from .backtest import calibrate_live
    from .ingest.warehouse import connect

    con = connect()
    try:
        report = calibrate_live.run(con, season=season, upto_gw=upto)
        if report is None:
            typer.echo(
                "No stored projections to score. Projections are recorded when you run "
                "`fpl plan`, so this fills in from the first gameweek you plan for."
            )
            return
        typer.echo(report.summary())
    finally:
        con.close()


def _parse_gameweeks(spec: str | None) -> list[int] | None:
    """'1-5,20' -> [1, 2, 3, 4, 5, 20]; None or 'all' -> None (every gameweek)."""
    if spec is None or spec.strip().lower() == "all":
        return None
    chosen: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            low, high = part.split("-", 1)
            chosen.extend(range(int(low), int(high) + 1))
        elif part:
            chosen.append(int(part))
    return chosen


@backtest_app.command("panel")
def backtest_panel(
    seasons: str = typer.Option("all", help="Comma-separated seasons, or 'all' completed ones."),
    horizon: int = typer.Option(8, help="Gameweeks projected from each deadline."),
    draws: int = typer.Option(2000, help="Monte Carlo draws per deadline."),
    workers: int = typer.Option(1, help="Seasons replayed in parallel processes."),
    gameweeks: str = typer.Option(None, help="Deadlines to replay, e.g. '1-38' or '20,21'."),
) -> None:
    """Replay every deadline of completed seasons and store the as-of projections.

    This is the ten-season backtest of the whole pipeline and the price history the option
    values are measured from. Roughly twenty seconds a deadline; use --workers to parallelise.
    """
    from .backtest import panel as panel_module
    from .ingest.warehouse import connect

    # Read the season list and close again before the workers start: DuckDB refuses a
    # read-only open while another process holds the file read-write.
    con = connect(read_only=True)
    try:
        chosen = (
            panel_module.panel_seasons(con)
            if seasons == "all"
            else [s.strip() for s in seasons.split(",")]
        )
    finally:
        con.close()

    paths = panel_module.build_panel(
        chosen,
        horizon=horizon,
        n_draws=draws,
        workers=workers,
        gameweeks=_parse_gameweeks(gameweeks),
    )
    con = connect()
    try:
        loaded = panel_module.load_into_warehouse(con, paths)
    finally:
        con.close()
    typer.echo(f"loaded {loaded:,} panel rows from {len(paths)} season(s)")


@backtest_app.command("components")
def backtest_components(
    seasons: str = typer.Option("all", help="Comma-separated seasons, or 'all' completed ones."),
    every: int = typer.Option(4, help="Replay every n-th deadline of each season."),
    draws: int = typer.Option(2000, help="Monte Carlo draws per deadline."),
    workers: int = typer.Option(1, help="Seasons replayed in parallel processes."),
) -> None:
    """The engine against history, scoring component by component and position by position.

    Totals hide offsetting errors; this is the check that finds them. Writes the player rows
    to data/backtest/components_<model version>.csv and prints the comparison.
    """
    from .backtest import components as components_module
    from .backtest import panel as panel_module
    from .ingest.warehouse import connect
    from .paths import BACKTEST
    from .sim.project import MODEL_VERSION

    con = connect(read_only=True)
    try:
        chosen = (
            panel_module.panel_seasons(con)
            if seasons == "all"
            else [s.strip() for s in seasons.split(",")]
        )
    finally:
        con.close()

    frame = components_module.measure(chosen, every=every, n_draws=draws, workers=workers)
    BACKTEST.mkdir(parents=True, exist_ok=True)
    path = BACKTEST / f"components_{MODEL_VERSION}.csv"
    frame.to_csv(path, index=False)
    typer.echo(components_module.report(frame))
    typer.echo(f"rows: {path}")


@backtest_app.command("score")
def backtest_score(
    seasons: str = typer.Option("all", help="Comma-separated seasons, or 'all'."),
    weeks_ahead: int = typer.Option(0, help="Horizon to report; -1 for every horizon."),
    panel_version: str = typer.Option(
        None, help="Score a panel version's parquet files (e.g. panel.1) instead of the warehouse."
    ),
) -> None:
    """Score the projection panel against what happened, per season and horizon."""
    from .backtest import panel as panel_module
    from .ingest.warehouse import connect

    sources = panel_module.panel_files(panel_version) if panel_version else None
    if panel_version and not sources:
        typer.echo(f"No parquet files for panel version {panel_version}.")
        return
    con = connect(read_only=True)
    try:
        chosen = None if seasons == "all" else [s.strip() for s in seasons.split(",")]
        table = panel_module.score_panel(con, chosen, sources=sources)
    finally:
        con.close()
    if table.empty:
        typer.echo("No panel rows stored. Run `fpl backtest panel` first.")
        return
    if weeks_ahead >= 0:
        table = table[table["weeks_ahead"] == weeks_ahead]
    typer.echo(table.to_string(index=False))


@backtest_app.command("manager")
def backtest_manager(
    seasons: str = typer.Option("all", help="Comma-separated seasons, or 'all' with panel rows."),
    policy: str = typer.Option("current", help="current | hold — the planner to replay."),
    workers: int = typer.Option(1, help="Seasons replayed in parallel processes."),
    config: str = typer.Option(
        None,
        help="Planner knobs to change, e.g. 'bench_weight=0.3,terminal_beta=0.5,floor:3xc=8'.",
    ),
    max_gameweek: int = typer.Option(None, help="Stop after this gameweek (for quick checks)."),
    panel_version: str = typer.Option(
        None, help="Replay a panel version's parquet files (e.g. panel.1) instead of the warehouse."
    ),
) -> None:
    """Play completed seasons with the planner and score them in points.

    The number every policy change is judged by. Needs the projection panel; roughly half a
    minute a gameweek with the default solver limit.
    """
    from .backtest import manager as manager_module
    from .ingest.warehouse import connect

    if policy not in manager_module.POLICIES:
        raise typer.BadParameter(f"policy must be one of {manager_module.POLICIES}")

    if seasons != "all":
        chosen = [s.strip() for s in seasons.split(",")]
    elif panel_version:
        chosen = sorted(
            p.name.split(".")[0] for p in manager_module.PANEL.glob(f"*.{panel_version}.parquet")
        )
    else:
        con = connect(read_only=True)
        try:
            chosen = [
                r[0]
                for r in con.execute(
                    "SELECT DISTINCT season FROM projection_panel ORDER BY season"
                ).fetchall()
            ]
        finally:
            con.close()
    if not chosen:
        typer.echo("No panel rows stored. Run `fpl backtest panel` first.")
        return

    settings = manager_module.PolicyConfig.parse(config)
    table = manager_module.replay_seasons(
        chosen,
        policy=policy,
        config=settings,
        max_gameweek=max_gameweek,
        workers=workers,
        panel_version=panel_version,
    )
    typer.echo(table.to_string(index=False))
    typer.echo(
        f"\n{policy} [{settings.tag()}]: {table['points'].mean():.0f} points a season "
        f"over {len(table)} season(s)"
    )


@backtest_app.command("summary")
def backtest_summary(
    policy: str = typer.Option("current", help="Whose traces to summarise."),
    config: str = typer.Option(None, help="The traces' planner config, as given to `manager`."),
    panel_version: str = typer.Option(None, help="The panel version the traces were played on."),
) -> None:
    """Season totals rebuilt from a run's trace files."""
    from .backtest import manager as manager_module

    settings = manager_module.PolicyConfig.parse(config)
    table = manager_module.summaries(policy, settings, panel_version)
    if table.empty:
        typer.echo("No traces found.")
        return
    typer.echo(table.to_string(index=False))
    typer.echo(f"\n{policy} [{settings.tag()}]: {table['points'].mean():.0f} points a season over {len(table)} season(s)")


@backtest_app.command("options")
def backtest_options(
    policy: str = typer.Option("current", help="Whose traces to measure on."),
    config: str = typer.Option(None, help="The traces' planner config, as given to `manager`."),
    panel_version: str = typer.Option(None, help="The panel version the traces were played on."),
) -> None:
    """Measure the transfer, bank and bench option values over the paper manager's weeks."""
    from .backtest import manager as manager_module
    from .backtest import panel as panel_module
    from .ingest.warehouse import connect
    from .options import value as value_module

    settings = manager_module.PolicyConfig.parse(config)
    tag = manager_module.run_tag(policy, settings, panel_version)
    traces = sorted(manager_module.BACKTEST.glob(f"manager_{tag}_20*.csv"))
    if not traces:
        typer.echo(f"No traces for {tag}. Run `fpl backtest manager` first.")
        return
    sources = panel_module.panel_files(panel_version) if panel_version else None
    con = connect(read_only=True)
    try:
        weeks = value_module.measure_from_traces(con, traces, panel_sources=sources)
    finally:
        con.close()
    bench, per_season = value_module.bench_weight_from_traces(traces)
    typer.echo("BENCH  substitutes' points as a share of what the plan expected of the bench")
    typer.echo(per_season.to_string(index=False))
    typer.echo(f"  bench_weight = {bench:.3f}\n")
    typer.echo("TRANSFERS AND MONEY  swap gains available at the next deadline, per week")
    typer.echo(
        weeks.groupby("season")[
            ["gain_1", "gain_2", "transfer_value", "bank_tenth", "bank_half", "bank_million"]
        ].mean().round(3).to_string()
    )
    typer.echo(
        f"  banked_transfer_value = {weeks['transfer_value'].mean():.3f} pts a week "
        f"(second-best swap {weeks['gain_2'].mean():.2f} pts; {len(weeks)} weeks)"
    )
    typer.echo(
        f"  bank: 0.1m = {weeks['bank_tenth'].mean():.3f}, 0.5m = {weeks['bank_half'].mean():.3f}, "
        f"1.0m = {weeks['bank_million'].mean():.3f} pts"
    )


@backtest_app.command("chips")
def backtest_chips(
    policy: str = typer.Option("current", help="Whose traces to measure on."),
    config: str = typer.Option(None, help="The traces' planner config, as given to `manager`."),
    panel_version: str = typer.Option(None, help="The panel version the traces were played on."),
) -> None:
    """Chip timing: what each chip was worth each week, and how the rules compare."""
    from .backtest import manager as manager_module
    from .backtest import panel as panel_module
    from .ingest.sources import CURRENT_SEASON
    from .ingest.warehouse import connect
    from .optimise import milp
    from .options import chips as chips_module

    settings = manager_module.PolicyConfig.parse(config)
    tag = manager_module.run_tag(policy, settings, panel_version)
    traces = sorted(manager_module.BACKTEST.glob(f"manager_{tag}_20*.csv"))
    if not traces:
        typer.echo(f"No traces for {tag}. Run `fpl backtest manager` first.")
        return
    sources = panel_module.panel_files(panel_version) if panel_version else None
    con = connect(read_only=True)
    try:
        gains = chips_module.chip_gains_from_traces(con, traces, panel_sources=sources)
        windows = milp.ChipWindows.from_warehouse(con, CURRENT_SEASON)
    finally:
        con.close()
    gains.to_csv(manager_module.BACKTEST / f"chip_gains_{tag}.csv", index=False)
    typer.echo(chips_module.report(gains, windows))


@backtest_app.command("revisions")
def backtest_revisions(
    panel_version: str = typer.Option(None, help="Read a panel version's files, else the warehouse."),
) -> None:
    """How far projections move between deadlines, by position and certainty."""
    from .backtest import panel as panel_module
    from .ingest.warehouse import connect
    from .options import revisions as revisions_module

    sources = panel_module.panel_files(panel_version) if panel_version else None
    con = connect(read_only=True)
    try:
        table = revisions_module.revisions(con, sources=sources)
    finally:
        con.close()
    if table.empty:
        typer.echo("No panel rows to measure revisions on.")
        return
    typer.echo(revisions_module.summarise(table).to_string(index=False))
    jumps = table[table["p_full_before"] >= revisions_module.JUMP_FROM]
    typer.echo(
        f"\n{len(table):,} player-weeks; a likely starter collapses to a likely absentee by the "
        f"next deadline in {jumps['jump'].mean():.2%} of weeks"
    )




def _me() -> dict:
    """The advised team from config/me.json, or an empty dict."""
    import json

    from .paths import CONFIG

    path = CONFIG / "me.json"
    return json.loads(path.read_text()) if path.exists() else {}


@study_app.command("crawl")
def study_crawl(
    top: int = typer.Option(10_000, help="Read the overall standings down to this rank in full."),
    band: bool = typer.Option(True, help="Also sample standings pages below the top."),
    band_to: int = typer.Option(250_000, help="Deepest rank the sampled band reaches."),
    band_every: int = typer.Option(4, help="Read every n-th standings page in the band."),
    prior_rank: int = typer.Option(
        50_000, help="Band managers whose decisions are fetched: last season inside this rank."
    ),
    rate: float = typer.Option(10.0, help="Requests a second, across all threads."),
    workers: int = typer.Option(8, help="Concurrent requests."),
    stages: str = typer.Option(
        "standings,history,transfers,picks", help="Comma-separated stages to run."
    ),
    incremental: bool = typer.Option(
        False, help="Weekly refresh: top pages, and only the studied managers' histories."
    ),
) -> None:
    """Collect the top managers' histories, transfers and squads into data/warehouse/managers.duckdb.

    Resumable: each stage skips what it already has for the latest finished gameweek. A full
    crawl is about 100,000 requests; run it detached. Re-run weekly with --incremental (about
    40,000 requests) after the gameweek ingest.
    """
    from .study.crawl import CrawlConfig, crawl

    me = _me()
    config = CrawlConfig(
        top=top,
        band=band,
        band_to=band_to,
        band_every=band_every,
        prior_rank=prior_rank,
        leagues=tuple(league["id"] for league in me.get("leagues", [])),
        me=me.get("entry_id"),
        rate=rate,
        workers=workers,
        stages=tuple(s.strip() for s in stages.split(",") if s.strip()),
        incremental=incremental,
    )
    summary = crawl(config)
    for key, value in summary.items():
        typer.echo(f"{key:<20} {value}")


def _study_connection(managers_db: str | None):
    """The warehouse read-only, with the managers database attached beside it."""
    from .ingest.warehouse import connect
    from .study.store import MANAGERS_DB, attach_managers

    con = connect(read_only=True)
    attach_managers(con, managers_db or MANAGERS_DB)
    return con


@study_app.command("panel")
def study_panel(
    gameweeks: str = typer.Option("1-38", help="Deadlines to replay, e.g. '1-10'."),
    draws: int = typer.Option(2000, help="Monte Carlo draws per deadline."),
    season: str = typer.Option("2026-27", help="Season."),
) -> None:
    """Replay this season's deadlines with the current model, into data/panel/study/.

    Kept out of projection_panel, which the live option values and `backtest manager` read.
    Deadlines not yet reached are skipped. Run `fpl ingest market` first so the replay has
    this season's prices and order flow.
    """
    from .ingest.warehouse import connect
    from .study.missed import build_study_panel

    con = connect(read_only=True)
    try:
        upcoming = con.execute(
            "SELECT min(event) FROM events WHERE season = ? AND NOT finished", [season]
        ).fetchone()[0]
    finally:
        con.close()
    chosen = [g for g in (_parse_gameweeks(gameweeks) or []) if upcoming is None or g <= upcoming]
    typer.echo(build_study_panel(season, chosen, n_draws=draws))


@study_app.command("report")
def study_report(
    season: str = typer.Option("2026-27", help="Season."),
    managers_db: str = typer.Option(None, help="A managers database other than the default."),
) -> None:
    """What the top managers transferred, whether the model backed it, and what you missed."""
    from .study import report as report_module

    con = _study_connection(managers_db)
    try:
        typer.echo(report_module.build(con, season, _me().get("entry_id")))
    finally:
        con.close()


@study_app.command("signal")
def study_signal(
    season: str = typer.Option("2026-27", help="Season."),
    managers_db: str = typer.Option(None, help="A managers database other than the default."),
) -> None:
    """Whether the top managers' transfers predict the points and minutes the model misses."""
    from .study import report as report_module

    con = _study_connection(managers_db)
    try:
        typer.echo(report_module.build_signal(con, season, _me().get("entry_id")))
    finally:
        con.close()


@study_app.command("export")
def study_export(
    season: str = typer.Option("2026-27", help="Season."),
    raw: bool = typer.Option(True, help="Also export the raw tables as gzipped CSV."),
    managers_db: str = typer.Option(None, help="A managers database other than the default."),
) -> None:
    """Write cohort flows and the raw decisions under data/snapshots, the copy that outlasts FPL's."""
    from .study import export as export_module

    con = _study_connection(managers_db)
    try:
        last = con.execute("SELECT max(gw) FROM m.manager_gw WHERE season = ?", [season]).fetchone()[0]
        typer.echo(export_module.export_aggregates(con, season, last))
        if raw:
            for path in export_module.export_raw(con, season):
                typer.echo(path)
    finally:
        con.close()


# ------------------------------------------------------------------ reinforcement learning


def _rl_seasons(seasons: str | None) -> list[str]:
    from .rl.season import REPLAY_SEASONS

    return [s.strip() for s in seasons.split(",")] if seasons else list(REPLAY_SEASONS)


@rl_app.command("cache")
def rl_cache(
    seasons: str = typer.Option(None, help="Comma-separated seasons (default: the nine replayed)."),
    refresh: bool = typer.Option(False, help="Rebuild even if cached."),
) -> None:
    """Read each season's deadlines, prices, outcomes and opening squad into the RL cache."""
    from .ingest.warehouse import connect
    from .rl import season as season_module

    con = connect(read_only=True)
    try:
        for season in _rl_seasons(seasons):
            data = season_module.load(con, season, refresh=refresh)
            typer.echo(f"{season}: {len(data.deadlines)} deadlines, opening squad bank {data.start_bank / 10:.1f}m")
    finally:
        con.close()


@rl_app.command("elite")
def rl_elite(
    managers: int = typer.Option(3000, help="Managers sampled from the fair cohorts."),
    workers: int = typer.Option(6, help="Worker processes."),
    season: str = typer.Option("2026-27", help="Season of the crawl."),
    managers_db: str = typer.Option(None, help="A managers database other than the default."),
) -> None:
    """Clone the top managers' roll, hit and chip decisions into a policy."""
    from .rl import elite

    con = _study_connection(managers_db)
    try:
        states = elite.build_states(con, season, managers=managers, workers=workers)
    finally:
        con.close()
    _, report = elite.fit(states)
    typer.echo(
        f"{int(report['states']):,} states. GW{int(report['held_out_gw'])} predicted from the earlier "
        f"deadlines: {report['accuracy']:.1%} of choices (always the commonest: {report['majority']:.1%})"
    )
    shares = elite.describe(states)
    typer.echo("Choices by deadline (transfers, +chip), weighted:")
    typer.echo((shares[[c for c in shares.columns if shares[c].max() >= 0.02]] * 100).round(1).to_string())


@rl_app.command("collect")
def rl_collect(
    rule: int = typer.Option(60, help="Seasons per season, rule with random thresholds, 10% random."),
    rule30: int = typer.Option(40, help="... with 30% random actions."),
    elite_seasons: int = typer.Option(30, "--elite", help="... the cloned top managers, 15% random."),
    random: int = typer.Option(10, help="... uniformly random legal actions."),
    seasons: str = typer.Option(None, help="Comma-separated seasons."),
    workers: int = typer.Option(6, help="Worker processes."),
    seed: int = typer.Option(0, help="Random seed."),
) -> None:
    """Play seasons in the fast environment and store every deadline as a transition."""
    from .rl import elite, fqi

    plan = {"rule": rule, "rule30": rule30, "random": random}
    models = {}
    if elite_seasons and elite.MODEL.exists():
        plan["model:elite"] = elite_seasons
        models["elite"] = elite.MODEL
    data = fqi.collect(_rl_seasons(seasons), {k: v for k, v in plan.items() if v}, models=models,
                       workers=workers, seed=seed)
    path = fqi.TRANSITIONS / "base.npz"
    data.save(path)
    typer.echo(f"{len(data):,} transitions -> {path}")


def _rl_table(totals, order: list[str]) -> str:
    table = totals.pivot(index="season", columns="policy", values="points")
    table = table[[c for c in order if c in table.columns]]
    lines = [table.to_string(), "", "mean  " + "  ".join(f"{c}={table[c].mean():,.0f}" for c in table.columns)]
    return "\n".join(lines)


@rl_app.command("train")
def rl_train(
    reward: str = typer.Option("actual", help="'actual' points or the projection's 'expected'."),
    iterations: int = typer.Option(8, help="Fitted Q-iterations per fit."),
    rounds: int = typer.Option(1, help="Refinement rounds with each fold's own policy."),
    refine: int = typer.Option(12, help="Seasons per training season in each refinement round."),
    seasons: str = typer.Option(None, help="Comma-separated seasons."),
    workers: int = typer.Option(6, help="Worker processes."),
) -> None:
    """Fit one policy per held-out season and play each on the season it never saw."""
    from .rl import elite, fqi

    names = _rl_seasons(seasons)
    base = fqi.Transitions.load(fqi.TRANSITIONS / "base.npz")
    models = {"elite": elite.MODEL} if elite.MODEL.exists() else {}
    paths = fqi.train_folds(base, names, reward=reward, iterations=iterations, rounds=rounds,
                            refine_episodes=refine, workers=workers, models=models)
    name = f"rl-{reward}"
    entries = [(s, name, paths[s]) for s in names] + [(s, "rule", None) for s in names]
    if elite.MODEL.exists():
        entries += [(s, "elite", elite.MODEL) for s in names]
    traces = fqi.evaluate(entries, workers=workers)
    out = fqi.MODELS / reward / "evaluation.csv"
    traces.to_csv(out, index=False)
    totals = fqi.season_totals(traces)
    typer.echo("Fast environment, each season played by a policy that never saw it (points):")
    typer.echo(_rl_table(totals, [name, "rule", "elite"]))
    for against in ("rule", "elite"):
        row = fqi.paired(totals, name, against)
        if row:
            typer.echo(f"{name} vs {against}: {row['diff']:+.0f} a season (SE {row['se']:.0f}), "
                       f"up in {row['up']} of {row['seasons']}")
    typer.echo(totals[totals["policy"] == name][["season", "hits", "transfers", "chips"]].to_string(index=False))
    typer.echo(f"Traces: {out}")


@rl_app.command("replay")
def rl_replay(
    policy: str = typer.Option("rl-actual", help="'rl-actual', 'rl-expected', 'rule' or 'elite'."),
    seasons: str = typer.Option(None, help="Comma-separated seasons."),
    workers: int = typer.Option(6, help="Worker processes."),
) -> None:
    """The policy's choices executed by the planner's solver, paired against the planner."""
    from .rl import elite, fqi, replay

    names = _rl_seasons(seasons)
    if policy.startswith("rl-"):
        reward = policy.split("-", 1)[1]
        specs = {s: (policy, str(fqi.model_path(reward, s))) for s in names}
    elif policy == "elite":
        specs = {s: ("elite", str(elite.MODEL)) for s in names}
    else:
        specs = {s: ("rule", None) for s in names}
    table = replay.replay_seasons(specs, workers=workers)
    typer.echo(table[["season", "points", "hits", "transfers", "chips"]].to_string(index=False))
    row = replay.against_planner(table)
    typer.echo(f"{policy}: {row['ours']:,.0f} a season against the planner's {row['planner']:,.0f}: "
               f"{row['diff']:+.0f} (SE {row['se']:.0f}), up in {row['up']} of {row['seasons']}")


if __name__ == "__main__":
    app()
