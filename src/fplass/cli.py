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
app.add_typer(backtest_app, name="backtest")


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
) -> None:
    """Recommend this week's transfers, captain and chip strategy."""
    from .advise import advise, format_report
    from .ingest.warehouse import connect

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


@backtest_app.command("score")
def backtest_score(
    seasons: str = typer.Option("all", help="Comma-separated seasons, or 'all'."),
    weeks_ahead: int = typer.Option(0, help="Horizon to report; -1 for every horizon."),
) -> None:
    """Score the projection panel against what happened, per season and horizon."""
    from .backtest import panel as panel_module
    from .ingest.warehouse import connect

    con = connect(read_only=True)
    try:
        chosen = None if seasons == "all" else [s.strip() for s in seasons.split(",")]
        table = panel_module.score_panel(con, chosen)
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
    solver_seconds: int = typer.Option(20, help="Per-solve time limit."),
    wildcard_candidates: int = typer.Option(1, help="Gameweeks tried for a wildcard each week."),
    max_gameweek: int = typer.Option(None, help="Stop after this gameweek (for quick checks)."),
) -> None:
    """Play completed seasons with the planner and score them in points.

    The number every policy change is judged by. Needs the projection panel; roughly half a
    minute a gameweek with the default solver limit.
    """
    from .backtest import manager as manager_module
    from .ingest.warehouse import connect

    if policy not in manager_module.POLICIES:
        raise typer.BadParameter(f"policy must be one of {manager_module.POLICIES}")

    con = connect(read_only=True)
    try:
        if seasons == "all":
            chosen = [
                r[0]
                for r in con.execute(
                    "SELECT DISTINCT season FROM projection_panel ORDER BY season"
                ).fetchall()
            ]
        else:
            chosen = [s.strip() for s in seasons.split(",")]
    finally:
        con.close()
    if not chosen:
        typer.echo("No panel rows stored. Run `fpl backtest panel` first.")
        return

    table = manager_module.replay_seasons(
        chosen,
        policy=policy,
        solver_time_limit=solver_seconds,
        wildcard_candidates=wildcard_candidates,
        max_gameweek=max_gameweek,
        workers=workers,
    )
    typer.echo(table.to_string(index=False))
    typer.echo(
        f"\n{policy}: {table['points'].mean():.0f} points a season over {len(table)} season(s)"
    )


if __name__ == "__main__":
    app()
