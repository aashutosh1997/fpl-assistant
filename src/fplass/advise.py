"""The top-level advisor: everything joined up into one weekly recommendation.

This is the entry point the CLI and the Claude Code skill both call. It runs the whole pipeline in
the order the decisions actually depend on each other:

1. Read your live squad, bank and free transfers from the FPL API.
2. Simulate the horizon, giving joint point samples for every player.
3. Value the chips across the whole remaining season and fix a roadmap, so the transfer solver
   cannot burn them for short-horizon gain.
4. Generate a pool of near-optimal transfer plans under that roadmap.
5. Re-score the pool against simulated mini-league rivals and pick the one that most improves your
   league position, which is usually — but not always — the same as the one that scores most.
6. Decide, for each recommended transfer, whether to execute now or wait for team news.

Each stage is separately testable and separately inspectable, and the returned object carries the
intermediate results so a recommendation can be interrogated rather than merely accepted.
"""

from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .api import FPLAPIError, FPLClient
from .features import flow as flow_module
from .ingest.sources import CURRENT_SEASON
from .options import revisions as revisions_module
from .options import value as value_module
from .optimise import chips as chips_module
from .optimise import league as league_module
from .optimise import milp
from .prices import calibrate as calibrate_module
from .prices import decide as decide_module
from .sim import project

log = logging.getLogger(__name__)

DEFAULT_HORIZON = 8
DEFAULT_DRAWS = 10_000
PLAN_POOL_SIZE = 6

# Everything before this gameweek's deadline is an unlimited squad build, not a transfer week.
FIRST_GAMEWEEK = 1

# Cap on how far the order-flow layer may move a live probability, in log-odds. The layer is
# fitted on complete weeks of flow; a live plan sees a partial week extrapolated, so it is held
# to a smaller move than history would allow (about 0.9 -> 0.77 at most for a nailed starter).
LIVE_FLOW_MAX_SHIFT = 1.0


@dataclass(slots=True)
class Recommendation:
    """A complete weekly recommendation, with its workings attached."""

    gameweek: int
    plan: milp.Plan
    roadmap: chips_module.ChipRoadmap
    comparison: pd.DataFrame
    league_metrics: dict[str, float]
    price_advice: pd.DataFrame
    sell_alerts: pd.DataFrame
    buy_alerts: pd.DataFrame
    expected_points: pd.DataFrame
    players: pd.DataFrame
    squad_before: list[int]
    deadline: dt.datetime | None
    notes: list[str] = field(default_factory=list)
    # What flexibility is worth to this squad this week, from sampled next weeks; None until
    # the projection panel exists to sample from.
    option_values: value_module.OptionValues | None = None

    def transfer_summary(self) -> str:
        names = dict(zip(self.players["element"], self.players["web_name"], strict=True))
        gw = self.gameweek
        out = [names.get(e, str(e)) for e in self.plan.transfers_out.get(gw, [])]
        into = [names.get(e, str(e)) for e in self.plan.transfers_in.get(gw, [])]
        if not into and not out:
            return "Roll your transfer — nothing clears the bar this week."
        hit = self.plan.hits.get(gw, 0)
        cost = f" (-{hit * 4} pts)" if hit else ""
        return f"{', '.join(out)} -> {', '.join(into)}{cost}"


def load_squad_state(
    client: FPLClient, entry_id: int, gameweek: int, players: pd.DataFrame
) -> milp.SquadState:
    """Read your current squad, bank and free transfers from the API.

    Picks are public only after a gameweek's deadline, so we read the most recently completed
    gameweek and reconstruct purchase prices from your transfer history — those determine selling
    prices through the 50% sell-on fee, and getting them wrong makes plans that cannot be executed.
    """
    history = client.entry_history(entry_id)
    played = [row for row in history.get("current", []) if row.get("event")]
    last_gameweek = max((row["event"] for row in played), default=0)

    if last_gameweek == 0:
        # Pre-season: the squad is built from scratch with unlimited changes, so the transfer
        # budget is the squad size rather than a free-transfer count.
        log.info("entry %d has no completed gameweeks yet", entry_id)
        return milp.SquadState(players={}, bank=1000, free_transfers=milp.SQUAD_SIZE)

    picks = client.entry_picks(entry_id, last_gameweek)
    squad = [p["element"] for p in picks.get("picks", [])]

    entry = client.entry(entry_id)
    bank = int(entry.get("last_deadline_bank") or 0)

    # Purchase prices: start from the current price, then override with what you actually paid
    # wherever the transfer history records it.
    price_by_element = dict(zip(players["element"], players["price"], strict=True))
    purchase = {e: int(price_by_element.get(e, 0)) for e in squad}
    try:
        for transfer in client.entry_transfers(entry_id):
            if transfer["element_in"] in purchase:
                purchase[transfer["element_in"]] = int(transfer["element_in_cost"])
    except FPLAPIError:
        log.warning("could not read transfer history; assuming current prices were paid")

    chips_used = {
        f"{row['name']}:{0 if row['event'] < 20 else 1}"
        for row in history.get("chips", [])
        if row.get("name") and row.get("event")
    }

    free_transfers = _free_transfers(history, last_gameweek)
    log.info(
        "entry %d: %d players, bank %.1fm, %d free transfer(s), chips used %s",
        entry_id,
        len(squad),
        bank / 10,
        free_transfers,
        sorted(chips_used) or "none",
    )
    return milp.SquadState(
        players=purchase, bank=bank, free_transfers=free_transfers, chips_used=chips_used
    )


def _free_transfers(history: dict, last_gameweek: int) -> int:
    """Reconstruct banked free transfers from the transfer counts in your history.

    FPL does not expose this directly. You gain one per gameweek and may bank up to five, so it can
    be replayed from how many transfers you made each week. Wildcard and Free Hit weeks are skipped
    because their transfers are free and do not consume the balance.

    **Gameweek 1 is not a gameweek for this purpose.** Everything before the season's first
    deadline is an unlimited squad build — effectively a free wildcard — so it neither spends a
    transfer nor banks one. You enter gameweek 2 with exactly one free transfer no matter what you
    did beforehand. Counting gameweek 1 as an ordinary week credits a transfer that does not exist,
    and a plan built on a phantom transfer is one that takes an unplanned -4 to execute.
    """
    banked = 1  # what you hold entering gameweek 2
    chip_weeks = {row["event"]: row["name"] for row in history.get("chips", [])}

    for row in sorted(history.get("current", []), key=lambda r: r.get("event", 0)):
        gameweek = row.get("event")
        if not gameweek or gameweek > last_gameweek:
            continue
        if gameweek == FIRST_GAMEWEEK:
            continue
        if chip_weeks.get(gameweek) in {"wildcard", "freehit"}:
            banked = min(banked + 1, milp.MAX_BANKED_TRANSFERS)
            continue
        made = int(row.get("event_transfers", 0))
        banked = min(max(banked - made, 0) + 1, milp.MAX_BANKED_TRANSFERS)
    return max(banked, 1)


def advise(
    con,
    *,
    entry_id: int | None = None,
    league_id: int | None = None,
    squad: milp.SquadState | None = None,
    gameweek: int | None = None,
    horizon: int = DEFAULT_HORIZON,
    n_draws: int = DEFAULT_DRAWS,
    objective: str = "league",
    season: str = CURRENT_SEASON,
    models: project.ProjectionModels | None = None,
    price_snapshots: pd.DataFrame | None = None,
    solver_time_limit: int = 120,
) -> Recommendation:
    """Produce a full weekly recommendation. See the module docstring for the stages."""
    notes: list[str] = []

    with FPLClient() as client:
        players = project.live_player_table(con, client=client, season=season)
        target_gameweek = gameweek or project.next_gameweek(con, season)

        if squad is None:
            if entry_id is None:
                raise ValueError("provide either entry_id or an explicit squad")
            squad = load_squad_state(client, entry_id, target_gameweek, players)

        league_state = None
        if league_id is not None and entry_id is not None:
            league_state = league_module.load_league(
                client, league_id, entry_id, gameweek=max(target_gameweek - 1, 0)
            )

        deadline = _deadline(con, season, target_gameweek)

    # ---- 2. simulate the horizon
    models = models or project.fit_models(con, season=season)
    availability = players[["element", "status", "chance_of_playing_next_round"]]
    flow = _live_flow(con, players, season, target_gameweek, deadline)
    result, player_matches, models = project.project(
        con,
        models=models,
        start_gameweek=target_gameweek,
        horizon=horizon,
        n_draws=n_draws,
        availability=availability,
        # Recorded before the deadline so `fpl calibrate` can score it afterwards. This is the
        # only way the model learns whether its own advice was any good.
        store=True,
        flow=flow,
        flow_max_shift=LIVE_FLOW_MAX_SHIFT,
    )
    expected_points = result.expected_points

    windows = milp.ChipWindows.from_warehouse(con, season)

    # ---- 3. chip roadmap, decided over the season rather than the horizon
    baseline = milp.solve(
        expected_points,
        players,
        squad,
        windows,
        allow_chips=False,
        time_limit=solver_time_limit,
    )
    roadmap = chips_module.build_roadmap(
        con,
        result.points,
        result.elements,
        result.gameweeks,
        players,
        squad,
        baseline,
        windows,
        season,
    )
    notes.extend(roadmap.notes)

    # ---- 4. a pool of near-optimal plans under that roadmap
    plans: list[milp.Plan] = []
    forbidden: list[dict] = []
    for _ in range(PLAN_POOL_SIZE):
        plan = milp.solve(
            expected_points,
            players,
            squad,
            windows,
            chip_schedule=roadmap.schedule,
            forbidden=forbidden,
            time_limit=solver_time_limit,
        )
        if plan.status not in {"Optimal", "Not Solved"}:
            break
        plans.append(plan)
        forbidden.append({gw: plan.squads[gw] for gw in plan.squads})
    if not plans:
        plans = [baseline]

    # ---- 5. pick under the mini-league objective
    if league_state is not None and league_state.rivals:
        ownership = players.set_index("element")["selected_by_percent"].fillna(0.0)
        rival_totals = league_module.simulate_rivals(
            league_state, result.points, result.elements, players, ownership
        )
        chosen, comparison = league_module.choose(
            plans, league_state, rival_totals, result.points, result.elements, objective=objective
        )
        metrics = league_module.evaluate(
            chosen, league_state, rival_totals, result.points, result.elements
        )
        notes.append(_league_note(league_state, metrics))
    else:
        chosen = plans[0]
        comparison = pd.DataFrame(
            [{"plan": i, "expected_points": p.objective} for i, p in enumerate(plans)]
        )
        metrics = {"expected_points": float(chosen.objective)}
        if league_id is None:
            notes.append(
                "No league given, so this maximises expected points. Pass a league id to optimise "
                "for finishing above your rivals instead, which can favour different picks."
            )

    # ---- 6. price timing for the recommended transfers
    price_advice, sells, buys = _price_stage(
        con,
        chosen,
        players,
        squad,
        target_gameweek,
        deadline,
        expected_points,
        windows,
        price_snapshots,
        notes,
        solver_time_limit,
    )

    notes.extend(_minutes_risk_notes(chosen, player_matches, players, target_gameweek))
    notes.extend(_flow_notes(chosen, player_matches, players, target_gameweek))
    option_values = _option_values(
        con, expected_points, player_matches, players, squad, target_gameweek
    )

    return Recommendation(
        gameweek=target_gameweek,
        plan=chosen,
        roadmap=roadmap,
        comparison=comparison,
        league_metrics=metrics,
        price_advice=price_advice,
        sell_alerts=sells,
        buy_alerts=buys,
        expected_points=expected_points,
        players=players,
        squad_before=sorted(squad.elements),
        deadline=deadline,
        notes=notes,
        option_values=option_values,
    )


def _option_values(
    con,
    expected_points: pd.DataFrame,
    player_matches: pd.DataFrame,
    players: pd.DataFrame,
    squad: milp.SquadState,
    gameweek: int,
) -> value_module.OptionValues | None:
    """Price this squad's flexibilities under next weeks sampled from the panel's revisions."""
    if not squad.players:
        return None
    try:
        stored = con.execute("SELECT count(*) FROM projection_panel").fetchone()[0]
    except Exception:  # pragma: no cover - table absent on an old warehouse
        return None
    if not stored:
        return None
    remaining = [g for g in expected_points.columns if g > gameweek]
    if not remaining:
        return None
    try:
        table = revisions_module.revisions(con)
        sampler = revisions_module.RevisionSampler.fit(table)
        p_full = (
            player_matches[player_matches["event"] == gameweek]
            .groupby("element")["p_full_base"]
            .max()
        )
        return value_module.live_option_values(
            expected_points[remaining], p_full, players, squad, sampler
        )
    except Exception as exc:  # pragma: no cover - never lose a plan over a diagnostic
        log.warning("could not value the squad's options: %s", exc)
        return None


def _live_flow(
    con, players: pd.DataFrame, season: str, gameweek: int, deadline: dt.datetime | None
) -> pd.DataFrame | None:
    """This week's order flow so far, scaled by how much of the transfer window has passed."""
    needed = {"transfers_in_event", "transfers_out_event", "selected_by_percent", "total_players"}
    if not needed <= set(players.columns) or gameweek <= FIRST_GAMEWEEK:
        return None
    elapsed = 1.0
    previous = _deadline(con, season, gameweek - 1)
    if deadline is not None and previous is not None and deadline > previous:
        now = dt.datetime.now(dt.UTC)
        elapsed = (now - previous).total_seconds() / (deadline - previous).total_seconds()
        elapsed = min(max(elapsed, 0.0), 1.0)
    return flow_module.live_flow(players, elapsed=elapsed)


def _flow_notes(
    plan: milp.Plan, player_matches: pd.DataFrame, players: pd.DataFrame, gameweek: int
) -> list[str]:
    """Name the squad members the market moved this week, so the shift can be argued with."""
    if "flow_shift" not in player_matches.columns:
        return []
    shift = (
        player_matches[player_matches["event"] == gameweek]
        .groupby("element")["flow_shift"]
        .max()
    )
    names = players.set_index("element")["web_name"]
    moved = [
        (e, float(shift.get(e, 0.0)))
        for e in plan.squads.get(gameweek, [])
        if abs(float(shift.get(e, 0.0))) >= 0.05
    ]
    if not moved:
        return []
    moved.sort(key=lambda item: item[1])
    described = ", ".join(f"{names.get(e, e)} {w:+.0%}" for e, w in moved)
    return [
        "Order flow, already priced into this plan (chance of an hour, this gameweek only): "
        f"{described}. Managers' transfers before the deadline predicted absences in ten seasons "
        "of history; the live move is capped and scaled for the part of the week seen so far."
    ]


def _minutes_risk_notes(
    plan: milp.Plan, player_matches: pd.DataFrame, players: pd.DataFrame, gameweek: int
) -> list[str]:
    """Name the recommended players whose minutes a new signing has just made uncertain.

    The projection has already been widened for them (see :mod:`fplass.features.arrivals`), so
    the plan accounts for the risk; this makes the accounting visible, because a reader who sees
    a captain pick with a settled role beat a higher-ceiling one should know why.
    """
    if "minutes_risk" not in player_matches.columns:
        return []
    risk = (
        player_matches[player_matches["event"] == gameweek]
        .groupby("element")["minutes_risk"]
        .max()
    )
    names = players.set_index("element")["web_name"]
    squad = plan.squads.get(gameweek, [])
    lineup = set(plan.lineups.get(gameweek, []))
    flagged = [(e, float(risk.get(e, 0.0))) for e in squad if risk.get(e, 0.0) > 0]
    if not flagged:
        return []
    flagged.sort(key=lambda item: -item[1])
    described = ", ".join(
        f"{names.get(e, e)} ({'starting' if e in lineup else 'bench'}, widened {w:.0%})"
        for e, w in flagged
    )
    return [
        "Minutes risk from new signings, already priced into this plan: "
        f"{described}. Their club bought a competitor for the shirt after the last recorded "
        "gameweek. Historically that costs established starters about 4% of their chance of an "
        "hour in the first match, and nothing after the lineup is seen, so the projections are "
        "widened by a tenth until then — a nudge toward the settled alternative, not a verdict."
    ]


def _league_note(league: league_module.LeagueState, metrics: dict[str, float]) -> str:
    beaten = metrics["expected_rivals_beaten"]
    total = metrics["n_rivals"]
    deficits = league.deficits()
    behind = int((deficits > 0).sum())
    stance = (
        "chasing, so differentials are favoured"
        if behind > total / 2
        else "ahead of most rivals, so the safer template is favoured"
    )
    return (
        f"{league.name}: behind {behind} of {int(total)} rivals — {stance}. "
        f"This plan expects to finish above {beaten:.1f} of them."
    )


def _deadline(con, season: str, gameweek: int) -> dt.datetime | None:
    row = con.execute(
        "SELECT deadline_time FROM events WHERE season = ? AND event = ?", [season, gameweek]
    ).fetchone()
    if not row or row[0] is None:
        return None
    value = pd.Timestamp(row[0])
    return value.tz_localize("UTC").to_pydatetime() if value.tz is None else value.to_pydatetime()


def _price_stage(
    con,
    plan,
    players,
    squad,
    gameweek,
    deadline,
    expected_points,
    windows,
    price_snapshots,
    notes,
    solver_time_limit,
):
    """Fit or fall back on a price model, then time the recommended transfers."""
    from .paths import PRICE_SNAPSHOTS

    snapshot = players.copy()
    model = None
    try:
        history = (
            price_snapshots
            if price_snapshots is not None
            else calibrate_module.load_snapshots(PRICE_SNAPSHOTS)
        )
        model = calibrate_module.fit(history)
        latest = history.sort_values("ts").groupby("element", as_index=False).tail(1)
        snapshot = players.merge(
            latest.drop(columns=[c for c in ("web_name", "team") if c in latest.columns]),
            on="element",
            how="left",
            suffixes=("", "_snap"),
        )
    except (FileNotFoundError, KeyError) as exc:
        notes.append(
            f"No price snapshot history yet ({exc}); price timing uses the classical "
            "net-transfer model until the hourly logger has run for a few days."
        )

    if model is not None:
        probabilities = model.predict(snapshot)
        probabilities["source"] = "calibrated"
        if model.calibrating:
            notes.append(
                "FPL's own price model still reports itself as calibrating, so these "
                "probabilities are provisional."
            )
    else:
        # FPL's own fields are now decoded and are more direct than anything we can fit this
        # early; the classical net-transfer heuristic is only needed if they are absent.
        if "price_change_percent" in snapshot.columns:
            probabilities = calibrate_module.official_projection(
                snapshot, hours_to_deadline=None
            )
        else:
            probabilities = calibrate_module.classical_model(snapshot)

    days = 3.0
    if deadline is not None:
        days = max((deadline - dt.datetime.now(dt.UTC)).total_seconds() / 86400.0, 0.0)

    def solve_with_bank(extra: int) -> float:
        adjusted = milp.SquadState(
            players=dict(squad.players),
            bank=squad.bank + extra,
            free_transfers=squad.free_transfers,
            chips_used=set(squad.chips_used),
        )
        return milp.solve(
            expected_points,
            players,
            adjusted,
            windows,
            allow_chips=False,
            time_limit=solver_time_limit,
        ).objective

    try:
        points_per_tenth = decide_module.budget_shadow_price(solve_with_bank)
    except Exception as exc:  # pragma: no cover - solver hiccup should not sink the report
        log.warning("could not compute budget shadow price: %s", exc)
        points_per_tenth = 0.05

    targets = players[players["element"].isin(plan.transfers_in.get(gameweek, []))][
        ["element", "web_name"]
    ]
    advice = (
        decide_module.decide(
            targets, probabilities, days_to_deadline=days, points_per_tenth=points_per_tenth
        )
        if len(targets)
        else pd.DataFrame()
    )

    watchlist = list(
        expected_points.sum(axis=1).nlargest(40).index.difference(pd.Index(sorted(squad.elements)))
    )
    return (
        advice,
        decide_module.sell_alerts(sorted(squad.elements), probabilities, players),
        decide_module.buy_alerts(watchlist, probabilities, players),
    )


def format_report(recommendation: Recommendation) -> str:
    """A readable summary of the recommendation, for the terminal or a chat reply."""
    names = dict(
        zip(recommendation.players["element"], recommendation.players["web_name"], strict=True)
    )
    gw = recommendation.gameweek
    lines = [f"=== Gameweek {gw} ==="]

    if recommendation.deadline:
        remaining = recommendation.deadline - dt.datetime.now(dt.UTC)
        hours = remaining.total_seconds() / 3600
        lines.append(
            f"Deadline {recommendation.deadline:%a %d %b %H:%M UTC} "
            f"({hours:.0f}h away)" if hours > 0 else "Deadline has passed"
        )

    lines += ["", "TRANSFER", "  " + recommendation.transfer_summary()]

    captain = recommendation.plan.captains.get(gw)
    if captain:
        lines.append(f"  Captain: {names.get(captain, captain)}")

    if len(recommendation.price_advice):
        lines += ["", "PRICE TIMING"]
        for row in recommendation.price_advice.itertuples():
            lines.append(f"  {row.web_name}: {row.recommendation} — {row.reason}")

    if len(recommendation.sell_alerts):
        lines += ["", "PRICE FALL RISK (players you own)"]
        for row in recommendation.sell_alerts.head(5).itertuples():
            lines.append(f"  {row.web_name}: {row.p_fall:.0%} chance of a drop, {row.urgency}")

    lines += ["", recommendation.roadmap.summary()]
    lines += ["", "PLAN", recommendation.plan.summary(names)]

    positions = dict(
        zip(recommendation.players["element"], recommendation.players["position"], strict=True)
    )
    lines += [
        "",
        "LINEUPS",
        recommendation.plan.lineups_summary(names, positions, recommendation.expected_points),
    ]

    if recommendation.option_values is not None:
        values = recommendation.option_values
        gw = recommendation.gameweek
        moves = len(recommendation.plan.transfers_in.get(gw, []))
        lines += [
            "",
            "OPTIONS (what flexibility is worth to this squad)",
            "  " + values.summary(),
            f"  best single swap under this week's projection: {values.best_swap_now:.2f} pts; "
            f"the plan makes {moves} transfer(s) this week",
        ]

    if recommendation.league_metrics:
        lines += ["", "LEAGUE"]
        for key, value in recommendation.league_metrics.items():
            lines.append(f"  {key}: {value:.2f}")

    if recommendation.notes:
        lines += ["", "NOTES"] + [f"  - {n}" for n in recommendation.notes]

    return "\n".join(lines)
