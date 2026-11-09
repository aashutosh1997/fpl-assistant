"""The top-level advisor: everything joined up into one weekly recommendation.

This is the entry point the CLI and the Claude Code skill both call. It runs the whole pipeline in
the order the decisions actually depend on each other:

1. Read your live squad, bank and free transfers from the FPL API.
2. Simulate the horizon, giving joint point samples for every player.
3. Value the chips across the whole remaining season and fix a roadmap, so the transfer solver
   cannot burn them for short-horizon gain.
4. Generate a pool of plans under that roadmap, each a different decision for the coming week.
5. Re-score the pool against simulated mini-league rivals and rank it, so the manager sees the
   best few options and how close the call is rather than a single answer
   (:mod:`fplass.optimise.alternatives`).
6. Decide, for each transfer the options make this week, whether to execute now or wait for news.

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
from .options import news as news_module
from .options import revisions as revisions_module
from .options import value as value_module
from .optimise import alternatives as alternatives_module
from .optimise import chips as chips_module
from .optimise import league as league_module
from .optimise import milp
from .optimise.policy import PolicyConfig, split_horizon
from .prices import calibrate as calibrate_module
from .prices import decide as decide_module
from .sim import project

log = logging.getLogger(__name__)

DEFAULT_HORIZON = 8
DEFAULT_DRAWS = 10_000
# Candidate plans solved, each a different squad for the coming week; a roll is added on top
# when none of them rolls.
PLAN_POOL_SIZE = 8
# Ranked options shown to the manager.
DEFAULT_OPTIONS = 5
# Revised projections the options are re-scored under for their robustness.
ROBUSTNESS_REVISIONS = 300

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
    # The ranked alternatives, best first; ``plan`` is the first of them.
    options: list[alternatives_module.PlanOption] = field(default_factory=list)
    # How close each chip still held is to being worth playing (see alternatives.chip_calls).
    chip_calls: pd.DataFrame | None = None
    objective: str = "league"
    league_name: str | None = None
    free_transfers: int = 1

    def transfer_summary(self) -> str:
        names = dict(zip(self.players["element"], self.players["web_name"], strict=True))
        summary = move_summary(self.plan, self.gameweek, names, self.free_transfers)
        if summary.startswith("Roll"):
            return "Roll your transfer — nothing clears the bar this week."
        return summary


def move_summary(
    plan: milp.Plan, gameweek: int, names: dict[int, str], free_transfers: int
) -> str:
    """This week's decision in a plan, in one line: the moves and any hit, or the roll."""
    chip = plan.chips.get(gameweek)
    label = f" [{chips_module.CHIP_LABELS.get(chip, chip)}]" if chip else ""
    out = [names.get(e, str(e)) for e in plan.transfers_out.get(gameweek, [])]
    into = [names.get(e, str(e)) for e in plan.transfers_in.get(gameweek, [])]
    if chip == "freehit":
        return f"Free Hit — a one-week squad, then yours returns{label}"
    if not into and not out:
        if chip == "wildcard":
            return f"Keep the squad{label}"
        if free_transfers >= milp.MAX_BANKED_TRANSFERS:
            cap = milp.MAX_BANKED_TRANSFERS
            return f"Roll — already at {cap} free transfers, so one is lost{label}"
        return f"Roll — {free_transfers + 1} free transfers next week{label}"
    hit = plan.hits.get(gameweek, 0)
    cost = f" (-{hit * int(milp.HIT_COST)})" if hit else ""
    return f"{', '.join(out)} -> {', '.join(into)}{cost}{label}"


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
    be replayed from how many transfers you made each week. Wildcard and Free Hit weeks leave the
    balance exactly as it was: their transfers are free and do not consume it, and the week does
    not add one either.

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
            # The balance is kept through the chip week but does not grow in it. Crediting one
            # here made Esteciao's 2026-27 GW8 count 4 where FPL showed 3: a GW3 Wildcard, one
            # transfer in each of GW4 and GW5, then two rolls.
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
    config: PolicyConfig | None = None,
    n_options: int = DEFAULT_OPTIONS,
) -> Recommendation:
    """Produce a full weekly recommendation. See the module docstring for the stages.

    Args:
        config: The planner's knobs — bench weight, banked transfer value, terminal value, chip
            rule. Defaults to the values the ten-season replay has earned; the same object the
            replay runs, so what is measured is what is played.
        n_options: How many ranked options to return (roll is added as a reference when it
            falls outside them).
    """
    notes: list[str] = []
    config = config or PolicyConfig()
    if config.horizon != horizon:
        config = PolicyConfig(**{**{f: getattr(config, f) for f in config.__slots__}, "horizon": horizon})

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
    availability = players[["element", "status", "chance_of_playing_next_round", "news"]]
    flow = _live_flow(con, players, season, target_gameweek, deadline)
    result, player_matches, models = project.project(
        con,
        models=models,
        start_gameweek=target_gameweek,
        horizon=config.projection_horizon(),
        n_draws=n_draws,
        availability=availability,
        # Recorded before the deadline so `fpl calibrate` can score it afterwards. This is the
        # only way the model learns whether its own advice was any good.
        store=True,
        flow=flow,
        flow_max_shift=LIVE_FLOW_MAX_SHIFT,
    )
    # The planning horizon, and the terminal value read from the weeks projected beyond it.
    expected_points, terminal = split_horizon(result.expected_points, config)
    options = {**config.solve_options(), "terminal_value": terminal}
    horizon_slots = [int(np.where(result.gameweeks == g)[0][0]) for g in expected_points.columns]
    samples = result.points[:, :, horizon_slots]
    gameweeks = np.array(list(expected_points.columns))

    windows = milp.ChipWindows.from_warehouse(con, season)
    chip_floors = config.thresholds(windows, season)

    # ---- 3. chip roadmap, decided over the season rather than the horizon
    baseline = milp.solve(
        expected_points,
        players,
        squad,
        windows,
        allow_chips=False,
        time_limit=solver_time_limit,
        **options,
    )
    roadmap = chips_module.build_roadmap(
        con,
        samples,
        result.elements,
        gameweeks,
        players,
        squad,
        baseline,
        windows,
        season,
        min_gain=chip_floors,
        solve_options=options,
    )
    notes.extend(roadmap.notes)

    # ---- 4. a pool of plans under that roadmap, each a different decision this week
    def solve_candidate(**extra) -> milp.Plan:
        return milp.solve(
            expected_points,
            players,
            squad,
            windows,
            chip_schedule=roadmap.schedule,
            time_limit=solver_time_limit,
            **options,
            **extra,
        )

    plans = alternatives_module.distinct_plans(
        solve_candidate,
        target_gameweek,
        squad,
        pool_size=PLAN_POOL_SIZE,
        chip=roadmap.schedule.get(target_gameweek),
    ) or [baseline]

    # ---- 5. rank the pool, under the mini-league objective when there is a league
    p_full = (
        player_matches[player_matches["event"] == target_gameweek]
        .groupby("element")["p_full_base"]
        .max()
    )
    sampler = _revision_sampler(con)
    projections, projection_elements = _revised_projections(
        sampler, plans, expected_points, p_full, players
    )
    rival_totals = None
    against_rivals = league_state is not None and bool(league_state.rivals)
    if against_rivals:
        ownership = players.set_index("element")["selected_by_percent"].fillna(0.0)
        rival_totals = league_module.simulate_rivals(
            league_state, samples, result.elements, players, ownership
        )
    ranked = alternatives_module.rank(
        plans,
        samples,
        result.elements,
        target_gameweek,
        league=league_state,
        rival_totals=rival_totals,
        objective=objective,
        projections=projections,
        projection_elements=projection_elements,
        n_show=n_options,
    )
    chosen = ranked[0].plan
    names = dict(zip(players["element"], players["web_name"], strict=True))
    comparison = pd.DataFrame(
        [
            {
                "rank": o.rank,
                "move": move_summary(o.plan, target_gameweek, names, squad.free_transfers),
                "reference": o.reference,
                "tags": ",".join(o.tags),
                **o.metrics,
            }
            for o in ranked
        ]
    )
    if against_rivals:
        metrics = league_module.evaluate(
            chosen, league_state, rival_totals, samples, result.elements
        )
        notes.append(_league_note(league_state, metrics))
    else:
        metrics = {"expected_points": float(ranked[0].metrics["expected_points"])}
        if league_id is None:
            notes.append(
                "No league given, so this maximises expected points. Pass a league id to optimise "
                "for finishing above your rivals instead, which can favour different picks."
            )
    calls = alternatives_module.chip_calls(
        roadmap.valuations, chip_floors, target_gameweek, squad.chips_used, roadmap.schedule
    )

    # ---- 6. price timing for every transfer the options make this week
    price_advice, sells, buys = _price_stage(
        con,
        [o.plan for o in ranked],
        players,
        squad,
        target_gameweek,
        deadline,
        expected_points,
        windows,
        price_snapshots,
        notes,
        solver_time_limit,
        options,
    )

    others = [o.plan for o in ranked[1:]]
    notes.extend(_minutes_risk_notes(chosen, player_matches, players, target_gameweek, others))
    notes.extend(_flow_notes(chosen, player_matches, players, target_gameweek, others))
    option_values = _option_values(
        expected_points, p_full, players, squad, target_gameweek, sampler
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
        options=ranked,
        chip_calls=calls,
        objective=objective if against_rivals else "points",
        league_name=league_state.name if against_rivals else None,
        free_transfers=squad.free_transfers,
    )


def _revision_sampler(con) -> revisions_module.RevisionSampler | None:
    """The bootstrap of week-to-week projection revisions, once the projection panel exists."""
    try:
        stored = con.execute("SELECT count(*) FROM projection_panel").fetchone()[0]
    except Exception:  # pragma: no cover - table absent on an old warehouse
        return None
    if not stored:
        return None
    try:
        return revisions_module.RevisionSampler.fit(revisions_module.revisions(con))
    except Exception as exc:  # pragma: no cover - never lose a plan over a diagnostic
        log.warning("could not measure projection revisions: %s", exc)
        return None


def _revised_projections(
    sampler: revisions_module.RevisionSampler | None,
    plans: list[milp.Plan],
    expected_points: pd.DataFrame,
    p_full: pd.Series,
    players: pd.DataFrame,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """The horizon's projection, revised as projections are between deadlines, for the players
    any candidate fields; the options' robustness is measured under these."""
    if sampler is None or len(plans) < 2:
        return None, None
    fielded = sorted({e for plan in plans for lineup in plan.lineups.values() for e in lineup})
    expected = expected_points.reindex(fielded).fillna(0.0)
    try:
        revised = sampler.sample(
            expected,
            p_full,
            players,
            draws=ROBUSTNESS_REVISIONS,
            rng=np.random.default_rng(0),
        )
    except Exception as exc:  # pragma: no cover - never lose a plan over a diagnostic
        log.warning("could not revise the projections: %s", exc)
        return None, None
    return revised, np.array(fielded)


def _option_values(
    expected_points: pd.DataFrame,
    p_full: pd.Series,
    players: pd.DataFrame,
    squad: milp.SquadState,
    gameweek: int,
    sampler: revisions_module.RevisionSampler | None,
) -> value_module.OptionValues | None:
    """Price this squad's flexibilities under next weeks sampled from the panel's revisions."""
    if not squad.players or sampler is None:
        return None
    remaining = [g for g in expected_points.columns if g > gameweek]
    if not remaining:
        return None
    try:
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


def _fielded_now(plan: milp.Plan, others: list[milp.Plan], gameweek: int) -> list[int]:
    """Everyone in the coming week's squad of the plan or any alternative, plan's first."""
    seen = list(plan.squads.get(gameweek, []))
    for other in others:
        seen += [e for e in other.squads.get(gameweek, []) if e not in seen]
    return seen


def _flow_notes(
    plan: milp.Plan,
    player_matches: pd.DataFrame,
    players: pd.DataFrame,
    gameweek: int,
    others: list[milp.Plan] | None = None,
) -> list[str]:
    """Name the players the market moved this week, so the shift can be argued with: the
    squad's, and those of any alternative option."""
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
        for e in _fielded_now(plan, others or [], gameweek)
        if abs(float(shift.get(e, 0.0))) >= 0.05
    ]
    if not moved:
        return []
    moved.sort(key=lambda item: item[1])
    described = ", ".join(f"{names.get(e, e)} {w:+.0%}" for e, w in moved)
    return [
        "Order flow, already priced into these options (chance of an hour, this gameweek only): "
        f"{described}. Managers' transfers before the deadline predicted absences in ten seasons "
        "of history; the live move is capped and scaled for the part of the week seen so far."
    ]


def _minutes_risk_notes(
    plan: milp.Plan,
    player_matches: pd.DataFrame,
    players: pd.DataFrame,
    gameweek: int,
    others: list[milp.Plan] | None = None,
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
    squad = set(plan.squads.get(gameweek, []))
    lineup = set(plan.lineups.get(gameweek, []))
    flagged = [
        (e, float(risk.get(e, 0.0)))
        for e in _fielded_now(plan, others or [], gameweek)
        if risk.get(e, 0.0) > 0
    ]
    if not flagged:
        return []
    flagged.sort(key=lambda item: -item[1])

    def role(e: int) -> str:
        if e in lineup:
            return "starting"
        return "bench" if e in squad else "in another option"

    described = ", ".join(f"{names.get(e, e)} ({role(e)}, widened {w:.0%})" for e, w in flagged)
    return [
        "Minutes risk from new signings, already priced into these options: "
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
    plans,
    players,
    squad,
    gameweek,
    deadline,
    expected_points,
    windows,
    price_snapshots,
    notes,
    solver_time_limit,
    options=None,
):
    """Fit or fall back on a price model, then time the transfers the options make this week.

    ``plans`` is the ranked options, best first; every player any of them buys this week gets
    buy-now-or-wait advice, in the order the options rank.
    """
    options = options or {}
    if isinstance(plans, milp.Plan):
        plans = [plans]
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
            "net-transfer model until the daily logger has run for a few days."
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

    # News risk measured from the logged windows, when there are enough of them.
    risk_table = None
    try:
        snapshots = price_snapshots if price_snapshots is not None else calibrate_module.load_snapshots(PRICE_SNAPSHOTS)
        risk_table = news_module.news_risk_table(
            snapshots, news_module.deadlines_from_warehouse(con, CURRENT_SEASON)
        )
    except (FileNotFoundError, KeyError, ValueError):
        risk_table = None
    if risk_table is not None:
        notes.append(
            "News risk before a deadline is now measured from the logged snapshots: "
            + ", ".join(f"{int(d)}d {float(v):.1%}" for d, v in risk_table.items() if d in (1, 2, 4, 7))
            + " (the built-in table assumed 6%, 11%, 18%, 24%)."
        )

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
            **options,
        ).objective

    try:
        points_per_tenth = decide_module.budget_shadow_price(solve_with_bank)
    except Exception as exc:  # pragma: no cover - solver hiccup should not sink the report
        log.warning("could not compute budget shadow price: %s", exc)
        points_per_tenth = 0.05

    buying: list[int] = []
    for plan in plans:
        buying += [e for e in plan.transfers_in.get(gameweek, []) if e not in buying]
    targets = (
        players.set_index("element")
        .reindex(buying)
        .dropna(subset=["web_name"])
        .reset_index()[["element", "web_name"]]
    )
    advice = (
        decide_module.decide(
            targets,
            probabilities,
            days_to_deadline=days,
            points_per_tenth=points_per_tenth,
            risk_table=risk_table,
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
    """A readable summary of the recommendation, for the terminal or a chat reply.

    It leads with the ranked options, because the choice between them is the manager's: the
    planner's first choice, the alternatives it weighed, and how close the call is.
    """
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

    if recommendation.options:
        lines += ["", *_options_table(recommendation, names)]
        lines += ["", "PATHS (each option over the horizon; later weeks are re-planned weekly)"]
        for option in recommendation.options:
            label = f"#{option.rank}" + (" (reference)" if option.reference else "")
            move = move_summary(option.plan, gw, names, recommendation.free_transfers)
            lines.append(f"Option {label}: {move}")
            lines += ["  " + line for line in option.plan.summary(names).splitlines()]
    else:
        lines += ["", "TRANSFER", "  " + recommendation.transfer_summary()]
        captain = recommendation.plan.captains.get(gw)
        if captain:
            lines.append(f"  Captain: {names.get(captain, captain)}")

    chips = _chip_lines(recommendation, names)
    if chips:
        lines += ["", "CHIPS (gain if played, against what holding it is worth)", *chips]

    if len(recommendation.price_advice):
        lines += ["", "PRICE TIMING (every player an option buys this week)"]
        for row in recommendation.price_advice.itertuples():
            lines.append(f"  {row.web_name}: {row.recommendation} — {row.reason}")

    if len(recommendation.sell_alerts):
        lines += ["", "PRICE FALL RISK (players you own)"]
        for row in recommendation.sell_alerts.head(5).itertuples():
            lines.append(f"  {row.web_name}: {row.p_fall:.0%} chance of a drop, {row.urgency}")

    lines += ["", recommendation.roadmap.summary()]
    if not recommendation.options:
        lines += ["", "PLAN", recommendation.plan.summary(names)]

    positions = dict(
        zip(recommendation.players["element"], recommendation.players["position"], strict=True)
    )
    lines += [
        "",
        "LINEUPS (option #1)" if recommendation.options else "LINEUPS",
        recommendation.plan.lineups_summary(names, positions, recommendation.expected_points),
    ]

    if recommendation.option_values is not None:
        values = recommendation.option_values
        moves = len(recommendation.plan.transfers_in.get(gw, []))
        lines += [
            "",
            "FLEXIBILITY (what it is worth to this squad)",
            "  " + values.summary(),
            f"  best single swap under this week's projection: {values.best_swap_now:.2f} pts; "
            f"option #1 makes {moves} transfer(s) this week",
        ]

    if recommendation.league_metrics:
        lines += ["", "LEAGUE (option #1)" if recommendation.options else "LEAGUE"]
        for key, value in recommendation.league_metrics.items():
            lines.append(f"  {key}: {value:.2f}")

    if recommendation.notes:
        lines += ["", "NOTES"] + [f"  - {n}" for n in recommendation.notes]

    return "\n".join(lines)


def _options_table(recommendation: Recommendation, names: dict[int, str]) -> list[str]:
    """The ranked options as a table, with what each column means underneath."""
    options = recommendation.options
    gw = recommendation.gameweek
    rivals = all("expected_rivals_beaten" in o.metrics for o in options)
    robust = all("robustness" in o.metrics for o in options)

    if recommendation.objective == "points" or not rivals:
        basis = "ranked on the planner's value (expected points, plus banked transfers)"
    elif recommendation.objective == "blend":
        basis = f"ranked on rivals beaten in {recommendation.league_name} and points together"
    else:
        basis = f"ranked on rivals beaten in {recommendation.league_name}"
    lines = [f"OPTIONS — {basis}; the call is yours"]

    header = f"  {'#':>3}  {'8-GW pts':>8}  {'vs #1':>5}  {f'GW{gw}':>5}"
    if rivals:
        header += f"  {'Rivals':>6}  {'Win':>4}"
    header += f"  {'P10-P90':>9}"
    if robust:
        header += f"  {'Robust':>6}"
    header += "  This week"
    lines.append(header)

    for option in options:
        m = option.metrics
        rank = f"{option.rank}{'*' if option.reference else ''}"
        gap = "—" if option.rank == 1 else f"{m['gap']:+.1f}"
        row = (
            f"  {rank:>3}  {m['expected_points']:>8.1f}  {gap:>5}"
            f"  {option.plan.expected_points.get(gw, 0.0):>5.1f}"
        )
        if rivals:
            row += f"  {m['expected_rivals_beaten']:>6.2f}  {m['win_probability']:>4.0%}"
        row += f"  {m['points_p10']:>4.0f}-{m['points_p90']:<4.0f}"
        if robust:
            row += f"  {m['robustness']:>6.0%}"
        captain = names.get(option.plan.captains.get(gw, -1), "?")
        move = move_summary(option.plan, gw, names, recommendation.free_transfers)
        tags = f"  [{', '.join(option.tags)}]" if option.tags else ""
        lines.append(f"{row}  {move}; (C) {captain}{tags}")

    lines.append(
        "  8-GW pts: expected points over the horizon, net of hits. P10-P90: the range eight in "
        "ten simulated outcomes fall in."
    )
    on_value = recommendation.objective == "points" or not rivals
    if robust:
        best = "the planner's best" if on_value else "the candidate with the most points"
        lines.append(
            f"  Robust: how often the option is still {best} when the projections move as much "
            "as they have between deadlines in nine replayed seasons, injuries included. Under "
            "about 50% for #1 means a close call."
        )
    if any(o.tags for o in options):
        lines.append(
            "  [ ]: what an option leads the shown options on — "
            + ", ".join(
                f"{tag} = {label}" for tag, label in alternatives_module.TAG_LABELS.items()
            )
            + "."
        )
    for option in options:
        if option.reference:
            why = []
            if alternatives_module.is_roll(option.plan, gw):
                why.append("the do-nothing reference")
            if "value" in option.tags:
                why.append("the planner's own pick")
            lines.append(
                f"  {option.rank}*: ranks {option.rank} of the candidates, shown as "
                f"{' and '.join(why) or 'a reference'}."
            )
    pick = next((o for o in options if "value" in o.tags), None)
    if pick is not None and pick.rank != 1:
        lines.append(
            f"  On points plus what it prices a banked transfer at — the economy the replay "
            f"measured — the planner would choose #{pick.rank}. The league ranking counts only "
            "points scored, so it gives an unspent transfer no value."
        )
    return lines


def _chip_lines(recommendation: Recommendation, names: dict[int, str]) -> list[str]:
    """One line per chip still held: its gain now against the floor, and its best week."""
    calls = recommendation.chip_calls
    if calls is None or calls.empty:
        return []
    gw = recommendation.gameweek
    lines = []
    for row in calls.itertuples():
        label = row.label + (" (2nd half)" if row.window else "")
        if pd.isna(row.now_mean):
            now = "not playable this week"
        else:
            extra = ""
            if row.chip == "3xc":
                captain = str(row.note).removeprefix("captain ").strip()
                who = names.get(int(captain), captain) if captain.lstrip("-").isdigit() else ""
                extra = f" (P90 +{row.now_upside:.1f}{', on ' + who if who else ''})"
            if row.scheduled == gw:
                verdict = "play it: it clears the floor"
            elif row.scheduled:
                verdict = f"held for GW{row.scheduled}, where it is worth more"
            elif row.short_by > 0:
                verdict = f"hold, short by {row.short_by:.1f}"
            else:
                verdict = "clears the floor, but another chip has the week"
            now = (
                f"+{row.now_mean:.1f} now{extra} against {row.now_floor:.1f} to hold — {verdict}"
            )
        best = (
            f" Best in the horizon: GW{row.best_gameweek} +{row.best_mean:.1f} against "
            f"{row.best_floor:.1f}."
            if row.best_gameweek != gw
            else ""
        )
        lines.append(f"  {label}: {now}.{best}")
    return lines
