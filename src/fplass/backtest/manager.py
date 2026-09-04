"""The paper manager: the whole policy replayed over a season and scored in points.

Every component of this system has its own validation — the scoring engine reproduces history
exactly, the minutes model has a Brier score, the team model a held-out log-likelihood. None of
that says whether the *decisions* are any good. A planner can be fed perfect projections and
still lose points by churning, by burning chips in October, by carrying a bench that never plays.
The only measure of a policy is the points it scores, and the only honest way to get that number
is to play the season with it.

So this module plays the season. Given the projection panel (what the model knew before each
deadline, see :mod:`fplass.backtest.panel`) it builds the gameweek-one squad, then week by week
runs the same planner the live advisor runs — chip roadmap, transfer solve, lineup, captain —
executes the plan against real prices with the real sell-on fee, and scores it against what the
players actually did, with the game's own automatic substitutions and captain rules. The result
is a season total that can be compared with a manager's, and a per-gameweek trace that can be
diffed when the policy changes.

This is the number every change to the planner is judged by. A constant replaced by a measured
option value either lifts the ten-season total or it does not.

What is deliberately *not* replayed: the mini-league objective (there are no rivals in history)
and the price-timing stage (prices are taken as they stood at each deadline). Chip windows are
the live season's, whatever season is replayed, because the question is how the policy we run
today would have fared, not what was legal in 2019. Availability news is absent from history, so
the replayed manager is blind to injuries until a player has missed a match — the order-flow
layer narrows that gap once it is fitted.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..ingest.sources import CURRENT_SEASON
from ..ingest.warehouse import connect
from ..optimise import chips as chips_module
from ..optimise import milp
from ..paths import BACKTEST, PANEL
from ..sim import project
from . import panel as panel_module

log = logging.getLogger(__name__)

STARTING_BANK = 1000  # tenths: the 100.0m every manager starts with
POLICY_CURRENT = "current"
POLICY_HOLD = "hold"
POLICIES = (POLICY_CURRENT, POLICY_HOLD)

# Weeks beyond the planning horizon whose projection feeds the terminal value.
TERMINAL_WEEKS = 4


@dataclass(slots=True)
class PolicyConfig:
    """The knobs of the planner, so a replay can measure what each is worth.

    The defaults are the live planner's current constants. A replay with one of them changed,
    paired season by season against the baseline, is how a constant earns its replacement.
    """

    horizon: int = 8
    bench_weight: float = 0.12
    banked_transfer_value: float = 0.25
    terminal_beta: float = 0.0  # share of the projection beyond the horizon credited at its end
    terminal_bank_value: float = 0.0  # points per 0.1m left in the bank at the horizon's end
    chip_floors: dict[str, float] | None = None  # None: the roadmap's defaults
    wildcard_candidates: int = 1
    solver_time_limit: int = 20

    @classmethod
    def parse(cls, spec: str | None) -> PolicyConfig:
        """``"bench_weight=0.3,terminal_beta=0.5"`` -> a config; chip floors as ``3xc:8``."""
        config = cls()
        if not spec:
            return config
        floors: dict[str, float] = {}
        for part in spec.split(","):
            if not part.strip():
                continue
            key, raw = part.split("=", 1)
            key, raw = key.strip(), raw.strip()
            if ":" in key:
                chip = key.split(":", 1)[1]
                floors[chip] = float(raw)
                continue
            current = getattr(config, key)
            setattr(config, key, type(current)(raw) if current is not None else float(raw))
        if floors:
            base = dict(chips_module.DEFAULT_MIN_GAIN)
            base.update(floors)
            config.chip_floors = base
        return config

    def solve_options(self) -> dict:
        return {
            "bench_weight": self.bench_weight,
            "banked_transfer_value": self.banked_transfer_value,
            "terminal_bank_value": self.terminal_bank_value,
        }

    def tag(self) -> str:
        """A short label for output files: only the knobs that differ from the defaults."""
        default = PolicyConfig()
        parts = []
        for name in ("horizon", "bench_weight", "banked_transfer_value", "terminal_beta",
                     "terminal_bank_value"):
            if getattr(self, name) != getattr(default, name):
                parts.append(f"{name}={getattr(self, name)}")
        if self.chip_floors:
            parts.append("floors=" + "-".join(f"{k}{v:g}" for k, v in sorted(self.chip_floors.items())))
        return ",".join(parts) or "default"


def split_horizon(
    expected: pd.DataFrame, config: PolicyConfig
) -> tuple[pd.DataFrame, pd.Series | None]:
    """The planning horizon, and the terminal value read from the weeks beyond it."""
    columns = list(expected.columns)
    inside = expected[columns[: config.horizon]]
    beyond = columns[config.horizon : config.horizon + TERMINAL_WEEKS]
    if config.terminal_beta and beyond:
        return inside, config.terminal_beta * expected[beyond].sum(axis=1)
    return inside, None

FORMATION_MIN = {"GKP": 1, "DEF": 3, "MID": 2, "FWD": 1}


@dataclass(slots=True)
class GameweekRecord:
    """What happened in one replayed gameweek."""

    gameweek: int
    points: int  # net of hits
    raw_points: int  # before hits, after captaincy and substitutions
    hits: int
    chip: str | None
    transfers_in: list[int]
    transfers_out: list[int]
    captain: int
    captain_points: int  # the extra copy (or two) of the armband holder's score
    auto_subs: int
    sub_points: int  # points delivered by the substitutes who came on
    bench_points: int  # points left on the bench after substitutions
    bench_expected: float  # what the plan expected the four bench players to score
    bank: int  # after this gameweek's transfers
    free_transfers: int  # entering the gameweek
    squad_value: int  # selling value of the squad plus bank, after transfers
    expected: float  # what the plan expected this week to score


@dataclass(slots=True)
class SeasonReplay:
    season: str
    policy: str
    records: list[GameweekRecord] = field(default_factory=list)

    @property
    def total(self) -> int:
        return sum(r.points for r in self.records)

    def summary(self) -> dict[str, object]:
        chips = ",".join(f"{r.chip}@GW{r.gameweek}" for r in self.records if r.chip)
        return {
            "season": self.season,
            "policy": self.policy,
            "gameweeks": len(self.records),
            "points": self.total,
            "expected": round(sum(r.expected for r in self.records), 1),
            "hits": sum(r.hits for r in self.records),
            "transfers": sum(len(r.transfers_in) for r in self.records[1:]),
            "auto_subs": sum(r.auto_subs for r in self.records),
            "sub_points": sum(r.sub_points for r in self.records),
            "bench_expected": round(sum(r.bench_expected for r in self.records), 1),
            "bench_points": sum(r.bench_points for r in self.records),
            "captain_points": sum(r.captain_points for r in self.records),
            "chips": chips,
            "final_value": self.records[-1].squad_value if self.records else 0,
        }

    def trace(self) -> pd.DataFrame:
        rows = []
        for r in self.records:
            rows.append(
                {
                    "season": self.season,
                    "policy": self.policy,
                    "gameweek": r.gameweek,
                    "points": r.points,
                    "raw_points": r.raw_points,
                    "expected": round(r.expected, 2),
                    "hits": r.hits,
                    "chip": r.chip or "",
                    "transfers_in": " ".join(str(e) for e in r.transfers_in),
                    "transfers_out": " ".join(str(e) for e in r.transfers_out),
                    "captain": r.captain,
                    "captain_points": r.captain_points,
                    "auto_subs": r.auto_subs,
                    "sub_points": r.sub_points,
                    "bench_expected": round(r.bench_expected, 2),
                    "bench_points": r.bench_points,
                    "bank": r.bank,
                    "free_transfers": r.free_transfers,
                    "squad_value": r.squad_value,
                }
            )
        return pd.DataFrame(rows)


# ------------------------------------------------------------------ the game's rules


def formation_is_legal(elements: list[int], positions: dict[int, str]) -> bool:
    counts: dict[str, int] = {}
    for e in elements:
        counts[positions[e]] = counts.get(positions[e], 0) + 1
    if len(elements) != milp.LINEUP_SIZE or counts.get("GKP", 0) != 1:
        return False
    return all(counts.get(pos, 0) >= need for pos, need in FORMATION_MIN.items())


def auto_substitute(
    lineup: list[int],
    bench: list[int],
    positions: dict[int, str],
    played: dict[int, bool],
) -> tuple[list[int], int]:
    """Apply FPL's automatic substitutions.

    A starter who did not play is replaced by the first substitute, in bench order, who did play
    and whose introduction leaves a legal formation. Goalkeepers only ever swap with goalkeepers.
    ``bench`` is in priority order with the goalkeeper anywhere in it.

    Returns:
        The eleven who count, and how many substitutions were made.
    """
    eleven = list(lineup)
    made = 0
    bench_keepers = [b for b in bench if positions[b] == "GKP"]
    bench_outfield = [b for b in bench if positions[b] != "GKP"]
    used: set[int] = set()

    for starter in lineup:
        if played.get(starter, False):
            continue
        if positions[starter] == "GKP":
            for keeper in bench_keepers:
                if keeper not in used and played.get(keeper, False):
                    eleven[eleven.index(starter)] = keeper
                    used.add(keeper)
                    made += 1
                    break
            continue
        for sub in bench_outfield:
            if sub in used or not played.get(sub, False):
                continue
            candidate = list(eleven)
            candidate[candidate.index(starter)] = sub
            if formation_is_legal(candidate, positions):
                eleven = candidate
                used.add(sub)
                made += 1
                break
    return eleven, made


def score_gameweek(
    lineup: list[int],
    bench: list[int],
    captain: int,
    vice: int | None,
    chip: str | None,
    positions: dict[int, str],
    points: dict[int, int],
    minutes: dict[int, int],
) -> tuple[int, int, int, int]:
    """Points a squad scored, the way the game scores it.

    Returns:
        ``(total, captain_extra, bench_points, auto_subs, sub_points)``. ``total`` includes the
        captain's extra copy and whatever the substitutes who came on scored; hits are the
        caller's business.
    """
    everyone = list(lineup) + list(bench)
    played = {e: minutes.get(e, 0) > 0 for e in everyone}

    if chip == "bboost":
        eleven, subs = everyone, 0
    else:
        eleven, subs = auto_substitute(lineup, bench, positions, played)

    base = sum(points.get(e, 0) for e in eleven)
    multiplier = 3 if chip == "3xc" else 2
    armband = captain if played.get(captain, False) else vice
    if armband is None or not played.get(armband, False) or armband not in eleven:
        extra = 0
    else:
        extra = (multiplier - 1) * points.get(armband, 0)
    benched = sum(points.get(e, 0) for e in everyone if e not in eleven)
    came_on = sum(points.get(e, 0) for e in eleven if e not in set(lineup))
    return base + extra, extra, benched, subs, came_on


def next_free_transfers(available: int, made: int, chip: str | None) -> tuple[int, int]:
    """Free transfers carried into next week, and hits paid this week.

    One is gained each week and at most five are held. A wildcard or free hit puts the week's
    transfers outside the economy: nothing is spent and the balance still grows by one.
    """
    if chip in {"wildcard", "freehit"}:
        return min(available + 1, milp.MAX_BANKED_TRANSFERS), 0
    hits = max(made - available, 0)
    return min(max(available - made, 0) + 1, milp.MAX_BANKED_TRANSFERS), hits


# ------------------------------------------------------------------ data access


def panel_expected_points(
    con, season: str, as_of_gw: int, *, source: Path | None = None
) -> pd.DataFrame:
    """The panel's expected points as of one deadline: players by target gameweek.

    Reads the warehouse table, or a season's parquet file when ``source`` is given (so a season
    can be replayed as soon as its panel worker has finished, before the final load).
    """
    if source is None:
        frame = con.execute(
            "SELECT element, target_gw, ep_mean FROM projection_panel "
            "WHERE season = ? AND as_of_gw = ?",
            [season, as_of_gw],
        ).fetchdf()
    else:
        frame = con.execute(
            "SELECT element, target_gw, ep_mean FROM read_parquet(?) "
            "WHERE season = ? AND as_of_gw = ?",
            [str(source), season, as_of_gw],
        ).fetchdf()
    if frame.empty:
        raise ValueError(f"no panel rows for {season} as of GW{as_of_gw}")
    table = frame.pivot(index="element", columns="target_gw", values="ep_mean").fillna(0.0)
    table.columns = [int(c) for c in table.columns]
    return table.sort_index()


def panel_source(season: str, version: str | None = None) -> Path | None:
    """The season's parquet file for a panel version, if the panel worker has written one.

    Panel files are named ``<season>.<version>.parquet``; a version lets two panels — say one
    built with the order-flow layer and one without — be replayed side by side while the
    warehouse table holds only the latest.
    """
    candidates = (
        [PANEL / f"{season}.{version}.parquet"]
        if version
        else sorted(PANEL.glob(f"{season}*.parquet"))
    )
    for path in candidates:
        if path.exists():
            return path
    return None


def actual_gameweek(con, season: str, gameweek: int) -> tuple[dict[int, int], dict[int, int]]:
    """Points and minutes each player recorded in a gameweek, summed over its fixtures."""
    frame = con.execute(
        "SELECT element, sum(total_points) AS points, sum(minutes) AS minutes "
        "FROM player_gw WHERE season = ? AND gw = ? GROUP BY element",
        [season, gameweek],
    ).fetchdf()
    points = dict(zip(frame["element"].astype(int), frame["points"].astype(int), strict=True))
    minutes = dict(zip(frame["element"].astype(int), frame["minutes"].astype(int), strict=True))
    return points, minutes


# ------------------------------------------------------------------ policies


def chips_remaining(state: milp.SquadState, windows: milp.ChipWindows) -> bool:
    for chip, instances in windows.windows.items():
        for window in range(len(instances)):
            if f"{chip}:{window}" not in state.chips_used:
                return True
    return False


def plan_current(
    con,
    season: str,
    gameweek: int,
    expected: pd.DataFrame,
    players: pd.DataFrame,
    state: milp.SquadState,
    windows: milp.ChipWindows,
    config: PolicyConfig,
) -> milp.Plan:
    """The live advisor's policy, minus the rival stage: roadmap the chips, then solve.

    The chip valuations in :mod:`fplass.optimise.chips` read joint samples. The panel stores
    moments, so they are handed a single "draw" holding the means: every mean gain is then exact
    (a sum of means is the mean of the sum) and only the reported upside collapses to the mean,
    which the roadmap does not schedule on.
    """
    inside, terminal = split_horizon(expected, config)
    options = {**config.solve_options(), "terminal_value": terminal}
    time_limit = config.solver_time_limit
    baseline = milp.solve(
        inside, players, state, windows, allow_chips=False, time_limit=time_limit, **options
    )
    if not chips_remaining(state, windows):
        return baseline

    samples = inside.to_numpy(dtype="float64")[None, :, :]
    roadmap = chips_module.build_roadmap(
        con,
        samples,
        inside.index.to_numpy(),
        np.array(list(inside.columns)),
        players,
        state,
        baseline,
        windows,
        season,
        min_gain=config.chip_floors,
        wildcard_candidates=config.wildcard_candidates,
        solver_time_limit=time_limit,
        solve_options=options,
    )
    if not roadmap.schedule:
        return baseline
    plan = milp.solve(
        inside,
        players,
        state,
        windows,
        chip_schedule=roadmap.schedule,
        time_limit=time_limit,
        **options,
    )
    if plan.status not in {"Optimal", "Not Solved"}:
        log.warning("%s GW%d: chip plan %s, falling back to the chip-free plan", season, gameweek, plan.status)
        return baseline
    return plan


def plan_hold(
    con,
    season: str,
    gameweek: int,
    expected: pd.DataFrame,
    players: pd.DataFrame,
    state: milp.SquadState,
    windows: milp.ChipWindows,
    config: PolicyConfig,
) -> milp.Plan:
    """Build the opening squad, then never touch it: the value of doing nothing.

    After gameweek one the solver only ever sees the fifteen players owned, so no transfer is
    possible and it just picks the eleven and the captain each week.
    """
    inside, _ = split_horizon(expected, config)
    time_limit = config.solver_time_limit
    if not state.players:
        return milp.solve(
            inside, players, state, windows, allow_chips=False, time_limit=time_limit
        )
    owned = players[players["element"].isin(state.elements)]
    frozen = milp.SquadState(
        players=dict(state.players), bank=state.bank, free_transfers=0, chips_used=set(state.chips_used)
    )
    return milp.solve(
        inside,
        owned,
        frozen,
        windows,
        gameweeks=[gameweek],
        allow_chips=False,
        time_limit=time_limit,
    )


PLANNERS = {POLICY_CURRENT: plan_current, POLICY_HOLD: plan_hold}


# ------------------------------------------------------------------ the replay


def bench_order(
    squad: list[int], lineup: list[int], expected: pd.Series, positions: dict[int, str]
) -> list[int]:
    """Substitutes in priority order: the goalkeeper first, then outfielders by expectation."""
    bench = [e for e in squad if e not in set(lineup)]
    keepers = [e for e in bench if positions[e] == "GKP"]
    outfield = sorted(
        (e for e in bench if positions[e] != "GKP"), key=lambda e: -float(expected.get(e, 0.0))
    )
    return keepers + outfield


def execute_gameweek(
    con,
    season: str,
    gameweek: int,
    plan: milp.Plan,
    expected: pd.DataFrame,
    players: pd.DataFrame,
    state: milp.SquadState,
    windows: milp.ChipWindows,
) -> tuple[GameweekRecord, milp.SquadState]:
    """Carry out one gameweek of a plan: move the money, score the team, advance the state."""
    price = dict(zip(players["element"].astype(int), players["price"].astype(int), strict=True))
    positions = dict(zip(players["element"].astype(int), players["position"], strict=True))

    chip = plan.chips.get(gameweek)
    ins = list(plan.transfers_in.get(gameweek, []))
    outs = list(plan.transfers_out.get(gameweek, []))
    lineup = list(plan.lineups[gameweek])
    squad = list(plan.squads[gameweek])

    next_ft, hits = next_free_transfers(state.free_transfers, len(ins), chip)
    if hits != plan.hits.get(gameweek, 0):
        log.warning(
            "%s GW%d: replay charges %d hit(s), the plan reported %d",
            season, gameweek, hits, plan.hits.get(gameweek, 0),
        )

    raised = sum(state.selling_price(e, price[e]) for e in outs)
    spent = sum(price[e] for e in ins)
    bank = state.bank + raised - spent
    if bank < 0:
        log.warning("%s GW%d: plan overspends by %.1fm", season, gameweek, -bank / 10)

    week = expected[gameweek] if gameweek in expected.columns else pd.Series(dtype="float64")
    bench = bench_order(squad, lineup, week, positions)
    captain = plan.captains.get(gameweek, lineup[0])
    others = [e for e in lineup if e != captain]
    vice = max(others, key=lambda e: float(week.get(e, 0.0))) if others else None

    points, minutes = actual_gameweek(con, season, gameweek)
    raw, captain_extra, benched, subs, came_on = score_gameweek(
        lineup, bench, captain, vice, chip, positions, points, minutes
    )
    bench_expected = float(sum(float(week.get(e, 0.0)) for e in bench))

    if chip == "freehit":
        # The free-hit squad plays this week; the real squad and the bank are untouched.
        holdings = dict(state.players)
        bank = state.bank
    else:
        holdings = {e: p for e, p in state.players.items() if e not in set(outs)}
        for e in ins:
            holdings[e] = price[e]

    chips_used = set(state.chips_used)
    if chip:
        for window in windows.legal(chip, gameweek):
            if f"{chip}:{window}" not in chips_used:
                chips_used.add(f"{chip}:{window}")
                break

    new_state = milp.SquadState(
        players=holdings, bank=bank, free_transfers=next_ft, chips_used=chips_used
    )
    value = bank + sum(new_state.selling_price(e, price.get(e, p)) for e, p in holdings.items())
    record = GameweekRecord(
        gameweek=gameweek,
        points=raw - int(milp.HIT_COST) * hits,
        raw_points=raw,
        hits=hits,
        chip=chip,
        transfers_in=ins,
        transfers_out=outs,
        captain=captain,
        captain_points=captain_extra,
        auto_subs=subs,
        sub_points=came_on,
        bench_points=benched,
        bench_expected=bench_expected,
        bank=bank,
        free_transfers=state.free_transfers,
        squad_value=value,
        expected=float(plan.expected_points.get(gameweek, 0.0)),
    )
    return record, new_state


def replay_season(
    con,
    season: str,
    *,
    policy: str = POLICY_CURRENT,
    config: PolicyConfig | None = None,
    max_gameweek: int | None = None,
    source: Path | None = None,
) -> SeasonReplay:
    """Play one season with a policy, from the opening squad to the final whistle."""
    planner = PLANNERS[policy]
    config = config or PolicyConfig()
    windows = milp.ChipWindows.from_warehouse(con, CURRENT_SEASON)
    calendar = panel_module.season_gameweeks(con, season)
    state = milp.SquadState(players={}, bank=STARTING_BANK, free_transfers=milp.SQUAD_SIZE)
    replay = SeasonReplay(season=season, policy=policy)
    started = time.time()

    for gameweek in calendar:
        if max_gameweek is not None and gameweek > max_gameweek:
            break
        tick = time.time()
        expected = panel_expected_points(con, season, gameweek, source=source)
        players = project.current_players(con, season, as_of_gameweek=gameweek)
        plan = planner(con, season, gameweek, expected, players, state, windows, config)
        record, state = execute_gameweek(
            con, season, gameweek, plan, expected, players, state, windows
        )
        replay.records.append(record)
        log.info(
            "%s GW%-2d %s: %3d pts (exp %5.1f) hits %d chip %-8s ft %d bank %4.1f  [%4.1fs, %.0fs]",
            season,
            gameweek,
            policy,
            record.points,
            record.expected,
            record.hits,
            record.chip or "-",
            record.free_transfers,
            record.bank / 10,
            time.time() - tick,
            time.time() - started,
        )
    return replay


def _replay_task(args: tuple) -> dict[str, object]:
    season, policy, config, max_gameweek, out_dir, version = args
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s"
    )
    logging.getLogger("fplass").setLevel(logging.WARNING)
    log.setLevel(logging.INFO)
    con = connect(read_only=True)
    try:
        replay = replay_season(
            con,
            season,
            policy=policy,
            config=config,
            max_gameweek=max_gameweek,
            source=_source_for(con, season, version),
        )
    finally:
        con.close()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tag = run_tag(policy, config, version)
    replay.trace().to_csv(out / f"manager_{tag}_{season}.csv", index=False)
    summary = replay.summary()
    summary["panel"] = version or "warehouse"
    summary["config"] = config.tag()
    return summary


def summary_from_trace(path: Path) -> dict[str, object]:
    """A season's summary row rebuilt from its trace file."""
    trace = pd.read_csv(path)
    chips = ",".join(
        f"{r.chip}@GW{r.gameweek}" for r in trace.itertuples() if isinstance(r.chip, str) and r.chip
    )
    transfers = trace["transfers_in"].iloc[1:].map(
        lambda cell: len(cell.split()) if isinstance(cell, str) else 0
    )
    return {
        "season": str(trace["season"].iloc[0]),
        "policy": str(trace["policy"].iloc[0]),
        "gameweeks": int(len(trace)),
        "points": int(trace["points"].sum()),
        "expected": round(float(trace["expected"].sum()), 1),
        "hits": int(trace["hits"].sum()),
        "transfers": int(transfers.sum()),
        "auto_subs": int(trace["auto_subs"].sum()),
        "sub_points": int(trace["sub_points"].sum()),
        "bench_expected": round(float(trace["bench_expected"].sum()), 1),
        "bench_points": int(trace["bench_points"].sum()),
        "captain_points": int(trace["captain_points"].sum()),
        "chips": chips,
        "final_value": int(trace["squad_value"].iloc[-1]),
    }


def summaries(policy: str, config: PolicyConfig, version: str | None, out_dir: Path = BACKTEST) -> pd.DataFrame:
    """Summary rows for every season trace of a run, rebuilt from the files."""
    tag = run_tag(policy, config, version)
    rows = [summary_from_trace(p) for p in sorted(out_dir.glob(f"manager_{tag}_20*.csv"))]
    table = pd.DataFrame(rows)
    if not table.empty:
        table["panel"] = version or "warehouse"
        table["config"] = config.tag()
        table.to_csv(out_dir / f"manager_{tag}_summary.csv", index=False)
    return table


def run_tag(policy: str, config: PolicyConfig, version: str | None) -> str:
    parts = [policy, config.tag()]
    if version:
        parts.append(version)
    return ".".join(parts)


def _source_for(con, season: str, version: str | None) -> Path | None:
    """A named panel version reads its parquet file; otherwise the warehouse, else any file."""
    if version:
        path = panel_source(season, version)
        if path is None:
            raise FileNotFoundError(f"no panel file for {season} version {version}")
        return path
    return None if _warehouse_has(con, season) else panel_source(season)


def _warehouse_has(con, season: str) -> bool:
    return (
        con.execute(
            "SELECT count(*) FROM projection_panel WHERE season = ?", [season]
        ).fetchone()[0]
        > 0
    )


def replay_seasons(
    seasons: list[str],
    *,
    policy: str = POLICY_CURRENT,
    config: PolicyConfig | None = None,
    max_gameweek: int | None = None,
    workers: int = 1,
    out_dir: Path = BACKTEST,
    panel_version: str | None = None,
) -> pd.DataFrame:
    """Replay several seasons, in parallel processes when asked; one summary row each.

    Traces go to ``out_dir/manager_<policy>.<config>[.<panel>]_<season>.csv``. The calling
    process must not hold the warehouse open read-write while this runs.
    """
    config = config or PolicyConfig()
    tasks = [(s, policy, config, max_gameweek, str(out_dir), panel_version) for s in seasons]
    if workers <= 1 or len(tasks) == 1:
        rows = [_replay_task(t) for t in tasks]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            rows = list(pool.map(_replay_task, tasks))
    table = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / f"manager_{run_tag(policy, config, panel_version)}_summary.csv", index=False)
    return table
