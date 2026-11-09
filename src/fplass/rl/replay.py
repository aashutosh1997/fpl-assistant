"""The learner's choices executed by the planner's solver, scored as the planner is scored.

At each deadline the policy sees the same observation as in the fast environment and picks a
transfer count and a chip. The solver then plans the usual eight weeks with that week pinned:
exactly that many transfers (any number under a Wildcard or Free Hit), that chip and no other
this week, and no chips in the weeks after, which are the policy's to decide when it gets there.
The plan's first week is carried out by :func:`fplass.backtest.manager.execute_gameweek`, the
function every replay of the planner goes through, so the two totals differ only by the
decisions.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from ..backtest import manager
from ..ingest.warehouse import connect
from ..optimise import milp
from ..optimise.policy import PolicyConfig, split_horizon
from ..paths import BACKTEST
from . import fqi, policies
from . import season as season_module
from .features import ACTIONS, observe

log = logging.getLogger(__name__)

PLANNER_BASELINE = BACKTEST / "manager_current.banked_transfer_value=3.0,chips=continuation.panel.5_summary.csv"


def _policy(spec: tuple[str, str | None]):
    name, path = spec
    if path is None:
        return policies.RulePolicy()
    return policies.ScoredPolicy(fqi.load_model(Path(path)), name)


def replay_season(con, season: str, spec: tuple[str, str | None], *, config: PolicyConfig | None = None) -> manager.SeasonReplay:
    config = config or PolicyConfig()
    data = season_module.read(season_module.cache_path(season, season_module.DEFAULT_PANEL))
    policy = _policy(spec)
    rng = np.random.default_rng(0)
    state = milp.SquadState(players={}, bank=manager.STARTING_BANK, free_transfers=milp.SQUAD_SIZE)
    replay = manager.SeasonReplay(season=season, policy=spec[0])
    started = time.time()
    for t, gw in enumerate(data.calendar):
        expected = data.ep[gw]
        players = data.players[gw]
        seen = state if t else milp.SquadState(players=dict(data.start), bank=data.start_bank, free_transfers=1)
        obs = observe(data, t, seen)
        count, chip = ACTIONS[int(policy.act(obs, rng))]
        inside, terminal = split_horizon(expected, config)
        options = {**config.solve_options(), "terminal_value": terminal}
        pinned = None if t == 0 or chip in ("wildcard", "freehit") else {gw: count}
        plan = milp.solve(
            inside, players, state, data.windows, chip_schedule={gw: chip} if chip else {},
            transfers=pinned, time_limit=config.solver_time_limit, **options,
        )
        if plan.status not in {"Optimal", "Not Solved"} or gw not in plan.squads:
            log.warning("%s GW%d: %s with %s transfers is %s; solving without the count",
                        season, gw, chip or "no chip", count, plan.status)
            plan = milp.solve(
                inside, players, state, data.windows, chip_schedule={gw: chip} if chip else {},
                time_limit=config.solver_time_limit, **options,
            )
        record, state = manager.execute_gameweek(con, season, gw, plan, expected, players, state, data.windows)
        replay.records.append(record)
        log.info("%s GW%-2d %s: %3d pts, %d transfers%s [%.0fs]", season, gw, spec[0], record.points,
                 len(record.transfers_in) if t else 0, f" {chip}" if chip else "", time.time() - started)
    return replay


def _task(args) -> dict[str, object]:
    season, spec, out_dir = args
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    logging.getLogger("fplass").setLevel(logging.WARNING)
    log.setLevel(logging.INFO)
    con = connect(read_only=True)
    con.execute("SET threads TO 1")
    try:
        replay = replay_season(con, season, spec)
    finally:
        con.close()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    replay.trace().to_csv(out / f"manager_rl.{spec[0]}_{season}.csv", index=False)
    return replay.summary()


def replay_seasons(
    specs: dict[str, tuple[str, str | None]], *, workers: int = 6, out_dir: Path = BACKTEST
) -> pd.DataFrame:
    """Replay each season with its own policy (``season -> (name, model path)``)."""
    import os

    for key, value in manager.WORKER_THREAD_ENV.items():
        os.environ.setdefault(key, value)
    tasks = [(season, spec, str(out_dir)) for season, spec in specs.items()]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(_task, tasks))
    table = pd.DataFrame(rows)
    name = next(iter(specs.values()))[0]
    table.to_csv(out_dir / f"manager_rl.{name}_summary.csv", index=False)
    return table


def against_planner(table: pd.DataFrame, baseline: Path = PLANNER_BASELINE) -> dict[str, float]:
    """Points a season over the planner's own replay, paired by season."""
    planner = pd.read_csv(baseline).set_index("season")["points"]
    ours = table.set_index("season")["points"]
    diff = (ours - planner.reindex(ours.index)).dropna()
    return {
        "seasons": len(diff), "ours": float(ours.mean()), "planner": float(planner.reindex(ours.index).mean()),
        "diff": float(diff.mean()), "se": float(diff.std(ddof=1) / np.sqrt(len(diff))) if len(diff) > 1 else float("nan"),
        "up": int((diff > 0).sum()),
    }
