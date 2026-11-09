"""Everything a replayed season needs, read once and cached.

A season of the fast environment touches the warehouse dozens of times per gameweek if it asks
for what it needs as it goes. Here it is read once per season — the projection at every deadline,
the player pool and prices, what everyone scored, how many fixtures each club has — and pickled,
so a worker process plays hundreds of seasons from a file it loads in a fraction of a second.
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from ..backtest import manager
from ..backtest import panel as panel_module
from ..ingest.sources import CURRENT_SEASON
from ..optimise import chips as chips_module
from ..optimise import milp
from ..optimise.policy import PolicyConfig, split_horizon
from ..paths import RL
from ..sim import project

log = logging.getLogger(__name__)

CACHE = RL / "cache"
DEFAULT_PANEL = "panel.5"
REPLAY_SEASONS = (
    "2017-18", "2018-19", "2019-20", "2020-21", "2021-22",
    "2022-23", "2023-24", "2024-25", "2025-26",
)
PLAYER_COLUMNS = ["element", "web_name", "position", "team_id", "price"]


@dataclass(slots=True)
class SeasonData:
    """One season's deadlines: what the model expected, what it cost, and what happened."""

    season: str
    calendar: list[int]
    ep: dict[int, pd.DataFrame]  # deadline -> players x target gameweek
    players: dict[int, pd.DataFrame]  # deadline -> PLAYER_COLUMNS
    points: dict[int, dict[int, int]]  # gameweek -> element -> points
    minutes: dict[int, dict[int, int]]
    fixtures: dict[int, dict[int, int]]  # gameweek -> club -> fixtures
    windows: milp.ChipWindows
    position_of: dict[int, str] = field(default_factory=dict)  # positions never change in a season
    start: dict[int, int] = field(default_factory=dict)  # the GW1 squad, element -> price
    start_bank: int = manager.STARTING_BANK
    horizon: int = 8

    def week(self, gw: int) -> pd.Series:
        """The deadline's projection for its own gameweek."""
        table = self.ep[gw]
        return table[gw] if gw in table.columns else pd.Series(0.0, index=table.index)

    def horizon_ep(self, gw: int, weeks: int | None = None) -> pd.Series:
        """The deadline's projection summed over the planning horizon."""
        table = self.ep[gw]
        return table[list(table.columns)[: weeks or self.horizon]].sum(axis=1)

    def index(self, gw: int) -> int:
        return self.calendar.index(gw)

    @property
    def deadlines(self) -> list[int]:
        """The gameweeks the panel has a projection for: all of them, for a finished season."""
        return [g for g in self.calendar if g in self.ep]

    def prices(self, gw: int) -> dict[int, int]:
        frame = self.players[gw]
        return dict(zip(frame["element"].astype(int), frame["price"].astype(int), strict=True))


def cache_path(season: str, version: str) -> Path:
    return CACHE / f"{season}.{version}.pkl"


def _panel_frame(con, source: Path, season: str) -> pd.DataFrame:
    return con.execute(
        "SELECT element, as_of_gw, target_gw, ep_mean FROM read_parquet(?) WHERE season = ?",
        [str(source), season],
    ).fetchdf()


def build(
    con,
    season: str,
    *,
    version: str = DEFAULT_PANEL,
    source: Path | None = None,
    config: PolicyConfig | None = None,
    start: bool = True,
) -> SeasonData:
    """Read a season from the warehouse and its panel file.

    ``start`` solves the gameweek-one squad the way the planner builds it without chips, so
    every policy in the fast environment starts from the same fifteen.
    """
    config = config or PolicyConfig()
    source = source or manager.panel_source(season, version)
    if source is None:
        raise FileNotFoundError(f"no panel file for {season} version {version}")
    # The whole season's calendar, even where the panel stops (the current season): the
    # features count weeks and chip windows left against it.
    calendar = panel_module.season_gameweeks(con, season)
    frame = _panel_frame(con, source, season)
    ep: dict[int, pd.DataFrame] = {}
    for gw, group in frame.groupby("as_of_gw"):
        if int(gw) not in calendar:
            continue
        table = group.pivot(index="element", columns="target_gw", values="ep_mean").fillna(0.0)
        table.columns = [int(c) for c in table.columns]
        ep[int(gw)] = table.sort_index()
    deadlines = [g for g in calendar if g in ep]

    players = {
        gw: project.current_players(con, season, as_of_gameweek=gw)[PLAYER_COLUMNS]
        .assign(price=lambda f: f["price"].astype(int))
        .reset_index(drop=True)
        for gw in deadlines
    }
    position_of: dict[int, str] = {}
    for frame_gw in players.values():
        position_of.update(zip(frame_gw["element"].astype(int), frame_gw["position"], strict=True))
    points: dict[int, dict[int, int]] = {}
    minutes: dict[int, dict[int, int]] = {}
    for gw in deadlines:
        points[gw], minutes[gw] = manager.actual_gameweek(con, season, gw)
    counts = chips_module.fixture_counts(con, season, calendar)
    fixtures = {
        int(gw): dict(zip(g["team"].astype(int), g["fixtures"].astype(int), strict=True))
        for gw, g in counts.groupby("event")
    }
    windows = milp.ChipWindows.from_warehouse(con, CURRENT_SEASON)
    data = SeasonData(
        season=season, calendar=calendar, ep=ep, players=players, points=points,
        minutes=minutes, fixtures=fixtures, windows=windows, position_of=position_of,
        horizon=config.horizon,
    )
    if start:
        data.start, data.start_bank = opening_squad(data, config)
    return data


def opening_squad(data: SeasonData, config: PolicyConfig) -> tuple[dict[int, int], int]:
    """The planner's gameweek-one squad without chips: element -> price, and the bank left."""
    gw = data.calendar[0]
    inside, terminal = split_horizon(data.ep[gw], config)
    plan = milp.solve(
        inside,
        data.players[gw],
        milp.SquadState(players={}, bank=manager.STARTING_BANK, free_transfers=milp.SQUAD_SIZE),
        data.windows,
        allow_chips=False,
        time_limit=config.solver_time_limit,
        terminal_value=terminal,
        **config.solve_options(),
    )
    price = data.prices(gw)
    squad = {int(e): price[int(e)] for e in plan.squads[gw]}
    return squad, manager.STARTING_BANK - sum(squad.values())


def load(con, season: str, *, version: str = DEFAULT_PANEL, refresh: bool = False, **kwargs) -> SeasonData:
    """A season from the cache, building it on a miss."""
    path = cache_path(season, version)
    if path.exists() and not refresh:
        return read(path)
    data = build(con, season, version=version, **kwargs)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    log.info("cached %s (%d deadlines) -> %s", season, len(data.calendar), path)
    return data


def read(path: Path) -> SeasonData:
    with Path(path).open("rb") as handle:
        return pickle.load(handle)
