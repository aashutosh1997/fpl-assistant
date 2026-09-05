"""The planner's knobs, shared by the live advisor and the ten-season replay.

Every constant here was once typed into a function signature. Now it is a field on one object
that both `fpl plan` and `fpl backtest manager` read, so the replay judges exactly the planner
the advisor runs, and a value that earns its place in the replay changes the live plan by
changing one default.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import logging

import pandas as pd

from ..paths import MEASURED_CHIP_GAINS
from . import chips as chips_module
from . import milp

log = logging.getLogger(__name__)

# Weeks beyond the planning horizon whose projection feeds the terminal value.
TERMINAL_WEEKS = 4


@dataclass(slots=True)
class PolicyConfig:
    """The knobs of the planner, so a replay can measure what each is worth."""

    horizon: int = 8
    bench_weight: float = milp.DEFAULT_BENCH_WEIGHT
    banked_transfer_value: float = milp.DEFAULT_BANKED_TRANSFER_VALUE
    terminal_beta: float = 0.0  # share of the projection beyond the horizon credited at its end
    terminal_bank_value: float = 0.0  # points per 0.1m left in the bank at the horizon's end
    chip_floors: dict[str, float] | None = None  # None: the roadmap's defaults
    # "floors": flat floors. "continuation": the measured expected-best-later thresholds from a
    # chip-gains file (see `fpl backtest chips`), fitted with the replayed season left out.
    # Continuation is the default: replayed over nine seasons it scored 43 points a season more
    # than the floors (seven seasons of nine up), because a flat floor leaves chips unplayed
    # while the rule plays each by the end of its window for what it is then worth.
    chip_rule: str = "continuation"
    chip_gains: str | None = str(MEASURED_CHIP_GAINS)
    wildcard_candidates: int = 1
    # A backstop only: the solver stops on the MIP gap, so replays are load-independent.
    solver_time_limit: int = 90

    @classmethod
    def parse(cls, spec: str | None) -> PolicyConfig:
        """``"bench_weight=0.3,terminal_beta=0.5"`` -> a config; chip floors as ``floor:3xc=8``."""
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
            if current is None:
                setattr(config, key, raw if key in ("chip_gains",) else float(raw))
            else:
                setattr(config, key, type(current)(raw))
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
        if self.chip_rule != "floors":
            parts.append(f"chips={self.chip_rule}")
        return ",".join(parts) or "default"

    def projection_horizon(self) -> int:
        """Gameweeks to project: the horizon, plus the weeks a terminal value reads."""
        return self.horizon + (TERMINAL_WEEKS if self.terminal_beta else 0)

    def thresholds(
        self, windows: milp.ChipWindows, season: str
    ) -> dict[str, float] | Callable[[str, int], float] | None:
        """The chip floor to hand the roadmap, for one season."""
        if self.chip_rule != "continuation":
            return self.chip_floors
        from ..options import chips as chip_options

        path = self.chip_gains
        if not path or not __import__("pathlib").Path(path).exists():
            log.warning(
                "chip gains file %s not found; chips fall back to flat floors", path
            )
            return self.chip_floors
        gains = pd.read_csv(path)
        return chip_options.ContinuationThresholds.from_gains(
            gains, windows, exclude_season=season, floors=self.chip_floors
        ).floor_for


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
