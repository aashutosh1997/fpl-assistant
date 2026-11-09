"""Policies over the learner's actions: rules, exploration, a cloned elite, and a learned value.

Each has ``act(observation, rng) -> action index`` and never returns an illegal action. The rule
is the planner's logic written as thresholds — a free transfer for a swap worth three, a hit for
one worth more than the four it costs, a chip when its gain clears a floor or its window is
about to close — so randomising its thresholds per season is a cheap way to collect varied,
mostly sensible seasons to learn from.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np

from .features import ACTIONS, N_ACTIONS, ROLL, Observation, action_index


def _legal(obs: Observation, transfers: int, chip: str | None) -> int:
    """The action asked for, or the nearest legal one: fewer transfers, then no chip."""
    for k in range(transfers, -1, -1):
        index = action_index(k, chip)
        if obs.mask[index]:
            return index
    if chip is not None:
        return _legal(obs, transfers, None)
    return ROLL


@dataclass(slots=True)
class RulePolicy:
    """The planner's thresholds as rules. Gains are horizon points; chip floors week points."""

    free_gain: float = 3.0
    hit_gain: float = 8.0
    bboost: float = 14.0
    triple: float = 9.0
    wildcard: float = 25.0
    freehit: float = 20.0
    name: str = "rule"

    def chip(self, n: dict[str, float]) -> str | None:
        # A window about to close forces its chips out, the most valuable first.
        closing = n["chips_now"]
        tests = (
            ("wildcard", sum(max(n[f"gain_{i}"], 0.0) for i in range(1, 6)), self.wildcard),
            ("freehit", n["best_xi_gap"], self.freehit),
            ("bboost", n["bench_week"], self.bboost),
            ("3xc", n["captain_week"], self.triple),
        )
        for chip, value, floor in tests:
            if n[f"{chip}_now"] and (value >= floor or n[f"{chip}_window_left"] <= closing):
                return chip
        return None

    def act(self, obs: Observation, rng=None) -> int:
        n = obs.named
        chip = self.chip(n)
        if chip in ("wildcard", "freehit"):
            return _legal(obs, 0, chip)
        free = int(n["free_transfers"])
        count = 0
        for i in range(1, 4):
            gain = n[f"gain_{i}"]
            if gain > (self.free_gain if i <= free else self.hit_gain):
                count = i
            else:
                break
        return _legal(obs, count, chip)


RULE_RANGES = {
    "free_gain": (0.5, 6.0),
    "hit_gain": (5.0, 16.0),
    "bboost": (8.0, 22.0),
    "triple": (6.0, 14.0),
    "wildcard": (12.0, 45.0),
    "freehit": (10.0, 35.0),
}


def random_rule(rng: np.random.Generator) -> RulePolicy:
    return RulePolicy(**{k: float(rng.uniform(*v)) for k, v in RULE_RANGES.items()}, name="rule*")


@dataclass(slots=True)
class RandomPolicy:
    name: str = "random"

    def act(self, obs: Observation, rng: np.random.Generator) -> int:
        return int(rng.choice(np.flatnonzero(obs.mask)))


class Epsilon:
    """Another policy, with a random legal action a share of the time. A ``RandomRule`` base
    draws fresh thresholds each season."""

    def __init__(self, base, epsilon: float, *, rerandomise: bool = False):
        self.base = base
        self.epsilon = epsilon
        self.rerandomise = rerandomise
        self.name = f"{getattr(base, 'name', 'policy')}+e{epsilon:g}"

    def begin(self, rng: np.random.Generator) -> None:
        if self.rerandomise:
            self.base = random_rule(rng)
        if hasattr(self.base, "begin"):
            self.base.begin(rng)

    def act(self, obs: Observation, rng: np.random.Generator) -> int:
        if rng.random() < self.epsilon:
            return int(rng.choice(np.flatnonzero(obs.mask)))
        return int(self.base.act(obs, rng))


class ScoredPolicy:
    """Pick the legal action a model scores highest: a value per action for a Q-function, a
    probability per action for a classifier cloned from the top managers."""

    def __init__(self, model, name: str):
        self.model = model
        self.name = name

    def scores(self, obs: Observation) -> np.ndarray:
        return np.asarray(self.model.action_scores(obs.vector[None, :]))[0]

    def act(self, obs: Observation, rng=None) -> int:
        scores = np.where(obs.mask, self.scores(obs), -np.inf)
        return int(np.argmax(scores))


def describe(policy) -> str:
    if isinstance(policy, RulePolicy):
        return "rule(" + ", ".join(f"{f.name}={getattr(policy, f.name):g}" for f in fields(policy) if f.name != "name") + ")"
    return getattr(policy, "name", type(policy).__name__)


__all__ = [
    "ACTIONS", "N_ACTIONS", "RulePolicy", "RandomPolicy", "Epsilon", "ScoredPolicy",
    "random_rule", "describe",
]
