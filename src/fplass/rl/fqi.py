"""Fitted Q-iteration: the value of each action at a deadline, learned from played seasons.

The learner is a gradient-boosted regression of what an action is worth — the points of its own
gameweek plus the best that can be done from the state it leads to — on the deadline's features
and the action. It starts from the seasons' actual returns under the policies that collected
them and is refitted on its own one-step look-aheads, so each pass prices every action by the
best continuation the last pass found rather than by what the collecting policy happened to do.

Two rewards can be learned from. ``actual`` is the points the team scored, the objective; it is
also as noisy as football, and with nine seasons the learner can fit a season's luck. ``expected``
is what the deadline's projection said the fielded team would score: the planner's own
objective, free of luck, which a learner can only beat the planner on by sequencing (when to
roll, when to chip) rather than by remembering which captain hauled. Both are always evaluated
on actual points, each season held out from the model that plays it.
"""

from __future__ import annotations

import logging
import pickle
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from ..backtest.manager import WORKER_THREAD_ENV
from ..paths import RL
from . import env as env_module
from . import policies
from . import season as season_module
from .features import FEATURES, N_ACTIONS

log = logging.getLogger(__name__)

TRANSITIONS = RL / "transitions"
MODELS = RL / "models"
WEEKS_LEFT = FEATURES.index("weeks_left")


@dataclass(slots=True)
class Transitions:
    """Every deadline of every collected season, with what followed it."""

    season: np.ndarray
    episode: np.ndarray
    t: np.ndarray
    x: np.ndarray
    mask: np.ndarray
    action: np.ndarray
    reward: np.ndarray  # actual points, net of hits
    expected: np.ndarray  # projected points of the fielded team, net of hits
    x_next: np.ndarray
    mask_next: np.ndarray
    done: np.ndarray
    behaviour: np.ndarray

    def __len__(self) -> int:
        return len(self.action)

    def take(self, keep: np.ndarray) -> Transitions:
        return Transitions(**{f: getattr(self, f)[keep] for f in self.__slots__})

    @classmethod
    def concat(cls, parts: list[Transitions]) -> Transitions:
        parts = [p for p in parts if len(p)]
        offset, shifted = 0, []
        for p in parts:
            moved = p.take(np.ones(len(p), dtype=bool))
            moved.episode = p.episode + offset
            offset = int(moved.episode.max()) + 1
            shifted.append(moved)
        return cls(**{f: np.concatenate([getattr(p, f) for p in shifted]) for f in cls.__slots__})

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **{f: getattr(self, f) for f in self.__slots__})

    @classmethod
    def load(cls, path: Path) -> Transitions:
        with np.load(path, allow_pickle=False) as z:
            return cls(**{f: z[f] for f in cls.__slots__})


def from_episodes(episodes: list[env_module.Episode], behaviour: str) -> Transitions:
    rows: dict[str, list] = {f: [] for f in Transitions.__slots__}
    for number, episode in enumerate(episodes):
        obs = episode.observations
        for i, step in enumerate(episode.steps):
            last = i == len(episode.steps) - 1
            rows["season"].append(episode.season)
            rows["episode"].append(number)
            rows["t"].append(step.t)
            rows["x"].append(obs[i].vector)
            rows["mask"].append(obs[i].mask)
            rows["action"].append(step.action)
            rows["reward"].append(step.points)
            rows["expected"].append(step.expected)
            rows["x_next"].append(np.zeros_like(obs[i].vector) if last else obs[i + 1].vector)
            rows["mask_next"].append(np.zeros(N_ACTIONS, dtype=bool) if last else obs[i + 1].mask)
            rows["done"].append(last)
            rows["behaviour"].append(behaviour)
    if not rows["action"]:
        return Transitions(**{f: np.array([]) for f in Transitions.__slots__})
    return Transitions(
        season=np.array(rows["season"], dtype="U8"),
        episode=np.array(rows["episode"], dtype="int64"),
        t=np.array(rows["t"], dtype="int64"),
        x=np.vstack(rows["x"]),
        mask=np.vstack(rows["mask"]),
        action=np.array(rows["action"], dtype="int64"),
        reward=np.array(rows["reward"], dtype="float64"),
        expected=np.array(rows["expected"], dtype="float64"),
        x_next=np.vstack(rows["x_next"]),
        mask_next=np.vstack(rows["mask_next"]),
        done=np.array(rows["done"], dtype=bool),
        behaviour=np.array(rows["behaviour"], dtype="U32"),
    )


# ------------------------------------------------------------------ the value model


class QModel:
    """Q(s, a) as a boosted tree over the features and the action (a categorical column), fitted
    to the return less its mean for the weeks left, which carries most of its variance."""

    def __init__(self, regressor: HistGradientBoostingRegressor, baseline: np.ndarray, reward: str):
        self.regressor = regressor
        self.baseline = baseline
        self.reward = reward

    @staticmethod
    def design(x: np.ndarray, action: np.ndarray) -> np.ndarray:
        return np.column_stack([x, action.astype("float64")])

    def _base(self, x: np.ndarray) -> np.ndarray:
        weeks = np.clip(x[:, WEEKS_LEFT].astype(int), 0, len(self.baseline) - 1)
        return self.baseline[weeks]

    def values(self, x: np.ndarray) -> np.ndarray:
        """Q for every action: rows of ``x`` by ``N_ACTIONS``."""
        n = len(x)
        stacked = np.repeat(x, N_ACTIONS, axis=0)
        actions = np.tile(np.arange(N_ACTIONS), n)
        q = self.regressor.predict(self.design(stacked, actions)).reshape(n, N_ACTIONS)
        return q + self._base(x)[:, None]

    def action_scores(self, x: np.ndarray) -> np.ndarray:
        return self.values(x)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_model(path: Path):
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def returns_to_go(rewards: np.ndarray, episode: np.ndarray, t: np.ndarray) -> np.ndarray:
    order = np.lexsort((-t, episode))
    out = np.zeros_like(rewards, dtype="float64")
    running, current = 0.0, None
    for i in order:
        if episode[i] != current:
            running, current = 0.0, episode[i]
        running += rewards[i]
        out[i] = running
    return out


def _regressor(seed: int) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        max_iter=250, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=80,
        l2_regularization=1.0, categorical_features=[len(FEATURES)], random_state=seed,
    )


def fit(data: Transitions, *, reward: str = "actual", iterations: int = 8, seed: int = 0) -> QModel:
    """Fitted Q-iteration over a batch of transitions."""
    r = data.reward if reward == "actual" else data.expected
    returns = returns_to_go(r, data.episode, data.t)
    weeks = data.x[:, WEEKS_LEFT].astype(int)
    baseline = np.zeros(max(weeks.max() + 2, 40))
    frame = pd.DataFrame({"weeks": weeks, "g": returns}).groupby("weeks")["g"].mean()
    baseline[frame.index.to_numpy()] = frame.to_numpy()
    design = QModel.design(data.x, data.action)

    model = QModel(_regressor(seed), baseline, reward)
    model.regressor.fit(design, returns - baseline[weeks])
    for _ in range(iterations):
        nxt = model.values(data.x_next)
        nxt = np.where(data.mask_next, nxt, -np.inf).max(axis=1)
        target = r + np.where(data.done, 0.0, nxt)
        model = QModel(_regressor(seed), baseline, reward)
        model.regressor.fit(design, target - baseline[weeks])
    return model


# ------------------------------------------------------------------ collection


def behaviour(spec: str, models: dict[str, Path] | None = None):
    """A collecting policy by name: ``rule``/``rule30`` (randomised thresholds, 10% or 30%
    random actions), ``random``, or ``model:<name>`` (a fitted model, 15% random)."""
    if spec == "rule":
        return policies.Epsilon(policies.RulePolicy(), 0.1, rerandomise=True)
    if spec == "rule30":
        return policies.Epsilon(policies.RulePolicy(), 0.3, rerandomise=True)
    if spec == "random":
        return policies.RandomPolicy()
    if spec.startswith("model:"):
        name = spec.split(":", 1)[1]
        model = load_model((models or {})[name])
        return policies.Epsilon(policies.ScoredPolicy(model, name), 0.15)
    raise ValueError(f"unknown behaviour {spec}")


def _collect_task(args) -> Transitions:
    season, episodes, seed, spec, models = args
    data = season_module.read(season_module.cache_path(season, season_module.DEFAULT_PANEL))
    env = env_module.SeasonEnv(data)
    policy = behaviour(spec, models)
    rng = np.random.default_rng(seed)
    played = [env_module.play(env, policy, rng) for _ in range(episodes)]
    return from_episodes(played, spec)


def _pool(workers: int):
    import os

    for key, value in WORKER_THREAD_ENV.items():
        os.environ.setdefault(key, value)
    return ProcessPoolExecutor(max_workers=workers)


def collect(
    seasons: list[str],
    plan: dict[str, int],
    *,
    models: dict[str, Path] | None = None,
    workers: int = 6,
    seed: int = 0,
    chunk: int = 10,
) -> Transitions:
    """Play ``plan[spec]`` seasons of each behaviour on every season, in parallel chunks."""
    tasks = []
    rng = np.random.default_rng(seed)
    for season in seasons:
        for spec, count in plan.items():
            for start in range(0, count, chunk):
                tasks.append((season, min(chunk, count - start), int(rng.integers(1 << 31)), spec, models))
    started = time.time()
    with _pool(workers) as pool:
        parts = list(pool.map(_collect_task, tasks))
    data = Transitions.concat(parts)
    log.info("collected %d transitions (%d seasons played) in %.0fs",
             len(data), len(np.unique(data.episode)), time.time() - started)
    return data


# ------------------------------------------------------------------ evaluation


def _evaluate_task(args) -> pd.DataFrame:
    season, name, model_path = args
    data = season_module.read(season_module.cache_path(season, season_module.DEFAULT_PANEL))
    env = env_module.SeasonEnv(data)
    if model_path is None:
        policy = policies.RulePolicy()
    else:
        policy = policies.ScoredPolicy(load_model(model_path), name)
    episode = env_module.play(env, policy, np.random.default_rng(0), keep=False)
    return episode.trace().assign(policy=name)


def evaluate(entries: list[tuple[str, str, Path | None]], *, workers: int = 6) -> pd.DataFrame:
    """Play each (season, policy name, model path) once, greedily; None is the default rule."""
    with _pool(workers) as pool:
        return pd.concat(list(pool.map(_evaluate_task, entries)), ignore_index=True)


def season_totals(traces: pd.DataFrame) -> pd.DataFrame:
    return traces.groupby(["policy", "season"]).agg(
        points=("points", "sum"), expected=("expected", "sum"), hits=("hits", "sum"),
        transfers=("transfers", "sum"),
        chips=("chip", lambda c: ",".join(f"{x}" for x in c if x)),
    ).reset_index()


def paired(totals: pd.DataFrame, policy: str, against: str) -> dict[str, float]:
    """Mean points a season of ``policy`` over ``against``, paired by season."""
    table = totals.pivot(index="season", columns="policy", values="points")
    if policy not in table or against not in table:
        return {}
    diff = (table[policy] - table[against]).dropna()
    return {
        "policy": policy, "against": against, "seasons": len(diff),
        "mean": float(table[policy].mean()), "diff": float(diff.mean()),
        "se": float(diff.std(ddof=1) / np.sqrt(len(diff))) if len(diff) > 1 else float("nan"),
        "up": int((diff > 0).sum()),
    }


# ------------------------------------------------------------------ the held-out experiment


def model_path(reward: str, fold: str) -> Path:
    return MODELS / reward / f"{fold}.pkl"


def train_folds(
    base: Transitions,
    seasons: list[str],
    *,
    reward: str = "actual",
    iterations: int = 8,
    rounds: int = 1,
    refine_episodes: int = 12,
    workers: int = 6,
    seed: int = 0,
    models: dict[str, Path] | None = None,
) -> dict[str, Path]:
    """One model per held-out season, and one on every season (``all``) for live use.

    Each fold learns only from seasons other than the one it will play. A refinement round
    plays the fold's own policy (15% random) on its training seasons and refits on everything,
    which is where fitted Q-iteration stops being bounded by the policies that collected the
    first batch.
    """
    paths: dict[str, Path] = {}
    folds = [*seasons, "all"]
    data = {fold: base.take(base.season != fold) for fold in folds}
    for round_number in range(rounds + 1):
        for fold in folds:
            started = time.time()
            model = fit(data[fold], reward=reward, iterations=iterations, seed=seed)
            paths[fold] = model_path(reward, fold)
            model.save(paths[fold])
            log.info("%s fold %s round %d: %d transitions, %.0fs",
                     reward, fold, round_number, len(data[fold]), time.time() - started)
        if round_number == rounds:
            break
        for fold in folds:
            train = [s for s in seasons if s != fold]
            extra = collect(
                train, {f"model:{fold}": refine_episodes},
                models={**(models or {}), fold: paths[fold]}, workers=workers,
                seed=seed + 1000 * (round_number + 1) + folds.index(fold),
            )
            data[fold] = Transitions.concat([data[fold], extra])
    return paths
