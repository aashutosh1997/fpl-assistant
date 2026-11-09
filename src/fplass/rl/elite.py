"""The top managers' decisions as a policy: what they did, from the state they were in.

The study's crawl holds, for every studied manager and deadline, the squad carried in, the bank,
the transfers made and the chip played. Rebuilding the free transfers each had (one a week,
five at most, untouched by a Wildcard or Free Hit) and the chips left, each deadline becomes the
learner's observation with the manager's choice as the label, and a classifier fitted to them
is a policy that plays like the cohort: when they roll, when they take a hit, when they chip.

It is fitted on one season's first ten deadlines, so it says little about January and May; it
is used as one of the policies that collect seasons for the learner, and played on its own as
a benchmark, not trusted as a strategy. Purchase prices are not in the crawl, so each squad
sells at the current price: a few tenths of budget at most.
"""

from __future__ import annotations

import logging
import pickle
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from ..backtest import manager
from ..backtest.manager import WORKER_THREAD_ENV
from ..optimise import milp
from ..paths import RL
from ..study import cohorts as cohorts_module
from ..study import missed
from ..study.store import attach_managers
from . import features as features_module
from . import season as season_module
from .features import FEATURES, N_ACTIONS, action_index

log = logging.getLogger(__name__)

STUDY_VERSION = "study"
STATES = RL / "elite_states.csv.gz"
MODEL = RL / "models" / "elite.pkl"
COHORTS = ("prior50k", "deadline")


def free_transfers(made: dict[int, int], chips: dict[int, str], last: int) -> dict[int, int]:
    """Free transfers held at each deadline from GW2: one to start, one more a week, five at
    most, none spent under a Wildcard or Free Hit."""
    held = {2: 1}
    for gw in range(2, last):
        held[gw + 1], _ = manager.next_free_transfers(held[gw], made.get(gw, 0), chips.get(gw))
    return held


def chips_used_before(chips: dict[int, str], gw: int, windows: milp.ChipWindows) -> set[str]:
    used: set[str] = set()
    for played, chip in sorted(chips.items()):
        if played >= gw:
            break
        for window in windows.legal(chip, played):
            if f"{chip}:{window}" not in used:
                used.add(f"{chip}:{window}")
                break
    return used


def label(made: int, chip: str | None, gw: int) -> int:
    if gw == 1:
        made = 0
    return action_index(made, chip)


def _states_task(args) -> list[dict]:
    cache, rows = args
    data = season_module.read(cache)
    out = []
    for row in rows:
        gw = row["gw"]
        t = data.calendar.index(gw)
        price = data.prices(gw)
        state = milp.SquadState(
            players={e: price.get(e, 45) for e in row["squad"]},
            bank=int(row["bank"]), free_transfers=int(row["ft"]), chips_used=set(row["used"]),
        )
        obs = features_module.observe(data, t, state)
        action = row["action"]
        adjusted = False
        if not obs.mask[action]:
            count, chip = features_module.ACTIONS[action]
            for k in range(count, -1, -1):
                candidate = action_index(k, chip)
                if obs.mask[candidate]:
                    action, adjusted = candidate, True
                    break
            else:
                action, adjusted = features_module.ROLL, True
        out.append(
            {"entry": row["entry"], "gw": gw, "weight": row["weight"], "action": action,
             "adjusted": adjusted, "legal": "".join("1" if m else "0" for m in obs.mask),
             **dict(zip(FEATURES, obs.vector, strict=True))}
        )
    return out


def build_states(
    con, season: str, *, managers: int = 3000, seed: int = 0, workers: int = 6,
    cohorts: tuple[str, ...] = COHORTS,
) -> pd.DataFrame:
    """Observations and labels for a sample of the fair cohorts' managers at every deadline."""
    attach_managers(con)
    source = missed.study_panel_path(season)
    data = season_module.load(con, season, version=STUDY_VERSION, source=source, start=False)
    cache = season_module.cache_path(season, STUDY_VERSION)
    deadlines = data.deadlines
    last = max(deadlines)

    members = cohorts_module.memberships(con, season, deadlines)
    members = members[members["cohort"].isin(cohorts)]
    members = members.groupby(["gw", "entry"], as_index=False)["weight"].max()
    rng = np.random.default_rng(seed)
    entries = members["entry"].unique()
    chosen = set(rng.choice(entries, size=min(managers, len(entries)), replace=False).tolist())
    members = members[members["entry"].isin(chosen)]

    squads = cohorts_module.load_squads(con, season, last)
    kept = squads.kept[squads.kept["entry"].isin(chosen)]
    held = {k: list(g["element"].astype(int)) for k, g in kept.groupby(["entry", "gw"])}
    history = con.execute(
        "SELECT entry, gw, bank, event_transfers, event_transfers_cost, chip "
        "FROM m.manager_gw WHERE season = ?", [season],
    ).fetchdf()
    history = history[history["entry"].isin(chosen)]

    rows, checked, agreed = [], 0, 0
    for entry, group in history.groupby("entry"):
        group = group.set_index("gw")
        made = group["event_transfers"].fillna(0).astype(int).to_dict()
        chips = group["chip"].dropna().to_dict()
        bank = group["bank"].fillna(0).astype(int).to_dict()
        ft = free_transfers(made, chips, last + 1)
        for gw in range(2, last + 1):
            if chips.get(gw) in ("wildcard", "freehit") or gw not in group.index:
                continue
            checked += 1
            paid = int(group.at[gw, "event_transfers_cost"] or 0) // int(milp.HIT_COST)
            agreed += paid == max(made.get(gw, 0) - ft.get(gw, 1), 0)
        weights = members[members["entry"] == entry].set_index("gw")["weight"]
        for gw in deadlines:
            if gw not in weights.index:
                continue
            squad = held.get((entry, gw - 1)) if gw > 1 else held.get((entry, 1))
            if not squad or len(squad) != milp.SQUAD_SIZE:
                continue
            rows.append(
                {
                    "entry": int(entry), "gw": gw, "weight": float(weights[gw]), "squad": squad,
                    "bank": bank.get(gw - 1, bank.get(gw, 0)) if gw > 1 else bank.get(1, 0),
                    "ft": ft.get(gw, 1), "used": sorted(chips_used_before(chips, gw, data.windows)),
                    "action": label(made.get(gw, 0), chips.get(gw), gw),
                }
            )
    if checked:
        log.info("free transfers rebuilt: hits agree with the record in %.1f%% of %d manager-weeks",
                 100 * agreed / checked, checked)

    started = time.time()
    chunks = [rows[i : i + 400] for i in range(0, len(rows), 400)]
    import os

    for key, value in WORKER_THREAD_ENV.items():
        os.environ.setdefault(key, value)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        parts = list(pool.map(_states_task, [(cache, c) for c in chunks]))
    states = pd.DataFrame([r for p in parts for r in p])
    log.info("elite states: %d from %d managers in %.0fs (%.1f%% of labels moved to a legal action)",
             len(states), states["entry"].nunique(), time.time() - started, 100 * states["adjusted"].mean())
    STATES.parent.mkdir(parents=True, exist_ok=True)
    states.to_csv(STATES, index=False, compression="gzip")
    return states


class EliteModel:
    """A classifier over the learner's actions; scores are its probabilities."""

    def __init__(self, classifier: HistGradientBoostingClassifier):
        self.classifier = classifier

    def action_scores(self, x: np.ndarray) -> np.ndarray:
        proba = self.classifier.predict_proba(x)
        out = np.zeros((len(x), N_ACTIONS))
        out[:, self.classifier.classes_.astype(int)] = proba
        return out

    def save(self, path: Path = MODEL) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)


def _classifier(seed: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=200, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=40,
        l2_regularization=1.0, random_state=seed,
    )


def fit(states: pd.DataFrame, *, seed: int = 0) -> tuple[EliteModel, dict[str, float]]:
    """Fit on every deadline; report how well the last deadline is predicted by the others."""
    x = states[list(FEATURES)].to_numpy(dtype="float64")
    y = states["action"].to_numpy(dtype=int)
    w = states["weight"].to_numpy(dtype="float64")
    last = states["gw"].max()
    train, test = states["gw"] < last, states["gw"] == last
    check = _classifier(seed).fit(x[train], y[train], sample_weight=w[train])
    held = EliteModel(check)
    scores = held.action_scores(x[test])
    mask = np.array([[c == "1" for c in legal] for legal in states.loc[test, "legal"]], dtype=bool)
    pred = np.argmax(np.where(mask, scores, -np.inf), axis=1)
    common = np.bincount(y[train], weights=w[train], minlength=N_ACTIONS).argmax()
    report = {
        "states": float(len(states)),
        "held_out_gw": float(last),
        "accuracy": float(np.average(pred == y[test], weights=w[test])),
        "majority": float(np.average(y[test] == common, weights=w[test])),
    }
    model = EliteModel(_classifier(seed).fit(x, y, sample_weight=w))
    model.save()
    return model, report


def describe(states: pd.DataFrame) -> pd.DataFrame:
    """The cohort's action shares at each deadline."""
    labels = {i: (f"{k}" + (f"+{c}" if c else "")) for i, (k, c) in enumerate(features_module.ACTIONS)}
    table = states.assign(choice=states["action"].map(labels)).pivot_table(
        index="gw", columns="choice", values="weight", aggfunc="sum", fill_value=0.0
    )
    return table.div(table.sum(axis=1), axis=0)
