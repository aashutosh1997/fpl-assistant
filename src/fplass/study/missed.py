"""The top managers' transfers, scored by the model that would have judged them and by what happened.

For every net transfer a cohort member made outside a Wildcard or Free Hit week, four numbers:

``model_h``
    What the model, as it stood at that deadline, expected the swap to gain over its full
    eight-week horizon: the replayed projection of the player bought minus the player sold.
    Positive is "the model backs this transfer".
``model_w`` / ``real_w``
    The model's expected gain and the actual gain over the same window, the deadline's week
    and up to three after, cut at the last gameweek played. The pair is like for like.
``alt_*``
    The model's own choice for the same sale: the best replacement by horizon projection in the
    same position, affordable on the sale price plus the manager's bank, not already owned.
    ``real_w - alt_real_w`` is whether the manager's pick beat the model's, given the same sale.

The projection is the leak-free replay of this season (``fpl backtest panel`` for 2026-27,
written under ``data/panel/study``). It is the model as the code stands now, cut at each
deadline, so it answers "would the current model have backed this?" — with one thing it cannot
see: availability flags, which history never recorded. Moves around a flagged player are tagged
from the price snapshots, so the gap is visible rather than blamed on the model's numbers.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..backtest import calibrate_live
from ..backtest import panel as panel_module
from ..options import news as news_module
from ..paths import PANEL
from . import cohorts as cohorts_module

log = logging.getLogger(__name__)

STUDY_PANEL = PANEL / "study"
HORIZON = 8
WINDOW = 4
POSITION_NAMES = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}


def study_panel_path(season: str) -> Path:
    return STUDY_PANEL / f"{season}.{panel_module.PANEL_VERSION}.parquet"


def build_study_panel(season: str, gameweeks: list[int], *, n_draws: int = 2_000) -> Path:
    """Replay this season's deadlines into a parquet file kept out of ``projection_panel``.

    The live planner's option values and ``backtest manager`` read every season in that table;
    a partial current season there would change both.
    """
    panel_module._single_threaded_environment()
    return panel_module.build_season(
        season, gameweeks=gameweeks, n_draws=n_draws, out_dir=STUDY_PANEL
    )


@dataclass(slots=True)
class Inputs:
    season: str
    last_played: int
    deadlines: list[int]
    values: pd.DataFrame  # (gw, element) -> ep_h, ep_w, real_w, weeks, p_full, minutes
    prices: pd.DataFrame  # (gw, element) -> price at that deadline
    players: pd.DataFrame  # element -> web_name, team, position, newcomer
    flags: pd.DataFrame  # (gw, element) flagged at the deadline
    squads: cohorts_module.Squads
    chips: pd.DataFrame
    members: pd.DataFrame
    banks: pd.DataFrame  # (entry, gw) -> bank after that gameweek
    crowd: pd.DataFrame  # (gw, element) -> transfers_in, transfers_out


def _values(con, season: str, panel_path: Path, last_played: int) -> tuple[pd.DataFrame, list[int]]:
    panel = con.execute(
        "SELECT as_of_gw, target_gw, element, ep_mean, p_full FROM read_parquet(?) WHERE season = ?",
        [str(panel_path), season],
    ).fetchdf()
    actual = con.execute(
        "SELECT gw, element, sum(total_points) AS points, sum(minutes) AS minutes "
        "FROM player_gw WHERE season = ? GROUP BY 1, 2",
        [season],
    ).fetchdf()
    points = actual.pivot(index="element", columns="gw", values="points").fillna(0.0)
    minutes = actual.pivot(index="element", columns="gw", values="minutes").fillna(0.0)

    deadlines = sorted(panel["as_of_gw"].unique().tolist())
    frames = []
    for t in deadlines:
        sub = panel[panel["as_of_gw"] == t]
        end = min(t + WINDOW - 1, last_played)
        horizon = sub[sub["target_gw"] < t + HORIZON].groupby("element")["ep_mean"].sum()
        frame = pd.DataFrame({"ep_h": horizon})
        frame["p_full"] = sub[sub["target_gw"] == t].groupby("element")["p_full"].max()
        if end >= t:
            weeks = [g for g in range(t, end + 1) if g in points.columns]
            frame["ep_w"] = sub[sub["target_gw"].between(t, end)].groupby("element")["ep_mean"].sum()
            frame["real_w"] = points.reindex(columns=weeks).sum(axis=1)
            frame["minutes"] = minutes.get(t)
            frame["weeks"] = len(weeks)
        else:
            frame["ep_w"] = np.nan
            frame["real_w"] = np.nan
            frame["minutes"] = np.nan
            frame["weeks"] = 0
        frame.index.name = "element"
        frame = frame.reset_index()
        frame.insert(0, "gw", t)
        frames.append(frame)
    values = pd.concat(frames, ignore_index=True)
    # A player the projection never saw scores what he scored, against an expectation of nought.
    values["ep_w"] = values["ep_w"].fillna(0.0).where(values["weeks"] > 0)
    values["real_w"] = values["real_w"].fillna(0.0).where(values["weeks"] > 0)
    return values.set_index(["gw", "element"]), deadlines


def deadline_flags(snapshots: pd.DataFrame | None, deadlines: dict[int, pd.Timestamp]) -> pd.DataFrame:
    """Each player's status at the last snapshot before each deadline.

    ``flagged`` is any status but available; ``doubt`` a chance of playing of 75% or less.
    """
    columns = ["gw", "element", "flagged", "chance"]
    if snapshots is None or snapshots.empty:
        return pd.DataFrame(columns=columns)
    frame = snapshots[["ts", "element", "status", "chance_of_playing_next_round"]].copy()
    frame["ts"] = pd.to_datetime(frame["ts"], utc=True, format="ISO8601", errors="coerce")
    frame = frame.dropna(subset=["ts"]).sort_values("ts")
    rows = []
    for gw, deadline in sorted(deadlines.items()):
        before = frame[frame["ts"] <= deadline]
        if before.empty or (deadline - before["ts"].max()) > pd.Timedelta(days=2):
            continue
        last = before.groupby("element").last()
        rows.append(
            pd.DataFrame(
                {
                    "gw": gw,
                    "element": last.index.astype(int),
                    "flagged": (last["status"].fillna("a") != "a").to_numpy(),
                    "chance": pd.to_numeric(last["chance_of_playing_next_round"], errors="coerce").to_numpy(),
                }
            )
        )
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=columns)


def load_inputs(con, season: str, *, me: int | None = None, panel_path: Path | None = None) -> Inputs:
    """Everything the study reads, from the warehouse and the attached managers database."""
    panel_path = panel_path or study_panel_path(season)
    if not panel_path.exists():
        raise FileNotFoundError(f"no study panel at {panel_path}; run `fpl study panel` first")
    last_played = con.execute(
        "SELECT max(gw) FROM player_gw WHERE season = ?", [season]
    ).fetchone()[0]
    values, deadlines = _values(con, season, panel_path, last_played)

    prices = con.execute(
        "SELECT gw, element, any_value(value) AS price FROM player_gw WHERE season = ? GROUP BY 1, 2",
        [season],
    ).fetchdf()
    players = con.execute(
        """
        SELECT pl.element, pl.web_name, t.short_name AS team, pl.element_type, pl.start_cost,
               pl.code NOT IN (
                   SELECT DISTINCT code FROM players WHERE season < ? AND code IS NOT NULL
               ) AS newcomer
        FROM players pl LEFT JOIN teams t ON t.season = pl.season AND t.team_id = pl.team_id
        WHERE pl.season = ?
        """,
        [season, season],
    ).fetchdf()
    players["position"] = players["element_type"].map(POSITION_NAMES)
    # A deadline's price: the gameweek's recorded value, carried forward to a deadline with no
    # row yet, and the starting price before any is recorded.
    grid = pd.MultiIndex.from_product([deadlines, players["element"]], names=["gw", "element"])
    priced = prices.set_index(["gw", "element"])["price"].reindex(grid)
    priced = priced.groupby(level="element").ffill()
    start = players.set_index("element")["start_cost"]
    priced = priced.fillna(
        pd.Series(grid.get_level_values("element").map(start).to_numpy(), index=grid)
    )
    prices = priced.rename("price").reset_index()

    flags = deadline_flags(
        calibrate_live.load_price_snapshots(), news_module.deadlines_from_warehouse(con, season)
    )
    crowd = con.execute(
        "SELECT gw, element, any_value(transfers_in) AS transfers_in, "
        "any_value(transfers_out) AS transfers_out, any_value(selected) AS selected "
        "FROM player_gw WHERE season = ? GROUP BY 1, 2",
        [season],
    ).fetchdf()

    squads = cohorts_module.load_squads(con, season, max(deadlines))
    chips = cohorts_module.chip_table(con, season)
    members = cohorts_module.memberships(con, season, deadlines, me=me)
    banks = con.execute(
        "SELECT entry, gw, bank FROM m.manager_gw WHERE season = ?", [season]
    ).fetchdf()
    return Inputs(
        season=season,
        last_played=int(last_played),
        deadlines=deadlines,
        values=values,
        prices=prices,
        players=players.set_index("element"),
        flags=flags,
        squads=squads,
        chips=chips,
        members=members,
        banks=banks,
        crowd=crowd,
    )


# --------------------------------------------------------------------------------- scoring


def score_moves(moves: pd.DataFrame, values: pd.DataFrame) -> pd.DataFrame:
    """Attach the model's and the actual gain to each (gw, element_out, element_in) move."""
    out = values.add_suffix("_out")
    into = values.add_suffix("_in")
    scored = moves.join(out, on=["gw", "element_out"]).join(into, on=["gw", "element_in"])
    for column in ("ep_h", "ep_w", "real_w"):
        for side in ("_out", "_in"):
            if side == "_out" or column != "real_w":
                scored[column + side] = scored[column + side].fillna(0.0)
    scored["model_h"] = scored["ep_h_in"] - scored["ep_h_out"]
    scored["model_w"] = scored["ep_w_in"] - scored["ep_w_out"]
    scored["real_w"] = scored["real_w_in"] - scored["real_w_out"]
    scored["weeks"] = scored["weeks_in"].fillna(scored["weeks_out"])
    return scored


def model_alternatives(scored: pd.DataFrame, inputs: Inputs) -> pd.DataFrame:
    """The model's best replacement for each sale, on the manager's budget at the deadline.

    Budget is the sold player's price at the deadline plus the bank carried in; the sell-on fee
    and the club limit are ignored, so the model's choice is if anything a little too generous.
    """
    prices = inputs.prices.set_index(["gw", "element"])["price"]
    position = inputs.players["element_type"]
    bank = inputs.banks.set_index(["entry", "gw"])["bank"]
    kept = {
        k: set(g["element"])
        for k, g in inputs.squads.kept.groupby(["entry", "gw"])
    }
    ranked: dict[tuple[int, int], list[tuple[int, float, float, float]]] = {}
    values = inputs.values.reset_index()
    values = values.join(prices, on=["gw", "element"])
    values["element_type"] = values["element"].map(position)
    for (gw, kind), group in values.dropna(subset=["price", "element_type"]).groupby(
        ["gw", "element_type"]
    ):
        group = group.sort_values("ep_h", ascending=False)
        ranked[(int(gw), int(kind))] = list(
            zip(group["element"], group["price"], group["ep_h"], group["real_w"], strict=True)
        )

    alt_in, alt_h, alt_real = [], [], []
    for row in scored.itertuples(index=False):
        kind = position.get(row.element_out)
        budget = prices.get((row.gw, row.element_out), np.nan) + bank.get((row.entry, row.gw - 1), 0)
        owned = kept.get((row.entry, row.gw - 1), set())
        choice = None
        for element, price, ep_h, real_w in ranked.get((row.gw, kind), []):
            if price <= budget and element not in owned:
                choice = (element, ep_h, real_w)
                break
        if choice is None:
            alt_in.append(np.nan)
            alt_h.append(np.nan)
            alt_real.append(np.nan)
        else:
            alt_in.append(choice[0])
            alt_h.append(choice[1] - row.ep_h_out)
            alt_real.append(choice[2] - row.real_w_out if pd.notna(choice[2]) else np.nan)
    scored = scored.copy()
    scored["alt_in"] = alt_in
    scored["alt_model_h"] = alt_h
    scored["alt_real_w"] = alt_real
    return scored


def scored_moves(inputs: Inputs) -> pd.DataFrame:
    """Every cohort member's net transfers, outside chip weeks, scored."""
    moves = cohorts_module.weighted_moves(inputs.members, inputs.squads, inputs.chips)
    moves = moves[moves["gw"].isin(inputs.deadlines)]
    scored = score_moves(moves, inputs.values)
    scored = model_alternatives(scored, inputs)
    flags = inputs.flags.set_index(["gw", "element"])["flagged"] if not inputs.flags.empty else None
    if flags is not None:
        scored["flag_out"] = [
            bool(flags.get((g, e), False)) for g, e in zip(scored["gw"], scored["element_out"], strict=True)
        ]
        scored["flag_in"] = [
            bool(flags.get((g, e), False)) for g, e in zip(scored["gw"], scored["element_in"], strict=True)
        ]
    else:
        scored["flag_out"] = False
        scored["flag_in"] = False
    scored["newcomer_in"] = scored["element_in"].map(inputs.players["newcomer"]).fillna(False)
    return scored


# -------------------------------------------------------------------------------- summaries


def _wmean(values: pd.Series, weights: pd.Series) -> float:
    mask = values.notna()
    total = weights[mask].sum()
    return float((values[mask] * weights[mask]).sum() / total) if total > 0 else float("nan")


def _gw_bootstrap(frame: pd.DataFrame, stat, n: int = 500, seed: int = 7) -> float:
    """Standard error of ``stat(frame)`` resampling whole gameweeks."""
    weeks = frame["gw"].unique()
    if len(weeks) < 2:
        return float("nan")
    rng = np.random.default_rng(seed)
    groups = {g: frame[frame["gw"] == g] for g in weeks}
    draws = []
    for _ in range(n):
        sample = pd.concat([groups[g] for g in rng.choice(weeks, size=len(weeks))], ignore_index=True)
        draws.append(stat(sample))
    return float(np.nanstd(draws, ddof=1))


def _fit(frame: pd.DataFrame) -> tuple[float, float]:
    """Weighted least squares of actual on modelled gain: (intercept, slope)."""
    sub = frame.dropna(subset=["model_w", "real_w"])
    if len(sub) < 3 or sub["model_w"].std() == 0:
        return float("nan"), float("nan")
    slope, intercept = np.polyfit(sub["model_w"], sub["real_w"], 1, w=np.sqrt(sub["weight"]))
    return float(intercept), float(slope)


def verdicts(scored: pd.DataFrame) -> pd.DataFrame:
    """Per cohort: how the model judged its transfers, and how they turned out.

    Every gain is per transfer over the matched window (the deadline's week and up to three
    after), weighted by the managers each member stands for.
    """
    played = scored[scored["weeks"] > 0]
    rows = []
    for cohort, group in played.groupby("cohort"):
        w = group["weight"]
        backed = group["model_h"] > 0
        worked = group["real_w"] > 0
        intercept, slope = _fit(group)
        rows.append(
            {
                "cohort": cohort,
                "moves": len(group),
                "managers": group["entry"].nunique(),
                "model_backed": _wmean(backed.astype(float), w),
                "worked": _wmean(worked.astype(float), w),
                "backed_and_worked": _wmean((backed & worked).astype(float), w),
                "opposed_and_worked": _wmean((~backed & worked).astype(float), w),
                "model_gain": _wmean(group["model_w"], w),
                "actual_gain": _wmean(group["real_w"], w),
                "actual_gain_se": _gw_bootstrap(group, lambda f: _wmean(f["real_w"], f["weight"])),
                "beat_model_pick": _wmean(group["real_w"] - group["alt_real_w"], w),
                "beat_model_pick_se": _gw_bootstrap(
                    group, lambda f: _wmean(f["real_w"] - f["alt_real_w"], f["weight"])
                ),
                "same_as_model_pick": _wmean((group["element_in"] == group["alt_in"]).astype(float), w),
                "slope": slope,
                "intercept": intercept,
            }
        )
    return pd.DataFrame(rows).set_index("cohort")


def crowd_baseline(inputs: Inputs) -> dict[str, float]:
    """What the average manager's transfers gained over the same windows.

    Buys weighted by ``transfers_in``, sales by ``transfers_out``, for every deadline with a
    played window. The crowd's numbers include Wildcard and Free Hit transfers, which the
    cohorts' do not, and are an unpaired average; across many transfers the two agree, because
    each transfer swaps players of one position.
    """
    values = inputs.values.reset_index()
    crowd = inputs.crowd.merge(values, on=["gw", "element"], how="inner")
    crowd = crowd[crowd["weeks"] > 0].dropna(subset=["transfers_in", "transfers_out"])
    per_week = []
    for gw, group in crowd.groupby("gw"):
        tin, tout = group["transfers_in"].astype(float), group["transfers_out"].astype(float)
        if tin.sum() <= 0 or tout.sum() <= 0:
            continue
        per_week.append(
            {
                "gw": gw,
                "actual_gain": (group["real_w"] * tin).sum() / tin.sum()
                - (group["real_w"] * tout).sum() / tout.sum(),
                "model_gain": (group["ep_w"] * tin).sum() / tin.sum()
                - (group["ep_w"] * tout).sum() / tout.sum(),
                "transfers": tin.sum(),
            }
        )
    table = pd.DataFrame(per_week)
    if table.empty:
        return {}
    weights = table["transfers"]
    return {
        "actual_gain": float((table["actual_gain"] * weights).sum() / weights.sum()),
        "model_gain": float((table["model_gain"] * weights).sum() / weights.sum()),
    }


def consensus(scored: pd.DataFrame, cohort: str, gw: int, *, top: int = 8) -> pd.DataFrame:
    """A cohort's most common moves at one deadline, with the model's and the actual verdict."""
    group = scored[(scored["cohort"] == cohort) & (scored["gw"] == gw)]
    if group.empty:
        return group
    table = group.groupby(["element_out", "element_in"]).agg(
        weight=("weight", "sum"),
        managers=("entry", "nunique"),
        model_h=("model_h", "first"),
        model_w=("model_w", "first"),
        real_w=("real_w", "first"),
        alt_in=("alt_in", lambda s: s.mode().iloc[0] if s.notna().any() else np.nan),
        flag_out=("flag_out", "first"),
    )
    return table.sort_values("weight", ascending=False).head(top).reset_index()


def blind_spots(scored: pd.DataFrame, inputs: Inputs, cohorts: tuple[str, ...]) -> pd.DataFrame:
    """Why the model opposed transfers that worked, for the fair cohorts.

    One cause per move, first match wins: the sold player was flagged at the deadline (news the
    replay cannot see); the player bought is new to the league (his rates are stand-ins); the
    model's chance of an hour was wrong for either player in the deadline's week; otherwise the
    points simply beat the projection.
    """
    sub = scored[
        scored["cohort"].isin(cohorts) & (scored["weeks"] > 0)
        & (scored["model_h"] <= 0) & (scored["real_w"] > 0)
    ].copy()
    if sub.empty:
        return pd.DataFrame()
    minutes_miss = (
        ((sub["p_full_in"].fillna(0) < 0.5) & (sub["minutes_in"].fillna(0) >= 60))
        | ((sub["p_full_out"].fillna(0) >= 0.5) & (sub["minutes_out"].fillna(0) < 60))
    )
    sub["cause"] = np.select(
        [sub["flag_out"], sub["newcomer_in"].astype(bool), minutes_miss],
        ["news on the player sold", "player bought new to the league", "minutes"],
        default="points beat the projection",
    )
    table = sub.groupby(["cohort", "cause"]).apply(
        lambda g: pd.Series(
            {
                "moves": len(g),
                "weight": g["weight"].sum(),
                "actual_gain": _wmean(g["real_w"], g["weight"]),
                "model_gain": _wmean(g["model_w"], g["weight"]),
            }
        ),
        include_groups=False,
    )
    totals = scored[scored["cohort"].isin(cohorts) & (scored["weeks"] > 0)].groupby("cohort")["weight"].sum()
    table["share_of_all_moves"] = table["weight"] / table.index.get_level_values("cohort").map(totals)
    return table.reset_index()


def our_season(scored: pd.DataFrame, inputs: Inputs, me: int, cohorts: tuple[str, ...]) -> pd.DataFrame:
    """Per deadline: the fair cohorts' most common sales of players we held, and what we did.

    A move is "available to us" when we held the player sold going into the deadline and did
    not hold the player bought. ``we_did`` says whether we made that exact move.
    """
    kept = {k: set(g["element"]) for k, g in inputs.squads.kept.groupby(["entry", "gw"])}
    ours = scored[(scored["cohort"] == "us")]
    rows = []
    for gw in inputs.deadlines:
        held = kept.get((me, gw - 1), set())
        our_moves = {(o, i) for o, i in zip(
            ours.loc[ours["gw"] == gw, "element_out"], ours.loc[ours["gw"] == gw, "element_in"], strict=True
        )}
        for cohort in cohorts:
            group = scored[(scored["cohort"] == cohort) & (scored["gw"] == gw)]
            if group.empty:
                continue
            size = float(inputs.members[(inputs.members["cohort"] == cohort) & (inputs.members["gw"] == gw)]["weight"].sum())
            table = group.groupby(["element_out", "element_in"]).agg(
                weight=("weight", "sum"),
                model_h=("model_h", "first"),
                real_w=("real_w", "first"),
                weeks=("weeks", "first"),
                flag_out=("flag_out", "first"),
            ).reset_index()
            table["share"] = table["weight"] / size if size else np.nan
            table = table[table["element_out"].isin(held)]
            table = table.sort_values("weight", ascending=False).head(3)
            for r in table.itertuples(index=False):
                rows.append(
                    {
                        "gw": gw, "cohort": cohort, "out": int(r.element_out), "in": int(r.element_in),
                        "share": r.share, "model_h": r.model_h, "real_w": r.real_w,
                        "weeks": r.weeks, "flag_out": r.flag_out,
                        "we_did": (r.element_out, r.element_in) in our_moves,
                    }
                )
    return pd.DataFrame(rows)


def name(inputs: Inputs, element) -> str:
    if pd.isna(element):
        return "-"
    element = int(element)
    if element not in inputs.players.index:
        return str(element)
    row = inputs.players.loc[element]
    return f"{row['web_name']} ({row['team']})"
