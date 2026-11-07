"""Who counts as a top manager, and what their squads were at every deadline.

**Squads.** A squad is the first gameweek's picks with every later transfer applied in order.
The Free Hit is the one wrinkle: its transfers are in the feed like any others (the top manager's
GW5 Free Hit shows seven), but the squad reverts the week after, so those transfers shape that
gameweek's team and nothing after it. Wildcard transfers are permanent. The rebuild is checked
against each manager's latest picks.

**Cohorts.** Judging transfers by the managers at the top today flatters the transfers, because
those managers are at the top partly *because* their transfers worked. So the cohorts used for
any test are chosen without the results being tested:

``prior10k`` / ``prior50k``
    Last season's final rank inside 10,000 / 50,000. Fixed before this season began.
``deadline``
    Inside the top 10,000 after the gameweek *before* the deadline being studied. Known at the
    deadline; a manager who has since fallen below the crawl's reach is missing.

and two that are descriptive only:

``now``
    Today's top 10,000. What the leaders did, with the hindsight that implies.
``league``
    The mini-leagues being played.

Every manager carries the weight the crawl gave them (a band manager stands for ``band_every``
managers), so a cohort's shares are estimates for the whole population it describes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from .store import attach_managers

log = logging.getLogger(__name__)

COHORTS = ("prior10k", "prior50k", "deadline", "now", "league")
# The cohorts whose membership does not lean on the results being judged.
FAIR_COHORTS = ("prior10k", "prior50k", "deadline")
SQUAD_CHIPS = ("wildcard", "freehit")


@dataclass(slots=True)
class Squads:
    """Every manager's squad by gameweek, and how well the rebuild matched their picks.

    ``squads`` has one row per (entry, gw, element) for the team that played that gameweek;
    ``kept`` the same for the squad carried into the next deadline (which differs only after a
    Free Hit). ``moves`` the net transfers of each gameweek, chains collapsed: A out for B and
    then B out for C is one move, A for C.
    """

    squads: pd.DataFrame
    kept: pd.DataFrame
    moves: pd.DataFrame
    checked: int
    mismatched: int


def collapse_chains(pairs: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Net a gameweek's transfers in time order into (out, in) pairs.

    A player bought and sold in the same week is neither: A for B then B for C is A for C, and
    A for B then B for A is nothing.
    """
    net: list[list[int]] = []
    for out, into in pairs:
        for pair in net:
            if pair[1] == out:
                pair[1] = into
                break
        else:
            net.append([out, into])
    return [(o, i) for o, i in net if o != i]


def pair_by_position(
    outs: set[int], ins: set[int], positions: dict[int, str]
) -> list[tuple[int, int]]:
    """Pair a squad difference into (out, in) moves within each position, in id order.

    Used only where the transfer feed cannot explain a squad, so which defender is paired with
    which matters less than that every move is between players of the same position.
    """
    pairs: list[tuple[int, int]] = []
    for position in sorted({positions.get(e, "") for e in outs | ins}):
        o = sorted(e for e in outs if positions.get(e, "") == position)
        i = sorted(e for e in ins if positions.get(e, "") == position)
        pairs += list(zip(o, i, strict=False))
    return pairs


def rebuild_squads(
    picks: pd.DataFrame,
    transfers: pd.DataFrame,
    chips: pd.DataFrame,
    last_gw: int,
    positions: dict[int, str] | None = None,
) -> Squads:
    """Replay every manager's transfers on their first squad, deferring to picks where known.

    The transfers endpoint returns at most 500 records, newest first, so a manager who tinkers
    through a Wildcard loses their earliest transfers. Wherever a gameweek's picks are stored
    they are the truth: a rebuild that disagrees is replaced by them, and that week's moves are
    read from the squad difference instead (paired within position).

    Args:
        picks: ``entry, gw, element`` — at least each manager's first gameweek.
        transfers: ``entry, gw, element_in, element_out, time``.
        chips: ``entry, gw, chip`` for the gameweeks a chip was played.
        last_gw: The last gameweek to rebuild.
        positions: element -> position, for pairing a squad difference.
    """
    positions = positions or {}
    first = picks.groupby("entry")["gw"].min()
    picks_by = {k: set(g["element"]) for k, g in picks.groupby(["entry", "gw"])}
    transfers = transfers.sort_values(["entry", "time"])
    transfers_by = {
        k: list(zip(g["element_out"], g["element_in"], strict=True))
        for k, g in transfers.groupby(["entry", "gw"])
    }
    chip_by = {(r.entry, r.gw): r.chip for r in chips.itertuples(index=False)}

    squad_rows, kept_rows, move_rows = [], [], []
    checked = mismatched = 0
    for entry, start in first.items():
        persistent = set(picks_by[(entry, start)])
        squad_rows.append((entry, start, persistent))
        kept_rows.append((entry, start, persistent))
        for gw in range(start + 1, last_gw + 1):
            played = set(persistent)
            raw = transfers_by.get((entry, gw), [])
            for out, into in raw:
                played.discard(out)
                played.add(into)
            net = collapse_chains(raw)
            actual = picks_by.get((entry, gw))
            if actual is not None:
                checked += 1
                if actual != played:
                    mismatched += 1
                    played = set(actual)
                    net = pair_by_position(persistent - played, played - persistent, positions)
            chip = chip_by.get((entry, gw))
            squad_rows.append((entry, gw, played))
            if chip != "freehit":
                persistent = played
            kept_rows.append((entry, gw, persistent))
            for out, into in net:
                move_rows.append((entry, gw, out, into, chip))

    def explode(rows: list[tuple[int, int, set[int]]]) -> pd.DataFrame:
        return pd.DataFrame(
            [(e, g, el) for e, g, members in rows for el in members],
            columns=["entry", "gw", "element"],
        )

    moves = pd.DataFrame(move_rows, columns=["entry", "gw", "element_out", "element_in", "chip"])
    return Squads(explode(squad_rows), explode(kept_rows), moves, checked, mismatched)


def load_squads(con, season: str, last_gw: int) -> Squads:
    """Rebuild every crawled manager's squads from the managers database (attached as ``m``).

    ``con`` is a warehouse connection: positions come from its ``players`` table.
    """
    attach_managers(con)
    positions = dict(
        con.execute(
            "SELECT element, CAST(element_type AS VARCHAR) FROM players WHERE season = ?", [season]
        ).fetchall()
    )
    picks = con.execute(
        "SELECT entry, gw, element FROM m.manager_picks WHERE season = ?", [season]
    ).fetchdf()
    transfers = con.execute(
        "SELECT entry, gw, element_in, element_out, time FROM m.manager_transfers WHERE season = ?",
        [season],
    ).fetchdf()
    chips = con.execute(
        "SELECT entry, gw, chip FROM m.manager_gw WHERE season = ? AND chip IS NOT NULL", [season]
    ).fetchdf()
    squads = rebuild_squads(picks, transfers, chips, last_gw, positions)
    if squads.checked:
        log.info(
            "squad rebuild: %d of %d stored squads disagreed with the transfers and were "
            "taken from the picks (%.2f%%)",
            squads.mismatched, squads.checked, 100 * squads.mismatched / squads.checked,
        )
    return squads


def memberships(con, season: str, gameweeks: list[int], me: int | None = None) -> pd.DataFrame:
    """Which managers belong to each cohort at each deadline, with their weights.

    One row per (cohort, gw, entry, weight). Only managers whose decisions were fetched (with
    picks) can belong. ``deadline`` membership at GW t reads the rank after GW t-1 and nothing
    later; GW1 has no such rank and no members.
    """
    attach_managers(con)
    entries = con.execute(
        """
        SELECT e.entry, e.source, e.weight, e.standing_rank, e.standing_gw, e.past_rank_last
        FROM m.manager_entries e
        WHERE e.season = ?
          AND e.entry IN (SELECT DISTINCT entry FROM m.manager_picks WHERE season = ?)
        """,
        [season, season],
    ).fetchdf()
    ranks = con.execute(
        "SELECT entry, gw, overall_rank FROM m.manager_gw WHERE season = ?", [season]
    ).fetchdf()
    leagues = con.execute(
        "SELECT DISTINCT entry FROM m.manager_leagues WHERE season = ?", [season]
    ).fetchdf()
    latest = entries["standing_gw"].max()

    frames = []
    for gw in gameweeks:
        def tag(name: str, chosen: pd.DataFrame, weight=None) -> None:
            frame = chosen[["entry"]].copy()
            frame["weight"] = chosen["weight"].to_numpy() if weight is None else weight
            frame.insert(0, "gw", gw)
            frame.insert(0, "cohort", name)
            frames.append(frame)

        ranked = entries["source"].isin(["top", "band"])
        tag("prior10k", entries[ranked & (entries["past_rank_last"] <= 10_000)])
        tag("prior50k", entries[ranked & (entries["past_rank_last"] <= 50_000)])
        before = ranks[(ranks["gw"] == gw - 1) & (ranks["overall_rank"] <= 10_000)]["entry"]
        tag("deadline", entries[ranked & entries["entry"].isin(before)])
        tag(
            "now",
            entries[
                (entries["source"] == "top")
                & (entries["standing_gw"] == latest)
                & (entries["standing_rank"] <= 10_000)
            ],
            weight=1.0,
        )
        tag("league", entries[entries["entry"].isin(leagues["entry"])], weight=1.0)
        if me is not None:
            tag("us", entries[entries["entry"] == me], weight=1.0)
    return pd.concat(frames, ignore_index=True)


def cohort_flows(
    members: pd.DataFrame, squads: Squads, chips: pd.DataFrame
) -> pd.DataFrame:
    """Per cohort, gameweek and player: who owned, bought and sold him at the deadline.

    ``own_before`` is the weighted share of the cohort holding him going into the deadline,
    ``own_after`` after its transfers. ``bought``/``sold`` are shares of the cohort's managers
    who did not play a Wildcard or Free Hit that week (their rebuilds are not transfers in the
    usual sense), and ``net`` is the difference. ``size`` is the cohort's weighted size and
    ``size_moving`` its weight without the chip players.
    """
    kept = squads.kept.rename(columns={"gw": "kept_gw"})
    chip_weeks = chips[chips["chip"].isin(SQUAD_CHIPS)][["entry", "gw"]].assign(chip_week=True)

    rows = []
    for (cohort, gw), group in members.groupby(["cohort", "gw"]):
        if group.empty:
            continue
        weight = group.set_index("entry")["weight"]
        size = float(weight.sum())
        flagged = chip_weeks[chip_weeks["gw"] == gw].set_index("entry")["chip_week"]
        moving = weight[~weight.index.isin(flagged.index)]
        size_moving = float(moving.sum())

        before = kept[(kept["kept_gw"] == gw - 1) & kept["entry"].isin(weight.index)]
        after = squads.squads[(squads.squads["gw"] == gw) & squads.squads["entry"].isin(weight.index)]
        moves = squads.moves[(squads.moves["gw"] == gw) & squads.moves["entry"].isin(moving.index)]

        def share(frame: pd.DataFrame, column: str, total: float) -> pd.Series:
            if frame.empty or total <= 0:
                return pd.Series(dtype="float64")
            w = frame["entry"].map(weight)
            return w.groupby(frame[column]).sum() / total

        table = pd.concat(
            {
                "own_before": share(before, "element", size),
                "own_after": share(after, "element", size),
                "bought": share(moves, "element_in", size_moving),
                "sold": share(moves, "element_out", size_moving),
            },
            axis=1,
        ).fillna(0.0)
        table.index.name = "element"
        table = table.reset_index()
        table["net"] = table["bought"] - table["sold"]
        table.insert(0, "gw", gw)
        table.insert(0, "cohort", cohort)
        table["size"] = size
        table["size_moving"] = size_moving
        table["managers"] = len(weight)
        rows.append(table)
    if not rows:
        return pd.DataFrame(
            columns=["cohort", "gw", "element", "own_before", "own_after", "bought", "sold",
                     "net", "size", "size_moving", "managers"]
        )
    return pd.concat(rows, ignore_index=True)


def weighted_moves(members: pd.DataFrame, squads: Squads, chips: pd.DataFrame) -> pd.DataFrame:
    """Every non-chip net move by a cohort member, with the member's weight.

    One row per (cohort, entry, gw, element_out, element_in, weight); a manager in several
    cohorts appears once in each.
    """
    chip_weeks = chips[chips["chip"].isin(SQUAD_CHIPS)][["entry", "gw"]]
    moves = squads.moves.merge(chip_weeks.assign(skip=True), on=["entry", "gw"], how="left")
    moves = moves[moves["skip"].isna()].drop(columns=["skip", "chip"])
    return members.merge(moves, on=["entry", "gw"], how="inner")


def chip_table(con, season: str) -> pd.DataFrame:
    attach_managers(con)
    return con.execute(
        "SELECT entry, gw, chip FROM m.manager_gw WHERE season = ? AND chip IS NOT NULL", [season]
    ).fetchdf()


def describe(members: pd.DataFrame) -> pd.DataFrame:
    """Cohort sizes by gameweek: managers and the population they stand for."""
    table = members.groupby(["cohort", "gw"]).agg(
        managers=("entry", "size"), represents=("weight", "sum")
    )
    return table.reset_index()


def summarise_overlap(members: pd.DataFrame, gw: int) -> pd.DataFrame:
    """How many managers each pair of cohorts shares at one deadline."""
    sets = {c: set(g["entry"]) for c, g in members[members["gw"] == gw].groupby("cohort")}
    names = sorted(sets)
    return pd.DataFrame(
        [[len(sets[a] & sets[b]) for b in names] for a in names], index=names, columns=names
    )


__all__ = [
    "COHORTS",
    "FAIR_COHORTS",
    "Squads",
    "chip_table",
    "cohort_flows",
    "collapse_chains",
    "describe",
    "load_squads",
    "memberships",
    "rebuild_squads",
    "summarise_overlap",
    "weighted_moves",
]

