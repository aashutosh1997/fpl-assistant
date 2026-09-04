"""How much a projection moves between one deadline and the next.

This is the volatility of the underlying. Every flexibility the planner can hold — a free
transfer, cash, a bench player, a chip — is worth something only because next week's projection
will not be this week's: a starter gets injured, a signing displaces an incumbent, a striker's
club turns out to be worse than the model thought. The planner cannot value any of that from a
single projection, and every constant it uses in place of a value (``banked_transfer_value``,
``bench_weight``, the chip floors) is a guess at a quantity that the projection panel now lets us
measure directly.

The measurement pairs each deadline's projection with the next deadline's projection of the
*same* target gameweeks, so the mechanical roll of the horizon (one week dropping out, one
arriving) is not counted as a revision. For each player and week the record carries how his
remaining-horizon expected points and his chance of a full appearance in the coming gameweek
changed, plus a flag for the case that matters most for a bench and a banked transfer: a likely
starter collapsing to a likely absentee, which is what an injury or a suspension looks like from
the outside.

The sampler is a bootstrap. Revisions are pooled by position, price tier and how certain the
player's minutes were, and a plausible next-week projection is this week's with each player's
remaining horizon scaled by a revision drawn from his pool. Nothing is fitted; the tails,
including the injuries, are the ones that happened.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..features.rates import price_tier

log = logging.getLogger(__name__)

# Full-appearance probability at the deadline, bucketed: the certainty a player's minutes had.
CERTAINTY_EDGES = (0.0, 0.2, 0.6, 0.85, 1.0001)
CERTAINTY_LABELS = ("fringe", "rotation", "regular", "nailed")

# A jump is a regular or nailed starter whose chance of an hour collapses by the next deadline.
JUMP_FROM = 0.6
JUMP_TO = 0.2

# Relative revisions are measured against at least this many points per remaining week, so a
# player projected for nothing cannot post an infinite percentage move.
FLOOR_PER_WEEK = 0.5

# Pools are keyed by position, certainty and whether the player is premium-priced (8.0m and
# up). Splitting further by price tier left keys whose spread differed by a factor of two
# between halves of the history; these are stable to within about 20%. Pools smaller than this
# fall back to the position-and-certainty pool, then to the position.
MIN_POOL = 500
PREMIUM_PRICE = 80


def _panel_sql(sources: list[Path] | None) -> str:
    if sources:
        files = ", ".join(f"'{p}'" for p in sources)
        return f"read_parquet([{files}])"
    return "projection_panel"


def revisions(
    con, seasons: list[str] | None = None, *, sources: list[Path] | None = None
) -> pd.DataFrame:
    """One row per player and consecutive pair of deadlines.

    Columns: ``horizon_before``/``horizon_after`` sum the projections of the common target
    gameweeks (those from the later deadline on, as seen from each), ``next_before``/
    ``next_after`` are the coming gameweek's projection from each deadline, ``p_full_before``/
    ``p_full_after`` likewise for the chance of an hour. ``rel`` is the relative revision of the
    remaining horizon and ``jump`` the collapse flag.
    """
    season_filter = ""
    params: list[object] = []
    if seasons:
        season_filter = f"WHERE season IN ({', '.join('?' for _ in seasons)})"
        params = list(seasons)
    frame = con.execute(
        f"""
        WITH panel AS (SELECT * FROM {_panel_sql(sources)} {season_filter}),
        deadlines AS (SELECT DISTINCT season, as_of_gw FROM panel),
        ordered AS (
            SELECT season, as_of_gw,
                   lead(as_of_gw) OVER (PARTITION BY season ORDER BY as_of_gw) AS next_gw
            FROM deadlines
        ),
        pairs AS (
            SELECT p.season, p.element, p.as_of_gw, o.next_gw, p.target_gw,
                   p.ep_mean AS ep_before, q.ep_mean AS ep_after,
                   p.p_full AS p_before, q.p_full AS p_after
            FROM panel p
            JOIN ordered o ON o.season = p.season AND o.as_of_gw = p.as_of_gw
            JOIN panel q ON q.season = p.season AND q.element = p.element
                        AND q.as_of_gw = o.next_gw AND q.target_gw = p.target_gw
            WHERE p.target_gw >= o.next_gw
        ),
        per_week AS (
            SELECT season, element, as_of_gw, next_gw,
                   count(*) AS weeks,
                   sum(ep_before) AS horizon_before, sum(ep_after) AS horizon_after,
                   max(CASE WHEN target_gw = next_gw THEN ep_before END) AS next_before,
                   max(CASE WHEN target_gw = next_gw THEN ep_after END) AS next_after,
                   max(CASE WHEN target_gw = next_gw THEN p_before END) AS p_full_before,
                   max(CASE WHEN target_gw = next_gw THEN p_after END) AS p_full_after
            FROM pairs GROUP BY 1, 2, 3, 4
        )
        SELECT w.*,
               CASE pl.element_type WHEN 1 THEN 'GKP' WHEN 2 THEN 'DEF'
                                    WHEN 3 THEN 'MID' ELSE 'FWD' END AS position,
               v.price
        FROM per_week w
        JOIN players pl ON pl.season = w.season AND pl.element = w.element
        LEFT JOIN (
            SELECT season, element, gw, any_value(value) AS price FROM player_gw GROUP BY 1, 2, 3
        ) v ON v.season = w.season AND v.element = w.element AND v.gw = w.as_of_gw
        """,
        params,
    ).fetchdf()
    if frame.empty:
        return frame

    frame["tier"] = price_tier(frame["price"]).fillna("rotation")
    frame["premium"] = np.where(frame["price"].fillna(0) >= PREMIUM_PRICE, "premium", "standard")
    frame["certainty"] = pd.cut(
        frame["p_full_before"].fillna(0.0),
        bins=list(CERTAINTY_EDGES),
        labels=list(CERTAINTY_LABELS),
        right=False,
    ).astype("object")
    floor = FLOOR_PER_WEEK * frame["weeks"]
    frame["rel"] = (frame["horizon_after"] - frame["horizon_before"]) / np.maximum(
        frame["horizon_before"], floor
    )
    frame["jump"] = (
        (frame["p_full_before"] >= JUMP_FROM) & (frame["p_full_after"] <= JUMP_TO)
    ).astype(float)
    return frame


def summarise(table: pd.DataFrame) -> pd.DataFrame:
    """Spread and jump rate of revisions by position, price tier and certainty."""
    if table.empty:
        return table
    grouped = table.groupby(["position", "premium", "certainty"], observed=True)
    out = grouped.agg(
        n=("rel", "size"),
        mean_rel=("rel", "mean"),
        sd_rel=("rel", "std"),
        p10_rel=("rel", lambda s: s.quantile(0.10)),
        p90_rel=("rel", lambda s: s.quantile(0.90)),
        p_jump=("jump", "mean"),
        next_before=("next_before", "mean"),
        next_after=("next_after", "mean"),
    )
    return out.round(3).reset_index()


@dataclass(slots=True)
class RevisionSampler:
    """Bootstrap next-week projections from measured revisions."""

    pools: dict[tuple[str, str, str], np.ndarray] = field(default_factory=dict)
    by_certainty: dict[tuple[str, str], np.ndarray] = field(default_factory=dict)
    fallback: dict[str, np.ndarray] = field(default_factory=dict)
    seasons: tuple[str, ...] = ()

    @classmethod
    def fit(cls, table: pd.DataFrame) -> RevisionSampler:
        sampler = cls(seasons=tuple(sorted(table["season"].unique())))
        for (position, premium, certainty), group in table.groupby(
            ["position", "premium", "certainty"], observed=True
        ):
            if len(group) >= MIN_POOL:
                sampler.pools[(str(position), str(premium), str(certainty))] = (
                    group["rel"].to_numpy(dtype="float64")
                )
        for (position, certainty), group in table.groupby(
            ["position", "certainty"], observed=True
        ):
            if len(group) >= MIN_POOL:
                sampler.by_certainty[(str(position), str(certainty))] = (
                    group["rel"].to_numpy(dtype="float64")
                )
        for position, group in table.groupby("position", observed=True):
            sampler.fallback[str(position)] = group["rel"].to_numpy(dtype="float64")
        return sampler

    def pool_for(self, position: str, premium: str, certainty: str) -> np.ndarray:
        found = self.pools.get((position, premium, certainty))
        if found is None:
            found = self.by_certainty.get((position, certainty))
        if found is None:
            found = self.fallback.get(position, np.zeros(1))
        return found

    def sample(
        self,
        expected: pd.DataFrame,
        p_full: pd.Series,
        players: pd.DataFrame,
        *,
        draws: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """``(draws, n_players, n_gameweeks)`` next-week projections of ``expected``.

        Args:
            expected: Players (index) by target gameweeks (columns), this week's projection of
                the weeks that will remain next week.
            p_full: Each player's chance of an hour in the coming gameweek, indexed like
                ``expected``.
            players: Carries ``element``, ``position`` and ``price`` for the keys.
        """
        info = players.set_index("element").reindex(expected.index)
        premium = pd.Series(
            np.where(
                pd.to_numeric(info["price"], errors="coerce").fillna(0) >= PREMIUM_PRICE,
                "premium",
                "standard",
            ),
            index=expected.index,
        )
        certainty = pd.cut(
            p_full.reindex(expected.index).fillna(0.0),
            bins=list(CERTAINTY_EDGES),
            labels=list(CERTAINTY_LABELS),
            right=False,
        ).astype(str)
        positions = info["position"].fillna("MID").astype(str)

        base = expected.to_numpy(dtype="float64")
        out = np.empty((draws, *base.shape), dtype="float64")
        for i, element in enumerate(expected.index):
            pool = self.pool_for(positions.iloc[i], premium.iloc[i], certainty.iloc[i])
            factors = 1.0 + rng.choice(pool, size=draws)
            out[:, i, :] = np.clip(factors[:, None] * base[i][None, :], 0.0, None)
        return out


def stability(table: pd.DataFrame) -> pd.DataFrame:
    """Does the spread of revisions in one half of the seasons predict the other half's?

    Splits seasons alternately, compares the standard deviation of relative revisions per key,
    and reports the ratio; a bootstrap from history is only as good as this is stable.
    """
    seasons = sorted(table["season"].unique())
    first = table[table["season"].isin(seasons[::2])]
    second = table[table["season"].isin(seasons[1::2])]
    keys = ["position", "premium", "certainty"]
    a = first.groupby(keys, observed=True)["rel"].agg(["size", "std"]).rename(
        columns={"size": "n_a", "std": "sd_a"}
    )
    b = second.groupby(keys, observed=True)["rel"].agg(["size", "std"]).rename(
        columns={"size": "n_b", "std": "sd_b"}
    )
    out = a.join(b, how="inner")
    out["ratio"] = out["sd_b"] / out["sd_a"]
    return out.round(3).reset_index()
