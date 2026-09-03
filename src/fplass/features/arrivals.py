"""Minutes uncertainty from new signings.

A transfer window closes, a club signs two midfielders on deadline day, and every midfielder it
already had is suddenly less certain to start — but nothing in the data says so until the next
match is played. The minutes model reads each incumbent's own recent minutes, which were earned
before the signing existed, and the recalibration layer reads friendlies and ownership, both of
which predate it too. Left alone the model is confidently wrong about exactly the players whose
minutes just became a coin flip.

The honest response is not to guess who starts. It is to *widen* the projection for the players
the signing competes with, so that the optimiser sees the risk and prices it: a squad built on
three of a club's midfielders looks worse, a stronger bench looks better, and a player with the
same mean but a settled role wins the tie. The widening is applied before the lineup constraint,
so the club still fields the right number of players — the probability mass simply spreads across
the competing group rather than sitting on the pre-window hierarchy.

What counts as an arrival is read from the warehouse, not typed in: a player whose club now
differs from the club he played for in this season's recorded gameweeks, or who is registered
today but appeared in none of them while his club did. The former is a transfer; the latter is a
deadline-day registration.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# How much of an incumbent's full-appearance probability is pulled toward the club-position
# average per competing arrival. Deliberately a widening, not a verdict: 0.3 leaves a nailed
# starter clearly ahead of a squad player while making him a visibly worse captain than a rival
# with no such doubt. Capped so a club that signs three midfielders does not flatten them all.
SHRINK_PER_ARRIVAL = 0.3
MAX_SHRINK = 0.5

# An arrival competes for a starting place when he cost at least this fraction of the
# incumbent's price. A 4.5m squad-filler does not threaten an 8.5m starter; a 6.9m one does.
COMPETING_PRICE_RATIO = 0.75

# Below this price (tenths) FPL is saying the arrival is a squad player, and he only competes
# with incumbents priced no higher than himself. Prices are compressed at the bottom of the
# scale — a 4.5m centre-back is three quarters of a 5.9m one — so the ratio alone would let a
# deadline-day backup widen an entire settled defence.
SQUAD_PLAYER_PRICE = 50


def competes(arrival_price: int, incumbent_price: float) -> bool:
    """Whether a signing at ``arrival_price`` threatens an incumbent at ``incumbent_price``."""
    if arrival_price >= incumbent_price:
        return True
    if arrival_price < SQUAD_PLAYER_PRICE:
        return False
    return arrival_price >= COMPETING_PRICE_RATIO * incumbent_price


@dataclass(slots=True)
class Arrival:
    element: int
    web_name: str
    team_id: int
    position: str
    price: int  # tenths
    origin: str  # "transfer from <club>" or "new registration"


def detect_arrivals(con, season: str) -> list[Arrival]:
    """Players whose current club is not the one this season's results show them at."""
    frame = con.execute(
        """
        WITH played AS (
            SELECT p.element,
                   max(CASE WHEN p.was_home THEN f.team_h ELSE f.team_a END) AS played_for,
                   count(*) AS appearances
            FROM player_gw p
            JOIN fixtures f ON f.season = p.season AND f.fixture_id = p.fixture_id
            WHERE p.season = ?
            GROUP BY p.element
        ),
        club_played AS (
            SELECT team_id, count(*) AS matches FROM (
                SELECT team_h AS team_id FROM fixtures WHERE season = ? AND finished
                UNION ALL
                SELECT team_a FROM fixtures WHERE season = ? AND finished
            ) GROUP BY team_id
        )
        SELECT pl.element, pl.web_name, pl.team_id,
               CASE pl.element_type WHEN 1 THEN 'GKP' WHEN 2 THEN 'DEF'
                                    WHEN 3 THEN 'MID' ELSE 'FWD' END AS position,
               pl.start_cost AS price,
               played.played_for, played.appearances,
               th.short_name AS from_club, COALESCE(cp.matches, 0) AS club_matches
        FROM players pl
        LEFT JOIN played ON played.element = pl.element
        LEFT JOIN teams th ON th.season = pl.season AND th.team_id = played.played_for
        LEFT JOIN club_played cp ON cp.team_id = pl.team_id
        WHERE pl.season = ?
        """,
        [season, season, season, season],
    ).fetchdf()

    arrivals: list[Arrival] = []
    for row in frame.itertuples():
        if pd.notna(row.played_for) and int(row.played_for) != int(row.team_id):
            origin = f"transfer from {row.from_club}"
        elif pd.isna(row.played_for) and int(row.club_matches) > 0:
            origin = "new registration"
        else:
            continue
        arrivals.append(
            Arrival(
                element=int(row.element),
                web_name=str(row.web_name),
                team_id=int(row.team_id),
                position=str(row.position),
                price=int(row.price) if pd.notna(row.price) else 0,
                origin=origin,
            )
        )
    if arrivals:
        log.info(
            "%d arrivals since the last recorded gameweek: %s",
            len(arrivals),
            ", ".join(f"{a.web_name} ({a.origin})" for a in arrivals),
        )
    return arrivals


def disruption(
    players: pd.DataFrame, arrivals: list[Arrival], *, per_arrival: float = SHRINK_PER_ARRIVAL
) -> pd.Series:
    """Shrink weight in ``[0, MAX_SHRINK]`` per element, indexed like ``players``.

    An incumbent is disrupted by every arrival at his club and position priced at least
    :data:`COMPETING_PRICE_RATIO` of his own. Arrivals are disrupted too — their minutes were
    earned somewhere else — by the same amount as the incumbents they threaten, so that a
    signing does not walk in as a certain starter over players the model knows.

    Args:
        players: Must carry ``element``, ``team_id``, ``position`` and ``price`` (tenths).
    """
    weight = pd.Series(0.0, index=players.index)
    if not arrivals:
        return weight

    by_club_position: dict[tuple[int, str], list[Arrival]] = {}
    for arrival in arrivals:
        by_club_position.setdefault((arrival.team_id, arrival.position), []).append(arrival)

    arrival_ids = {a.element for a in arrivals}
    price = pd.to_numeric(players["price"], errors="coerce").fillna(0.0)
    for idx, row in players.iterrows():
        competitors = by_club_position.get((int(row["team_id"]), str(row["position"])), [])
        if not competitors:
            continue
        own_price = float(price.at[idx])
        if int(row["element"]) in arrival_ids:
            # The arrival is uncertain in proportion to the incumbents he has to displace.
            threatened = [c for c in competitors if c.element != int(row["element"])]
            count = max(len(threatened), 0) + 1
        else:
            count = sum(1 for c in competitors if competes(c.price, own_price))
        weight.at[idx] = min(per_arrival * count, MAX_SHRINK)
    return weight


def widen(
    probabilities: pd.DataFrame, players: pd.DataFrame, weight: pd.Series
) -> pd.DataFrame:
    """Pull each disrupted player's ``p_full`` toward his club-position mean.

    The mean is taken over the competing group *including* the arrival, so a signing who is
    himself well rated pulls the incumbents down and they pull him down. With equal weights the
    group's total is preserved exactly and only the certainty is lost, which is the intent; where
    weights differ the lineup constraint applied afterwards restores the club's total.

    Groups are per fixture when ``players`` carries an ``event`` column, so a player-match frame
    spanning several gameweeks is not averaged across them.
    """
    out = probabilities.copy()
    if weight.max() <= 0:
        return out
    group = players["team_id"].astype(str) + ":" + players["position"].astype(str)
    if "event" in players.columns:
        group = group + ":" + players["event"].astype(str)
    group_mean = out["p_full"].groupby(group.to_numpy()).transform("mean")
    w = weight.to_numpy(dtype="float64")
    new_full = (1 - w) * out["p_full"].to_numpy() + w * group_mean.to_numpy()
    # Keep the three classes coherent, as the recalibration layer does.
    old_remaining = (1.0 - out["p_full"]).clip(lower=1e-9)
    out["p_cameo"] = (out["p_cameo"] / old_remaining) * (1.0 - new_full)
    out["p_full"] = new_full
    out["p_none"] = 1.0 - out["p_full"] - out["p_cameo"]
    return out
