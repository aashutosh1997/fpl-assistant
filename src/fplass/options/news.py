"""When availability news arrives, timed from the hourly snapshots.

The buy-now-or-wait decision in :mod:`fplass.prices.decide` needs the chance that news which
would change your mind arrives between now and the deadline. It carried a hand-set table by
days to go. The price logger records every player's status and ``chance_of_playing`` hourly,
which is exactly the timing data that table was guessing at: for each transfer window, the
players who were unflagged at its start and the hour at which a flag appeared.

Two windows of 2026/27 gave 79 such events among roughly 600 unflagged players a week: 38% in
the two days before the deadline, 11% in the final day, and a weekly rate near 6.5%, against a
guessed 24% for a week out. The table is rebuilt from all logged windows on every run, so it
sharpens as the season goes, and the constants remain the fallback until two windows exist.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# A player is flagged when his status leaves "available" or his chance of playing is cut to
# half or less; a 75% knock is a doubt the model already discounts, not news that changes a buy.
FLAG_CHANCE = 50
MIN_WINDOWS = 2


def deadlines_from_warehouse(con, season: str) -> dict[int, pd.Timestamp]:
    rows = con.execute(
        "SELECT event, deadline_time FROM events WHERE season = ? AND deadline_time IS NOT NULL "
        "ORDER BY event",
        [season],
    ).fetchall()
    return {int(e): pd.Timestamp(t).tz_localize("UTC") for e, t in rows}


def flag_events(
    snapshots: pd.DataFrame, deadlines: dict[int, pd.Timestamp]
) -> tuple[pd.DataFrame, dict[int, int]]:
    """Flags that appeared during each transfer window, and how many players were clear at its start.

    Returns:
        ``(events, clear)``: one row per (gameweek, element) with ``days_before`` the deadline
        the flag first appeared, and per gameweek the count of players who were unflagged at
        the window's first snapshot — the denominator.
    """
    frame = snapshots.copy()
    # Either snapshot loader may hand over naive or aware timestamps; compare in UTC.
    frame["ts"] = pd.to_datetime(frame["ts"], errors="coerce", utc=True)
    frame = frame.dropna(subset=["ts"])
    frame["chance"] = pd.to_numeric(frame["chance_of_playing_next_round"], errors="coerce")
    frame["flagged"] = (frame["status"].fillna("a") != "a") | (frame["chance"] <= FLAG_CHANCE)
    frame = frame.sort_values("ts")

    rows: list[dict[str, object]] = []
    clear: dict[int, int] = {}
    ordered = sorted(deadlines.items())
    for i, (gw, deadline) in enumerate(ordered):
        start = ordered[i - 1][1] if i else frame["ts"].min() - pd.Timedelta(hours=1)
        window = frame[(frame["ts"] > start) & (frame["ts"] <= deadline)]
        if window["ts"].nunique() < 2:
            continue
        first = window.groupby("element").first()
        unflagged = first[~first["flagged"]].index
        clear[gw] = int(len(unflagged))
        flagged = window[window["flagged"] & window["element"].isin(unflagged)]
        onset = flagged.groupby("element")["ts"].min()
        for element, ts in onset.items():
            rows.append(
                {
                    "gameweek": gw,
                    "element": int(element),
                    "days_before": (deadline - ts).total_seconds() / 86400.0,
                }
            )
    return pd.DataFrame(rows, columns=["gameweek", "element", "days_before"]), clear


def news_risk_table(
    snapshots: pd.DataFrame | None, deadlines: dict[int, pd.Timestamp]
) -> pd.Series | None:
    """P(a flag appears within the final ``d`` days before the deadline), for d = 0..7.

    ``None`` until at least :data:`MIN_WINDOWS` complete windows have been logged.
    """
    if snapshots is None or snapshots.empty or not deadlines:
        return None
    events, clear = flag_events(snapshots, deadlines)
    windows = [gw for gw, n in clear.items() if n > 0]
    if len(windows) < MIN_WINDOWS:
        return None
    denominator = float(sum(clear[gw] for gw in windows))
    days = np.arange(0, 8)
    counts = [float((events["days_before"] <= d + 1).sum()) for d in days]
    table = pd.Series(np.array(counts) / denominator, index=days, name="news_risk")
    log.info(
        "news risk measured from %d windows, %d flags among %d clear player-weeks: %s",
        len(windows),
        len(events),
        int(denominator),
        {int(d): round(float(v), 3) for d, v in table.items()},
    )
    return table
