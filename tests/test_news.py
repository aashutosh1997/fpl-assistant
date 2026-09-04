"""Tests for the measured news-risk table."""

from __future__ import annotations

import pandas as pd

from fplass.options import news
from fplass.prices import decide


def _snapshots() -> pd.DataFrame:
    """Two windows, three players; player 2 is flagged two days before the second deadline."""
    rows = []
    hours = pd.date_range("2026-08-22 00:00", "2026-09-04 12:00", freq="6h", tz="UTC")
    for ts in hours:
        for element in (1, 2, 3):
            flagged = element == 2 and ts >= pd.Timestamp("2026-09-02 17:30", tz="UTC")
            rows.append(
                {
                    "ts": ts,
                    "element": element,
                    "status": "i" if flagged else "a",
                    "chance_of_playing_next_round": 0 if flagged else None,
                }
            )
    return pd.DataFrame(rows)


DEADLINES = {
    2: pd.Timestamp("2026-08-28 17:30", tz="UTC"),
    3: pd.Timestamp("2026-09-04 17:30", tz="UTC"),
}


def test_flags_are_timed_against_the_deadline_they_precede():
    events, clear = news.flag_events(_snapshots(), DEADLINES)
    assert clear == {2: 3, 3: 3}
    assert len(events) == 1
    event = events.iloc[0]
    assert event["gameweek"] == 3 and event["element"] == 2
    assert 1.9 < event["days_before"] < 2.1, "first flagged snapshot is 18:00 two days before"


def test_risk_table_needs_two_windows_and_accumulates_by_day():
    table = news.news_risk_table(_snapshots(), DEADLINES)
    assert table is not None
    assert table.loc[0] == 0.0, "index d is the risk within the final d+1 days: none in the last"
    assert abs(table.loc[1] - 1 / 6) < 1e-9, "one flag among six clear player-weeks, two days out"
    assert table.loc[7] == table.loc[1]
    assert news.news_risk_table(_snapshots(), {3: DEADLINES[3]}) is None
    assert news.news_risk_table(None, DEADLINES) is None


def test_decide_uses_the_measured_table_when_given():
    table = pd.Series([0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07], index=range(8))
    assert decide.news_risk(2.5, table) == 0.02
    assert decide.news_risk(2.5) == decide.NEWS_RISK_BY_DAYS[2]
    assert decide.news_risk(30.0, table) == 0.07
