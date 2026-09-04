"""Tests for the projection panel: the as-of replay of completed seasons.

What these pin is that a replayed deadline sees only what was knowable at it. A panel that
leaks — a January signing projected in August, a mover projected for his end-of-season club, a
rate fed minutes from the gameweek being projected — would report accuracy the live model can
never reach, and every option value measured from it would be too small.
"""

from __future__ import annotations

import pandas as pd
import pytest

from fplass.backtest import panel
from fplass.features import rates, teams
from fplass.ingest.sources import CURRENT_SEASON
from fplass.sim.project import current_players

SEASON = "2024-25"


@pytest.fixture(scope="module")
def season_available(con):
    played = con.execute(
        "SELECT count(DISTINCT gw) FROM player_gw WHERE season = ?", [SEASON]
    ).fetchone()[0]
    if played < 38:
        pytest.skip(f"{SEASON} not fully ingested")


def _element(con, name: str) -> int:
    row = con.execute(
        "SELECT element FROM players WHERE season = ? AND web_name = ?", [SEASON, name]
    ).fetchone()
    if row is None:
        pytest.skip(f"{name} not in the {SEASON} pool")
    return int(row[0])


def test_panel_seasons_exclude_the_live_and_the_earliest(con):
    seasons = panel.panel_seasons(con)
    assert CURRENT_SEASON not in seasons
    earliest = con.execute("SELECT min(season) FROM player_gw").fetchone()[0]
    assert earliest not in seasons, "no results exist before the first season's first deadline"
    assert seasons == sorted(seasons)


def test_the_pool_is_the_players_registered_by_the_deadline(con, season_available):
    """A January signing must not exist in August; by May everyone is there."""
    august = current_players(con, SEASON, as_of_gameweek=1)
    may = current_players(con, SEASON, as_of_gameweek=38)
    season_end = current_players(con, SEASON)
    assert len(august) < len(may)
    assert set(may["element"]) == set(season_end["element"])
    assert set(august["element"]) <= set(may["element"])


def test_a_mid_season_mover_is_projected_for_the_club_he_was_at(con, season_available):
    """Rashford: Manchester United until his February 2025 loan to Aston Villa.

    The season-end table says Villa for the whole season, which would have projected him for
    Villa's fixtures all autumn.
    """
    element = _element(con, "Rashford")
    clubs = dict(
        con.execute(
            "SELECT short_name, team_id FROM teams WHERE season = ? AND short_name IN ('MUN', 'AVL')",
            [SEASON],
        ).fetchall()
    )
    before = current_players(con, SEASON, as_of_gameweek=20).set_index("element")
    after = current_players(con, SEASON, as_of_gameweek=30).set_index("element")
    assert before.at[element, "team_id"] == clubs["MUN"]
    assert before.at[element, "team"] == "MUN"
    assert after.at[element, "team_id"] == clubs["AVL"]
    assert current_players(con, SEASON).set_index("element").at[element, "team_id"] == clubs["AVL"]


def test_the_price_is_the_one_recorded_for_the_gameweek(con, season_available):
    element = _element(con, "Haaland")
    value = con.execute(
        "SELECT any_value(value) FROM player_gw WHERE season = ? AND element = ? AND gw = 20",
        [SEASON, element],
    ).fetchone()[0]
    players = current_players(con, SEASON, as_of_gameweek=20).set_index("element")
    assert players.at[element, "price"] == value
    assert players["price"].dtype.kind == "i"


def test_rates_and_results_stop_at_the_deadline(con, season_available):
    """The per-90 rates and team strength must not see the gameweek being projected."""
    element = _element(con, "Haaland")
    code = con.execute(
        "SELECT code FROM players WHERE season = ? AND element = ?", [SEASON, element]
    ).fetchone()[0]
    sequence = panel.gameweek_sequence(con, SEASON)
    cutoff_seq = sequence[20]

    table, _ = rates.build(con, up_to_season=SEASON, up_to_gw_seq=cutoff_seq)
    expected = con.execute(
        """
        SELECT sum(p.minutes) FROM player_gw_derived p
        JOIN players pl ON pl.season = p.season AND pl.element = p.element
        WHERE pl.code = ? AND (p.season < ? OR (p.season = ? AND p.gw_seq < ?))
        """,
        [code, SEASON, SEASON, cutoff_seq],
    ).fetchone()[0]
    assert table.set_index("code").at[code, "raw_minutes"] == expected

    cutoff = panel.deadline_cutoff(con, SEASON, 20)
    assert cutoff == pd.Timestamp("2025-01-04 12:30:00")
    results = teams.load_results(con, up_to=cutoff)
    assert (results["kickoff_time"] < cutoff).all()
    assert ((results["season"] == SEASON) & (results["event"] == 19)).any()
    assert not ((results["season"] == SEASON) & (results["event"] >= 20)).any()


def test_a_replayed_deadline_projects_the_week_that_followed(con, season_available):
    """GW20 of 2024-25, replayed blind: sensible probabilities, and rank skill on the outcome."""
    context = panel.season_context(con, SEASON)
    frame = panel.project_deadline(con, context, 20, horizon=2, n_draws=500)

    assert set(frame["target_gw"]) == {20, 21}
    assert frame["p_full"].between(0, 1).all()
    assert (frame["as_of_gw"] == 20).all() and (frame["season"] == SEASON).all()
    assert set(frame["element"]) == set(current_players(con, SEASON, as_of_gameweek=20)["element"])

    week = frame[frame["target_gw"] == 20]
    clubs = current_players(con, SEASON, as_of_gameweek=20)[["element", "team_id"]]
    fielded = week.merge(clubs, on="element").groupby("team_id")["p_full"].sum()
    assert fielded.between(9.5, 10.6).all(), fielded.to_dict()

    actual = con.execute(
        "SELECT element, sum(total_points) AS total_points FROM player_gw "
        "WHERE season = ? AND gw = 20 GROUP BY element",
        [SEASON],
    ).fetchdf()
    joined = week.merge(actual, on="element")
    assert joined["ep_mean"].corr(joined["total_points"], method="spearman") > 0.35


def test_scoring_the_panel_needs_rows(con):
    stored = con.execute("SELECT count(*) FROM projection_panel").fetchone()[0]
    if stored == 0:
        assert panel.score_panel(con).empty
        return
    table = panel.score_panel(con)
    assert {"season", "weeks_ahead", "spearman", "minutes_brier"} <= set(table.columns)
    assert (table["weeks_ahead"] >= 0).all()
