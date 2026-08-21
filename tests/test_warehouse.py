"""Tests for the warehouse: cross-season joins, the leakage guard, and derived columns."""

from __future__ import annotations

from fplass.ingest.warehouse import POST_HOC_COLUMNS


def test_code_joins_across_every_season(con):
    """``code`` is the stable cross-season player id; ``element`` is reassigned annually.

    Getting this backwards would silently attribute one player's history to another, so assert
    that the same code genuinely tracks one person across many seasons.
    """
    seasons = con.execute("SELECT count(DISTINCT season) FROM players").fetchone()[0]
    long_careers = con.execute(
        """
        SELECT count(*) FROM (
            SELECT code FROM players GROUP BY code HAVING count(DISTINCT season) >= 8
        )
        """
    ).fetchone()[0]
    assert long_careers > 20, (
        f"only {long_careers} players span 8+ of {seasons} seasons; the code join looks broken"
    )

    # And the reverse: element ids really are reused, which is why we do not join on them.
    reused = con.execute(
        """
        SELECT count(*) FROM (
            SELECT element FROM players GROUP BY element HAVING count(DISTINCT code) > 1
        )
        """
    ).fetchone()[0]
    assert reused > 100, "expected element ids to be reused across seasons"


def test_every_player_gw_row_has_a_position(con):
    """Position is backfilled from players_raw for the seasons whose gameweek files omit it."""
    missing = con.execute(
        "SELECT count(*) FROM player_gw WHERE position IS NULL"
    ).fetchone()[0]
    assert missing == 0


def test_as_of_view_hides_post_hoc_columns(con):
    """The leakage guard: same-gameweek market data must be unreachable from the training view.

    ``selected``, ``value`` and the transfer counts are only knowable after a gameweek is played.
    A model that reads them for the gameweek it is predicting has the answer in its features.
    """
    columns = {
        r[0]
        for r in con.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = 'player_gw_as_of'"
        ).fetchall()
    }
    leaked = columns & set(POST_HOC_COLUMNS)
    assert not leaked, f"post-hoc columns reachable from the training view: {sorted(leaked)}"

    for column in POST_HOC_COLUMNS:
        assert f"prev_{column}" in columns, f"missing lagged column prev_{column}"


def test_as_of_lag_is_the_previous_gameweek(con):
    """``prev_value`` on gameweek N must equal ``value`` on that player's gameweek N-1."""
    mismatches = con.execute(
        """
        WITH ordered AS (
            SELECT season, element, gw_seq, fixture_id, value,
                   LAG(value) OVER (PARTITION BY season, element ORDER BY gw_seq, fixture_id)
                       AS expected_prev
            FROM player_gw_derived
            WHERE season = '2025-26'
        )
        SELECT count(*)
        FROM ordered o
        JOIN player_gw_as_of a USING (season, element, fixture_id)
        WHERE o.expected_prev IS NOT NULL
          AND (a.prev_value IS NULL OR a.prev_value <> o.expected_prev)
        """
    ).fetchone()[0]
    assert mismatches == 0


def test_gw_seq_is_gapless_and_fixes_the_covid_season(con):
    """2019-20's gameweeks jump 29 -> 39; a rolling window on raw ``gw`` would span the break."""
    gaps = con.execute(
        """
        SELECT count(*) FROM (
            SELECT season, gw_seq, LAG(gw_seq) OVER (PARTITION BY season ORDER BY gw_seq) AS prev
            FROM (SELECT DISTINCT season, gw_seq FROM player_gw_derived)
        ) WHERE prev IS NOT NULL AND gw_seq <> prev + 1
        """
    ).fetchone()[0]
    assert gaps == 0, "gw_seq should be contiguous within every season"

    covid = dict(
        con.execute(
            "SELECT DISTINCT gw, gw_seq FROM player_gw_derived "
            "WHERE season = '2019-20' AND gw IN (29, 39, 47)"
        ).fetchall()
    )
    assert covid[29] == 29
    assert covid[39] == 30, "the post-lockdown restart should follow gameweek 29 in sequence"
    assert covid[47] == 38, "2019-20 had 38 real rounds despite reaching gameweek label 47"


def test_defcon_reconstructed_for_opta_era_seasons(con):
    """CBIT/CBIRT counts exist for 2016-19 and from 2025-26, but not in between."""
    coverage = dict(
        con.execute(
            """
            SELECT season, round(100.0 * count(defcon_count) / count(*), 1)
            FROM player_gw_derived
            WHERE position NOT IN ('GK', 'GKP', 'AM')
            GROUP BY season
            """
        ).fetchall()
    )
    for season in ("2016-17", "2017-18", "2018-19"):
        assert coverage[season] > 85, f"{season} should have reconstructed DEFCON counts"
    for season in ("2020-21", "2022-23", "2024-25"):
        assert coverage[season] == 0, f"{season} published no defensive components"
    assert coverage["2025-26"] > 99


def test_defcon_definition_matches_actuals(con):
    """The reconstruction identity, verified against the season that publishes both.

    Defenders: CBI + tackles. Midfielders and forwards: CBI + tackles + recoveries.
    """
    rows = con.execute(
        """
        SELECT position,
               count(*) AS n,
               sum(CASE WHEN defensive_contribution
                        = clearances_blocks_interceptions + tackles
                        THEN 1 ELSE 0 END) AS matches_cbit,
               sum(CASE WHEN defensive_contribution
                        = clearances_blocks_interceptions + tackles + recoveries
                        THEN 1 ELSE 0 END) AS matches_cbirt
        FROM player_gw
        WHERE season = '2025-26' AND minutes > 0
        GROUP BY position
        """
    ).fetchall()
    by_position = {r[0]: r for r in rows}

    _, n_def, cbit_def, _ = by_position["DEF"]
    assert cbit_def == n_def, "defender DEFCON should be exactly CBI + tackles"

    for position in ("MID", "FWD"):
        _, n, _, cbirt = by_position[position]
        assert cbirt == n, f"{position} DEFCON should be exactly CBI + tackles + recoveries"


def test_match_player_stats_bps_agrees_with_player_gw(con):
    """The exploded fixture stats must agree with the gameweek table on BPS.

    ``match_player_stats`` is the training set for the reworked 2026/27 bonus model, so a parsing
    error in the fixture blob would quietly corrupt every bonus projection.
    """
    disagreements = con.execute(
        """
        SELECT count(*)
        FROM player_gw p
        JOIN match_player_stats s
          ON s.season = p.season AND s.fixture_id = p.fixture_id AND s.element = p.element
         AND s.identifier = 'bps'
        WHERE p.season = '2025-26' AND p.bps <> s.value
        """
    ).fetchone()[0]
    assert disagreements == 0
