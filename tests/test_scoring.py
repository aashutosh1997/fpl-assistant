"""Tests for the FPL points function.

The headline test is :func:`test_reproduces_every_season_exactly`. Every projection this system
makes depends on converting simulated event counts into points, so that conversion is checked
against a quarter of a million real, known-good rows rather than a handful of hand-written cases.
If it ever regresses, everything downstream is silently wrong, and this is the test that says so.
"""

from __future__ import annotations

import pandas as pd
import pytest

from fplass.scoring import (
    DEFCON_THRESHOLDS,
    NON_PLAYER_POSITIONS,
    ScoringRules,
    points_from_events,
    rules_for_season,
    verify_against_actuals,
)


# --------------------------------------------------------------- against actuals


def test_reproduces_every_season_exactly(con, complete_seasons):
    """Recomputed points must match FPL's own totals for every historical player-gameweek."""
    assert complete_seasons, "expected at least one complete season in the warehouse"

    failures = {}
    for season in complete_seasons:
        mismatched = verify_against_actuals(con, season, rules=rules_for_season(con, season))
        if len(mismatched):
            diffs = (mismatched["computed_total"] - mismatched["total_points"]).value_counts()
            failures[season] = (len(mismatched), diffs.head(3).to_dict())

    assert not failures, f"scoring engine diverges from actuals: {failures}"


def test_2025_26_row_count_is_stable(con):
    """Guard against upstream silently changing the dataset under us."""
    n = con.execute("SELECT count(*) FROM player_gw WHERE season = '2025-26'").fetchone()[0]
    # 29,757 rows upstream, less 10 exact duplicates from a mid-season player rename.
    assert n == 29_747, f"expected 29,747 rows for 2025-26, found {n}"


def test_assistant_manager_rows_are_excluded(con):
    """The 2024-25 Assistant Manager 'players' must not reach the scoring engine.

    They scored on club results rather than player actions, and the chip no longer exists.
    """
    n = con.execute(
        "SELECT count(*) FROM player_gw WHERE season = '2024-25' AND position = 'AM'"
    ).fetchone()[0]
    assert n > 0, "expected the 2024-25 dataset to contain Assistant Manager rows"
    assert "AM" in NON_PLAYER_POSITIONS


# ------------------------------------------------------------------ unit cases


@pytest.fixture
def rules() -> ScoringRules:
    """The 2026/27 ruleset, written out explicitly rather than read from the warehouse."""
    return ScoringRules(
        season="2026-27",
        goals_scored={"GKP": 10.0, "DEF": 6.0, "MID": 5.0, "FWD": 4.0},
        assists=dict.fromkeys(("GKP", "DEF", "MID", "FWD"), 3.0),
        clean_sheets={"GKP": 4.0, "DEF": 4.0, "MID": 1.0, "FWD": 0.0},
        goals_conceded={"GKP": -1.0, "DEF": -1.0, "MID": 0.0, "FWD": 0.0},
        defensive_contribution={"GKP": 0.0, "DEF": 2.0, "MID": 2.0, "FWD": 2.0},
    )


def score_one(rules: ScoringRules, **events) -> float:
    events.setdefault("position", "MID")
    events.setdefault("minutes", 90)
    return float(points_from_events(pd.DataFrame([events]), rules)["total"].iloc[0])


def test_appearance_points(rules):
    assert score_one(rules, minutes=0) == 0
    assert score_one(rules, minutes=1) == 1
    assert score_one(rules, minutes=59) == 1
    assert score_one(rules, minutes=60) == 2
    assert score_one(rules, minutes=90) == 2


def test_goals_by_position(rules):
    assert score_one(rules, position="FWD", goals_scored=1) == 2 + 4
    assert score_one(rules, position="MID", goals_scored=1) == 2 + 5
    assert score_one(rules, position="DEF", goals_scored=1) == 2 + 6
    # A goalkeeper scoring is worth 10 in 2026/27, up from 6.
    assert score_one(rules, position="GKP", goals_scored=1) == 2 + 10


def test_clean_sheet_requires_sixty_minutes(rules):
    assert score_one(rules, position="DEF", minutes=90, clean_sheets=1) == 2 + 4
    # Substituted at 59 minutes: appearance point only, no clean sheet even if the team kept one.
    assert score_one(rules, position="DEF", minutes=59, clean_sheets=1) == 1
    # Midfielders get 1, forwards nothing.
    assert score_one(rules, position="MID", clean_sheets=1) == 2 + 1
    assert score_one(rules, position="FWD", clean_sheets=1) == 2


def test_goals_conceded_is_per_two_and_defence_only(rules):
    assert score_one(rules, position="DEF", goals_conceded=1) == 2
    assert score_one(rules, position="DEF", goals_conceded=2) == 2 - 1
    assert score_one(rules, position="DEF", goals_conceded=3) == 2 - 1
    assert score_one(rules, position="DEF", goals_conceded=4) == 2 - 2
    # Midfielders and forwards are never docked for conceding.
    assert score_one(rules, position="MID", goals_conceded=4) == 2


def test_saves_are_per_three(rules):
    assert score_one(rules, position="GKP", saves=2) == 2
    assert score_one(rules, position="GKP", saves=3) == 2 + 1
    assert score_one(rules, position="GKP", saves=5) == 2 + 1
    assert score_one(rules, position="GKP", saves=6) == 2 + 2


@pytest.mark.parametrize(
    ("position", "threshold"),
    [("DEF", DEFCON_THRESHOLDS["DEF"]), ("MID", DEFCON_THRESHOLDS["MID"])],
)
def test_defcon_threshold_and_cap(rules, position, threshold):
    """DEFCON is a flat +2 at the threshold, and does not scale beyond it."""
    assert score_one(rules, position=position, defcon_count=threshold - 1) == 2
    assert score_one(rules, position=position, defcon_count=threshold) == 2 + 2
    # Capped: twice the threshold is still +2, not +4.
    assert score_one(rules, position=position, defcon_count=threshold * 2) == 2 + 2


def test_goalkeepers_score_no_defcon(rules):
    assert score_one(rules, position="GKP", defcon_count=50) == 2


def test_defcon_reconstructed_from_components(rules):
    """Without a defcon_count column, the count is rebuilt from its components.

    Defenders count CBI + tackles; midfielders and forwards also count recoveries. So the same
    action counts clear the bar for a midfielder but not a defender, and vice versa.
    """
    # Defender: 8 CBI + 2 tackles = 10 CBIT, exactly the threshold. Recoveries are ignored.
    assert (
        score_one(
            rules,
            position="DEF",
            clearances_blocks_interceptions=8,
            tackles=2,
            recoveries=99,
        )
        == 2 + 2
    )
    # Midfielder with the same 10 actions falls short of 12 until recoveries are counted.
    assert (
        score_one(rules, position="MID", clearances_blocks_interceptions=8, tackles=2) == 2
    )
    assert (
        score_one(
            rules, position="MID", clearances_blocks_interceptions=8, tackles=2, recoveries=2
        )
        == 2 + 2
    )


def test_cards_and_own_goals(rules):
    assert score_one(rules, yellow_cards=1) == 2 - 1
    assert score_one(rules, red_cards=1) == 2 - 3
    assert score_one(rules, own_goals=1) == 2 - 2
    assert score_one(rules, position="GKP", penalties_saved=1) == 2 + 5
    assert score_one(rules, penalties_missed=1) == 2 - 2


def test_bonus_can_be_excluded(rules):
    """The simulator allocates bonus by ranking BPS within a match, so it opts out here."""
    events = pd.DataFrame([{"position": "MID", "minutes": 90, "bonus": 3}])
    assert points_from_events(events, rules)["total"].iloc[0] == 2 + 3
    assert points_from_events(events, rules, include_bonus=False)["total"].iloc[0] == 2


def test_missing_event_columns_are_treated_as_zero(rules):
    """A caller supplying only minutes should get appearance points, not an error."""
    events = pd.DataFrame([{"position": "FWD", "minutes": 90}])
    assert points_from_events(events, rules)["total"].iloc[0] == 2


def test_historical_seasons_award_no_defcon(con):
    """DEFCON did not exist before 2025-26, even though we reconstruct the action counts."""
    old = rules_for_season(con, "2018-19")
    assert old.per_position("defensive_contribution")["DEF"] == 0.0
    current = rules_for_season(con, "2025-26")
    assert current.per_position("defensive_contribution")["DEF"] == 2.0


def test_goalkeeper_goal_was_worth_six_historically(con):
    """Pinned down by Alisson's May 2021 goal, which scored 6 rather than 10."""
    assert rules_for_season(con, "2020-21").per_position("goals_scored")["GKP"] == 6.0
    assert rules_for_season(con, "2026-27").per_position("goals_scored")["GKP"] == 10.0
