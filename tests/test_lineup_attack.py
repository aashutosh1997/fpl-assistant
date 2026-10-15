"""Tests for scaling a team's expected goals by the attackers the news leaves it.

Pinned against Liverpool before gameweek 7 of 2026-27: three forwards ruled out, and with the
team's expected goals unchanged their goals were handed whole to Wirtz.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fplass.features import lineup_attack


def _fixture():
    # One fixture: the home side has a striker and a midfielder, the away side one midfielder.
    return pd.DataFrame(
        {
            "fixture_id": [1, 1, 1],
            "is_home": [True, True, False],
            "goal_rate": [0.3, 0.4, 0.3],
            "xg_home": [1.6, 1.6, 1.6],
            "xg_away": [1.1, 1.1, 1.1],
        }
    )


def test_a_side_without_news_is_untouched():
    frame = _fixture()
    p_full = np.array([0.9, 0.9, 0.9])
    out = lineup_attack.scale_for_lineups(frame, p_full, p_full)
    assert out["xg_home"].tolist() == pytest.approx([1.6] * 3)
    assert out["xg_away"].tolist() == pytest.approx([1.1] * 3)


def test_losing_the_striker_costs_part_of_his_goals_not_all():
    frame = _fixture()
    usual = np.array([0.9, 0.9, 0.9])
    with_news = np.array([0.0, 1.0, 0.9])
    out = lineup_attack.scale_for_lineups(frame, with_news, usual)
    ratio = (0.4 * 1.0) / (0.3 * 0.9 + 0.4 * 0.9)
    expected = 1.6 * ratio**lineup_attack.ATTACK_ELASTICITY
    assert out["xg_home"].tolist() == pytest.approx([expected] * 3), "every row of the fixture agrees"
    assert ratio * 1.6 < expected < 1.6, "the replacements are worth something"
    assert out["xg_away"].tolist() == pytest.approx([1.1] * 3), "the other side's attack is its own"


def test_news_never_strengthens_an_attack():
    """An absent defender's minutes spread across the squad must not raise the team's goals."""
    frame = _fixture()
    usual = np.array([0.5, 0.5, 0.9])
    with_news = np.array([0.7, 0.6, 0.9])
    out = lineup_attack.scale_for_lineups(frame, with_news, usual)
    assert out["xg_home"].tolist() == pytest.approx([1.6] * 3)


def test_a_data_error_cannot_zero_a_team():
    frame = _fixture()
    out = lineup_attack.scale_for_lineups(frame, np.zeros(3), np.full(3, 0.9))
    assert out["xg_home"].iloc[0] == pytest.approx(1.6 * lineup_attack.MIN_SCALE)
