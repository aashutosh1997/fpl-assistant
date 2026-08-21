"""The FPL points function, reconstructed from raw event counts.

This is the single most safety-critical piece of the whole system. The Monte Carlo engine samples
event counts — goals, minutes, defensive actions — and then has to turn them into points. If that
conversion is wrong, every expected-points number, every transfer recommendation and every chip
decision downstream is wrong too, and in a way that no amount of clever simulation will reveal.

So it is written to be checked rather than trusted: :func:`points_from_events` recomputes points
for historical rows whose true ``total_points`` we already know, and the test suite asserts an
exact match across every 2025-26 player-gameweek. That test is the reason to believe any of the
projections.

Rules are read from the ``scoring_rules`` table (populated per season from FPL's own
``game_config.scoring``) rather than hardcoded, so backtesting an old season scores it under the
rules that season was actually played under.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

POSITIONS = ("GKP", "DEF", "MID", "FWD")

# The historical dataset writes goalkeepers as "GK"; the live API and scoring config use "GKP".
POSITION_ALIASES = {"GK": "GKP", "GKP": "GKP", "DEF": "DEF", "MID": "MID", "FWD": "FWD"}

# Rates that FPL applies per N actions rather than per action. These are not in game_config, which
# only gives the points value, so they live here — verified against actuals by the test suite.
GOALS_CONCEDED_PER_POINT = 2  # -1 per 2 conceded, goalkeepers and defenders only
SAVES_PER_POINT = 3  # +1 per 3 saves
MINUTES_FOR_FULL_APPEARANCE = 60

DEFCON_THRESHOLDS = {"DEF": 10, "MID": 12, "FWD": 12}

# Positions that are not footballers. element_type 5 ("AM") was the 2024-25 Assistant Manager
# chip, scored on its club's results rather than any player's actions. The chip was removed for
# 2025-26, so these rows are excluded from training and backtesting rather than modelled — there
# is no point learning a scoring mechanism that no longer exists.
NON_PLAYER_POSITIONS = frozenset({"AM"})

# FPL's rules changed during the eleven seasons we train on, and a backtest must score each
# season under the rules it was actually played under. Only the *deltas* from the current
# ruleset are listed; each was confirmed empirically by requiring the season to reproduce exactly
# (see tests/test_scoring.py).
#
#   defensive_contribution  Introduced in 2025-26. Before that, defenders still made the tackles,
#                           and we reconstruct the counts, but no points were awarded.
#   goals_scored[GKP]       Was 6, now 10. Pinned down by a single row: Alisson's goal for
#                           Liverpool in May 2021 scored 6, not 10.
SEASON_RULE_OVERRIDES: dict[str, dict[str, object]] = {
    season: {
        "defensive_contribution": dict.fromkeys(POSITIONS, 0.0),
        "goals_scored": {"GKP": 6.0, "DEF": 6.0, "MID": 5.0, "FWD": 4.0},
    }
    for season in (
        "2016-17",
        "2017-18",
        "2018-19",
        "2019-20",
        "2020-21",
        "2021-22",
        "2022-23",
        "2023-24",
        "2024-25",
    )
}


@dataclass(slots=True)
class ScoringRules:
    """Points per stat per position for one season."""

    season: str
    long_play: float = 2.0
    short_play: float = 1.0
    goals_scored: dict[str, float] = field(default_factory=dict)
    assists: dict[str, float] = field(default_factory=dict)
    clean_sheets: dict[str, float] = field(default_factory=dict)
    goals_conceded: dict[str, float] = field(default_factory=dict)
    defensive_contribution: dict[str, float] = field(default_factory=dict)
    saves: float = 1.0
    penalties_saved: float = 5.0
    penalties_missed: float = -2.0
    yellow_cards: float = -1.0
    red_cards: float = -3.0
    own_goals: float = -2.0
    bonus: float = 1.0

    @classmethod
    def from_warehouse(cls, con, season: str) -> ScoringRules:
        """Load a season's rules from the ``scoring_rules`` table."""
        rows = con.execute(
            "SELECT stat, position, points FROM scoring_rules WHERE season = ?", [season]
        ).fetchall()
        if not rows:
            raise ValueError(
                f"no scoring rules stored for {season}; run `fpl ingest current` first "
                "(rules are only published for the live season)"
            )

        by_position: dict[str, dict[str, float]] = {}
        scalar: dict[str, float] = {}
        for stat, position, points in rows:
            if position == "ALL":
                scalar[stat] = float(points)
            else:
                by_position.setdefault(stat, {})[position] = float(points)

        rules = cls(season=season)
        for stat in (
            "goals_scored",
            "assists",
            "clean_sheets",
            "goals_conceded",
            "defensive_contribution",
        ):
            per_position = by_position.get(stat)
            if per_position:
                setattr(rules, stat, per_position)
            elif stat in scalar:
                setattr(rules, stat, dict.fromkeys(POSITIONS, scalar[stat]))
        for stat in (
            "long_play",
            "short_play",
            "saves",
            "penalties_saved",
            "penalties_missed",
            "yellow_cards",
            "red_cards",
            "own_goals",
            "bonus",
        ):
            if stat in scalar:
                setattr(rules, stat, scalar[stat])
        return rules

    def per_position(self, stat: str) -> dict[str, float]:
        value = getattr(self, stat)
        return value if isinstance(value, dict) else dict.fromkeys(POSITIONS, float(value))

    def vector(self, stat: str, positions: pd.Series) -> np.ndarray:
        """Look up ``stat``'s points value for each row's position."""
        mapping = self.per_position(stat)
        return positions.map(mapping).fillna(0.0).to_numpy(dtype="float64")


def rules_for_season(con, season: str, *, live_season: str = "2026-27") -> ScoringRules:
    """The scoring rules in force for ``season``.

    FPL only publishes ``game_config.scoring`` for the live season, so historical seasons start
    from the live ruleset and apply the documented deltas in :data:`SEASON_RULE_OVERRIDES`.
    """
    rules = ScoringRules.from_warehouse(con, live_season)
    rules.season = season
    for stat, value in SEASON_RULE_OVERRIDES.get(season, {}).items():
        setattr(rules, stat, value)
    return rules


def normalise_positions(positions: pd.Series) -> pd.Series:
    return positions.map(POSITION_ALIASES).fillna(positions)


def points_from_events(
    events: pd.DataFrame, rules: ScoringRules, *, include_bonus: bool = True
) -> pd.DataFrame:
    """Compute FPL points from raw event counts, one row per player-match.

    Args:
        events: Needs ``position``, ``minutes`` and whichever event columns apply. Missing event
            columns are treated as zero, which is correct here (unlike at ingest time) because a
            caller asking for points has already decided what happened.
        rules: The season's scoring rules.
        include_bonus: Whether to add the ``bonus`` column. The simulator computes bonus
            separately by ranking BPS within a match, so it passes ``False``.

    Returns:
        A frame of per-component point columns plus ``total`` — component-wise so that a
        mismatch against actuals can be traced to the rule that caused it.
    """
    frame = events
    position = normalise_positions(frame["position"])
    minutes = pd.to_numeric(frame["minutes"], errors="coerce").fillna(0).to_numpy()

    def col(name: str) -> np.ndarray:
        if name not in frame.columns:
            return np.zeros(len(frame), dtype="float64")
        return pd.to_numeric(frame[name], errors="coerce").fillna(0).to_numpy(dtype="float64")

    out = pd.DataFrame(index=frame.index)

    # Appearance: 1 point for playing at all, 2 from 60 minutes.
    out["appearance"] = np.where(
        minutes >= MINUTES_FOR_FULL_APPEARANCE,
        rules.long_play,
        np.where(minutes > 0, rules.short_play, 0.0),
    )

    out["goals"] = col("goals_scored") * rules.vector("goals_scored", position)
    out["assists"] = col("assists") * rules.vector("assists", position)

    # Clean sheets require 60 minutes. The historical column already encodes that, but the
    # simulator does not, so the rule is enforced here and is idempotent either way.
    clean_sheet = np.where(minutes >= MINUTES_FOR_FULL_APPEARANCE, col("clean_sheets"), 0.0)
    out["clean_sheets"] = clean_sheet * rules.vector("clean_sheets", position)

    # Conceding costs a point per two goals, and only for goalkeepers and defenders. The rules
    # table stores -1; the "per 2" divisor is FPL's, not a per-position value.
    conceded_points = rules.vector("goals_conceded", position)
    out["goals_conceded"] = (
        np.floor(col("goals_conceded") / GOALS_CONCEDED_PER_POINT) * conceded_points
    )

    out["saves"] = np.floor(col("saves") / SAVES_PER_POINT) * rules.saves
    out["penalties_saved"] = col("penalties_saved") * rules.penalties_saved
    out["penalties_missed"] = col("penalties_missed") * rules.penalties_missed
    out["yellow_cards"] = col("yellow_cards") * rules.yellow_cards
    out["red_cards"] = col("red_cards") * rules.red_cards
    out["own_goals"] = col("own_goals") * rules.own_goals

    # Defensive contribution: a flat +2 once the position's action threshold is met, capped there.
    defcon_count = col("defcon_count") if "defcon_count" in frame.columns else _defcon(frame, col)
    threshold = position.map(DEFCON_THRESHOLDS).to_numpy(dtype="float64")
    hit = np.where(np.isnan(threshold), False, defcon_count >= threshold)
    out["defensive_contribution"] = hit * rules.vector("defensive_contribution", position)

    out["bonus"] = col("bonus") * rules.bonus if include_bonus else 0.0

    out["total"] = out.sum(axis=1)
    return out


def _defcon(frame: pd.DataFrame, col) -> np.ndarray:
    """Reconstruct the defensive-action count from its components.

    Defenders count clearances/blocks/interceptions plus tackles; midfielders and forwards also
    count ball recoveries. Verified exactly against 2025-26's ``defensive_contribution``.
    """
    position = normalise_positions(frame["position"])
    base = col("clearances_blocks_interceptions") + col("tackles")
    return np.where(position.isin(["MID", "FWD"]), base + col("recoveries"), base)


def verify_against_actuals(con, season: str, *, rules: ScoringRules | None = None) -> pd.DataFrame:
    """Recompute points for every stored player-gameweek and return the mismatching rows.

    An empty result means the scoring engine reproduces FPL exactly for that season. Any rows
    returned carry both the actual and recomputed totals plus every component, so the offending
    rule is visible directly.
    """
    rules = rules or rules_for_season(con, season)
    excluded = tuple(NON_PLAYER_POSITIONS)
    actual = con.execute(
        """
        SELECT season, element, fixture_id, gw, web_name, position, minutes, total_points,
               goals_scored, assists, clean_sheets, goals_conceded, own_goals,
               penalties_saved, penalties_missed, yellow_cards, red_cards, saves, bonus,
               defcon_count
        FROM player_gw_derived
        WHERE season = ? AND position NOT IN ?
        """,
        [season, excluded],
    ).fetchdf()

    computed = points_from_events(actual, rules)
    actual = actual.assign(computed_total=computed["total"], **{
        f"pts_{c}": computed[c] for c in computed.columns if c != "total"
    })
    mismatched = actual[actual["computed_total"] != actual["total_points"]]
    log.info(
        "%s: %d/%d rows reproduce exactly (%.4f%%)",
        season,
        len(actual) - len(mismatched),
        len(actual),
        100 * (1 - len(mismatched) / max(len(actual), 1)),
    )
    return mismatched
