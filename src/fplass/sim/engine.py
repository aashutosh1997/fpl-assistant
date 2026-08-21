"""The Monte Carlo engine: joint samples of every player's points, gameweek by gameweek.

The output is a single array of shape ``(n_draws, n_players, n_gameweeks)``. Everything downstream
— expected points, captaincy, chip valuation, mini-league win probability — is a reduction over
that array, which is the point: one simulation, many questions, all mutually consistent.

Three design decisions carry most of the value.

**Correlation is preserved, not assumed away.** Goals are allocated from a *sampled team total*,
so if Arsenal score four in a draw, several Arsenal players share them, and if Arsenal are kept
out, none do. Clean sheets for the whole back line resolve from the same scoreline. This is what
makes the engine able to evaluate Bench Boost (which stacks correlated risk) or a triple captain
(which is a bet on one team's upside) rather than silently treating fifteen players as independent.

**Common random numbers.** Every candidate squad is scored against the *same* draws. Comparing two
transfers then becomes a paired comparison, where the shared match outcomes cancel and only the
difference between the players remains. This cuts the variance of the comparison by roughly an
order of magnitude versus scoring each plan on fresh draws, and it is the difference between the
simulation being able to separate two similar transfers and drowning them in noise.

**Bonus is resolved inside the match.** BPS is sampled for all twenty-two-plus players, then ranked
to award 3/2/1. Bonus cannot be computed per player in isolation, and approximating it as an
expectation would systematically understate the ceiling of exactly the players worth captaining.

Memory is managed by chunking over draws: the intermediate per-player-match arrays are large, but
only one chunk exists at a time, so 50,000 draws costs no more memory than 1,000.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..features.bps import BPSModel, allocate_bonus
from ..scoring import ScoringRules, normalise_positions
from .match import sample_scorelines

log = logging.getLogger(__name__)

# Roughly a third of Premier League goals are unassisted (solo efforts, deflections, rebounds,
# penalties). Allocating every goal an assister would inflate midfield assist returns badly.
UNASSISTED_GOAL_SHARE = 0.35

# Minutes drawn within each class. The cameo range is wide because it covers both early
# substitutions off and late substitutions on.
CAMEO_MINUTES = (1, 59)
FULL_MINUTES = (60, 90)

DEFCON_THRESHOLDS = {"DEF": 10, "MID": 12, "FWD": 12, "GKP": 10**6}


@dataclass(slots=True)
class SimulationResult:
    """Joint point samples plus the metadata needed to interpret them."""

    points: np.ndarray  # (n_draws, n_players, n_gameweeks) int16
    elements: np.ndarray  # FPL element id per player column
    gameweeks: np.ndarray  # gameweek per depth slice
    minutes_played: np.ndarray  # (n_draws, n_players, n_gameweeks) int16
    n_draws: int

    @property
    def expected_points(self) -> pd.DataFrame:
        """Mean points per player per gameweek."""
        return pd.DataFrame(
            self.points.mean(axis=0), index=self.elements, columns=self.gameweeks
        )

    def horizon_total(self) -> pd.DataFrame:
        """Per-draw total across the whole horizon, one column per player."""
        return pd.DataFrame(self.points.sum(axis=2), columns=self.elements)

    def quantile(self, q: float) -> pd.DataFrame:
        """Per-player, per-gameweek quantile of points — the upside a captain is bought for."""
        return pd.DataFrame(
            np.quantile(self.points, q, axis=0), index=self.elements, columns=self.gameweeks
        )


def build_player_matches(
    players: pd.DataFrame, fixtures: pd.DataFrame, gameweeks: list[int]
) -> pd.DataFrame:
    """Expand players against the fixtures their club plays in each gameweek.

    A player appears once per fixture, so a double gameweek naturally produces two rows and a
    blank gameweek produces none. That falls out of the join rather than needing special cases,
    which is what lets the same code value a Free Hit in a blank and a Bench Boost in a double.
    """
    relevant = fixtures[fixtures["event"].isin(gameweeks)]

    home = relevant.merge(
        players, left_on="team_h", right_on="team_id", how="inner", suffixes=("", "_p")
    )
    home["is_home"] = True
    home["opponent"] = home["team_a"]

    away = relevant.merge(
        players, left_on="team_a", right_on="team_id", how="inner", suffixes=("", "_p")
    )
    away["is_home"] = False
    away["opponent"] = away["team_h"]

    combined = pd.concat([home, away], ignore_index=True)
    combined = combined.sort_values(["event", "fixture_id", "element"], ignore_index=True)
    combined["position"] = normalise_positions(combined["position"])
    return combined


def _sample_minutes(
    probabilities: np.ndarray, rng: np.random.Generator, n_draws: int
) -> np.ndarray:
    """Sample minutes for each (draw, player-match) from the three-class distribution."""
    n_rows = probabilities.shape[0]
    uniform = rng.random((n_draws, n_rows))

    p_none = probabilities[:, 0][None, :]
    p_cameo = probabilities[:, 1][None, :]

    minutes = np.zeros((n_draws, n_rows), dtype="int16")
    is_cameo = (uniform >= p_none) & (uniform < p_none + p_cameo)
    is_full = uniform >= p_none + p_cameo

    minutes[is_cameo] = rng.integers(
        CAMEO_MINUTES[0], CAMEO_MINUTES[1] + 1, size=int(is_cameo.sum()), dtype="int16"
    )
    minutes[is_full] = rng.integers(
        FULL_MINUTES[0], FULL_MINUTES[1] + 1, size=int(is_full.sum()), dtype="int16"
    )
    return minutes


def _allocate_to_players(
    team_totals: np.ndarray,
    weights: np.ndarray,
    group_rows: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Distribute each team's sampled goals among its players.

    Conditional on a team scoring ``g`` goals, which players scored them is a multinomial draw
    with weights proportional to each player's scoring rate times the minutes he played. Sampled
    by the sequential-binomial construction, which is exact and vectorises over draws.

    A quietly important consequence: because the weights sum to something close to the team's
    baseline goal rate, allocating a *sampled* total automatically scales every player's
    expectation by fixture difficulty. No separate difficulty multiplier is needed, and none can
    drift out of sync with the team model.

    Args:
        team_totals: ``(n_draws, n_groups)`` goals to distribute.
        weights: ``(n_draws, n_rows)`` non-negative allocation weights.
        group_rows: ``(n_groups, max_squad)`` row indices, padded with -1.
        rng: Random source.

    Returns:
        ``(n_draws, n_rows)`` counts summing to ``team_totals`` within each group.
    """
    n_draws, n_rows = weights.shape
    n_groups, max_squad = group_rows.shape
    allocated = np.zeros((n_draws, n_rows), dtype="int16")

    valid = group_rows >= 0
    safe_rows = np.where(valid, group_rows, 0)

    # (n_draws, n_groups, max_squad)
    grouped = weights[:, safe_rows] * valid[None, :, :]
    remaining_weight = grouped.sum(axis=2)
    remaining_goals = team_totals.astype("int32").copy()

    for slot in range(max_squad):
        share = np.divide(
            grouped[:, :, slot],
            remaining_weight,
            out=np.zeros_like(remaining_weight),
            where=remaining_weight > 0,
        )
        np.clip(share, 0.0, 1.0, out=share)
        drawn = rng.binomial(remaining_goals, share)

        rows = group_rows[:, slot]
        active = rows >= 0
        if active.any():
            np.add.at(allocated, (slice(None), rows[active]), drawn[:, active].astype("int16"))

        remaining_goals -= drawn
        remaining_weight -= grouped[:, :, slot]
        np.clip(remaining_weight, 0.0, None, out=remaining_weight)

    return allocated


def _group_rows(frame: pd.DataFrame, key: str) -> tuple[np.ndarray, np.ndarray]:
    """Build a padded ``(n_groups, max_size)`` index table of row positions per group."""
    codes, uniques = pd.factorize(frame[key], sort=True)
    sizes = np.bincount(codes, minlength=len(uniques))
    max_size = int(sizes.max()) if len(sizes) else 0

    table = np.full((len(uniques), max_size), -1, dtype="int64")
    cursor = np.zeros(len(uniques), dtype="int64")
    for row, group in enumerate(codes):
        table[group, cursor[group]] = row
        cursor[group] += 1
    return table, uniques


def simulate(
    player_matches: pd.DataFrame,
    minutes_probabilities: np.ndarray,
    rules: ScoringRules,
    bps_model: BPSModel,
    *,
    rho: float,
    n_draws: int = 10_000,
    seed: int = 20262027,
    chunk_size: int = 2_000,
) -> SimulationResult:
    """Run the simulation and return joint point samples.

    Args:
        player_matches: Output of :func:`build_player_matches`, carrying per-row ``goal_rate``,
            ``assist_rate``, ``defcon_rate``, ``save_rate``, ``card_rate``, ``xg_home``,
            ``xg_away``, ``position``, ``element``, ``event``, ``fixture_id``, ``is_home``.
        minutes_probabilities: ``(n_rows, 3)`` probabilities of none / cameo / full, aligned to
            ``player_matches``.
        rules: Season scoring rules.
        bps_model: Fitted BPS model, used to sample bonus.
        rho: Dixon-Coles correlation from the team-strength fit.
        n_draws: Number of Monte Carlo draws.
        seed: Fixed so that repeated planning runs are comparable and reproducible. Changing it
            between two candidate plans would destroy the paired-comparison advantage.
        chunk_size: Draws processed at once. Bounds peak memory.

    Returns:
        A :class:`SimulationResult`.
    """
    frame = player_matches.reset_index(drop=True)
    n_rows = len(frame)
    if n_rows == 0:
        raise ValueError("no player-matches to simulate")

    fixture_table, fixture_ids = _group_rows(frame, "fixture_id")
    fixture_index = pd.Series(np.arange(len(fixture_ids)), index=fixture_ids)

    # One "team-match" per side of each fixture: the unit goals are allocated within.
    frame["team_match"] = frame["fixture_id"].astype(str) + ":" + frame["is_home"].astype(str)
    team_table, team_keys = _group_rows(frame, "team_match")
    team_is_home = np.array([key.endswith("True") for key in team_keys])
    team_fixture = np.array(
        [fixture_index[type(fixture_ids[0])(key.split(":")[0])] for key in team_keys]
    )

    # Per-fixture expected goals, taken from the first row belonging to each fixture.
    first_row = fixture_table[:, 0]
    lam_home = frame["xg_home"].to_numpy(dtype="float64")[first_row]
    lam_away = frame["xg_away"].to_numpy(dtype="float64")[first_row]

    elements, element_index = np.unique(frame["element"].to_numpy()), None
    element_position = pd.Series(np.arange(len(elements)), index=elements)
    row_element = element_position[frame["element"].to_numpy()].to_numpy()

    gameweeks = np.array(sorted(frame["event"].unique()))
    gw_position = pd.Series(np.arange(len(gameweeks)), index=gameweeks)
    row_gameweek = gw_position[frame["event"].to_numpy()].to_numpy()

    position = frame["position"].to_numpy()
    goal_weight = frame["goal_rate"].to_numpy(dtype="float64")
    assist_weight = frame["assist_rate"].to_numpy(dtype="float64")
    defcon_rate = frame["defcon_rate"].to_numpy(dtype="float64")
    save_rate = frame["save_rate"].to_numpy(dtype="float64")
    card_rate = frame["card_rate"].to_numpy(dtype="float64")
    threshold = np.array([DEFCON_THRESHOLDS.get(p, 10**6) for p in position], dtype="float64")

    points = np.zeros((n_draws, len(elements), len(gameweeks)), dtype="int16")
    minutes_out = np.zeros((n_draws, len(elements), len(gameweeks)), dtype="int16")

    rng = np.random.default_rng(seed)
    fixture_match_ids = frame["fixture_id"].to_numpy()

    for start in range(0, n_draws, chunk_size):
        size = min(chunk_size, n_draws - start)

        home_goals, away_goals = sample_scorelines(lam_home, lam_away, rho, size, rng)

        # Goals each team-match's players must share, and goals its opponents scored.
        scored = np.where(
            team_is_home[None, :], home_goals[:, team_fixture], away_goals[:, team_fixture]
        )
        conceded = np.where(
            team_is_home[None, :], away_goals[:, team_fixture], home_goals[:, team_fixture]
        )

        minutes = _sample_minutes(minutes_probabilities, rng, size)
        played = minutes > 0
        share_of_90 = minutes / 90.0

        # Only players on the pitch can be allocated a goal, and for longer if they played longer.
        goal_weights = goal_weight[None, :] * share_of_90
        goals = _allocate_to_players(scored, goal_weights, team_table, rng)

        # Assists: allocate a share of the team's goals, excluding those that went unassisted.
        assisted = rng.binomial(scored, 1.0 - UNASSISTED_GOAL_SHARE)
        assist_weights = assist_weight[None, :] * share_of_90
        assists = _allocate_to_players(assisted, assist_weights, team_table, rng)

        # Scatter per-team quantities back out to each player row.
        row_conceded = np.zeros((size, n_rows), dtype="int16")
        for slot in range(team_table.shape[1]):
            rows = team_table[:, slot]
            active = rows >= 0
            if active.any():
                row_conceded[:, rows[active]] = conceded[:, active]

        clean_sheet = ((row_conceded == 0) & (minutes >= 60)).astype("int16")

        # Defensive contribution: a count with more spread than Poisson, so negative binomial.
        # Overdispersion matters because the payoff is a threshold — a Poisson would understate
        # how often a mid-rate player has a big defensive game and clears the bar.
        defcon_mean = np.maximum(defcon_rate[None, :] * share_of_90, 1e-9)
        defcon = _negative_binomial(defcon_mean, rng, dispersion=4.0)
        hit_defcon = (defcon >= threshold[None, :]) & played

        is_gkp = position == "GKP"
        saves = np.zeros((size, n_rows), dtype="int16")
        if is_gkp.any():
            # Save volume scales with how much the opponent threatens, which the sampled goals
            # conceded proxies for; add a base rate so a shut-out keeper still makes saves.
            expected_saves = (
                save_rate[None, :] * share_of_90 * (0.6 + 0.4 * row_conceded)
            )
            saves = np.where(
                is_gkp[None, :], rng.poisson(np.maximum(expected_saves, 1e-9)), 0
            ).astype("int16")

        yellows = rng.binomial(
            1, np.clip(card_rate[None, :] * share_of_90, 0, 0.95)
        ).astype("int16")

        events = pd.DataFrame(
            {
                "position": np.tile(position, size),
                "minutes": minutes.ravel(),
                "goals_scored": goals.ravel(),
                "assists": assists.ravel(),
                "clean_sheets": clean_sheet.ravel(),
                "goals_conceded": row_conceded.ravel(),
                "saves": saves.ravel(),
                "yellow_cards": yellows.ravel(),
                "defcon_count": defcon.ravel(),
            }
        )

        # Bonus: sample BPS for everyone, then rank within each match to award 3/2/1.
        expected_bps = bps_model.predict(events)
        bps_noise = rng.normal(0.0, 1.0, size=len(events)) * bps_model.residual_scale(events)
        sampled_bps = np.maximum(expected_bps + bps_noise, 0.0)

        # Match ids must be unique per (draw, fixture) so ranking never spans draws.
        draw_offset = np.repeat(np.arange(size), n_rows) * (fixture_match_ids.max() + 1)
        bonus = allocate_bonus(sampled_bps, draw_offset + np.tile(fixture_match_ids, size))
        events["bonus"] = bonus

        from ..scoring import points_from_events  # local import avoids a cycle at module load

        totals = points_from_events(events, rules)["total"].to_numpy(dtype="float64")
        chunk_points = np.rint(totals).astype("int16").reshape(size, n_rows)

        # Accumulate into (draw, player, gameweek); += handles double gameweeks, where a player
        # legitimately contributes twice to the same slot.
        np.add.at(
            points,
            (slice(start, start + size), row_element, row_gameweek),
            chunk_points,
        )
        np.add.at(
            minutes_out,
            (slice(start, start + size), row_element, row_gameweek),
            minutes,
        )

        log.debug("simulated draws %d-%d", start, start + size)

    del element_index
    return SimulationResult(
        points=points,
        elements=elements,
        gameweeks=gameweeks,
        minutes_played=minutes_out,
        n_draws=n_draws,
    )


def _negative_binomial(
    mean: np.ndarray, rng: np.random.Generator, *, dispersion: float
) -> np.ndarray:
    """Negative binomial counts with the given mean and a fixed dispersion.

    Parameterised so that ``variance = mean * (1 + mean / dispersion)``. Used for defensive
    actions, where a Poisson understates how often a player has an unusually busy game — and
    since DEFCON pays on crossing a threshold, that tail is exactly what matters.
    """
    n = dispersion
    p = n / (n + np.maximum(mean, 1e-9))
    return rng.negative_binomial(n, np.clip(p, 1e-9, 1 - 1e-9)).astype("int16")
