"""Sampling match scorelines from the Dixon-Coles model.

Dixon-Coles is not a bivariate distribution you can sample from directly. It is two independent
Poisson margins with a correction factor ``tau`` applied to the four lowest-scoring cells (0-0,
1-0, 0-1, 1-1), which is what fixes independent Poisson's well-known tendency to under-predict
draws. Because the correction only touches a handful of cells, the cleanest exact approach is to
build the joint probability table over a truncated grid of scorelines, apply the correction,
renormalise, and sample from the resulting categorical distribution.

Getting the low-scoring cells right is not a refinement. Clean sheets are worth 4 points to
goalkeepers and defenders, and the probability of a 0-0 or a 1-0 is precisely what a clean-sheet
projection is. An independent-Poisson simulator systematically misprices every defender in the
game, and therefore misprices Bench Boost, which is disproportionately defensive.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import poisson

# Scorelines above this are vanishingly rare and cost grid size to represent. A 9-9 draw has
# never happened in Premier League history; the truncation error is far below sampling noise.
MAX_GOALS = 9


def joint_scoreline_pmf(
    lam_home: np.ndarray, lam_away: np.ndarray, rho: float, *, max_goals: int = MAX_GOALS
) -> np.ndarray:
    """Joint probability of each scoreline, for each fixture.

    Args:
        lam_home: Expected home goals, one per fixture.
        lam_away: Expected away goals, one per fixture.
        rho: Dixon-Coles low-score dependence (negative in practice).
        max_goals: Grid truncation, inclusive.

    Returns:
        Array of shape ``(n_fixtures, max_goals + 1, max_goals + 1)`` summing to 1 per fixture.
    """
    lam_home = np.asarray(lam_home, dtype="float64").reshape(-1, 1, 1)
    lam_away = np.asarray(lam_away, dtype="float64").reshape(-1, 1, 1)

    goals = np.arange(max_goals + 1)
    home_pmf = poisson.pmf(goals.reshape(1, -1, 1), lam_home)
    away_pmf = poisson.pmf(goals.reshape(1, 1, -1), lam_away)
    joint = home_pmf * away_pmf

    # Dixon-Coles correction on the four low-scoring cells.
    tau = np.ones_like(joint)
    tau[:, 0, 0] = 1 - (lam_home * lam_away * rho)[:, 0, 0]
    tau[:, 1, 0] = 1 + (lam_away * rho)[:, 0, 0]
    tau[:, 0, 1] = 1 + (lam_home * rho)[:, 0, 0]
    tau[:, 1, 1] = 1 - rho

    joint = joint * np.clip(tau, 0.0, None)
    # Renormalise: both the correction and the grid truncation cost a little total mass.
    return joint / joint.sum(axis=(1, 2), keepdims=True)


def sample_scorelines(
    lam_home: np.ndarray,
    lam_away: np.ndarray,
    rho: float,
    n_draws: int,
    rng: np.random.Generator,
    *,
    max_goals: int = MAX_GOALS,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw correlated scorelines for every fixture.

    Returns:
        ``(home_goals, away_goals)``, each of shape ``(n_draws, n_fixtures)`` as int16.
    """
    pmf = joint_scoreline_pmf(lam_home, lam_away, rho, max_goals=max_goals)
    n_fixtures = pmf.shape[0]
    side = max_goals + 1

    flat = pmf.reshape(n_fixtures, -1)
    cumulative = np.cumsum(flat, axis=1)
    # Guard against floating-point shortfall leaving the last bin unreachable.
    cumulative[:, -1] = 1.0

    uniform = rng.random((n_draws, n_fixtures))
    # searchsorted per fixture; a small loop over fixtures is cheaper than building a 3D
    # comparison against the full cumulative table.
    cells = np.empty((n_draws, n_fixtures), dtype="int32")
    for f in range(n_fixtures):
        cells[:, f] = np.searchsorted(cumulative[f], uniform[:, f], side="right")
    np.clip(cells, 0, side * side - 1, out=cells)

    return (cells // side).astype("int16"), (cells % side).astype("int16")


def clean_sheet_probability(
    lam_home: np.ndarray, lam_away: np.ndarray, rho: float
) -> tuple[np.ndarray, np.ndarray]:
    """Analytic clean-sheet probability for each side, for validating the sampler."""
    pmf = joint_scoreline_pmf(lam_home, lam_away, rho)
    # The home side keeps a clean sheet when the away side fails to score.
    return pmf[:, :, 0].sum(axis=1), pmf[:, 0, :].sum(axis=1)
