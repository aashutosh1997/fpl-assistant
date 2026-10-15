"""A team's expected goals, adjusted for which of its attackers the news rules out.

The team-strength model rates a club on its results, and those results were earned by its usual
lineup. When the injury news takes attackers out of that lineup, the lineup constraint correctly
hands their minutes to club-mates, but with the team's expected goals unchanged it hands over
their goals too. Liverpool before gameweek 7 of 2026-27 is the case that exposed it: Isak,
Ekitiké and Gakpo all out with no return date, the club still projected to score 1.64 a match,
and Wirtz's expected goals over eight gameweeks rose from 7.9 to 10.8 because he inherited their
share. That was the whole of the optimiser's case for selling Rogers for him.

The correction scales each side's expected goals by the attacking weight it fields with the news
against the weight it would field without it:

    xg_adjusted = xg * (weight_with_news / weight_without_news) ** ATTACK_ELASTICITY

where a weight is the sum over the squad of goal rate times the probability of a sixty-minute
appearance, both lineups taken after the lineup constraint. A club with no news is untouched.

The exponent is measured, not chosen. Over 2,528 team-matches from 2022-23 to 2025-26, with goal
rates frozen before each season and each club's starting attacking weight compared with its own
previous ten matches, a Poisson fit of goals on the fixture rate and the weight ratio gives 0.44
(standard error 0.05); the same fit on team xG gives 0.43. Measured on sixty-minute appearances
rather than starts it is 0.51 on goals and 0.39 on xG. It is well below one because the
replacements are not nothing: a side missing a fifth of its usual attacking weight scores about
nine percent less, not twenty. Teams missing more than 15% of their usual weight scored 0.885 of
the fixture rate.

The baseline is the projection without the news rather than the club's recent matches, because a
projected weight and a realised one are not on the same scale: compared directly they put almost
every club above one, and Manchester City at 1.21, on squad turnover rather than injuries. The cost
is that an absence already visible in a player's recent minutes is not counted again — which is
also the conservative reading of a team rating that has partly absorbed it.

Only the attack is adjusted. A missing defender should raise the opponent's expected goals by the
same reasoning, but that has not been measured, and an unmeasured correction is not applied.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Elasticity of a team's goals to the attacking weight it fields; see the module docstring.
ATTACK_ELASTICITY = 0.44

# Bounds on the multiplier. News only takes players away, so it cannot strengthen an attack: a
# ratio above one is the lineup constraint spreading an absent defender's minutes across the
# squad, not a better forward line. The floor stops a data error from zeroing a team, and binds
# only when a side keeps under a third of its usual attacking weight.
MIN_SCALE, MAX_SCALE = 0.6, 1.0


def scale_for_lineups(
    player_matches: pd.DataFrame,
    p_full: np.ndarray,
    p_full_usual: np.ndarray,
    *,
    elasticity: float = ATTACK_ELASTICITY,
) -> pd.DataFrame:
    """Scale ``xg_home`` and ``xg_away`` by the attacking weight each side keeps after the news.

    Args:
        player_matches: One row per player-fixture with ``fixture_id``, ``is_home``,
            ``goal_rate``, ``xg_home`` and ``xg_away``.
        p_full: Each row's probability of a sixty-minute appearance with the news applied.
        p_full_usual: The same without the news.

    Returns:
        A copy with the expected goals scaled and the multipliers kept as ``attack_scale_home``
        and ``attack_scale_away``.
    """
    out = player_matches.reset_index(drop=True).copy()
    rate = out["goal_rate"].fillna(0.0).to_numpy(dtype="float64")
    side = pd.DataFrame(
        {
            "fixture_id": out["fixture_id"].to_numpy(),
            "is_home": out["is_home"].to_numpy(dtype=bool),
            "now": rate * np.asarray(p_full, dtype="float64"),
            "usual": rate * np.asarray(p_full_usual, dtype="float64"),
        }
    )
    weights = side.groupby(["fixture_id", "is_home"])[["now", "usual"]].sum()
    ratio = (weights["now"] / weights["usual"]).where(weights["usual"] > 0, 1.0)
    scale = np.clip(ratio**elasticity, MIN_SCALE, MAX_SCALE)

    n = len(out)
    home = pd.MultiIndex.from_arrays([out["fixture_id"].to_numpy(), np.ones(n, dtype=bool)])
    away = pd.MultiIndex.from_arrays([out["fixture_id"].to_numpy(), np.zeros(n, dtype=bool)])
    out["attack_scale_home"] = scale.reindex(home).fillna(1.0).to_numpy()
    out["attack_scale_away"] = scale.reindex(away).fillna(1.0).to_numpy()
    out["xg_home"] = out["xg_home"] * out["attack_scale_home"]
    out["xg_away"] = out["xg_away"] * out["attack_scale_away"]
    return out
