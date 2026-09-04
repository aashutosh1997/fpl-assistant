"""Buy now, or wait for team news?

This is the decision the price tracker exists to serve, and it is genuinely a trade-off rather
than a rule. Transferring in a player before he rises earns you 0.1m of team value. Waiting until
the deadline lets you see the press conference, the injury update and sometimes the confirmed
lineup — and avoids the far more expensive mistake of buying a player who turns out to be injured.

Most FPL advice resolves this with folklore ("never transfer before Friday", "always beat the
rise"). Both sides are quantifiable:

**The gain from buying early** is the probability of a rise before your intended deadline, times
what 0.1m is worth to you in points.

**The cost of buying early** is the option value you give up: the chance that news arrives which
would have changed your mind, times the damage of being wrong.

The first of those needs an exchange rate between money and points, and the usual approach — a
made-up constant like "0.1m is worth 0.3 points" — is unsatisfying because the true value depends
entirely on your situation. A manager with 3.0m in the bank gains almost nothing from another
0.1m; a manager who needs exactly 0.1m to afford the striker they want gains a great deal.

So we take the exchange rate from the optimiser itself: re-solve with a slightly larger budget and
measure how much the objective improves. That is the shadow price of the budget constraint, and it
is exactly what an extra 0.1m is worth *to you, this week*. It is expensive to compute — one extra
solve — but it is the honest number, and it correctly collapses to near zero when money is not
what is binding you.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Probability that a player's availability materially changes between now and the deadline, by how
# far out we are. These are the fallback: once two transfer windows have been logged the table is
# measured from the hourly snapshots instead (see fplass.options.news), and the first two windows
# of 2026/27 put the week-out figure near 0.07, not 0.24.
NEWS_RISK_BY_DAYS = {0: 0.02, 1: 0.06, 2: 0.11, 3: 0.15, 4: 0.18, 5: 0.20, 6: 0.22, 7: 0.24}

# Points typically lost by owning a player who turns out not to play, versus the alternative you
# would have picked with better information. Roughly a full starter's expected return.
NEWS_MISTAKE_COST = 4.0


@dataclass(slots=True)
class TransferTiming:
    """The recommendation for one intended transfer."""

    element: int
    name: str
    p_rise: float
    p_fall: float
    days_to_deadline: float
    value_of_rise: float
    option_value_of_waiting: float
    recommendation: str
    reason: str


def budget_shadow_price(
    solve_fn,
    *,
    increment: int = 1,
    baseline_objective: float | None = None,
) -> float:
    """Points gained per 0.1m of extra budget, from the optimiser itself.

    Args:
        solve_fn: Callable taking an integer bank increment (in tenths) and returning a plan
            objective. Usually a closure over the MILP.
        increment: Budget bump to price, in tenths of a million.
        baseline_objective: The objective at the current budget, if already known.

    Returns:
        Points per 0.1m. Zero when money is not the binding constraint, which is the correct and
        commonly overlooked answer.
    """
    base = baseline_objective if baseline_objective is not None else solve_fn(0)
    bumped = solve_fn(increment)
    value = max((bumped - base) / max(increment, 1), 0.0)
    log.debug("budget shadow price: %.3f points per 0.1m", value)
    return float(value)


def news_risk(days_to_deadline: float, table: pd.Series | None = None) -> float:
    """Probability that materially new team news arrives before the deadline.

    Args:
        table: A measured risk by whole days from :func:`fplass.options.news.news_risk_table`;
            the constants above are used when none is available.
    """
    day = int(np.clip(np.floor(days_to_deadline), 0, 7))
    if table is not None and day in table.index:
        return float(table.loc[day])
    return NEWS_RISK_BY_DAYS[day]


def decide(
    targets: pd.DataFrame,
    price_probabilities: pd.DataFrame,
    *,
    days_to_deadline: float,
    points_per_tenth: float,
    mistake_cost: float = NEWS_MISTAKE_COST,
    risk_table: pd.Series | None = None,
) -> pd.DataFrame:
    """Recommend buy-now or wait for each intended transfer.

    Args:
        targets: Players you intend to buy, with ``element`` and ``web_name``.
        price_probabilities: Output of the price model: ``element``, ``p_rise``, ``p_fall``.
        days_to_deadline: How long until you must commit.
        points_per_tenth: The budget shadow price from :func:`budget_shadow_price`.
        mistake_cost: Points lost by committing to a player who turns out not to play.
        risk_table: Measured news risk by days to go, when the snapshots can supply one.

    Returns:
        One row per target, with the two competing values made explicit so the recommendation can
        be argued with rather than merely obeyed.
    """
    merged = targets.merge(price_probabilities, on="element", how="left")
    merged["p_rise"] = merged["p_rise"].fillna(0.0)
    merged["p_fall"] = merged["p_fall"].fillna(0.0)

    risk = news_risk(days_to_deadline, risk_table)
    merged["value_of_rise"] = merged["p_rise"] * points_per_tenth
    # Waiting is only worth something if news could still change the decision, and only in
    # proportion to how much a wrong pick would cost.
    merged["option_value_of_waiting"] = risk * mistake_cost

    recommendations, reasons = [], []
    for row in merged.itertuples():
        if row.value_of_rise > row.option_value_of_waiting:
            recommendations.append("buy now")
            reasons.append(
                f"{row.p_rise:.0%} chance of a rise worth {row.value_of_rise:.2f} pts, "
                f"against {row.option_value_of_waiting:.2f} pts of news risk"
            )
        elif row.p_rise > 0.5 and points_per_tenth < 0.05:
            recommendations.append("wait")
            reasons.append(
                "likely to rise, but you have spare budget so the extra 0.1m buys you nothing"
            )
        else:
            recommendations.append("wait")
            reasons.append(
                f"news risk worth {row.option_value_of_waiting:.2f} pts exceeds the "
                f"{row.value_of_rise:.2f} pts of price gain"
            )

    merged["recommendation"] = recommendations
    merged["reason"] = reasons
    merged["days_to_deadline"] = days_to_deadline
    return merged


def sell_alerts(
    squad: list[int],
    price_probabilities: pd.DataFrame,
    players: pd.DataFrame,
    *,
    threshold: float = 0.6,
) -> pd.DataFrame:
    """Players you own who are likely to drop in price before the next change.

    A fall is not by itself a reason to sell — the player's expected points have not changed — but
    it is a reason to bring forward a transfer you were going to make anyway, and that is how the
    output is framed.
    """
    owned = price_probabilities[price_probabilities["element"].isin(squad)]
    at_risk = owned[owned["p_fall"] >= threshold]
    if at_risk.empty:
        return at_risk.assign(web_name=[], urgency=[])

    merged = at_risk.merge(players[["element", "web_name", "price"]], on="element", how="left")
    merged["urgency"] = np.where(merged["p_fall"] > 0.85, "tonight", "within a day or two")
    return merged.sort_values("p_fall", ascending=False, ignore_index=True)


def buy_alerts(
    watchlist: list[int],
    price_probabilities: pd.DataFrame,
    players: pd.DataFrame,
    *,
    threshold: float = 0.6,
) -> pd.DataFrame:
    """Players you are considering who are likely to rise before the next change."""
    watched = price_probabilities[price_probabilities["element"].isin(watchlist)]
    at_risk = watched[watched["p_rise"] >= threshold]
    if at_risk.empty:
        return at_risk.assign(web_name=[], urgency=[])

    merged = at_risk.merge(players[["element", "web_name", "price"]], on="element", how="left")
    merged["urgency"] = np.where(merged["p_rise"] > 0.85, "tonight", "within a day or two")
    return merged.sort_values("p_rise", ascending=False, ignore_index=True)
