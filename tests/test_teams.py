"""Tests for the Dixon-Coles team strength model."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fplass.features import teams as T


@pytest.fixture(scope="module")
def results(con) -> pd.DataFrame:
    return T.flag_promoted(T.load_results(con))


@pytest.fixture(scope="module")
def strength(results) -> T.TeamStrength:
    return T.fit(results)


def test_all_seasons_contribute_results(results):
    """Every season must reach the model, including the three with missing upstream files.

    2016-17 and 2017-18 have no fixtures.csv and 2016-17..2018-19 no teams.csv; both are
    reconstructed at ingest. If that regresses we silently lose a third of the match history —
    including two of only four seasons carrying defensive-action data.
    """
    per_season = results.groupby("season").size()
    assert len(per_season) >= 10, f"only {len(per_season)} seasons reached the model"
    for season in ("2016-17", "2017-18", "2018-19"):
        assert per_season.get(season, 0) > 370, f"{season} is missing matches"


def test_ratings_are_football_shaped(strength, con):
    """The strongest sides should come out on top; a sanity check that signs are not flipped."""
    names = dict(
        con.execute(
            "SELECT DISTINCT code, short_name FROM teams WHERE short_name IS NOT NULL"
        ).fetchall()
    )
    table = strength.effective_table()
    table["name"] = table["code"].map(names)
    top_five = set(table.head(5)["name"])
    assert {"MCI", "LIV", "ARS"} <= top_five, f"unexpected top attacks: {top_five}"


def test_promotion_correction_prevents_inflated_ratings(strength, con):
    """A club whose entire history is one promoted season must not be rated a top side.

    The promoted handicap and a club's own rating are not separately identified in that case. Left
    uncorrected, the handicap switches off the following season and appears as genuine quality —
    which rated Sunderland the third-best defence in the league, above Liverpool, on twenty-nine
    effective matches. The fix is the promoted-share correction in ``TeamStrength._rating``.
    """
    names = dict(
        con.execute(
            "SELECT DISTINCT code, short_name FROM teams WHERE short_name IS NOT NULL"
        ).fetchall()
    )
    effective = strength.effective_table()
    effective["name"] = effective["code"].map(names)

    mostly_promoted = effective[effective["promoted_share"] > 0.9]
    assert len(mostly_promoted) > 0, "expected some clubs with an all-promoted history"

    established = effective[effective["promoted_share"] < 0.1]
    top_defences = established.nlargest(3, "defence")["defence"].min()
    assert mostly_promoted["defence"].max() < top_defences, (
        "a club known only from a promoted season should not out-rate the established elite"
    )

    # And the raw fitted parameter really is inflated relative to the corrected one, which is
    # what makes the correction necessary rather than cosmetic.
    raw = strength.table().set_index("code")
    for code in mostly_promoted["code"]:
        corrected = effective.set_index("code").loc[code, "defence"]
        assert corrected < raw.loc[code, "defence"]


def test_home_advantage_is_positive_and_plausible(strength):
    assert 0.05 < strength.home_advantage < 0.40


def test_dixon_coles_rho_is_negative(strength):
    """A negative rho is what lifts 0-0 and 1-1 probability above independent Poisson.

    Clean sheets are worth 4 points to half a squad, so the low-score mass has to be right.
    """
    assert -0.35 < strength.rho < 0.0


def test_promotion_penalties_are_negative(strength):
    """Promoted sides score less and concede more; a positive penalty would mean a sign error."""
    assert strength.promoted_attack < 0
    assert strength.promoted_defence < 0
    assert strength.debutant_attack < 0
    assert strength.debutant_defence < 0


def test_debutant_prior_used_only_for_unrated_clubs(strength):
    """A club with no history falls back to the prior; a rated club never has it added on top."""
    unknown_code = -12345
    attack, defence = strength._rating(unknown_code, promoted=True)
    assert attack == strength.debutant_attack
    assert defence == strength.debutant_defence
    # The handicap is already inside the prior, so flagging promotion must not double-count it.
    assert strength._rating(unknown_code, promoted=False) == (attack, defence)


def test_thin_data_teams_are_shrunk_toward_average(strength):
    """A team with almost no observed matches must not be rated as an outlier.

    This is the failure the ridge tuning fixed: untuned, a team with seven effective matches was
    rated the fourth-best attack in the league.
    """
    table = strength.table()
    thin = table[table["matches"] < 5]
    assert len(thin) > 0, "expected some barely-observed teams in eleven seasons of data"
    assert thin["attack"].abs().max() < 0.10, (
        "teams with under five effective matches should sit near the league mean"
    )


def test_stronger_team_gets_higher_expected_goals(strength, con):
    """The headline use of the model: a good side at home to a weak side should dominate."""
    codes = dict(
        con.execute(
            "SELECT short_name, code FROM teams WHERE season = '2026-27' AND code IS NOT NULL"
        ).fetchall()
    )
    promotion = T.promotion_states(con, "2026-27")
    promoted = promotion["returning"] | promotion["debutant"]
    strong, weak = codes["MCI"], codes["COV"]

    lam_strong, lam_weak = strength.rates(strong, weak, away_promoted=weak in promoted)
    assert lam_strong > lam_weak
    assert lam_strong > 1.5, "a title contender at home should be expected to score freely"
    assert 0.2 < lam_weak < 1.5


def test_home_advantage_shows_up_in_rates(strength, con):
    """The same pairing must favour whichever side is at home."""
    codes = dict(
        con.execute(
            "SELECT short_name, code FROM teams WHERE season = '2026-27' AND code IS NOT NULL"
        ).fetchall()
    )
    a, b = codes["ARS"], codes["CHE"]
    a_home, b_away = strength.rates(a, b)
    b_home, a_away = strength.rates(b, a)
    assert a_home > a_away, "Arsenal should be expected to score more at home than away"
    assert b_home > b_away


def test_fixture_rates_cover_the_whole_season(strength, con):
    rates = T.fixture_rates(con, strength, "2026-27")
    assert len(rates) == 380
    assert rates["xg_home"].between(0.1, 5.0).all()
    assert rates["xg_away"].between(0.1, 5.0).all()
    # Home sides collectively outscore away sides.
    assert rates["xg_home"].mean() > rates["xg_away"].mean()


def test_promotion_states_classify_this_season(con):
    promotion = T.promotion_states(con, "2026-27")
    codes = dict(
        con.execute(
            "SELECT short_name, code FROM teams WHERE season = '2026-27' AND code IS NOT NULL"
        ).fetchall()
    )
    promoted = promotion["returning"] | promotion["debutant"]
    assert len(promoted) == 3, f"expected three promoted clubs, got {len(promoted)}"
    assert not (promotion["returning"] & promotion["debutant"]), "a club cannot be both"
    # Coventry have no prior season in our data window, so they are a debutant here.
    assert codes["COV"] in promotion["debutant"]


def test_time_decay_downweights_old_matches(results):
    """A shorter half-life must shift ratings toward recent form.

    Verified via the effective sample size, which is the decay weighting made visible.
    """
    slow = T.fit(results, decay=0.0005)
    fast = T.fit(results, decay=0.0060)
    assert sum(fast.match_counts.values()) < sum(slow.match_counts.values())


def test_ridge_controls_rating_spread(results):
    """More ridge means less spread. This is the knob cross-validation tuned."""
    light = T.fit(results, ridge=0.05)
    heavy = T.fit(results, ridge=50.0)
    assert np.std(list(heavy.attack.values())) < np.std(list(light.attack.values()))


def test_up_to_excludes_future_matches(con):
    """Backtests depend on this: fitting must be able to ignore everything after a cutoff."""
    cutoff = pd.Timestamp("2022-01-01")
    limited = T.load_results(con, up_to=cutoff)
    assert limited["kickoff_time"].max() < cutoff
    assert len(limited) < len(T.load_results(con))


def test_score_predictions_prefers_the_right_model(results):
    """Out-of-sample scoring must rank a fitted model above a deliberately blinded one."""
    ordered = results.sort_values("kickoff_time").reset_index(drop=True)
    cut = len(ordered) - 300
    train, holdout = ordered.iloc[:cut], ordered.iloc[cut:]

    fitted = T.fit(train, reference_time=train["kickoff_time"].max())
    flattened = T.fit(train, ridge=1e6, reference_time=train["kickoff_time"].max())

    assert (
        T.score_predictions(fitted, holdout)["mean_log_likelihood"]
        > T.score_predictions(flattened, holdout)["mean_log_likelihood"]
    ), "team ratings should beat treating every team as identical"
