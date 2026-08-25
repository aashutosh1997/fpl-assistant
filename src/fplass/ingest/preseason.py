"""Preseason friendlies: the only observation of the current squad hierarchy before a ball is kicked.

Gameweek 1 exposed the model's largest blind spot. It projects minutes from last season's rolling
form, which is the best available signal in October and close to worthless in August, because squad
roles are exactly what changes over a summer. Joao Pedro was rated 0.69 to play an hour and played
ninety; Dubravka was rated 0.93 and played none.

Both were knowable. Joao Pedro played 80 minutes a game across four friendlies. The information
existed; we simply were not reading it.

Source is the community dataset at github.com/olbauday/FPL-Core-Insights, which publishes preseason
matches with team-level expected goals and per-player minutes, goals and shots. It joins to FPL
cleanly and without fuzzy matching: its ``player_id`` *is* the FPL element id and its ``team_code``
is the FPL club code.

**Coverage is the important caveat.** Friendlies are published for 2026-27 only — earlier seasons in
that repository have no ``Friendlies`` directory. So this feature cannot be validated on historical
backtests, and everything downstream must degrade cleanly to "no preseason data" rather than
assuming it is present. That is also why it feeds a separate recalibration layer
(:mod:`fplass.features.adjust`) rather than the base minutes model: the base model is trained on ten
seasons in which this feature does not exist at all, and could never learn a weight for it.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass

import httpx
import pandas as pd

from ..paths import RAW
from .warehouse import upsert

log = logging.getLogger(__name__)

BASE = "https://raw.githubusercontent.com/olbauday/FPL-Core-Insights/main/data"

# Seasons known to publish a Friendlies directory. Checked rather than assumed: 2024-25 and 2025-26
# do not have one, which is why this list is explicit instead of derived.
SEASONS_WITH_FRIENDLIES: dict[str, str] = {"2026-27": "2026-2027"}

FILES = ("matches.csv", "playermatchstats.csv", "players.csv")


@dataclass(slots=True)
class PreseasonData:
    """Loaded preseason tables, already keyed on FPL ids."""

    players: pd.DataFrame  # one row per element: minutes, goals, xG across friendlies
    teams: pd.DataFrame  # one row per club code: goals and xG for and against
    n_matches: int


def _url(upstream_season: str, name: str) -> str:
    # The upstream path contains a literal space that must survive as %20.
    return f"{BASE}/{upstream_season}/By%20Tournament/Friendlies/GW0/{name}"


def fetch(season: str, *, refresh: bool = False, timeout: float = 120.0) -> dict[str, pd.DataFrame]:
    """Download the preseason files for a season, caching under ``data/raw``.

    Returns an empty mapping when the season publishes no friendlies, which is the normal case for
    every season before 2026-27.
    """
    upstream = SEASONS_WITH_FRIENDLIES.get(season)
    if upstream is None:
        log.info("%s: no preseason friendlies published upstream", season)
        return {}

    out: dict[str, pd.DataFrame] = {}
    for name in FILES:
        local = RAW / season / "friendlies" / name
        if local.exists() and not refresh and local.stat().st_size > 0:
            out[name] = pd.read_csv(local)
            continue

        local.parent.mkdir(parents=True, exist_ok=True)
        log.info("downloading preseason %s/%s", season, name)
        try:
            with httpx.Client(timeout=timeout, follow_redirects=True) as client:
                resp = client.get(_url(upstream, name))
        except httpx.TransportError as exc:
            log.error("preseason download failed (%s): %s", name, exc)
            return {}
        if resp.status_code == 404:
            log.warning("preseason %s/%s not available (404)", season, name)
            return {}
        resp.raise_for_status()

        local.write_bytes(resp.content)
        out[name] = pd.read_csv(io.BytesIO(resp.content))
    return out


def build(season: str, *, refresh: bool = False) -> PreseasonData | None:
    """Aggregate preseason friendlies into per-player and per-club summaries."""
    raw = fetch(season, refresh=refresh)
    if not raw:
        return None

    stats = raw["playermatchstats.csv"]
    matches = raw["matches.csv"]

    players = (
        stats.groupby("player_id")
        .agg(
            preseason_matches=("minutes_played", "size"),
            preseason_minutes=("minutes_played", "sum"),
            preseason_minutes_avg=("minutes_played", "mean"),
            preseason_minutes_max=("minutes_played", "max"),
            preseason_goals=("goals", "sum"),
            preseason_assists=("assists", "sum"),
            preseason_xg=("xg", "sum"),
            preseason_xa=("xa", "sum"),
        )
        .reset_index()
        .rename(columns={"player_id": "element"})
    )
    players.insert(0, "season", season)

    # A player's *share* of available preseason minutes is more comparable than the raw total,
    # because clubs play different numbers of friendlies and rest players at different rates.
    club_matches = _club_match_counts(matches)
    players["preseason_minutes_share"] = players["preseason_minutes_avg"] / 90.0

    teams = _team_summary(matches, season)
    teams = teams.merge(club_matches, on="code", how="left")

    log.info(
        "%s preseason: %d matches, %d players, %d clubs",
        season,
        len(matches),
        len(players),
        len(teams),
    )
    return PreseasonData(players=players, teams=teams, n_matches=len(matches))


def _club_match_counts(matches: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for side in ("home_team", "away_team"):
        codes = pd.to_numeric(matches[side], errors="coerce").dropna().astype(int)
        rows.append(codes.rename("code"))
    counts = pd.concat(rows).value_counts().rename("preseason_fixtures")
    return counts.reset_index().rename(columns={"index": "code"})


def _team_summary(matches: pd.DataFrame, season: str) -> pd.DataFrame:
    """Per-club preseason goals and expected goals, for and against.

    Only sides with an FPL club code are kept — friendlies are frequently against lower-league or
    foreign opposition that has no code, and those rows carry no information about our clubs beyond
    the result, which is already captured from our side.
    """
    rows = []
    for side, other in (("home", "away"), ("away", "home")):
        code = pd.to_numeric(matches[f"{side}_team"], errors="coerce")
        frame = pd.DataFrame(
            {
                "code": code,
                "goals_for": pd.to_numeric(matches[f"{side}_score"], errors="coerce"),
                "goals_against": pd.to_numeric(matches[f"{other}_score"], errors="coerce"),
                "xg_for": pd.to_numeric(
                    matches.get(f"{side}_expected_goals_xg"), errors="coerce"
                ),
                "xg_against": pd.to_numeric(
                    matches.get(f"{other}_expected_goals_xg"), errors="coerce"
                ),
            }
        )
        rows.append(frame[frame["code"].notna() & frame["goals_for"].notna()])

    combined = pd.concat(rows, ignore_index=True)
    combined["code"] = combined["code"].astype(int)
    summary = (
        combined.groupby("code")
        .agg(
            preseason_played=("goals_for", "size"),
            preseason_goals_for=("goals_for", "mean"),
            preseason_goals_against=("goals_against", "mean"),
            preseason_xg_for=("xg_for", "mean"),
            preseason_xg_against=("xg_against", "mean"),
        )
        .reset_index()
    )
    summary.insert(0, "season", season)
    return summary


def load(con, season: str, *, refresh: bool = False) -> dict[str, int]:
    """Load a season's preseason data into the warehouse."""
    data = build(season, refresh=refresh)
    if data is None:
        return {}
    return {
        "preseason_player": upsert(
            con, "preseason_player", data.players, ("season", "element")
        ),
        "preseason_team": upsert(con, "preseason_team", data.teams, ("season", "code")),
    }


def player_features(con, season: str) -> pd.DataFrame:
    """Per-element preseason features for the minutes adjustment layer.

    Always returns a row per element in the season, with ``preseason_observed`` distinguishing
    "played no preseason minutes" from "we have no preseason data at all". Conflating those two is
    the obvious trap: a player at a club we never observed is not the same as a player who was
    left out of every friendly.
    """
    rows = con.execute(
        """
        SELECT
            pl.element,
            COALESCE(ps.preseason_matches, 0)        AS preseason_matches,
            COALESCE(ps.preseason_minutes_avg, 0.0)  AS preseason_minutes_avg,
            COALESCE(ps.preseason_minutes_max, 0.0)  AS preseason_minutes_max,
            COALESCE(ps.preseason_goals, 0.0)        AS preseason_goals,
            COALESCE(ps.preseason_xg, 0.0)           AS preseason_xg,
            CASE WHEN ps.element IS NULL THEN 0.0 ELSE 1.0 END AS preseason_observed
        FROM players pl
        LEFT JOIN preseason_player ps
               ON ps.season = pl.season AND ps.element = pl.element
        WHERE pl.season = ?
        """,
        [season],
    ).fetchdf()
    return rows
