"""Downloading and caching the raw historical source files.

Source is the community dataset at github.com/vaastav/Fantasy-Premier-League, which mirrors the
official FPL API season by season from 2016-17 onward. Files are cached under ``data/raw`` so a
warehouse rebuild does not re-download ~200MB.
"""

from __future__ import annotations

import logging
from pathlib import Path

import httpx

from ..paths import RAW

log = logging.getLogger(__name__)

VAASTAV_BASE = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"

# Every season the upstream dataset covers, oldest first.
SEASONS: tuple[str, ...] = (
    "2016-17",
    "2017-18",
    "2018-19",
    "2019-20",
    "2020-21",
    "2021-22",
    "2022-23",
    "2023-24",
    "2024-25",
    "2025-26",
    "2026-27",
)

CURRENT_SEASON = "2026-27"


def season_files(season: str) -> dict[str, str]:
    """Relative paths of the files we consume for a season."""
    return {
        "merged_gw": f"{season}/gws/merged_gw.csv",
        "players_raw": f"{season}/players_raw.csv",
        "fixtures": f"{season}/fixtures.csv",
        "teams": f"{season}/teams.csv",
    }


def fetch(relative: str, *, refresh: bool = False, timeout: float = 120.0) -> Path | None:
    """Download one source file into the raw cache and return its local path.

    Returns ``None`` for a 404 — early seasons legitimately lack some files (2016-17 has no
    ``teams.csv``), and that is a fact about the data rather than an error to abort on.
    """
    local = RAW / relative
    if local.exists() and not refresh and local.stat().st_size > 0:
        return local

    url = f"{VAASTAV_BASE}/{relative}"
    local.parent.mkdir(parents=True, exist_ok=True)
    log.info("downloading %s", relative)

    try:
        with httpx.Client(timeout=timeout, follow_redirects=True) as client:
            resp = client.get(url)
    except httpx.TransportError as exc:
        log.error("failed to download %s: %s", relative, exc)
        raise

    if resp.status_code == 404:
        log.warning("%s not available upstream (404)", relative)
        return None
    resp.raise_for_status()

    tmp = local.with_suffix(local.suffix + ".tmp")
    tmp.write_bytes(resp.content)
    tmp.replace(local)
    return local


def fetch_season(season: str, *, refresh: bool = False) -> dict[str, Path | None]:
    """Download every source file for one season."""
    return {name: fetch(rel, refresh=refresh) for name, rel in season_files(season).items()}
