"""FPL public API client.

Only public, unauthenticated endpoints are used. Nothing here needs your login.

Two things this module exists to get right:

* **Cloudflare.** FPL sits behind Cloudflare and will 403 a bare client (notably from cloud
  runner IPs). We send a browser-like User-Agent and retry with backoff on 403/429/5xx.
* **Caching.** Backtests and repeated planning runs hammer the same endpoints. Responses are
  cached on disk with a per-endpoint TTL so a planning run is cheap and reproducible.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from .paths import CACHE

log = logging.getLogger(__name__)

BASE = "https://fantasy.premierleague.com/api"

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
)

HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-GB,en;q=0.9",
    "Referer": "https://fantasy.premierleague.com/",
}

RETRY_STATUS = frozenset({403, 408, 425, 429, 500, 502, 503, 504})

# Endpoints whose volatility differs sharply from the default TTL.
TTLS: dict[str, float] = {
    # Price fields move hourly, so never serve a stale bootstrap to the price layer;
    # callers that want a guaranteed-fresh read pass ttl=0 explicitly.
    "bootstrap-static/": 300.0,
    "fixtures/": 3600.0,
    "event-status/": 120.0,
}


class FPLAPIError(RuntimeError):
    """Raised when an endpoint cannot be fetched after retries."""


@dataclass(slots=True)
class FPLClient:
    """Thin, cached, retrying client for the FPL public API.

    Args:
        timeout: Per-request timeout in seconds.
        max_retries: Attempts per request before raising.
        cache_dir: Where cached JSON lives. ``None`` disables caching entirely.
        default_ttl: Cache lifetime in seconds for endpoints without a specific TTL.
    """

    timeout: float = 30.0
    max_retries: int = 5
    cache_dir: Path | None = CACHE
    default_ttl: float = 900.0
    _client: httpx.Client | None = None

    # ---------------------------------------------------------------- internals

    def _http(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(
                base_url=BASE, headers=HEADERS, timeout=self.timeout, follow_redirects=True
            )
        return self._client

    def _cache_path(self, endpoint: str) -> Path | None:
        if self.cache_dir is None:
            return None
        digest = hashlib.sha256(endpoint.encode()).hexdigest()[:20]
        safe = endpoint.strip("/").replace("/", "_") or "root"
        return self.cache_dir / f"{safe}.{digest}.json"

    def _read_cache(self, path: Path | None, ttl: float) -> Any | None:
        if path is None or ttl <= 0 or not path.exists():
            return None
        if time.time() - path.stat().st_mtime > ttl:
            return None
        try:
            return json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            return None

    def _write_cache(self, path: Path | None, payload: Any) -> None:
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        # Write-then-rename so a crashed run never leaves a half-written cache file that a
        # later run would happily parse as truncated-but-valid JSON.
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload))
        tmp.replace(path)

    # -------------------------------------------------------------------- fetch

    def get(self, endpoint: str, *, ttl: float | None = None) -> Any:
        """Fetch ``endpoint`` (relative to the API base), with caching and retries."""
        endpoint = endpoint.lstrip("/")
        effective_ttl = TTLS.get(endpoint, self.default_ttl) if ttl is None else ttl

        path = self._cache_path(endpoint)
        cached = self._read_cache(path, effective_ttl)
        if cached is not None:
            log.debug("cache hit %s", endpoint)
            return cached

        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                resp = self._http().get(endpoint)
                if resp.status_code == 200:
                    payload = resp.json()
                    self._write_cache(path, payload)
                    return payload
                if resp.status_code not in RETRY_STATUS:
                    raise FPLAPIError(f"{endpoint} -> HTTP {resp.status_code}")
                last_error = FPLAPIError(f"{endpoint} -> HTTP {resp.status_code}")
            except (httpx.TransportError, json.JSONDecodeError) as exc:
                last_error = exc

            # Exponential backoff with jitter. Jitter matters because many of these calls run
            # from a cron that fires on the hour alongside everyone else's.
            sleep = min(2**attempt, 30) * (0.5 + random.random())
            log.warning(
                "%s attempt %d/%d failed (%s); retrying in %.1fs",
                endpoint,
                attempt + 1,
                self.max_retries,
                last_error,
                sleep,
            )
            time.sleep(sleep)

        # A stale cache beats no data at all, especially for an hourly price snapshot.
        stale = self._read_cache(path, ttl=float("inf"))
        if stale is not None:
            log.error("%s unreachable; serving stale cache", endpoint)
            return stale
        raise FPLAPIError(f"{endpoint} failed after {self.max_retries} attempts") from last_error

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    def __enter__(self) -> FPLClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ------------------------------------------------------------- endpoints

    def bootstrap(self, *, ttl: float | None = None) -> dict[str, Any]:
        """Players, teams, events, chips, scoring rules and price-change fields."""
        return self.get("bootstrap-static/", ttl=ttl)

    def fixtures(self, *, event: int | None = None, ttl: float | None = None) -> list[dict]:
        """All 380 fixtures, or just one gameweek's."""
        ep = "fixtures/" if event is None else f"fixtures/?event={event}"
        return self.get(ep, ttl=ttl)

    def element_summary(self, element_id: int) -> dict[str, Any]:
        """Per-player history: this season by GW, past seasons, and upcoming fixtures."""
        return self.get(f"element-summary/{element_id}/")

    def event_status(self) -> dict[str, Any]:
        """Bonus-added / points-final status for the current gameweek."""
        return self.get("event-status/", ttl=120.0)

    def entry(self, entry_id: int) -> dict[str, Any]:
        """A manager's profile, including their league memberships."""
        return self.get(f"entry/{entry_id}/")

    def entry_history(self, entry_id: int) -> dict[str, Any]:
        """A manager's per-GW history, past seasons, and chips already played."""
        return self.get(f"entry/{entry_id}/history/")

    def entry_picks(self, entry_id: int, event: int) -> dict[str, Any]:
        """A manager's squad for a gameweek.

        Public only *after* that gameweek's deadline; raises before it.
        """
        return self.get(f"entry/{entry_id}/event/{event}/picks/", ttl=float("inf"))

    def entry_transfers(self, entry_id: int) -> list[dict]:
        """Every transfer a manager has made, with prices paid."""
        return self.get(f"entry/{entry_id}/transfers/")

    def league_standings(self, league_id: int, page: int = 1) -> dict[str, Any]:
        """One page (50 entries) of a classic league's standings."""
        return self.get(f"leagues-classic/{league_id}/standings/?page_standings={page}", ttl=600.0)

    def live(self, event: int) -> dict[str, Any]:
        """Live per-player stats and points for a gameweek."""
        return self.get(f"event/{event}/live/", ttl=60.0)
