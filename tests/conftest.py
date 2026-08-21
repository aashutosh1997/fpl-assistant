"""Shared fixtures.

The warehouse tests run against the real warehouse rather than synthetic data, because what they
are actually asserting is that our understanding of eleven seasons of real FPL data is correct.
They skip rather than fail when the warehouse has not been built, so a fresh clone can still run
the pure-logic tests.
"""

from __future__ import annotations

import pytest

from fplass.ingest.warehouse import connect
from fplass.paths import DB_PATH


@pytest.fixture(scope="session")
def con():
    if not DB_PATH.exists():
        pytest.skip(f"no warehouse at {DB_PATH}; run `fpl ingest history` first")
    connection = connect(read_only=True)
    try:
        yield connection
    finally:
        connection.close()


@pytest.fixture(scope="session")
def complete_seasons(con) -> list[str]:
    """Seasons with a full set of played gameweeks, so outcome assertions are meaningful."""
    rows = con.execute(
        "SELECT season FROM player_gw GROUP BY season HAVING count(DISTINCT gw) >= 38 "
        "ORDER BY season"
    ).fetchall()
    return [r[0] for r in rows]
