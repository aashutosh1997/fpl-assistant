"""Canonical filesystem locations. Everything resolves from the repo root."""

from __future__ import annotations

import os
from pathlib import Path

# src/fplass/paths.py -> repo root is three parents up.
ROOT = Path(__file__).resolve().parents[2]

DATA = Path(os.environ.get("FPLASS_DATA", ROOT / "data"))
SNAPSHOTS = DATA / "snapshots"
PRICE_SNAPSHOTS = SNAPSHOTS / "prices"
RAW = DATA / "raw"
WAREHOUSE = DATA / "warehouse"
DB_PATH = Path(os.environ.get("FPLASS_DB", WAREHOUSE / "fpl.duckdb"))
CACHE = Path(os.environ.get("FPLASS_CACHE", DATA / "cache"))
CONFIG = ROOT / "config"


def ensure_dirs() -> None:
    for d in (DATA, SNAPSHOTS, PRICE_SNAPSHOTS, RAW, WAREHOUSE, CACHE, CONFIG):
        d.mkdir(parents=True, exist_ok=True)
