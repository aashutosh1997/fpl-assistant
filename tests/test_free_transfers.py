"""Tests for reconstructing banked free transfers.

FPL does not expose this number, so it is replayed from transfer counts — and getting it wrong is
expensive in a way that is invisible until execution. A plan built on a free transfer that does not
exist is a plan that costs an unplanned -4 to carry out.
"""

from __future__ import annotations

from fplass.advise import _free_transfers
from fplass.optimise.milp import MAX_BANKED_TRANSFERS


def history(transfers_by_gw: dict[int, int], chips: dict[int, str] | None = None) -> dict:
    return {
        "current": [
            {"event": gw, "event_transfers": n} for gw, n in sorted(transfers_by_gw.items())
        ],
        "chips": [{"event": gw, "name": name} for gw, name in (chips or {}).items()],
    }


def test_gameweek_one_grants_no_free_transfer():
    """Everything before the first deadline is an unlimited squad build, not a transfer week.

    It neither spends a free transfer nor banks one, so you enter gameweek 2 with exactly one
    however you built the squad. Treating gameweek 1 as an ordinary week invented a second
    transfer, and the resulting plan recommended a triple move that would in fact have cost -4.
    """
    assert _free_transfers(history({1: 0}), 1) == 1
    # Even having played a chip in gameweek 1 changes nothing.
    assert _free_transfers(history({1: 0}, {1: "bboost"}), 1) == 1
    assert _free_transfers(history({1: 0}, {1: "3xc"}), 1) == 1


def test_rolling_banks_one():
    assert _free_transfers(history({1: 0, 2: 0}), 2) == 2
    assert _free_transfers(history({1: 0, 2: 0, 3: 0}), 3) == 3


def test_using_a_transfer_spends_it():
    assert _free_transfers(history({1: 0, 2: 1}), 2) == 1
    assert _free_transfers(history({1: 0, 2: 0, 3: 2}), 3) == 1


def test_banking_is_capped_at_five():
    rolled = {gw: 0 for gw in range(1, 13)}
    assert _free_transfers(history(rolled), 12) == MAX_BANKED_TRANSFERS


def test_never_drops_below_one():
    """Taking a big hit does not leave you with zero next week."""
    assert _free_transfers(history({1: 0, 2: 5}), 2) == 1


def test_wildcard_and_free_hit_do_not_consume_the_balance():
    """Their transfers are free, so the banked count still accrues that week."""
    assert _free_transfers(history({1: 0, 2: 0, 3: 8}, {3: "wildcard"}), 3) == 3
    assert _free_transfers(history({1: 0, 2: 0, 3: 11}, {3: "freehit"}), 3) == 3


def test_ignores_gameweeks_after_the_last_played():
    """A partially-populated history must not count weeks that have not happened."""
    assert _free_transfers(history({1: 0, 2: 0, 3: 0}), 2) == 2
