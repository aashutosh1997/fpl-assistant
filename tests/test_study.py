"""The elite-manager study: squad rebuilds, cohort membership, flows, scoring and the signal test.

All synthetic. The rebuild is the piece most likely to be quietly wrong — a Free Hit that never
reverts, or a transfer list truncated by the endpoint, shifts every squad after it — and the
``deadline`` cohort is the piece most likely to leak, by reading a rank from after the deadline
it is supposed to describe.
"""

from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd
import pytest

from fplass.study import cohorts, crawl, missed, signal, store

SQUAD = list(range(1, 16))


def _picks(entry: int, gw: int, elements) -> pd.DataFrame:
    return pd.DataFrame({"entry": entry, "gw": gw, "element": list(elements)})


def _transfers(entry: int, rows) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"entry": entry, "gw": gw, "element_out": o, "element_in": i,
             "time": pd.Timestamp("2026-08-01") + pd.Timedelta(hours=n)}
            for n, (gw, o, i) in enumerate(rows)
        ]
    )


def test_chains_within_a_week_net_out():
    assert cohorts.collapse_chains([(1, 2), (2, 3)]) == [(1, 3)]
    assert cohorts.collapse_chains([(1, 2), (2, 1)]) == []
    assert cohorts.collapse_chains([(1, 2), (4, 5)]) == [(1, 2), (4, 5)]


def test_a_free_hit_squad_reverts_the_week_after():
    transfers = _transfers(
        1,
        [
            (2, 1, 16),  # a normal transfer
            (3, 2, 17), (3, 3, 18),  # Free Hit
            (4, 4, 19),  # applied to the pre-Free-Hit squad
        ],
    )
    chips = pd.DataFrame({"entry": [1], "gw": [3], "chip": ["freehit"]})
    squads = cohorts.rebuild_squads(_picks(1, 1, SQUAD), transfers, chips, last_gw=4)
    played = {g: set(f["element"]) for g, f in squads.squads.groupby("gw")}
    kept = {g: set(f["element"]) for g, f in squads.kept.groupby("gw")}
    assert played[3] == (set(SQUAD) - {1, 2, 3}) | {16, 17, 18}
    assert kept[3] == kept[2] == (set(SQUAD) - {1}) | {16}
    assert played[4] == (set(SQUAD) - {1, 4}) | {16, 19}
    assert all(len(s) == 15 for s in played.values())
    moves = squads.moves.set_index("gw")
    assert moves.loc[3, "chip"].tolist() == ["freehit", "freehit"]


def test_stored_picks_override_a_truncated_transfer_list():
    """The endpoint returns 500 transfers at most; the earliest fall off."""
    # GW2's transfer (1 -> 16) is missing from the list, but GW2's picks are stored.
    picks = pd.concat([_picks(1, 1, SQUAD), _picks(1, 2, [16, *SQUAD[1:]])])
    transfers = _transfers(1, [(3, 2, 17)])
    positions = {e: "1" for e in range(1, 30)}
    squads = cohorts.rebuild_squads(picks, transfers, pd.DataFrame(columns=["entry", "gw", "chip"]),
                                    last_gw=3, positions=positions)
    assert squads.mismatched == 1
    played = {g: set(f["element"]) for g, f in squads.squads.groupby("gw")}
    assert played[3] == (set(SQUAD) - {1, 2}) | {16, 17}
    moves = squads.moves
    assert set(zip(moves["gw"], moves["element_out"], moves["element_in"], strict=True)) == {
        (2, 1, 16), (3, 2, 17)
    }


@pytest.fixture
def managers(tmp_path):
    path = tmp_path / "managers.duckdb"
    con = store.connect_managers(path)
    con.execute(
        """
        INSERT INTO manager_entries (season, entry, source, weight, standing_rank, standing_gw,
                                     started_gw, past_rank_last)
        VALUES ('2026-27', 1, 'top', 1, 50, 3, 1, 8000),
               ('2026-27', 2, 'band', 4, 90000, 3, 1, 40000),
               ('2026-27', 3, 'top', 1, 900, 3, 1, NULL)
        """
    )
    # Entry 3 is top 10k only after GW2: it must not be in the GW2 deadline cohort.
    con.execute(
        """
        INSERT INTO manager_gw (season, entry, gw, overall_rank) VALUES
            ('2026-27', 1, 1, 5000), ('2026-27', 1, 2, 4000),
            ('2026-27', 2, 1, 8000), ('2026-27', 2, 2, 60000),
            ('2026-27', 3, 1, 400000), ('2026-27', 3, 2, 3000)
        """
    )
    for entry in (1, 2, 3):
        con.execute(
            "INSERT INTO manager_picks (season, entry, gw, element) "
            f"SELECT '2026-27', {entry}, 1, unnest(range(1, 16))"
        )
    con.close()
    return path


def test_the_deadline_cohort_reads_only_the_rank_before_the_deadline(managers):
    con = duckdb.connect()
    store.attach_managers(con, managers)
    members = cohorts.memberships(con, "2026-27", [2, 3])
    deadline = members[members["cohort"] == "deadline"]
    assert set(deadline.loc[deadline["gw"] == 2, "entry"]) == {1, 2}
    assert set(deadline.loc[deadline["gw"] == 3, "entry"]) == {1, 3}
    # Weights come from the crawl: a band manager stands for four.
    assert deadline.set_index(["gw", "entry"]).loc[(2, 2), "weight"] == 4
    prior = members[(members["cohort"] == "prior10k") & (members["gw"] == 2)]
    assert set(prior["entry"]) == {1}
    prior50 = members[(members["cohort"] == "prior50k") & (members["gw"] == 2)]
    assert set(prior50["entry"]) == {1, 2}


def test_flows_weight_managers_and_leave_chip_weeks_out_of_transfers():
    picks = pd.concat([_picks(e, 1, SQUAD) for e in (1, 2, 3)])
    transfers = pd.concat([_transfers(1, [(2, 1, 16)]), _transfers(2, [(2, 1, 16)]),
                           _transfers(3, [(2, 1, 17), (2, 2, 18)])])
    chips = pd.DataFrame({"entry": [3], "gw": [2], "chip": ["wildcard"]})
    squads = cohorts.rebuild_squads(picks, transfers, chips, last_gw=2)
    members = pd.DataFrame({"cohort": "x", "gw": 2, "entry": [1, 2, 3], "weight": [1.0, 3.0, 1.0]})
    flows = cohorts.cohort_flows(members, squads, chips).set_index("element")
    assert flows.loc[16, "bought"] == pytest.approx(1.0)  # both non-chip managers, weight 4 of 4
    assert flows.loc[17, "bought"] == 0.0  # the Wildcard's move is not a transfer here
    assert flows.loc[1, "own_before"] == pytest.approx(1.0)
    assert flows.loc[1, "own_after"] == pytest.approx(0.0)
    assert flows.loc[16, "own_after"] == pytest.approx(4 / 5)
    assert flows.loc[1, "size"] == 5.0 and flows.loc[1, "size_moving"] == 4.0


def test_moves_are_scored_like_for_like():
    values = pd.DataFrame(
        {"gw": [5, 5], "element": [1, 2], "ep_h": [20.0, 26.0], "ep_w": [10.0, 12.0],
         "real_w": [4.0, 15.0], "weeks": [4, 4], "p_full": [0.9, 0.9], "minutes": [90, 90]}
    ).set_index(["gw", "element"])
    moves = pd.DataFrame({"gw": [5], "element_out": [1], "element_in": [2]})
    scored = missed.score_moves(moves, values).iloc[0]
    assert scored["model_h"] == pytest.approx(6.0)
    assert scored["model_w"] == pytest.approx(2.0)
    assert scored["real_w"] == pytest.approx(11.0)


def test_history_parsing_reads_last_season_and_chips():
    payload = {
        "past": [{"season_name": "2024/25", "rank": 120_000}, {"season_name": "2025/26", "rank": 8_000}],
        "chips": [{"event": 3, "name": "wildcard"}],
        "current": [{"event": 1, "overall_rank": 10}, {"event": 3, "overall_rank": 5}],
    }
    summary, rows = crawl.parse_history(7, payload, "2026-27")
    assert crawl.previous_season_name("2026-27") == "2025/26"
    assert summary["past_rank_last"] == 8_000
    assert summary["past_top10k"] == 1 and summary["past_top100k"] == 1
    assert summary["started_gw"] == 1
    assert rows.set_index("gw").loc[3, "chip"] == "wildcard"


def test_transfer_times_parse_with_and_without_fractional_seconds():
    payload = [
        {"event": 2, "element_in": 16, "element_out": 1, "time": "2026-08-20T10:00:00Z"},
        {"event": 3, "element_in": 17, "element_out": 2, "time": "2026-08-27T10:00:00.123456Z"},
    ]
    frame = crawl.parse_transfers(7, payload, "2026-27")
    assert frame["time"].notna().all() and len(frame) == 2


def test_the_signal_test_holds_out_whole_gameweeks_and_finds_a_planted_signal():
    rng = np.random.default_rng(3)
    frames = []
    for gw in range(2, 10):
        n = 300
        ep = rng.uniform(0, 8, n)
        net = rng.normal(0, 0.2, n)
        frames.append(
            pd.DataFrame(
                {
                    "gw": gw, "element": np.arange(n), "ep": ep, "p_full": rng.uniform(0.2, 0.95, n),
                    "elite_net": net, "elite_own": rng.uniform(0, 0.5, n),
                    "crowd_in": rng.uniform(0, 2, n), "crowd_out": rng.uniform(0, 2, n),
                    "crowd_own": rng.uniform(0, 8, n),
                    # Points the projection misses, carried by the cohort's buying.
                    "points": ep + 8 * net + rng.normal(0, 2, n),
                    "hour": rng.integers(0, 2, n),
                }
            )
        )
    rows = pd.concat(frames, ignore_index=True)
    folds = signal.held_out(rows)
    summary = signal.summarise(folds)
    assert summary.loc["calibrated", "folds"] == 8
    assert signal.verdict(summary, "elite", "rmse").startswith("signal")
    assert not signal.verdict(summary, "crowd", "rmse").startswith("signal")
