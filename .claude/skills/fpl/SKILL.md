---
name: fpl
description: Answer Fantasy Premier League questions for the 2026/27 season using this repo's simulator and optimiser — best transfer this week, who to captain, chip timing, price rise and fall warnings, whether to take a hit, and mini-league strategy. Use whenever the user asks about their FPL team, transfers, captaincy, chips, wildcards, or player prices.
---

# FPL 2026/27 advisor

This repo simulates the rest of the season and optimises transfers, captaincy and chips against it.
Use it rather than reasoning about players from memory — the model has eleven seasons of data, live
prices and live injury news, and your recollection of who is in form does not.

## Before answering anything

Check the data is current. Prices and injury news change daily, and the fixture list changes as cup
postponements create double and blank gameweeks.

```bash
.venv/bin/python -m fplass.cli ingest current   # live players, prices, fixtures, chips, rules
```

Run `ingest history` only if `data/warehouse/fpl.duckdb` is missing — it re-downloads every season.

Once a week, after a gameweek has finished (Monday or Tuesday), also refresh the elite-manager
study. The weekly crawl is about 55,000 requests (about 90 minutes; run it detached) and must run
locally, because FPL's Cloudflare blocks cloud runners:

```bash
.venv/bin/python -m fplass.cli ingest gameweek <N>         # the finished gameweek's results
.venv/bin/python -m fplass.cli ingest market               # prices, ownership, transfers per GW
.venv/bin/python -m fplass.cli study crawl --incremental   # top managers' histories and transfers
.venv/bin/python -m fplass.cli study panel --gameweeks 1-38   # the model replayed at each deadline
.venv/bin/python -m fplass.cli study export --no-raw       # cohort flows, data/snapshots/elite (<1 MB)
```

The raw tables (`study export` without `--no-raw`, about 18 MB at GW10 and growing) are the only
copy of the season's manager decisions once FPL wipes them in July. Commit them once a month and
always at the end of the season, not weekly: there is no Git LFS here, and each re-export adds a
whole new copy to the history.

## "What are the top managers doing that I'm not?"

```bash
.venv/bin/python -m fplass.cli study report    # their transfers, the model's verdict, what you missed
.venv/bin/python -m fplass.cli study signal    # whether their moves predict what the model misses
```

Read the cohorts honestly. "Top 10k today" is chosen by the results being judged, so its
transfers always look good; "last season top 10k/50k" and "top 10k at the deadline" are the fair
comparisons. A cohort consensus the model opposes is worth raising with the user, with the
reason the report gives (news on the player sold, a player new to the league, minutes, or plain
returns), but it is not by itself a reason to override the plan.

## The main question: "what should I do this week?"

```bash
.venv/bin/python -m fplass.cli plan --entry <TEAM_ID> --league <LEAGUE_ID>
```

`--entry` is the number in the user's `fantasy.premierleague.com/entry/<ID>/` URL. If you do not
have it, ask — the tool cannot read their squad without it.

Useful options:

- `--objective league` (default) maximises the expected number of mini-league rivals they finish
  above. `--objective points` maximises raw expected points. They genuinely disagree: a manager who
  is behind should take differentials, one who is ahead should hold the template.
- `--horizon 8` gameweeks to plan ahead. Longer is better for chip timing, slower to solve.
- `--draws 10000` Monte Carlo draws. Drop to 3000 for a quick answer, raise to 20000 for a close call.
- `--gameweek N` to plan a specific gameweek instead of the next one.

## Other questions

**"Build me a wildcard squad" / "what's the best team for £100m?"**

```bash
.venv/bin/python -m fplass.cli squad --budget 100 --horizon 8
```

**"Is anyone I own about to drop in price?"**

```bash
.venv/bin/python -m fplass.cli prices status
```

The 2026/27 API publishes official price-change projections. `data/snapshots/prices/` holds the
daily history that calibrates them; if it is thin, the answer falls back to the classical
net-transfer model and will say so.

## How to present the answer

Lead with the recommendation, then the reasoning. The tool prints a report — do not just paste it,
read it and explain what matters:

- **The transfer**, and explicitly whether taking a hit is worth it. "Roll your transfer" is a real
  and frequently correct answer; the optimiser prices a banked transfer, so if it says roll, say so.
- **The captain**, mentioning upside when it differs from the mean. A Triple Captain is a bet on a
  ceiling, not an average.
- **Chip timing**, including *why not now*. The roadmap deliberately holds chips when nothing clears
  the value floor.
- **Price urgency** only when it changes what they should do today.

## Things to be honest about

State these when relevant rather than projecting false confidence:

- **Bonus points are the weakest component early in 2026/27.** FPL reworked the BPS weights and did
  not publish them, so the model fits them from observed matches and starts from a 2025-26 prior.
  Expect bonus projections to firm up around GW5.
- **The price fields are new and undocumented.** Their meaning is calibrated from logged snapshots,
  and the model reports itself as provisional until enough price moves have been observed.
- **Rivals' current squads are unknowable.** `entry/{id}/event/{gw}/picks/` is public only after a
  deadline, so mini-league analysis uses their most recent completed gameweek plus a template-drift
  assumption, never their actual upcoming team.
- **Doubles and blanks do not exist yet.** Every gameweek currently has ten fixtures. Chip value,
  especially for Bench Boost and Free Hit, rises sharply once postponements land, which is why the
  second-half chips are usually worth holding.

## Repo layout

- `src/fplass/scoring.py` — the FPL points function; reproduces all 253,509 historical rows exactly
- `src/fplass/features/` — team strength (Dixon-Coles), minutes, per-90 rates, BPS
- `src/fplass/sim/` — the Monte Carlo engine
- `src/fplass/optimise/` — the transfer MILP, chip roadmap, mini-league objective
- `src/fplass/prices/` — snapshot logging, calibration, buy-now-or-wait
- `src/fplass/advise.py` — the pipeline the CLI calls
