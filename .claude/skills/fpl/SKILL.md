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
hourly history that calibrates them; if it is thin, the answer falls back to the classical
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
