# FPL 2026/27 expert

Simulates the rest of the Fantasy Premier League season and optimises transfers, captaincy and
chip timing against it — for winning mini-leagues, not just for scoring points.

## What it does

- **Simulates** every remaining fixture with a Dixon-Coles team model, a three-class minutes model
  and empirical-Bayes per-90 rates, producing joint samples of every player's points.
- **Optimises** squads, transfers, hits and chips together over an eight-gameweek horizon with a
  mixed-integer program, respecting the 50% sell-on fee and free-transfer banking.
- **Times chips** across the whole remaining season, so an eight-week horizon cannot burn them.
- **Targets your mini-leagues** by simulating rivals on the same random draws and maximising the
  number you finish above — which makes differentials emerge when you are behind and the template
  when you are ahead, without a risk dial to guess.
- **Tracks prices** hourly and decides buy-now versus wait-for-team-news using the optimiser's own
  shadow price on budget.
- **Replays ten seasons** to validate itself: every deadline of the completed seasons is projected
  again with only what was knowable at it, and the planner plays each season through and is
  scored the way the game scores it. Every constant in the planner is measured against that.
- **Reads the order book.** What eleven million managers did before a deadline predicts who then
  fails to appear; ten seasons of their transfers stand in for the availability news history never
  recorded, and the same signal is read live.

## Quick start

```bash
uv venv --python 3.12 && uv pip install -e ".[dev]"

python -m fplass.cli ingest history     # eleven seasons, ~1 min
python -m fplass.cli ingest current     # live prices, fixtures, chips, scoring rules
python -m fplass.cli ingest preseason   # friendlies: the best predictor of GW1 minutes

python -m fplass.cli plan --entry <YOUR_TEAM_ID> --league <LEAGUE_ID>
python -m fplass.cli squad --budget 100          # best squad from scratch
python -m fplass.cli prices status

# after each gameweek
python -m fplass.cli ingest gameweek 1
python -m fplass.cli calibrate                   # score past projections, refit the model

# the ten-season replay (hours, not minutes; run detached)
python -m fplass.cli backtest panel --workers 4    # as-of projections for every past deadline
python -m fplass.cli backtest score                # the pipeline's accuracy, season by horizon
python -m fplass.cli backtest manager --workers 4  # play each season with the planner, in points
python -m fplass.cli backtest options              # what a transfer, the bank and the bench are worth
python -m fplass.cli backtest chips                # chip timing: floors vs continuation vs hindsight
python -m fplass.cli backtest revisions            # how far projections move week to week
```

Your team id is the number in your `fantasy.premierleague.com/entry/<ID>/` URL. Everything uses
public endpoints; no login is needed or stored.

In Claude Code, the `/fpl` skill wraps all of this conversationally.

## The price logger

`.github/workflows/prices.yml` snapshots FPL's price fields every 20 minutes around the daily
00:00 UK price change and hourly otherwise, committing to `data/snapshots/prices/`. The commits
are the time series.

This needs a GitHub repo to run in. Push this repo, and the workflow starts on its own schedule.

## Why the numbers should be believed

The scoring engine recomputes `total_points` from raw events for **all 253,509 player-gameweeks
across ten complete seasons and matches FPL exactly** — including position-specific goal values,
the 60-minute clean-sheet rule, the 10/12 defensive-contribution thresholds and bonus tie handling.
Every projection is that function applied to simulated events, so it is the foundation worth
checking, and `pytest` checks it.

Other components are validated where they can be:

| Component | Validation |
|---|---|
| Scoring engine | Exact on 253,509 historical rows **and on GW1 2026/27 live** |
| Bonus allocation | Exact on 29,747 rows including ties |
| Team strength | Cross-validated; GW1 predicted 28.4 goals vs 30 actual, 5.2 clean sheets vs 6 |
| Player ranking | GW1 top-30 by projection scored 2.05x the league average |
| DEFCON reconstruction | Identity verified exactly against 2025-26 actuals |
| Minutes model | Brier skill 0.51–0.59 one week ahead in every replayed season (see below) |
| **Whole pipeline, replayed** | **Nine seasons, 2.2 million as-of projections: one-week Spearman 0.66–0.69, top-30 lift 3.2–4.5x** |

### What gameweek 1 taught us

The first real test was humbling: the model's own squad would have scored **34 against a 48
average**. The failure was concentrated entirely in minutes at the start of a season — Brier skill
was **0.251 in GW1 against 0.485 in backtest**, because the base model is trained on within-season
sequences and August has no current-season form to read.

Two signals fixed most of it, neither of which the base model can carry:

| Predicting GW1 60+ minutes | Brier (5-fold CV) |
|---|---|
| Base model | 0.1704 |
| + preseason friendly minutes | 0.1281 |
| + ownership | 0.1491 |
| **+ both** | **0.1161** |

Preseason had every one of the big misses right: Joao Pedro played 80 minutes a game across four
friendlies (model: 0.69, actual: 90 minutes and 11 points); Kinsky 67.5 (model: 0.08, actual: 90);
Dubravka 22.5 (model: 0.93, actual: 0).

These feed a **recalibration layer** (`features/adjust.py`) fitted on the season being played rather
than on history, so it improves every week and does nothing at all when there is nothing to
calibrate against. Re-running GW1 with it, the model's squad scores **42 instead of 34** — though
that number is in-sample, and GW2 is the honest test.

### The ten-season replay

The live table above scores one honest prediction a week. `fpl backtest panel` replays every
deadline of the nine completed seasons since 2017-18 the same way — team strength fitted on
results before the gameweek's first kickoff, the minutes model with the season held out, rates
cut at the gameweek, the player pool, clubs and prices as they stood that week — and stores the
projection for the next twelve gameweeks. Scored against what happened:

| Weeks ahead | Spearman (EP vs points) | Top-30 lift | Minutes Brier | Minutes skill |
|---|---|---|---|---|
| 0 (the deadline's week) | 0.67 | 3.7x | 0.097 | 0.55 |
| 1 | 0.63 | 3.5x | 0.118 | 0.45 |
| 3 | 0.58 | 3.3x | 0.141 | 0.35 |
| 7 | 0.53 | 3.1x | 0.162 | 0.25 |
| 11 | 0.50 | 3.0x | 0.176 | 0.19 |

Every season sits within a few hundredths of those means, and the live 2026/27 gameweeks land in
the same range (GW2: 0.64 and 3.9x), which is the point: the replay is an honest stand-in for
the live model. The first attempt was not — a query that kept "every season but this one"
quietly included the *future* seasons in a replay, and the minutes model showed no skill on any
replayed season but the last. That leak is now pinned by a test that compares serving-path and
training-path features player by player.

`fpl backtest manager` then plays each season with the planner exactly as `fpl plan` runs it:
chip roadmap, transfer solve, lineup, captain, executed at the deadline's real prices with the
sell-on fee, and scored with the game's automatic substitutions, vice-captain and chip rules.
Its season totals are the number every change to the planner is judged by.

### The order book

`selected`, `transfers_in` and `transfers_out` exist for every historical player-gameweek and
were quarantined as post-hoc. The timing is the other way round: ownership change tracks the
same gameweek's net transfers at 0.89–0.97 in every season, so a gameweek's flow happened
*before* its deadline. Among established starters, the top decile of owners selling then failed
to appear 19.5% of the time against a 6.7% base rate; the top two percent 34.5%. A small layer on
the base minutes model — a logistic on its log-odds and the two flows — improves the held-out
Brier score in all ten seasons by 2–4.5%, with coefficients stable to two decimals across
seasons. It is applied to the deadline's own gameweek, in the replay and live.

## Known limitations

Stated because they affect how much to trust an answer, not buried:

- **Bonus is the weakest component until roughly GW5.** FPL reworked the BPS weights for 2026/27
  and did not publish them; the model fits them from observed matches, starting from a 2025-26 prior.
- **The price fields are new and undocumented.** `price_change_percent`, `price_change_projections`
  and friends are calibrated from logged snapshots, with the classical net-transfer model as the
  fallback until enough moves are observed.
- **Rivals' upcoming squads are unknowable.** Picks are public only after a deadline, so mini-league
  analysis uses each rival's last completed gameweek plus a template-drift assumption.
- **A backtest cannot validate bonus or prices for 2026/27**, because the rules changed. It
  validates the engine — fixtures, minutes, goals, assists, clean sheets.
- **Doubles and blanks do not exist yet.** All 38 gameweeks currently have ten fixtures; chip values
  will shift materially as postponements land, and the roadmap recomputes on every run.

## Layout

```
src/fplass/
  api.py            FPL public API client (cached, retrying)
  scoring.py        the points function — the foundation everything rests on
  ingest/           eleven seasons into DuckDB, plus the leakage-proof as_of view
  features/         team strength, minutes, per-90 rates, BPS, the order-flow layer
  sim/              Monte Carlo engine
  optimise/         transfer MILP, chip roadmap, mini-league objective
  options/          what flexibility is worth: revisions, transfer/bank/bench values, chip timing, news risk
  backtest/         the projection panel, the paper manager, live calibration
  prices/           snapshots, calibration, buy-now-or-wait
  report/           the HTML dashboard published as an Artifact
  advise.py         the pipeline behind `fpl plan`
```
