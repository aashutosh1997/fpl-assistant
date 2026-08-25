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
| Minutes model | **This is the weak component — see below** |

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
  features/         team strength, minutes, per-90 rates, BPS
  sim/              Monte Carlo engine
  optimise/         transfer MILP, chip roadmap, mini-league objective
  prices/           snapshots, calibration, buy-now-or-wait
  report/           the HTML dashboard published as an Artifact
  advise.py         the pipeline behind `fpl plan`
```
