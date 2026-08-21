"""Rendering a recommendation as a self-contained HTML dashboard.

The design brief is set by what this system actually produces. Every number here is a *distribution*
— the Monte Carlo engine gives joint samples, not point estimates — and a dashboard that printed
single figures would throw away the most useful thing about it. So players are shown as a range
from the 10th to the 90th percentile with the mean marked, which makes the difference between a
reliable 5-point midfielder and a volatile 5-point striker visible at a glance. That difference is
exactly what a captaincy or Triple Captain decision turns on.

The fixture grid uses our own continuous expected-goals values rather than FPL's 1-to-5 difficulty
rating, because a coarse integer cannot distinguish a hard fixture from a nearly impossible one.

Output is a single file with no external requests beyond Google Fonts, so it can be published as an
Artifact and read anywhere.
"""

from __future__ import annotations

import datetime as dt
import html
import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

POSITION_ORDER = {"GKP": 0, "DEF": 1, "MID": 2, "FWD": 3}

CHIP_LABELS = {
    "bboost": "Bench Boost",
    "3xc": "Triple Captain",
    "freehit": "Free Hit",
    "wildcard": "Wildcard",
}

STYLE = """
:root{
  --ground:#F2F4F6; --surface:#FFFFFF; --surface-2:#E9EDF0; --line:#D3DBE1;
  --ink:#12212E; --ink-2:#4A5C6A; --ink-3:#7B8C99;
  --accent:#0F5C6B; --accent-soft:#DCEBEE;
  --rise:#2E6B3E; --fall:#B03A2E; --warn:#B5761E;
  --band:#C7D6DC; --band-strong:#0F5C6B;
  --shadow:0 1px 2px rgba(18,33,46,.06),0 8px 24px rgba(18,33,46,.06);
}
@media (prefers-color-scheme:dark){
  :root:not([data-theme="light"]){
    --ground:#0D161C; --surface:#132029; --surface-2:#1B2B35; --line:#263945;
    --ink:#DCE6EC; --ink-2:#9FB2BE; --ink-3:#6E838F;
    --accent:#4FB3C4; --accent-soft:#12313A;
    --rise:#5FBF77; --fall:#E0705C; --warn:#E0A64A;
    --band:#2B4450; --band-strong:#4FB3C4;
    --shadow:0 1px 2px rgba(0,0,0,.4),0 8px 24px rgba(0,0,0,.3);
  }
}
:root[data-theme="dark"]{
  --ground:#0D161C; --surface:#132029; --surface-2:#1B2B35; --line:#263945;
  --ink:#DCE6EC; --ink-2:#9FB2BE; --ink-3:#6E838F;
  --accent:#4FB3C4; --accent-soft:#12313A;
  --rise:#5FBF77; --fall:#E0705C; --warn:#E0A64A;
  --band:#2B4450; --band-strong:#4FB3C4;
  --shadow:0 1px 2px rgba(0,0,0,.4),0 8px 24px rgba(0,0,0,.3);
}

*{box-sizing:border-box}
body{
  margin:0; background:var(--ground); color:var(--ink);
  font-family:"Public Sans",ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif;
  font-size:15px; line-height:1.55; -webkit-font-smoothing:antialiased;
}
.wrap{max-width:1180px; margin:0 auto; padding:40px 24px 72px; display:flex; flex-direction:column; gap:28px}
h1,h2,h3{font-family:"Archivo",ui-sans-serif,system-ui,sans-serif; text-wrap:balance; margin:0}
h1{font-size:clamp(28px,4vw,40px); font-weight:700; letter-spacing:-.02em; line-height:1.08}
h2{font-size:13px; font-weight:700; letter-spacing:.1em; text-transform:uppercase; color:var(--ink-3)}
h3{font-size:17px; font-weight:600; letter-spacing:-.01em}
.num{font-family:"IBM Plex Mono",ui-monospace,SFMono-Regular,Menlo,monospace; font-variant-numeric:tabular-nums}

header{display:flex; flex-wrap:wrap; gap:20px; align-items:flex-end; justify-content:space-between;
  border-bottom:2px solid var(--ink); padding-bottom:18px}
.eyebrow{font-family:"IBM Plex Mono",ui-monospace,monospace; font-size:11px; letter-spacing:.16em;
  text-transform:uppercase; color:var(--accent); margin-bottom:8px}
.deadline{text-align:right}
.deadline .t{font-family:"IBM Plex Mono",monospace; font-size:26px; font-weight:600; font-variant-numeric:tabular-nums}
.deadline .l{font-size:12px; color:var(--ink-3); letter-spacing:.06em; text-transform:uppercase}

section{display:flex; flex-direction:column; gap:14px}
.card{background:var(--surface); border:1px solid var(--line); border-radius:10px;
  padding:20px 22px; box-shadow:var(--shadow)}

.decision{display:flex; flex-wrap:wrap; gap:28px; align-items:center; justify-content:space-between;
  border-left:4px solid var(--accent)}
.move{display:flex; align-items:center; gap:14px; flex-wrap:wrap; font-size:20px; font-weight:600;
  font-family:"Archivo",sans-serif}
.move .arrow{color:var(--ink-3); font-size:16px}
.out{color:var(--fall)} .in{color:var(--rise)}
.hit{font-family:"IBM Plex Mono",monospace; font-size:13px; color:var(--fall);
  border:1px solid var(--fall); border-radius:4px; padding:1px 7px}
.captain{text-align:right}
.captain .n{font-family:"Archivo",sans-serif; font-size:20px; font-weight:600}

.grid{display:grid; gap:20px; grid-template-columns:repeat(auto-fit,minmax(330px,1fr))}

table{width:100%; border-collapse:collapse; font-size:14px}
th{font-size:10px; letter-spacing:.1em; text-transform:uppercase; color:var(--ink-3);
  text-align:left; font-weight:600; padding:0 8px 8px; border-bottom:1px solid var(--line)}
td{padding:7px 8px; border-bottom:1px solid var(--line)}
tr:last-child td{border-bottom:none}
.pos{font-family:"IBM Plex Mono",monospace; font-size:10px; color:var(--ink-3); letter-spacing:.06em}
.bench td{opacity:.55}

/* Uncertainty bar: the 10th-90th percentile range with the mean marked. */
.range{position:relative; height:16px; min-width:110px; background:var(--surface-2); border-radius:3px}
.range .band{position:absolute; top:4px; height:8px; background:var(--band); border-radius:2px}
.range .mean{position:absolute; top:1px; width:2px; height:14px; background:var(--band-strong); border-radius:1px}
.scale{display:flex; justify-content:space-between; font-size:10px; color:var(--ink-3);
  font-family:"IBM Plex Mono",monospace; padding-top:4px}

.chips{display:flex; flex-direction:column; gap:0}
.chip-row{display:flex; gap:14px; align-items:baseline; padding:11px 0; border-bottom:1px solid var(--line)}
.chip-row:last-child{border-bottom:none}
.gw{font-family:"IBM Plex Mono",monospace; font-size:12px; color:var(--ink-3); min-width:46px; letter-spacing:.04em}
.chip-name{font-weight:600; flex:1}
.gain{font-family:"IBM Plex Mono",monospace; font-weight:600; color:var(--rise)}
.held{color:var(--ink-3); font-style:italic}

.pill{display:inline-block; font-family:"IBM Plex Mono",monospace; font-size:11px; font-weight:600;
  padding:2px 8px; border-radius:20px; letter-spacing:.03em}
.pill.rise{background:var(--accent-soft); color:var(--rise); border:1px solid var(--rise)}
.pill.fall{background:transparent; color:var(--fall); border:1px solid var(--fall)}
.pill.wait{background:transparent; color:var(--ink-3); border:1px solid var(--line)}

.heat{overflow-x:auto}
.heat table{font-size:12px; min-width:640px}
.heat td,.heat th{padding:0; border:none}
.heat .team{font-family:"IBM Plex Mono",monospace; font-size:11px; padding-right:10px;
  white-space:nowrap; color:var(--ink-2)}
.cell{width:34px; height:26px; text-align:center; vertical-align:middle;
  font-family:"IBM Plex Mono",monospace; font-size:10px; border-radius:3px; color:#fff}
.heat th.cell{background:none; color:var(--ink-3); font-weight:600}

.notes{display:flex; flex-direction:column; gap:9px; font-size:14px; color:var(--ink-2)}
.notes li{margin-left:2px}
ul{margin:0; padding-left:18px; display:flex; flex-direction:column; gap:8px}

footer{color:var(--ink-3); font-size:12px; border-top:1px solid var(--line); padding-top:16px;
  display:flex; flex-wrap:wrap; gap:6px 18px; justify-content:space-between}
@media (prefers-reduced-motion:reduce){*{animation:none!important; transition:none!important}}
"""


def _fmt_deadline(deadline: dt.datetime | None) -> tuple[str, str]:
    if deadline is None:
        return "—", "deadline unknown"
    remaining = deadline - dt.datetime.now(dt.UTC)
    hours = remaining.total_seconds() / 3600
    if hours < 0:
        return "closed", f"{deadline:%a %d %b %H:%M} UTC"
    if hours < 48:
        return f"{hours:.0f}h", f"{deadline:%a %d %b %H:%M} UTC"
    return f"{hours / 24:.0f}d", f"{deadline:%a %d %b %H:%M} UTC"


def _range_bar(low: float, mean: float, high: float, scale_max: float) -> str:
    """An uncertainty bar: the p10-p90 band with the mean marked."""
    span = max(scale_max, 1e-6)
    left = max(0.0, min(100.0, 100 * low / span))
    right = max(0.0, min(100.0, 100 * high / span))
    centre = max(0.0, min(100.0, 100 * mean / span))
    width = max(right - left, 1.2)
    return (
        f'<div class="range" role="img" aria-label="range {low:.1f} to {high:.1f}, '
        f'mean {mean:.1f} points">'
        f'<div class="band" style="left:{left:.1f}%;width:{width:.1f}%"></div>'
        f'<div class="mean" style="left:calc({centre:.1f}% - 1px)"></div>'
        f"</div>"
    )


def _heat_colour(expected_goals_for: float, expected_goals_against: float) -> str:
    """Colour a fixture by net expected goals — our continuous replacement for FPL's 1-5 rating."""
    net = expected_goals_for - expected_goals_against
    # Clamp to a plausible range and map onto a diverging red-to-teal ramp.
    t = float(np.clip((net + 1.6) / 3.2, 0, 1))
    if t < 0.5:
        ratio = t / 0.5
        r, g, b = 176 + (150 - 176) * ratio, 58 + (140 - 58) * ratio, 46 + (140 - 46) * ratio
    else:
        ratio = (t - 0.5) / 0.5
        r, g, b = 150 + (15 - 150) * ratio, 140 + (92 - 140) * ratio, 140 + (107 - 140) * ratio
    return f"rgb({r:.0f},{g:.0f},{b:.0f})"


def render(
    recommendation,
    samples: np.ndarray | None = None,
    elements: np.ndarray | None = None,
    fixture_rates: pd.DataFrame | None = None,
    *,
    title: str = "Gameweek Room",
) -> str:
    """Render a :class:`fplass.advise.Recommendation` to a standalone HTML page."""
    esc = html.escape
    rec = recommendation
    gw = rec.gameweek
    names = dict(zip(rec.players["element"], rec.players["web_name"], strict=True))
    info = rec.players.set_index("element")

    countdown, deadline_label = _fmt_deadline(rec.deadline)

    # ---- headline decision
    out_players = [esc(str(names.get(e, e))) for e in rec.plan.transfers_out.get(gw, [])]
    in_players = [esc(str(names.get(e, e))) for e in rec.plan.transfers_in.get(gw, [])]
    hits = rec.plan.hits.get(gw, 0)

    if in_players or out_players:
        move = (
            f'<span class="out">{", ".join(out_players)}</span>'
            f'<span class="arrow">&rarr;</span>'
            f'<span class="in">{", ".join(in_players)}</span>'
        )
        if hits:
            move += f'<span class="hit">-{hits * 4}</span>'
    else:
        move = "<span>Roll your transfer</span>"

    captain = names.get(rec.plan.captains.get(gw, -1), "—")

    # ---- squad table with uncertainty bars
    squad = rec.plan.squads.get(gw, [])
    lineup = set(rec.plan.lineups.get(gw, []))
    rows = []
    if squad:
        if samples is not None and elements is not None:
            index = {e: i for i, e in enumerate(elements)}
            slot = list(rec.expected_points.columns).index(gw)
            stats = {}
            for e in squad:
                if e in index:
                    column = samples[:, index[e], slot]
                    stats[e] = (
                        float(np.quantile(column, 0.10)),
                        float(column.mean()),
                        float(np.quantile(column, 0.90)),
                    )
                else:
                    stats[e] = (0.0, 0.0, 0.0)
        else:
            stats = {
                e: (0.0, float(rec.expected_points.at[e, gw]), 0.0)
                for e in squad
                if e in rec.expected_points.index
            }

        scale_max = max((v[2] for v in stats.values()), default=10.0) or 10.0
        for element in sorted(
            squad,
            key=lambda e: (POSITION_ORDER.get(info.loc[e, "position"], 9), -stats.get(e, (0, 0, 0))[1]),
        ):
            low, mean, high = stats.get(element, (0.0, 0.0, 0.0))
            starting = element in lineup
            captain_mark = " (C)" if element == rec.plan.captains.get(gw) else ""
            rows.append(
                f'<tr class="{"" if starting else "bench"}">'
                f'<td class="pos">{esc(str(info.loc[element, "position"]))}</td>'
                f'<td><strong>{esc(str(info.loc[element, "web_name"]))}</strong>{captain_mark}</td>'
                f'<td class="pos">{esc(str(info.loc[element, "team"] or ""))}</td>'
                f'<td class="num">{int(info.loc[element, "price"]) / 10:.1f}</td>'
                f"<td>{_range_bar(low, mean, high, scale_max)}</td>"
                f'<td class="num">{mean:.1f}</td>'
                f"</tr>"
            )
        scale_note = (
            f'<div class="scale"><span>0</span><span>expected points, GW{gw} '
            f"(bar = 10th&ndash;90th percentile)</span><span>{scale_max:.0f}</span></div>"
        )
    else:
        scale_note = ""

    squad_table = (
        "<table><thead><tr><th></th><th>Player</th><th>Club</th><th>Price</th>"
        f"<th>GW{gw} range</th><th>Mean</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
        + scale_note
    )

    # ---- chip roadmap
    chip_rows = []
    for chip_gw in sorted(rec.roadmap.schedule):
        chip = rec.roadmap.schedule[chip_gw]
        match = rec.roadmap.valuations[
            (rec.roadmap.valuations["chip"] == chip)
            & (rec.roadmap.valuations["gameweek"] == chip_gw)
        ]
        gain = match["mean_gain"].iloc[0] if len(match) else float("nan")
        chip_rows.append(
            f'<div class="chip-row"><span class="gw">GW{chip_gw}</span>'
            f'<span class="chip-name">{esc(CHIP_LABELS.get(chip, chip))}</span>'
            f'<span class="gain">+{gain:.1f}</span></div>'
        )
    scheduled = set(rec.roadmap.schedule.values())
    for chip in ("bboost", "3xc", "freehit", "wildcard"):
        if chip not in scheduled:
            chip_rows.append(
                f'<div class="chip-row"><span class="gw">&mdash;</span>'
                f'<span class="chip-name held">{esc(CHIP_LABELS[chip])}</span>'
                f'<span class="held">hold</span></div>'
            )

    # ---- price alerts
    price_rows = []
    for row in rec.price_advice.itertuples() if len(rec.price_advice) else []:
        pill = "rise" if row.recommendation == "buy now" else "wait"
        price_rows.append(
            f"<tr><td><strong>{esc(str(row.web_name))}</strong></td>"
            f'<td><span class="pill {pill}">{esc(row.recommendation)}</span></td>'
            f'<td class="num">{row.p_rise:.0%}</td></tr>'
        )
    for row in rec.sell_alerts.head(6).itertuples() if len(rec.sell_alerts) else []:
        price_rows.append(
            f"<tr><td><strong>{esc(str(row.web_name))}</strong></td>"
            f'<td><span class="pill fall">falling &middot; {esc(row.urgency)}</span></td>'
            f'<td class="num">{row.p_fall:.0%}</td></tr>'
        )
    price_table = (
        "<table><thead><tr><th>Player</th><th>Action</th><th>Prob</th></tr></thead><tbody>"
        + ("".join(price_rows) or '<tr><td colspan="3">No price pressure right now.</td></tr>')
        + "</tbody></table>"
    )

    # ---- fixture heatmap from our continuous expected goals
    heat = ""
    if fixture_rates is not None and len(fixture_rates):
        horizon = sorted(rec.expected_points.columns)[:8]
        window = fixture_rates[fixture_rates["event"].isin(horizon)]
        per_team: dict[str, dict[int, tuple[float, float, str]]] = {}
        for row in window.itertuples():
            per_team.setdefault(row.home, {})[int(row.event)] = (
                row.xg_home,
                row.xg_away,
                str(row.away).lower(),
            )
            per_team.setdefault(row.away, {})[int(row.event)] = (
                row.xg_away,
                row.xg_home,
                str(row.home).upper(),
            )
        ordered = sorted(
            per_team,
            key=lambda t: -np.mean(
                [v[0] - v[1] for v in per_team[t].values()] or [0]
            ),
        )
        header = "".join(f'<th class="cell">{g}</th>' for g in horizon)
        body = []
        for team in ordered:
            cells = []
            for g in horizon:
                entry = per_team[team].get(g)
                if entry is None:
                    cells.append('<td class="cell" style="background:var(--surface-2)">&mdash;</td>')
                else:
                    for_, against, opponent = entry
                    cells.append(
                        f'<td class="cell" style="background:{_heat_colour(for_, against)}" '
                        f'title="{esc(team)} vs {esc(opponent)}: {for_:.2f} - {against:.2f} xG">'
                        f"{esc(opponent[:3])}</td>"
                    )
            body.append(f'<tr><td class="team">{esc(team)}</td>{"".join(cells)}</tr>')
        heat = (
            '<section><h2>Fixture outlook</h2><div class="card heat">'
            "<table><thead><tr><th></th>"
            + header
            + "</tr></thead><tbody>"
            + "".join(body)
            + "</tbody></table>"
            '<p style="font-size:12px;color:var(--ink-3);margin:12px 0 0">'
            "Shaded by our model&rsquo;s net expected goals, not FPL&rsquo;s 1&ndash;5 rating. "
            "Uppercase means away. Teal is favourable.</p>"
            "</div></section>"
        )

    notes = "".join(f"<li>{esc(n)}</li>" for n in rec.notes)
    league_bits = " &middot; ".join(
        f"{esc(k.replace('_', ' '))} {v:.1f}" for k, v in rec.league_metrics.items()
    )

    return f"""<title>{esc(title)}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wght@500;600;700&family=IBM+Plex+Mono:wght@400;600&family=Public+Sans:wght@400;600&display=swap">
<style>{STYLE}</style>
<div class="wrap">
  <header>
    <div>
      <div class="eyebrow">Fantasy Premier League &middot; 2026/27</div>
      <h1>Gameweek {gw}</h1>
    </div>
    <div class="deadline">
      <div class="t num">{esc(countdown)}</div>
      <div class="l">{esc(deadline_label)}</div>
    </div>
  </header>

  <section>
    <h2>This week</h2>
    <div class="card decision">
      <div class="move">{move}</div>
      <div class="captain">
        <div class="l" style="font-size:11px;letter-spacing:.1em;text-transform:uppercase;color:var(--ink-3)">Captain</div>
        <div class="n">{esc(str(captain))}</div>
      </div>
    </div>
  </section>

  <div class="grid">
    <section>
      <h2>Squad</h2>
      <div class="card">{squad_table}</div>
    </section>
    <section>
      <h2>Chip roadmap</h2>
      <div class="card chips">{"".join(chip_rows)}</div>
      <h2 style="margin-top:8px">Price watch</h2>
      <div class="card">{price_table}</div>
    </section>
  </div>

  {heat}

  <section>
    <h2>What the model is unsure about</h2>
    <div class="card notes"><ul>{notes}</ul></div>
  </section>

  <footer>
    <span>{league_bits}</span>
    <span class="num">generated {dt.datetime.now(dt.UTC):%Y-%m-%d %H:%M} UTC</span>
  </footer>
</div>
"""


def write(recommendation, path: Path, **kwargs) -> Path:
    """Render and write the dashboard to ``path``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render(recommendation, **kwargs), encoding="utf-8")
    log.info("dashboard written to %s", path)
    return path
