"""The study, printed: what the top managers did, what the model made of it, and what we missed."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from ..paths import BACKTEST
from . import cohorts as cohorts_module
from . import missed as missed_module
from . import signal as signal_module

log = logging.getLogger(__name__)

LABELS = {
    "prior10k": "last season top 10k",
    "prior50k": "last season top 50k",
    "deadline": "top 10k at the deadline",
    "now": "top 10k today (hindsight)",
    "league": "your mini-leagues",
    "us": "you",
}
ORDER = ["prior10k", "prior50k", "deadline", "now", "league", "us"]


def ownership_gaps(inputs: missed_module.Inputs, me: int, cohort: str, *, top: int = 8) -> pd.DataFrame:
    """The squads carried into the next deadline: what the cohort holds that we don't, and back.

    With the model's horizon projection from the latest replayed deadline.
    """
    last = max(g for g in inputs.deadlines if g <= inputs.last_played)
    members = inputs.members[(inputs.members["cohort"] == cohort) & (inputs.members["gw"] == last)]
    weight = members.set_index("entry")["weight"]
    kept = inputs.squads.kept[inputs.squads.kept["gw"] == last]
    held = kept[kept["entry"].isin(weight.index)]
    share = held["entry"].map(weight).groupby(held["element"]).sum() / weight.sum()
    ours = set(kept.loc[kept["entry"] == me, "element"])
    upcoming = max(inputs.deadlines)
    horizon = inputs.values.xs(upcoming, level="gw")["ep_h"]
    index = pd.Index(sorted(set(share.index) | ours), name="element")
    frame = pd.DataFrame({"share": share.reindex(index).fillna(0.0)})
    frame["ours"] = frame.index.isin(ours)
    frame["ep_h"] = horizon.reindex(frame.index)
    frame["position"] = inputs.players["position"].reindex(frame.index)
    missing = frame[~frame["ours"]].sort_values("share", ascending=False).head(top)
    unusual = frame[frame["ours"]].sort_values("share").head(top)
    missing = missing.assign(kind="they hold, you don't")
    unusual = unusual.assign(kind="you hold, they mostly don't")
    out = pd.concat([missing, unusual])
    out.index.name = "element"
    return out.reset_index()


def _fmt_moves(inputs, frame: pd.DataFrame) -> list[str]:
    lines = []
    for r in frame.itertuples(index=False):
        actual = (
            f"  actual {r.real_w:+5.0f} over {int(r.weeks)}w" if r.weeks and pd.notna(r.real_w)
            else "  not played yet"
        )
        lines.append(
            f"    {missed_module.name(inputs, r.element_out):<26} -> "
            f"{missed_module.name(inputs, r.element_in):<26} {r.share:6.1%}  model {r.model_h:+5.1f}"
            f"{actual}{'  [sold player flagged]' if r.flag_out else ''}"
        )
    return lines


def build(con, season: str, me: int, *, out_dir: Path = BACKTEST) -> str:
    """The printed study; every table behind it is written to ``out_dir`` as CSV."""
    inputs = missed_module.load_inputs(con, season, me=me)
    scored = missed_module.scored_moves(inputs)
    verdicts = missed_module.verdicts(scored)
    crowd = missed_module.crowd_baseline(inputs)
    fair = ("prior10k", "prior50k", "deadline")
    blind = missed_module.blind_spots(scored, inputs, fair)
    ours = missed_module.our_season(scored, inputs, me, ("prior50k", "deadline"))
    gaps = ownership_gaps(inputs, me, "prior50k")

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / f"study_{season}"
    scored.to_csv(f"{stem}_moves.csv", index=False)
    verdicts.to_csv(f"{stem}_verdicts.csv")
    blind.to_csv(f"{stem}_blind_spots.csv", index=False)
    ours.to_csv(f"{stem}_ours.csv", index=False)
    gaps.to_csv(f"{stem}_ownership.csv", index=False)

    lines: list[str] = []
    played = [g for g in inputs.deadlines if g <= inputs.last_played]
    sizes = cohorts_module.describe(inputs.members)
    last = sizes[sizes["gw"] == max(played)].set_index("cohort")
    lines.append(f"=== The top managers against the model, {season} GW{min(played)}-{max(played)}")
    lines.append(
        f"Squads rebuilt from transfers: {inputs.squads.checked - inputs.squads.mismatched} of "
        f"{inputs.squads.checked} stored squads matched exactly; the rest were taken from the picks."
    )
    lines.append("Cohorts at the latest deadline (managers crawled, managers they stand for):")
    for cohort in ORDER:
        if cohort in last.index:
            lines.append(
                f"  {LABELS[cohort]:<28} {int(last.at[cohort, 'managers']):>6,} "
                f"{last.at[cohort, 'represents']:>9,.0f}"
            )
    lines.append("")

    lines.append(
        "Per transfer, outside Wildcard and Free Hit weeks, over the deadline's week and up to "
        "three after (points):"
    )
    lines.append(
        f"  {'cohort':<28}{'moves':>7}{'backed':>8}{'worked':>8}{'model':>7}{'actual':>8}"
        f"{'±':>5}{'vs model pick':>15}{'±':>5}{'same pick':>10}"
    )
    for cohort in ORDER:
        if cohort not in verdicts.index:
            continue
        v = verdicts.loc[cohort]
        lines.append(
            f"  {LABELS[cohort]:<28}{int(v['moves']):>7,}{v['model_backed']:>8.0%}{v['worked']:>8.0%}"
            f"{v['model_gain']:>7.1f}{v['actual_gain']:>8.1f}{v['actual_gain_se']:>5.1f}"
            f"{v['beat_model_pick']:>+15.1f}{v['beat_model_pick_se']:>5.1f}{v['same_as_model_pick']:>10.0%}"
        )
    if crowd:
        lines.append(
            f"  {'every manager (crowd)':<28}{'':>7}{'':>8}{'':>8}{crowd['model_gain']:>7.1f}"
            f"{crowd['actual_gain']:>8.1f}"
        )
    lines.append(
        "  backed: the model's 8-week projection favoured the player bought. vs model pick: the "
        "actual gain over the model's own best replacement for the same sale and budget."
    )
    lines.append("")

    if not blind.empty:
        lines.append("Transfers the model opposed that worked, by cause (fair cohorts):")
        for r in blind.sort_values(["cohort", "weight"], ascending=[True, False]).itertuples(index=False):
            lines.append(
                f"  {LABELS[r.cohort]:<28} {r.cause:<34} {r.share_of_all_moves:6.1%} of moves, "
                f"actual {r.actual_gain:+.1f} vs model {r.model_gain:+.1f}"
            )
        lines.append("")

    lines.append("Your season: the fair cohorts' most common sales of players you held, by deadline")
    for gw in inputs.deadlines[1:]:
        week = ours[ours["gw"] == gw] if not ours.empty else ours
        our_moves = scored[(scored["cohort"] == "us") & (scored["gw"] == gw)]
        did = ", ".join(
            f"{missed_module.name(inputs, o)} -> {missed_module.name(inputs, i)} "
            f"(model {m:+.1f}{f', actual {a:+.0f}' if pd.notna(a) else ''})"
            for o, i, m, a in zip(
                our_moves["element_out"], our_moves["element_in"], our_moves["model_h"],
                our_moves["real_w"], strict=True,
            )
        ) or "no transfer (or a chip week)"
        lines.append(f"  GW{gw}: you {did}")
        for cohort in ("prior50k", "deadline"):
            sub = week[week["cohort"] == cohort] if not week.empty else week
            if sub.empty:
                continue
            lines.append(f"   {LABELS[cohort]}:")
            sub = sub.rename(columns={"out": "element_out", "in": "element_in"})
            lines += _fmt_moves(inputs, sub)
    lines.append("")

    upcoming = max(inputs.deadlines)
    if upcoming > inputs.last_played:
        lines.append(f"GW{upcoming}, not yet played: the most common moves, with the model's 8-week verdict")
        for cohort in ("prior50k", "deadline"):
            table = missed_module.consensus(scored, cohort, upcoming, top=8)
            if table.empty:
                continue
            size = float(inputs.members[(inputs.members["cohort"] == cohort)
                                        & (inputs.members["gw"] == upcoming)]["weight"].sum())
            table["share"] = table["weight"] / size if size else float("nan")
            table["weeks"] = 0
            lines.append(f"   {LABELS[cohort]}:")
            lines += _fmt_moves(inputs, table)
        lines.append("")

    lines.append(f"Squads after GW{max(played)}: {LABELS['prior50k']} against yours "
                 f"(model's 8-week projection from GW{max(inputs.deadlines)})")
    for kind, group in gaps.groupby("kind", sort=False):
        lines.append(f"  {kind}:")
        for r in group.itertuples(index=False):
            model = f"   model {r.ep_h:5.1f}" if pd.notna(r.ep_h) else ""
            lines.append(
                f"    {missed_module.name(inputs, r.element):<28} {r.position:<4} "
                f"held by {r.share:6.1%}{model}"
            )
    lines.append("")
    lines.append(f"Rows: {stem}_*.csv")
    return "\n".join(lines)


def build_signal(con, season: str, me: int, cohorts: tuple[str, ...] = ("prior50k", "deadline")) -> str:
    inputs = missed_module.load_inputs(con, season, me=me)
    panel = con.execute(
        "SELECT * FROM read_parquet(?)", [str(missed_module.study_panel_path(season))]
    ).fetchdf()
    lines = [f"=== Do the top managers predict what the model misses? {season}, one gameweek held out at a time"]
    for cohort in cohorts:
        rows = signal_module.attach_outcomes(con, signal_module.build_rows(inputs, panel, cohort), season)
        folds = signal_module.held_out(rows)
        summary = signal_module.summarise(folds)
        lines.append(f"\nCohort: {LABELS.get(cohort, cohort)} ({len(rows):,} player-weeks, "
                     f"{rows['gw'].nunique()} gameweeks)")
        lines.append(
            f"  {'variant':<12}{'RMSE':>8}{'Spearman':>10}{'top-30 lift':>12}{'Brier (60+)':>13}"
        )
        for variant in ("projection", "calibrated", "crowd", "elite", "both"):
            if variant in summary.index:
                s = summary.loc[variant]
                lines.append(
                    f"  {variant:<12}{s['rmse']:>8.3f}{s['spearman']:>10.3f}{s['lift']:>12.2f}{s['brier']:>13.4f}"
                )
        for variant in ("crowd", "elite", "both"):
            lines.append(f"  {variant:<6} points: {signal_module.verdict(summary, variant, 'rmse')}")
            lines.append(f"  {variant:<6} minutes: {signal_module.verdict(summary, variant, 'brier')}")
    return "\n".join(lines)
