"""Chapter 4: classify quiet erosion and describe its incidence.

Uses only V-Dem and ERT. Nothing here touches authoritarian governance
preference, so the definition stays independent of the hypotheses.

Outputs
  results/tables/ch4_attrition.csv          reporting-standard counts per pathway
  results/tables/ch4_spells.csv             every spell (primary rule)
  results/tables/ch4_incidence_region_decade.csv
  results/tables/ch4_sensitivity.csv        spell counts across the parameter grid
  figures/ch4_fig1_worked_example.{png,pdf}
  figures/ch4_fig2_annual_incidence.{png,pdf}
  figures/ch4_fig3_sensitivity.{png,pdf}
  figures/ch4_fig4_trajectories.{png,pdf}
  figures/ch4_fig5_attrition.{png,pdf}
  QE_DATA_ROOT/processed/erosion_onsets.parquet
"""

from __future__ import annotations

import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from quiet_erosions import erosion as E
from quiet_erosions import plotstyle as S
from quiet_erosions.paths import FIGURES_DIR, PROC_DIR, TABLES_DIR, ensure_dir

R = E.PRIMARY


def main() -> None:
    S.apply()
    ensure_dir(FIGURES_DIR)
    ensure_dir(TABLES_DIR)
    panel = pd.read_parquet(PROC_DIR / "vdem_panel.parquet")

    cands, sps = {}, {}
    for pw in E.PATHWAYS:
        cands[pw] = E.candidates(panel, pw, R)
        sps[pw] = E.spells(cands[pw], panel, R)

    # Reporting-standard counts
    attr = pd.DataFrame([E.summarize(cands[pw]) for pw in E.PATHWAYS])
    attr["spells"] = [len(sps[pw]) for pw in E.PATHWAYS]
    attr["spells_nonprovisional"] = [int((~sps[pw].provisional).sum()) for pw in E.PATHWAYS]
    attr["countries"] = [sps[pw].country_text_id.nunique() for pw in E.PATHWAYS]
    attr["recovered"] = [int((~sps[pw].censored).sum()) for pw in E.PATHWAYS]
    attr["median_depth"] = [round(sps[pw].depth.median(), 3) for pw in E.PATHWAYS]
    attr.to_csv(TABLES_DIR / "ch4_attrition.csv", index=False)
    print(attr.to_string())

    spells = pd.concat(sps.values(), ignore_index=True)
    region = panel.drop_duplicates("country_text_id").set_index("country_text_id")
    spells["country_name"] = spells.country_text_id.map(region.country_name)
    spells["region"] = spells.country_text_id.map(region.e_regionpol_6C).map(S.REGION_LABELS)
    spells.sort_values(["pathway", "onset"]).to_csv(TABLES_DIR / "ch4_spells.csv", index=False)

    onsets = E.onset_panel(panel, R)
    onsets.to_parquet(PROC_DIR / "erosion_onsets.parquet", index=False)

    # Incidence by region and decade (onsets per 100 at-risk country-years)
    o = onsets.merge(
        panel[["country_text_id", "year", "e_regionpol_6C"]], on=["country_text_id", "year"]
    )
    o["decade"] = (o.year // 10 * 10).astype(str) + "s"
    o["region"] = o.e_regionpol_6C.map(S.REGION_LABELS)
    rows = []
    for (reg, dec), g in o.groupby(["region", "decade"]):
        row = {"region": reg, "decade": dec}
        for pw in E.PATHWAYS:
            n = g[f"risk_{pw}"].sum()
            row[f"{pw}_onsets"] = int(g[f"onset_{pw}"].sum())
            row[f"{pw}_per100"] = round(100 * g[f"onset_{pw}"].sum() / n, 2) if n else np.nan
        row["at_risk_country_years"] = int(g["risk_vertical"].sum())
        rows.append(row)
    pd.DataFrame(rows).to_csv(TABLES_DIR / "ch4_incidence_region_decade.csv", index=False)

    # Sensitivity grid
    grid = []
    for d, k, m, c in itertools.product(
        [0.10, 0.15, 0.20], [3, 4, 5], [1, 2, 3], ["point", "hpd", "none"]
    ):
        rule = E.Rule(delta=d, k=k, m=m, cred=c)
        row = dict(delta=d, k=k, m=m, cred=c)
        for pw in E.PATHWAYS:
            row[pw] = len(E.spells(E.candidates(panel, pw, rule), panel, rule))
        grid.append(row)
    orig = dict(delta="0.05 (osp)", k=4, m=2, cred="hpd")
    for pw in E.PATHWAYS:
        orig[pw] = len(E.spells(E.candidates(panel, pw, E.ORIGINAL), panel, E.ORIGINAL))
    grid.append(orig)
    grid = pd.DataFrame(grid)
    grid.to_csv(TABLES_DIR / "ch4_sensitivity.csv", index=False)

    fig_annual(onsets)
    fig_sensitivity(grid)
    fig_attrition(attr)
    fig_trajectories(panel, spells)
    fig_worked_example(panel, spells)


def fig_annual(onsets: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6, 3.2))
    yrs = onsets.groupby("year")
    for pw in E.PATHWAYS:
        n = yrs[f"onset_{pw}"].sum()
        risk = yrs[f"risk_{pw}"].sum()
        rate = (100 * n / risk).where(risk > 0)
        sm = rate.rolling(3, center=True, min_periods=2).mean()
        ax.plot(sm.index, sm.values, color=S.PATHWAY_COLORS[pw], label=S.PATHWAY_LABELS[pw], lw=1.6)
    ax.set_xlim(1990, 2019)
    ax.set_ylabel("Onsets per 100 at-risk country-years\n(3-year centered mean)")
    ax.legend(loc="upper left")
    S.save(fig, FIGURES_DIR / "ch4_fig2_annual_incidence")


def fig_sensitivity(grid: pd.DataFrame) -> None:
    g = grid[(grid.m == 2) & (grid.cred == "point")]
    fig, axes = plt.subplots(1, 3, figsize=(7, 2.4), sharey=True)
    for ax, pw in zip(axes, E.PATHWAYS, strict=True):
        mat = g.pivot(index="delta", columns="k", values=pw).sort_index(ascending=False)
        ax.imshow(mat.values, cmap="Greys", vmin=0, vmax=grid[list(E.PATHWAYS)].max().max())
        for i, j in itertools.product(range(mat.shape[0]), range(mat.shape[1])):
            v = mat.values[i, j]
            ax.text(
                j,
                i,
                int(v),
                ha="center",
                va="center",
                fontsize=8,
                color="white" if v > grid[list(E.PATHWAYS)].max().max() / 2 else "black",
            )
        ax.set_xticks(range(mat.shape[1]), [f"k={c}" for c in mat.columns])
        ax.set_yticks(range(mat.shape[0]), [f"δ={r:.2f}" for r in mat.index])
        ax.set_title(S.PATHWAY_LABELS[pw], color=S.PATHWAY_COLORS[pw], fontsize=8)
        ax.grid(False)
    S.save(fig, FIGURES_DIR / "ch4_fig3_sensitivity")


def fig_attrition(attr: pd.DataFrame) -> None:
    steps = [
        "magnitude_met",
        "failed_credibility",
        "excluded_episode",
        "failed_persistence",
        "qualifying_windows",
        "spells",
    ]
    labels = [
        "Decline ≥ δ",
        "− not credible",
        "− overlaps episode\nor EDI ≥ 0.10",
        "− not persistent",
        "Qualifying\nwindows",
        "Spells",
    ]
    fig, ax = plt.subplots(figsize=(6.5, 3))
    w = 0.26
    x = np.arange(len(steps))
    for i, pw in enumerate(E.PATHWAYS):
        vals = attr.set_index("pathway").loc[pw, steps].to_numpy(dtype=float)
        ax.bar(x + (i - 1) * w, vals, w, color=S.PATHWAY_COLORS[pw], label=S.PATHWAY_LABELS[pw])
    ax.set_xticks(x, labels, fontsize=7)
    ax.set_ylabel("Count")
    ax.legend(fontsize=7)
    S.save(fig, FIGURES_DIR / "ch4_fig5_attrition")


def _series(panel, cid, pw):
    var = E.PATHWAYS[pw].replace("_osp", "")
    g = panel[panel.country_text_id == cid].set_index("year")
    return g[var], g[f"{var}_codelow"], g[f"{var}_codehigh"], g


def fig_trajectories(panel: pd.DataFrame, spells: pd.DataFrame) -> None:
    sp = spells[~spells.provisional].sort_values("depth", ascending=False)
    picks = []
    for pw in E.PATHWAYS:  # deepest spell on each pathway
        picks.append(sp[sp.pathway == pw].iloc[0])
    multi = sp.groupby("country_text_id").pathway.nunique()
    multi = sp[sp.country_text_id.isin(multi[multi > 1].index)]
    if len(multi):
        picks.append(multi.iloc[0])
    rec = sp[~sp.censored]
    if len(rec):
        picks.append(rec.iloc[0])
    seen, chosen = set(), []
    for p in picks:
        key = (p.country_text_id, p.pathway)
        if key not in seen:
            seen.add(key)
            chosen.append(p)
    fig, axes = plt.subplots(2, 3, figsize=(7, 4.2), sharex=True)
    for ax, p in zip(axes.flat, chosen[:6], strict=False):
        y, lo, hi, g = _series(panel, p.country_text_id, p.pathway)
        ax.fill_between(y.index, lo, hi, color=S.PATHWAY_COLORS[p.pathway], alpha=0.15, lw=0)
        ax.plot(y.index, y, color=S.PATHWAY_COLORS[p.pathway], lw=1.4)
        ax.axvspan(p.onset, p.onset + R.k, color="#cbd5e0", alpha=0.5, lw=0)
        ep = g.index[g.aut_ep == 1]
        if len(ep):
            ax.scatter(ep, y.loc[ep], s=6, color="black", zorder=3)
        ax.set_title(f"{p.country_name}, {p.pathway} ({p.onset})", fontsize=8)
        ax.set_xlim(1995, 2023)
    for ax in axes.flat[len(chosen) :]:
        ax.axis("off")
    S.save(fig, FIGURES_DIR / "ch4_fig4_trajectories")


def fig_worked_example(panel: pd.DataFrame, spells: pd.DataFrame) -> None:
    p = (
        spells[(~spells.provisional) & (spells.pathway == "horizontal")]
        .sort_values("depth")
        .iloc[-1]
    )
    y, lo, hi, _ = _series(panel, p.country_text_id, "horizontal")
    t, a0 = p.onset, p.a0
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.fill_between(
        y.index,
        lo,
        hi,
        color=S.PATHWAY_COLORS["horizontal"],
        alpha=0.15,
        lw=0,
        label="68% interval",
    )
    ax.plot(
        y.index, y, color=S.PATHWAY_COLORS["horizontal"], lw=1.6, label="Horizontal accountability"
    )
    ax.axhline(a0 - R.delta, color=S.NEUTRAL, ls="--", lw=0.8)
    ax.text(
        y.index.min() + 0.5, a0 - R.delta, " A(t) − δ", va="bottom", fontsize=7, color=S.NEUTRAL
    )
    ax.axhline(a0 - R.delta / 2, color=S.NEUTRAL, ls=":", lw=0.8)
    ax.text(
        y.index.min() + 0.5,
        a0 - R.delta / 2,
        " A(t) − δ/2",
        va="bottom",
        fontsize=7,
        color=S.NEUTRAL,
    )
    ax.axvspan(t, t + R.k, color="#cbd5e0", alpha=0.5, lw=0, label=f"Window (k = {R.k})")
    ax.axvspan(
        t + R.k, t + R.k + R.m, color="#edf2f7", alpha=0.8, lw=0, label=f"Persistence (m = {R.m})"
    )
    ax.set_title(f"{p.country_name}: onset {t}", fontsize=9)
    ax.set_ylabel("Index (latent scale)")
    ax.legend(fontsize=7, loc="lower left")
    S.save(fig, FIGURES_DIR / "ch4_fig1_worked_example")


if __name__ == "__main__":
    main()
