"""Chapter 5: institutional targeting at the individual level (QE1).

Each outcome is standardized on its estimation sample and regressed on
standardized AGP (Volume 1 factor score) with sex, age and education
controls and country-wave fixed effects, using the trend-file design
weight; standard errors are clustered by country.

QE1 criterion (Chapter 2): the AGP coefficient on confidence in
constraining institutions (courts, parliament, press) is larger in
magnitude than the AGP coefficient on rating free elections as essential
to democracy, by a formal test of the difference. The test regresses the
within-respondent difference of the two standardized outcomes on AGP in
their common sample (waves 5-7).

Outputs
  results/tables/ch5_confidence.csv       one row per institution
  results/tables/ch5_essentials.csv       one row per democracy characteristic
  results/tables/ch5_qe1_test.csv         the criterion test and robustness
  results/tables/ch5_country_gaps.csv     targeting gap by country-wave
  figures/ch5_fig1_confidence.{png,pdf}
  figures/ch5_fig2_essentials.{png,pdf}
  figures/ch5_fig3_country_gaps.{png,pdf}
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).parent))

from quiet_erosions import plotstyle as S
from quiet_erosions.paths import FIGURES_DIR, PROC_DIR, TABLES_DIR, ensure_dir

CONF = {
    "conf_17": ("Courts", "constraint"),
    "conf_07": ("Parliament", "constraint"),
    "conf_04": ("Press", "constraint"),
    "conf_12": ("Political parties", "constraint"),
    "conf_11": ("Government", "executive"),
    "conf_02": ("Armed forces", "executive"),
    "conf_06": ("Police", "executive"),
    "conf_08": ("Civil service", "executive"),
}
ESS = {
    "ess_226": "Free elections",
    "ess_229": "Civil rights protect liberty",
    "ess_233": "Equal rights for women",
    "ess_224": "Tax the rich, subsidize the poor",
    "ess_227": "Unemployment aid",
    "ess_225": "Religious authorities interpret laws",
    "ess_228": "Army takes over if government incompetent",
}
CTRL = ["female", "age10", "edu_mid", "edu_high"]


def _z(s: pd.Series) -> pd.Series:
    return (s - s.mean()) / s.std()


def fit(df: pd.DataFrame, y: str, x: str = "agp_z", standardize: bool = True) -> dict:
    d = df.dropna(subset=[y, x, *CTRL, "weight"]).copy()
    d = d[d.weight > 0]
    if standardize:
        d[y] = _z(d[y])
    cols = [y, x, *CTRL]
    w = d.weight
    g = d.groupby("cell")
    wm = g[cols].transform(lambda c: np.average(c, weights=w.loc[c.index]))
    dm = d[cols] - wm
    X = dm[[x, *CTRL]]
    m = sm.WLS(dm[y], X, weights=w).fit(
        cov_type="cluster", cov_kwds={"groups": d.country_num.astype(int)}
    )
    b, se = m.params[x], m.bse[x]
    return dict(
        coef=b,
        se=se,
        p=m.pvalues[x],
        lo=b - 1.96 * se,
        hi=b + 1.96 * se,
        n=int(len(d)),
        countries=int(d.country_num.nunique()),
        cells=int(d.cell.nunique()),
    )


def prep() -> pd.DataFrame:
    df = pd.read_parquet(PROC_DIR / "qe_individual.parquet")
    df = df[df.agp_fs.notna()].copy()
    df["cell"] = (
        df.country_num.astype(int).astype(str) + "_" + df.wave_chron.astype(int).astype(str)
    )
    df["female"] = (df.X001 == 2).astype(float).where(df.X001.notna())
    df["age10"] = df.X003 / 10
    df["edu_mid"] = (df.X025R == 2).astype(float).where(df.X025R.notna())
    df["edu_high"] = (df.X025R == 3).astype(float).where(df.X025R.notna())
    df["agp_z"] = _z(df.agp_fs)
    df["agp2_z"] = _z(df[["agp_leader", "agp_army"]].mean(axis=1, skipna=False))
    for c in CONF:
        df[f"{c}_z"] = _z(df[c])
    df["constraint_idx"] = df[["conf_17_z", "conf_07_z", "conf_04_z"]].mean(axis=1, skipna=False)
    df["executive_idx"] = df[["conf_11_z", "conf_02_z", "conf_06_z"]].mean(axis=1, skipna=False)
    return df


def qe1_test(df: pd.DataFrame, x: str, label: str, sub: pd.Series | None = None) -> dict:
    d = df if sub is None else df[sub]
    d = d.dropna(subset=["constraint_idx", "ess_226"]).copy()
    d["c_z"], d["e_z"] = _z(d.constraint_idx), _z(d.ess_226)
    bc = fit(d, "c_z", x, standardize=False)
    be = fit(d, "e_z", x, standardize=False)
    # |b_c| - |b_e| when both are negative equals b_e - b_c; the difference
    # outcome e_z - c_z has coefficient b_e - b_c.
    d["diff"] = d.e_z - d.c_z
    bd = fit(d, "diff", x, standardize=False)
    return dict(
        spec=label,
        b_constraint=bc["coef"],
        se_constraint=bc["se"],
        b_elections=be["coef"],
        se_elections=be["se"],
        diff=bd["coef"],
        se_diff=bd["se"],
        p_diff=bd["p"],
        n=bd["n"],
        countries=bd["countries"],
        cells=bd["cells"],
    )


def _regime(df: pd.DataFrame) -> pd.DataFrame:
    """Attach the wave-year Regimes of the World category from agp_link."""
    link = pd.read_parquet(PROC_DIR / "agp_link.parquet")[
        ["country_text_id", "wave_chron", "regime"]
    ]
    xw = pd.read_parquet(PROC_DIR / "country_iso3_xwalk.parquet")
    from importlib import import_module

    sup = import_module("04_link_agp").XWALK_SUPPLEMENT
    extra = pd.DataFrame(list(sup.items()), columns=["country_num", "iso3"])
    xw = pd.concat([xw[~xw.country_num.isin(extra.country_num)], extra], ignore_index=True)
    link = link.merge(xw.rename(columns={"iso3": "country_text_id"}), on="country_text_id")
    return df.merge(
        link[["country_num", "wave_chron", "regime"]], on=["country_num", "wave_chron"], how="left"
    )


def country_gaps(df: pd.DataFrame) -> pd.DataFrame:
    d = df.dropna(subset=["constraint_idx", "ess_226", *CTRL]).copy()
    d["c_z"], d["e_z"] = _z(d.constraint_idx), _z(d.ess_226)
    rows = []
    for cell, g in d.groupby("cell"):
        if len(g) < 300:
            continue
        X = sm.add_constant(g[["agp_z", *CTRL]])
        bc = sm.WLS(g.c_z, X, weights=g.weight).fit().params.agp_z
        be = sm.WLS(g.e_z, X, weights=g.weight).fit().params.agp_z
        rows.append(
            dict(
                cell=cell,
                country_num=int(g.country_num.iloc[0]),
                wave=int(g.wave_chron.iloc[0]),
                n=len(g),
                b_constraint=bc,
                b_elections=be,
            )
        )
    return pd.DataFrame(rows)


def main() -> None:
    S.apply()
    ensure_dir(FIGURES_DIR)
    ensure_dir(TABLES_DIR)
    df = prep()

    conf = []
    for c, (lab, kind) in CONF.items():
        r = fit(df, c)
        conf.append(dict(item=c, institution=lab, kind=kind, **r))
    for c, lab in [("constraint_idx", "Constraining index"), ("executive_idx", "Executive index")]:
        conf.append(dict(item=c, institution=lab, kind="index", **fit(df, c)))
    conf = pd.DataFrame(conf)
    conf.to_csv(TABLES_DIR / "ch5_confidence.csv", index=False)
    print(conf.round(3).to_string())

    ess = pd.DataFrame([dict(item=c, characteristic=lab, **fit(df, c)) for c, lab in ESS.items()])
    ess.to_csv(TABLES_DIR / "ch5_essentials.csv", index=False)
    print(ess.round(3).to_string())

    tests = [
        qe1_test(df, "agp_z", "Primary: AGP factor score"),
        qe1_test(df, "agp2_z", "AGP without the democracy item"),
    ]
    reg = _regime(df)
    tests.append(qe1_test(reg, "agp_z", "Democracies only (RoW >= 2)", reg.regime >= 2))
    tests.append(qe1_test(reg, "agp_z", "Liberal democracies only (RoW = 3)", reg.regime == 3))
    tests.append(qe1_test(reg, "agp_z", "Autocracies only (RoW < 2)", reg.regime < 2))
    for w in (5, 6, 7):
        tests.append(qe1_test(df, "agp_z", f"Wave {w} only", df.wave_chron == w))
    tests = pd.DataFrame(tests)
    tests.to_csv(TABLES_DIR / "ch5_qe1_test.csv", index=False)
    print(tests.round(3).to_string())

    # Item decomposition and the elections-versus-liberties contrast
    extra = []
    for item in ["agp_leader", "agp_army", "agp_nodem"]:
        df[f"{item}_z"] = _z(df[item])
        for y in ["constraint_idx", "executive_idx", "ess_226", "ess_229"]:
            extra.append(dict(x=item, y=y, **fit(df, y, f"{item}_z")))
    d = df.dropna(subset=["ess_226", "ess_229"]).copy()
    d["el_minus_cr"] = _z(d.ess_226) - _z(d.ess_229)
    extra.append(
        dict(
            x="agp_z", y="elections minus civil rights", **fit(d, "el_minus_cr", standardize=False)
        )
    )
    extra = pd.DataFrame(extra)
    extra.to_csv(TABLES_DIR / "ch5_decomposition.csv", index=False)
    print(extra.round(3).to_string())

    gaps = country_gaps(df)
    gaps.to_csv(TABLES_DIR / "ch5_country_gaps.csv", index=False)
    print(
        "country-waves:",
        len(gaps),
        "constraint<0:",
        int((gaps.b_constraint < 0).sum()),
        "|constraint|>|elections|:",
        int((gaps.b_constraint.abs() > gaps.b_elections.abs()).sum()),
    )

    fig_coefs(conf[conf.kind != "index"], "institution", "ch5_fig1_confidence", "kind")
    fig_coefs(ess, "characteristic", "ch5_fig2_essentials", None)
    fig_gaps(gaps)


def fig_coefs(t: pd.DataFrame, lab: str, name: str, kind: str | None) -> None:
    t = t.iloc[::-1]
    fig, ax = plt.subplots(figsize=(5.5, 0.35 * len(t) + 0.8))
    y = np.arange(len(t))
    colors = (
        [S.PATHWAY_COLORS["horizontal"] if k == "constraint" else S.NEUTRAL for k in t[kind]]
        if kind
        else [S.NEUTRAL] * len(t)
    )
    ax.errorbar(t.coef, y, xerr=1.96 * t.se, fmt="none", ecolor=colors, lw=1.2)
    ax.scatter(t.coef, y, color=colors, zorder=3, s=18)
    ax.axvline(0, color="black", lw=0.6)
    ax.set_yticks(y, t[lab])
    ax.set_xlabel("Standardized coefficient on AGP (95% CI)")
    ax.grid(axis="y", visible=False)
    S.save(fig, FIGURES_DIR / name)


def fig_gaps(g: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.scatter(g.b_elections, g.b_constraint, s=10, color=S.NEUTRAL, alpha=0.7)
    lim = [
        min(g.b_constraint.min(), g.b_elections.min()) - 0.02,
        max(g.b_constraint.max(), g.b_elections.max()) + 0.02,
    ]
    ax.plot(lim, lim, color="black", lw=0.5, ls=":")
    ax.plot(lim, [-v for v in lim], color="black", lw=0.5, ls=":")
    ax.axhline(0, color="black", lw=0.6)
    ax.axvline(0, color="black", lw=0.6)
    ax.set_xlabel("AGP coefficient: free elections as essential")
    ax.set_ylabel("AGP coefficient: confidence in\ncourts, parliament and press")
    S.save(fig, FIGURES_DIR / "ch5_fig3_country_gaps")


if __name__ == "__main__":
    main()
