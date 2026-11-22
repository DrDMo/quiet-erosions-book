"""Chapters 6-8: demand and accountability change on the three pathways.

QE2 (Chapter 2): the AGP coefficient is negative for post-wave change in at
least two of the three accountability indices and positive for the hazard of
quiet-erosion onset.
QE3: the horizontal and diagonal coefficients exceed the vertical in
magnitude, with at least one difference distinguishable from zero, in a joint
model of the three outcomes.

Change models (primary: h = 4) use at-risk country-waves from agp_link:
  d_index(t, t+h) = b * AGP_z(t) + c * index(t) + d * log GDPpc(t)
                    + wave FE + country FE,  SE clustered by country.
The joint model stacks the three outcomes with pathway-specific intercepts,
slopes on every covariate, and country-by-pathway fixed effects, so that the
AGP coefficients equal the separate estimates and their differences can be
tested with country-clustered covariance.

Hazard models use the country-year frame agp_hazard (AGP carried at most 8
years): logit onset ~ AGP_z + index level + log GDPpc + decade, clustered SE.

Outputs
  results/tables/ch6_change_models.csv    every pathway x horizon x spec
  results/tables/ch7_qe3_joint.csv        joint model coefficients and differences
  results/tables/ch6_hazard_models.csv    onset hazard per pathway
  figures/ch7_fig1_pathway_coefs.{png,pdf}
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from quiet_erosions import erosion as E
from quiet_erosions import plotstyle as S
from quiet_erosions.paths import FIGURES_DIR, PROC_DIR, TABLES_DIR, ensure_dir

PW = list(E.PATHWAYS)


def load_link() -> pd.DataFrame:
    d = pd.read_parquet(PROC_DIR / "agp_link.parquet")
    d = d[d.at_risk == 1].copy()
    d["agp_z"] = (d.agp_score - d.agp_score.mean()) / d.agp_score.std()
    d["log_gdp"] = np.log(d.gdppc.astype(float))
    d["wave"] = d.wave_chron.astype(int).astype(str)
    for c in d.columns:
        if c.startswith("d_"):
            d[c] = d[c].astype(float)
    return d


def change_model(d: pd.DataFrame, pw: str, h: int, spec: str) -> dict:
    y = f"d_{pw}_{h}"
    cols = [y, "agp_z", f"{pw}_t0", "log_gdp", "wave", "country_text_id", "region"]
    s = d[cols].dropna().copy()
    rhs = f"agp_z + {pw}_t0 + log_gdp + C(wave)"
    if spec == "country_fe":
        rhs += " + C(country_text_id)"
    elif spec == "region_fe":
        rhs += " + C(region)"
    m = smf.ols(f"{y} ~ {rhs}", data=s).fit(
        cov_type="cluster", cov_kwds={"groups": s.country_text_id.astype("category").cat.codes}
    )
    multi = s.country_text_id.value_counts()
    return dict(
        pathway=pw,
        horizon=h,
        spec=spec,
        coef=m.params.agp_z,
        se=m.bse.agp_z,
        p=m.pvalues.agp_z,
        n=int(m.nobs),
        countries=int(s.country_text_id.nunique()),
        countries_multi=int((multi > 1).sum()),
        r2_within=m.rsquared,
    )


def joint(d: pd.DataFrame, h: int, spec: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for pw in PW:
        s = d[[f"d_{pw}_{h}", "agp_z", f"{pw}_t0", "log_gdp", "wave", "country_text_id", "region"]]
        s = s.rename(columns={f"d_{pw}_{h}": "dy", f"{pw}_t0": "lvl"}).assign(pw=pw)
        rows.append(s)
    st = pd.concat(rows).dropna()
    # keep country-waves observed on all three outcomes
    st = st[st.groupby(["country_text_id", "wave"]).pw.transform("count") == 3]
    fe = "C(country_text_id):C(pw)" if spec == "country_fe" else "C(region):C(pw)"
    f = f"dy ~ 0 + C(pw) + C(pw):agp_z + C(pw):lvl + C(pw):log_gdp + C(pw):C(wave) + {fe}"
    m = smf.ols(f, data=st).fit(
        cov_type="cluster", cov_kwds={"groups": st.country_text_id.astype("category").cat.codes}
    )
    names = {pw: f"C(pw)[{pw}]:agp_z" for pw in PW}
    coefs = pd.DataFrame(
        [
            dict(spec=spec, horizon=h, pathway=pw, coef=m.params[n], se=m.bse[n], p=m.pvalues[n])
            for pw, n in names.items()
        ]
    )
    diffs = []
    for a, b in [("horizontal", "vertical"), ("diagonal", "vertical"), ("horizontal", "diagonal")]:
        t = m.t_test(f"{names[a]} - {names[b]} = 0")
        diffs.append(
            dict(
                spec=spec,
                horizon=h,
                contrast=f"{a} - {b}",
                diff=float(np.squeeze(t.effect)),
                se=float(np.squeeze(t.sd)),
                p=float(np.squeeze(t.pvalue)),
                n_country_waves=int(len(st) / 3),
            )
        )
    return coefs, pd.DataFrame(diffs)


def hazard(pw: str) -> dict:
    hz = pd.read_parquet(PROC_DIR / "agp_hazard.parquet")
    v = E.PATHWAYS[pw].replace("_osp", "")
    s = hz[hz[f"risk_{pw}"] == 1].copy()
    s["agp_z"] = (s.agp_score - s.agp_score.mean()) / s.agp_score.std()
    s["log_gdp"] = np.log(s.e_gdppc.astype(float))
    s["lvl"] = s[v]
    s["decade"] = (s.year // 10 * 10).astype(str)
    s["y"] = s[f"onset_{pw}"]
    s = s[["y", "agp_z", "lvl", "log_gdp", "decade", "country_text_id"]].dropna()
    out = dict(pathway=pw, n=len(s), countries=s.country_text_id.nunique(), events=int(s.y.sum()))
    try:
        m = smf.logit("y ~ agp_z + lvl + log_gdp + C(decade)", data=s).fit(
            disp=False,
            cov_type="cluster",
            cov_kwds={"groups": s.country_text_id.astype("category").cat.codes},
        )
        out.update(
            coef=m.params.agp_z,
            se=m.bse.agp_z,
            p=m.pvalues.agp_z,
            converged=m.mle_retvals["converged"],
        )
        m0 = smf.logit("y ~ agp_z", data=s).fit(
            disp=False,
            cov_type="cluster",
            cov_kwds={"groups": s.country_text_id.astype("category").cat.codes},
        )
        out.update(coef_bivar=m0.params.agp_z, se_bivar=m0.bse.agp_z, p_bivar=m0.pvalues.agp_z)
    except Exception as e:  # too few events to estimate
        out.update(error=str(e)[:80])
    return out


COMPONENTS = {
    "vertical": ["v2xel_frefair", "v2psparban", "v2psbars"],
    "horizontal": ["v2x_jucon", "v2xlg_legcon", "v2juhcind"],
    "diagonal": ["v2x_freexp_altinf", "v2mecenefm", "v2xcs_ccsi"],
}


def components(d: pd.DataFrame, h: int = 4) -> pd.DataFrame:
    """Same within-country specification on standardized component changes."""
    p = pd.read_parquet(PROC_DIR / "vdem_panel.parquet").set_index(["country_text_id", "year"])
    rows = []
    for pw, comps in COMPONENTS.items():
        for c in comps:
            s = d[["country_text_id", "wave_year", "agp_z", "log_gdp", "wave"]].copy()
            t0 = [
                p[c].get((k, y), np.nan)
                for k, y in zip(s.country_text_id, s.wave_year, strict=True)
            ]
            th = [
                p[c].get((k, y + h), np.nan)
                for k, y in zip(s.country_text_id, s.wave_year, strict=True)
            ]
            s["lvl"], s["dy"] = t0, np.array(th) - np.array(t0)
            s = s.dropna()
            s["dy"] = s.dy / s.dy.std()
            m = smf.ols("dy ~ agp_z + lvl + log_gdp + C(wave) + C(country_text_id)", data=s).fit(
                cov_type="cluster",
                cov_kwds={"groups": s.country_text_id.astype("category").cat.codes},
            )
            rows.append(
                dict(
                    pathway=pw,
                    component=c,
                    coef=m.params.agp_z,
                    se=m.bse.agp_z,
                    p=m.pvalues.agp_z,
                    n=int(m.nobs),
                )
            )
    return pd.DataFrame(rows)


def main() -> None:
    S.apply()
    ensure_dir(TABLES_DIR)
    ensure_dir(FIGURES_DIR)
    d = load_link()

    rows = []
    for pw in PW:
        for h in (2, 4, 6):
            for spec in ("country_fe", "region_fe"):
                rows.append(change_model(d, pw, h, spec))
    ch = pd.DataFrame(rows)
    ch.to_csv(TABLES_DIR / "ch6_change_models.csv", index=False)
    print(ch.round(3).to_string())

    jc, jd = [], []
    for h in (2, 4, 6):
        for spec in ("country_fe", "region_fe"):
            c, df_ = joint(d, h, spec)
            jc.append(c)
            jd.append(df_)
    jc, jd = pd.concat(jc), pd.concat(jd)
    pd.concat([jc.assign(kind="coef"), jd.assign(kind="contrast")]).to_csv(
        TABLES_DIR / "ch7_qe3_joint.csv", index=False
    )
    print(jc.round(3).to_string())
    print(jd.round(3).to_string())

    hz = pd.DataFrame([hazard(pw) for pw in PW])
    hz.to_csv(TABLES_DIR / "ch6_hazard_models.csv", index=False)
    print(hz.round(3).to_string())

    comp = components(d)
    comp.to_csv(TABLES_DIR / "ch7_components.csv", index=False)
    print(comp.round(3).to_string())

    fig_coefs(ch)
    fig_binned(d)
    fig_components(comp)


def fig_coefs(ch: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.8), sharey=True)
    for ax, spec, title in zip(
        axes, ("country_fe", "region_fe"), ("Within country", "Region fixed effects"), strict=True
    ):
        for i, pw in enumerate(PW):
            g = ch[(ch.pathway == pw) & (ch.spec == spec)].sort_values("horizon")
            x = np.arange(len(g)) + (i - 1) * 0.22
            ax.errorbar(
                x,
                g.coef,
                yerr=1.96 * g.se,
                fmt="o",
                ms=4,
                color=S.PATHWAY_COLORS[pw],
                label=S.PATHWAY_LABELS[pw],
                lw=1.2,
            )
        ax.axhline(0, color="black", lw=0.6)
        ax.set_xticks(range(3), ["2 years", "4 years", "6 years"])
        ax.set_title(title, fontsize=8)
    axes[0].set_ylabel("Change in index per SD of AGP\n(latent units, 95% CI)")
    axes[0].legend(fontsize=7, loc="lower left")
    S.save(fig, FIGURES_DIR / "ch7_fig1_pathway_coefs")


COMP_LABELS = {
    "v2xel_frefair": "Clean elections",
    "v2psparban": "Party bans (absence)",
    "v2psbars": "Barriers to parties (absence)",
    "v2x_jucon": "Judicial constraints",
    "v2xlg_legcon": "Legislative constraints",
    "v2juhcind": "High court independence",
    "v2x_freexp_altinf": "Expression and alt. information",
    "v2mecenefm": "Media censorship (absence)",
    "v2xcs_ccsi": "Core civil society",
}


def fig_components(comp: pd.DataFrame) -> None:
    t = comp.iloc[::-1].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    y = np.arange(len(t))
    for i, r in t.iterrows():
        col = S.PATHWAY_COLORS[r.pathway]
        ax.errorbar(r.coef, i, xerr=1.96 * r.se, fmt="o", ms=4, color=col, lw=1.2)
    ax.axvline(0, color="black", lw=0.6)
    ax.set_yticks(y, [COMP_LABELS[c] for c in t.component])
    ax.set_xlabel("Four-year change (SD) per SD of AGP, within country (95% CI)")
    ax.grid(axis="y", visible=False)
    S.save(fig, FIGURES_DIR / "ch8_fig1_components")


def fig_binned(d: pd.DataFrame, h: int = 4) -> None:
    """Partial-residual binned scatter of 4-year change on AGP, by pathway."""
    fig, axes = plt.subplots(1, 3, figsize=(7, 2.5), sharey=True)
    for ax, pw in zip(axes, PW, strict=True):
        y = f"d_{pw}_{h}"
        s = d[[y, "agp_z", f"{pw}_t0", "log_gdp", "wave", "country_text_id"]].dropna()
        ry = smf.ols(f"{y} ~ {pw}_t0 + log_gdp + C(wave) + C(country_text_id)", data=s).fit().resid
        rx = (
            smf.ols(f"agp_z ~ {pw}_t0 + log_gdp + C(wave) + C(country_text_id)", data=s).fit().resid
        )
        b = pd.qcut(rx, 10, labels=False)
        g = pd.DataFrame({"x": rx, "y": ry, "b": b}).groupby("b").mean()
        ax.scatter(rx, ry, s=5, color=S.PATHWAY_COLORS[pw], alpha=0.25, lw=0)
        ax.scatter(g.x, g.y, s=22, color=S.PATHWAY_COLORS[pw], zorder=3)
        beta = np.polyfit(rx, ry, 1)
        xs = np.linspace(rx.min(), rx.max(), 2)
        ax.plot(xs, np.polyval(beta, xs), color="black", lw=0.8)
        ax.axhline(0, color="black", lw=0.4)
        ax.set_ylim(-0.6, 0.6)
        ax.set_title(S.PATHWAY_LABELS[pw], fontsize=8, color=S.PATHWAY_COLORS[pw])
        ax.set_xlabel("AGP (residual)", fontsize=7)
    axes[0].set_ylabel("Four-year change (residual)")
    S.save(fig, FIGURES_DIR / "ch6_fig1_binned_change")


if __name__ == "__main__":
    main()
