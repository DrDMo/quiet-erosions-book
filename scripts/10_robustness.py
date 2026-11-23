"""Chapter 11: robustness of the main estimates.

A. Post-wave change models (QE2/QE3), within country, h = 4, under:
   primary; _osp outcome scale; fieldwork-year alignment; simple-mean AGP;
   without wave 7; without log GDP; dropping each region in turn; and
   measurement uncertainty, by re-estimating on 200 draws from a normal
   approximation to each country-year posterior (sd = half the 68% interval)
   and combining with Rubin's rules.
B. QE1: courts confidence excluding wave 4, when the item was asked in fewer
   than half of the countries.
C. Cross-lagged country-wave models (forward: index on lagged AGP; reverse:
   AGP on lagged index), with and without country fixed effects.
D. QE6 under alternative prior-erosion windows and quiet-erosion rules.

Outputs: results/tables/ch11_change_robustness.csv, ch11_posterior.csv,
         ch11_qe1_courts.csv, ch11_crosslag.csv, ch11_escalation_alt.csv,
         figures/ch11_fig1_ledger.{png,pdf}
"""

from __future__ import annotations

import sys
import warnings
from importlib import import_module
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from quiet_erosions import erosion as E
from quiet_erosions import plotstyle as S
from quiet_erosions.paths import FIGURES_DIR, PROC_DIR, TABLES_DIR

sys.path.insert(0, str(Path(__file__).parent))
warnings.filterwarnings("ignore")
PW = list(E.PATHWAYS)
RNG = np.random.default_rng(20261122)
SUP = import_module("04_link_agp").XWALK_SUPPLEMENT


def _xwalk() -> pd.DataFrame:
    xw = pd.read_parquet(PROC_DIR / "country_iso3_xwalk.parquet")
    extra = pd.DataFrame(list(SUP.items()), columns=["country_num", "iso3"])
    return pd.concat([xw[~xw.country_num.isin(extra.country_num)], extra], ignore_index=True)


def base() -> tuple[pd.DataFrame, pd.DataFrame]:
    d = pd.read_parquet(PROC_DIR / "agp_link.parquet")
    d = d[d.at_risk == 1].copy()
    d["agp_z"] = (d.agp_score - d.agp_score.mean()) / d.agp_score.std()
    d["log_gdp"] = np.log(d.gdppc.astype(float))
    d["wave"] = d.wave_chron.astype(int).astype(str)
    ind = pd.read_parquet(
        PROC_DIR / "qe_individual.parquet",
        columns=["country_num", "wave_chron", "year", "agp_mean", "weight"],
    )
    ind = ind.merge(_xwalk(), on="country_num")
    fy = ind.groupby(["iso3", "wave_chron"]).year.median().round().rename("field_year")
    sm = (
        ind.dropna(subset=["agp_mean"])
        .groupby(["iso3", "wave_chron"])
        .apply(lambda g: np.average(g.agp_mean, weights=g.weight.clip(lower=1e-9)))
        .rename("agp_simple")
    )
    d = d.merge(fy, left_on=["country_text_id", "wave_chron"], right_index=True, how="left")
    d = d.merge(sm, left_on=["country_text_id", "wave_chron"], right_index=True, how="left")
    d["agp_simple_z"] = (d.agp_simple - d.agp_simple.mean()) / d.agp_simple.std()
    p = (
        pd.read_parquet(PROC_DIR / "vdem_panel.parquet")
        .set_index(["country_text_id", "year"])
        .sort_index()
    )
    return d, p


def _get(p, var, cids, years):
    idx = pd.MultiIndex.from_arrays([cids, years])
    return p[var].reindex(idx).to_numpy(dtype=float)


def fit(
    s: pd.DataFrame, x: str = "agp_z", ctrl: str = " + log_gdp", fe: str = " + C(country_text_id)"
):
    f = f"dy ~ {x} + y0{ctrl} + C(wave){fe}"
    m = smf.ols(f, data=s).fit(
        cov_type="cluster", cov_kwds={"groups": s.country_text_id.astype("category").cat.codes}
    )
    return m.params[x], m.bse[x], int(m.nobs)


def frame(d, p, pw, scale="latent", t_col="wave_year", x="agp_z", h=4):
    v = E.PATHWAYS[pw] if scale == "osp" else E.PATHWAYS[pw].replace("_osp", "")
    t = d[t_col].astype(int).to_numpy()
    s = d[
        ["country_text_id", "wave", "log_gdp", "region", "agp_z", "agp_simple_z", "wave_chron"]
    ].copy()
    s["y0"] = _get(p, v, d.country_text_id, t)
    s["dy"] = _get(p, v, d.country_text_id, t + h) - s.y0
    return s.dropna(subset=["dy", "y0", x, "log_gdp"])


def change_robustness(d, p) -> pd.DataFrame:
    rows = []

    def add(spec, pw, res):
        rows.append(dict(spec=spec, pathway=pw, coef=res[0], se=res[1], n=res[2]))

    for pw in PW:
        add("Primary", pw, fit(frame(d, p, pw)))
        add("0-1 (osp) scale", pw, fit(frame(d, p, pw, scale="osp")))
        add(
            "Fieldwork-year alignment",
            pw,
            fit(frame(d.dropna(subset=["field_year"]), p, pw, t_col="field_year")),
        )
        add("Simple-mean AGP", pw, fit(frame(d, p, pw, x="agp_simple_z"), x="agp_simple_z"))
        add("Without wave 7", pw, fit(frame(d[d.wave_chron != 7], p, pw)))
        add("Without log GDP", pw, fit(frame(d, p, pw), ctrl=""))
        for r, lab in S.REGION_LABELS.items():
            s = frame(d[d.region != r], p, pw)
            add(f"Without {lab}", pw, fit(s))
    return pd.DataFrame(rows)


def posterior(d, p, draws: int = 200) -> pd.DataFrame:
    rows = []
    for pw in PW:
        v = E.PATHWAYS[pw].replace("_osp", "")
        sd = (p[f"{v}_codehigh"] - p[f"{v}_codelow"]) / 2
        t = d.wave_year.astype(int).to_numpy()
        mu0, mu1 = _get(p, v, d.country_text_id, t), _get(p, v, d.country_text_id, t + 4)
        s0, s1 = (
            sd.reindex(pd.MultiIndex.from_arrays([d.country_text_id, t])).to_numpy(),
            sd.reindex(pd.MultiIndex.from_arrays([d.country_text_id, t + 4])).to_numpy(),
        )
        b, se = [], []
        for _ in range(draws):
            y0 = mu0 + RNG.standard_normal(len(mu0)) * s0
            y1 = mu1 + RNG.standard_normal(len(mu1)) * s1
            s = (
                d[["country_text_id", "wave", "log_gdp", "agp_z"]]
                .assign(y0=y0, dy=y1 - y0)
                .dropna()
            )
            c, e, _ = fit(s)
            b.append(c)
            se.append(e)
        b, se = np.array(b), np.array(se)
        total = np.sqrt((se**2).mean() + (1 + 1 / draws) * b.var(ddof=1))
        rows.append(
            dict(
                pathway=pw,
                mean_coef=b.mean(),
                combined_se=total,
                sampling_se=np.sqrt((se**2).mean()),
                between_draw_sd=b.std(ddof=1),
                share_negative=(b < 0).mean(),
                draws=draws,
            )
        )
    return pd.DataFrame(rows)


def qe1_courts() -> pd.DataFrame:
    m = import_module("05_ch5_targeting")
    df = m.prep()
    out = []
    for lab, sub in [
        ("Waves 3-7", df),
        ("Waves 3, 5-7 (excluding wave 4)", df[df.wave_chron != 4]),
    ]:
        r = m.fit(sub, "conf_17")
        out.append(dict(sample=lab, **r))
    return pd.DataFrame(out)


def crosslag(p) -> pd.DataFrame:
    lk = pd.read_parquet(PROC_DIR / "agp_link.parquet")
    lk = lk.sort_values(["country_text_id", "wave_year"])
    lk["agp_z"] = (lk.agp_score - lk.agp_score.mean()) / lk.agp_score.std()
    rows = []
    for pw in PW:
        v = E.PATHWAYS[pw].replace("_osp", "")
        g = lk[["country_text_id", "wave_year", "wave_chron", "agp_z"]].copy()
        g["acc"] = _get(p, v, g.country_text_id, g.wave_year.astype(int))
        g["acc_z"] = (g.acc - g.acc.mean()) / g.acc.std()
        g["agp_lag"] = g.groupby("country_text_id").agp_z.shift(1)
        g["acc_lag"] = g.groupby("country_text_id").acc_z.shift(1)
        g["wave"] = g.wave_chron.astype(int).astype(str)
        g = g.dropna(subset=["agp_lag", "acc_lag", "acc_z", "agp_z"])
        for fe, lab in [("", "pooled, wave FE"), (" + C(country_text_id)", "country and wave FE")]:
            grp = {"groups": g.country_text_id.astype("category").cat.codes}
            fwd = smf.ols(f"acc_z ~ agp_lag + acc_lag + C(wave){fe}", data=g).fit(
                cov_type="cluster", cov_kwds=grp
            )
            rev = smf.ols(f"agp_z ~ acc_lag + agp_lag + C(wave){fe}", data=g).fit(
                cov_type="cluster", cov_kwds=grp
            )
            rows.append(
                dict(
                    pathway=pw,
                    spec=lab,
                    forward=fwd.params.agp_lag,
                    forward_se=fwd.bse.agp_lag,
                    forward_p=fwd.pvalues.agp_lag,
                    reverse=rev.params.acc_lag,
                    reverse_se=rev.bse.acc_lag,
                    reverse_p=rev.pvalues.acc_lag,
                    n=int(fwd.nobs),
                    countries=g.country_text_id.nunique(),
                )
            )
    return pd.DataFrame(rows)


def escalation_alt(p) -> pd.DataFrame:
    m = import_module("09_escalation_reversal")
    panel = pd.read_parquet(PROC_DIR / "vdem_panel.parquet")
    rows = []
    for delta in (0.10, 0.15, 0.20):
        rule = E.Rule(delta=delta)
        sp = pd.concat([E.spells(E.candidates(panel, pw, rule), panel, rule) for pw in PW])
        sp = sp[~sp.provisional]
        d = (
            panel[(panel.v2x_regime >= 2) & panel.year.between(1996, 2022)]
            .sort_values(["country_text_id", "year"])
            .copy()
        )
        d["prev_ep"] = d.groupby("country_text_id").aut_ep.shift(1)
        d = d[(d.prev_ep == 0)].dropna(subset=["e_gdppc"])
        d["ep_start"] = (d.aut_ep == 1).astype(int)
        d["log_gdp"] = np.log(d.e_gdppc)
        d["iso3"] = d.country_text_id
        on = sp.groupby("country_text_id").onset.apply(list).to_dict()
        for prior in (4, 6, 8, 10):
            lo_off, hi_off = prior + rule.k, rule.k + 1
            d["prior_qe"] = [
                int(any(y - lo_off <= s <= y - hi_off for s in on.get(c, [])))
                for c, y in zip(d.country_text_id, d.year, strict=True)
            ]
            r = dict(
                delta=delta,
                prior_years=prior,
                n=len(d),
                prior_n=int(d.prior_qe.sum()),
                onsets_after_prior=int(d.loc[d.prior_qe == 1, "ep_start"].sum()),
                rate_prior=d.loc[d.prior_qe == 1, "ep_start"].mean(),
                rate_none=d.loc[d.prior_qe == 0, "ep_start"].mean(),
            )
            try:
                f = m.logit("ep_start ~ prior_qe + v2x_polyarchy + log_gdp", d)
                r.update(coef=f.params.prior_qe, se=f.bse.prior_qe, p=f.pvalues.prior_qe)
            except Exception:
                pass
            rows.append(r)
    return pd.DataFrame(rows)


def fig_ledger(rob: pd.DataFrame, post: pd.DataFrame) -> None:
    specs = [s for s in rob.spec.unique()]
    fig, ax = plt.subplots(figsize=(6.2, 0.26 * (len(specs) + 1) + 0.9))
    y = np.arange(len(specs) + 1)
    for i, pw in enumerate(PW):
        g = rob[rob.pathway == pw].set_index("spec").loc[specs]
        off = (i - 1) * 0.22
        ax.errorbar(
            g.coef,
            y[:-1] + off,
            xerr=1.96 * g.se,
            fmt="o",
            ms=3,
            lw=0.9,
            color=S.PATHWAY_COLORS[pw],
            label=S.PATHWAY_LABELS[pw],
        )
        q = post[post.pathway == pw].iloc[0]
        ax.errorbar(
            q.mean_coef,
            y[-1] + off,
            xerr=1.96 * q.combined_se,
            fmt="s",
            ms=3,
            lw=0.9,
            color=S.PATHWAY_COLORS[pw],
        )
    ax.axvline(0, color="black", lw=0.6)
    ax.set_yticks(y, specs + ["Measurement uncertainty (200 draws)"], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("AGP coefficient on four-year change, within country (95% CI)")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(axis="y", visible=False)
    S.save(fig, FIGURES_DIR / "ch11_fig1_ledger")


def main() -> None:
    S.apply()
    d, p = base()
    rob = change_robustness(d, p)
    rob.to_csv(TABLES_DIR / "ch11_change_robustness.csv", index=False)
    print(rob.round(3).to_string())
    post = posterior(d, p)
    post.to_csv(TABLES_DIR / "ch11_posterior.csv", index=False)
    print(post.round(3).to_string())
    fig_ledger(rob, post)
    cl = crosslag(p)
    cl.to_csv(TABLES_DIR / "ch11_crosslag.csv", index=False)
    print(cl.round(3).to_string())
    c = qe1_courts()
    c.to_csv(TABLES_DIR / "ch11_qe1_courts.csv", index=False)
    print(c.round(3).to_string())
    e = escalation_alt(p)
    e.to_csv(TABLES_DIR / "ch11_escalation_alt.csv", index=False)
    print(e.round(3).to_string())


if __name__ == "__main__":
    main()
