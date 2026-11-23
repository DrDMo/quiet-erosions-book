"""Chapter 10: escalation (QE6) and reversal (QE7).

QE6. The Volume 1 episode-onset model (country-wave logit of ERT onset within
the wave window on AGP; Molnar 2026, ch. 6) is re-estimated on its own sample
of 325 country-waves and on the corrected 347, with and without an indicator
of prior quiet erosion: a non-provisional quiet-erosion spell on any pathway
whose four-year window closed in the six years before the wave window opens.
The window must have closed because the Chapter 4 rule excludes any window
that overlaps an ERT episode, so an onset in the four years before an episode
cannot exist by construction; counting such years would bias the test
against QE6. Supported if prior erosion is
a positive, significant predictor and the AGP coefficient attenuates. The
share of the AGP coefficient that coincides with prior erosion is
(b0 - b1) / b0 with a country-cluster bootstrap CI. A country-year version
for democracies (episode start in t on AGP carried forward and an active or
recent quiet-erosion spell) is reported as a secondary specification.

QE7. Among non-provisional spells with two AGP waves before onset, recovery
(an index returning above A(t) - delta/2) is compared between spells that
began under declining and under rising AGP (change between the two most
recent waves at or before onset). Supported if recovery is more likely
under declining AGP, by a test at conventional levels.

Outputs: results/tables/ch10_escalation.csv, ch10_escalation_cy.csv,
         ch10_reversal.csv, ch10_reversal_spells.csv
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

from quiet_erosions.paths import PROC_DIR, TABLES_DIR

WAVE_WINDOW = {3: (1995, 1999), 4: (1999, 2004), 5: (2004, 2009), 6: (2009, 2015), 7: (2015, 2022)}
SUPPLEMENT = {
    40: "AUT",
    56: "BEL",
    197: "CYP",
    208: "DNK",
    352: "ISL",
    372: "IRL",
    442: "LUX",
    470: "MLT",
    620: "PRT",
    915: "XKX",
}
PRIOR = 6
K = 4  # Chapter 4 window length


def _prior_range(y: int) -> tuple[int, int]:
    """Onset years whose window [s, s+K] closed within PRIOR years before y."""
    return y - PRIOR - K, y - K - 1


warnings.filterwarnings("ignore")
RNG = np.random.default_rng(20261122)


def ds_panel(corrected: bool) -> pd.DataFrame:
    agp = pd.read_parquet(PROC_DIR / "agp_country_wave.parquet")
    xw = pd.read_parquet(PROC_DIR / "country_iso3_xwalk.parquet")
    if corrected:
        extra = pd.DataFrame(list(SUPPLEMENT.items()), columns=["country_num", "iso3"])
        xw = pd.concat([xw[~xw.country_num.isin(extra.country_num)], extra], ignore_index=True)
    agp = agp.merge(xw, on="country_num", how="inner")
    ert = pd.read_csv(
        PROC_DIR.parent / "raw" / "vdem_ert" / "ert.csv",
        low_memory=False,
        usecols=["country_text_id", "aut_ep_start_year"],
    )
    starts = ert.dropna().drop_duplicates()
    sp = pd.read_csv(TABLES_DIR / "ch4_spells.csv")
    sp = sp[~sp.provisional]
    rows = []
    for r in agp.itertuples():
        lo, hi = WAVE_WINDOW[int(r.wave_chron)]
        s = starts[(starts.country_text_id == r.iso3) & starts.aut_ep_start_year.between(lo, hi)]
        q = sp[(sp.country_text_id == r.iso3) & sp.onset.between(*_prior_range(lo))]
        rows.append(
            dict(
                iso3=r.iso3,
                wave=int(r.wave_chron),
                agp=r.agp_score,
                onset=int(len(s) > 0),
                prior_qe=int(len(q) > 0),
            )
        )
    return pd.DataFrame(rows)


def logit(f, d):
    return smf.logit(f, data=d).fit(
        disp=False, cov_type="cluster", cov_kwds={"groups": d.iso3.astype("category").cat.codes}
    )


def escalation() -> pd.DataFrame:
    out = []
    for corrected in (False, True):
        d = ds_panel(corrected)
        m0, m1 = logit("onset ~ agp", d), logit("onset ~ agp + prior_qe", d)
        m2 = logit("onset ~ agp + prior_qe + C(wave)", d)
        share = (m0.params.agp - m1.params.agp) / m0.params.agp
        isos = d.iso3.unique()
        by = {c: g for c, g in d.groupby("iso3")}
        bs = []
        for _ in range(1000):
            pick = RNG.choice(isos, len(isos), replace=True)
            b = pd.concat([by[c].assign(iso3=f"{c}_{i}") for i, c in enumerate(pick)])
            try:
                b0 = smf.logit("onset ~ agp", data=b).fit(disp=False).params.agp
                b1 = smf.logit("onset ~ agp + prior_qe", data=b).fit(disp=False).params.agp
                bs.append((b0 - b1) / b0)
            except Exception:
                continue
        lo, hi = np.percentile(bs, [2.5, 97.5])
        out.append(
            dict(
                sample="corrected (347)" if corrected else "Volume 1 (325)",
                n=len(d),
                events=int(d.onset.sum()),
                prior_qe_n=int(d.prior_qe.sum()),
                onsets_after_prior_qe=int(d[d.prior_qe == 1].onset.sum()),
                agp_m0=m0.params.agp,
                agp_m0_se=m0.bse.agp,
                agp_m1=m1.params.agp,
                agp_m1_se=m1.bse.agp,
                prior_m1=m1.params.prior_qe,
                prior_m1_se=m1.bse.prior_qe,
                prior_m1_p=m1.pvalues.prior_qe,
                prior_m2=m2.params.prior_qe,
                prior_m2_se=m2.bse.prior_qe,
                agp_m2=m2.params.agp,
                share=share,
                share_lo=lo,
                share_hi=hi,
            )
        )
    return pd.DataFrame(out)


def escalation_cy() -> pd.DataFrame:
    p = pd.read_parquet(PROC_DIR / "vdem_panel.parquet")
    hz = pd.read_parquet(PROC_DIR / "agp_hazard.parquet")[["country_text_id", "year", "agp_score"]]
    sp = pd.read_csv(TABLES_DIR / "ch4_spells.csv")
    sp = sp[~sp.provisional]
    d = (
        p[(p.v2x_regime >= 2) & p.year.between(1996, 2022)]
        .sort_values(["country_text_id", "year"])
        .copy()
    )
    d["prev_ep"] = d.groupby("country_text_id").aut_ep.shift(1)
    d = d[d.prev_ep == 0]  # at risk of episode start
    d["ep_start"] = (d.aut_ep == 1).astype(int)

    def recent(c, y):
        s = sp[(sp.country_text_id == c) & sp.onset.between(*_prior_range(y))]
        return int(len(s) > 0)

    def ongoing(c, y):
        s = sp[(sp.country_text_id == c) & (sp.onset <= y - K - 1)]
        s = s[s.recovery_year.isna() | (s.recovery_year >= y)]
        return int(len(s) > 0)

    d["prior_qe"] = [recent(c, y) for c, y in zip(d.country_text_id, d.year, strict=True)]
    d["ongoing_qe"] = [ongoing(c, y) for c, y in zip(d.country_text_id, d.year, strict=True)]
    d = d.merge(hz, on=["country_text_id", "year"], how="left")
    d["log_gdp"] = np.log(d.e_gdppc)
    d["iso3"] = d.country_text_id
    rows = []
    for lab, s, f in [
        (
            "democracies, all years",
            d.dropna(subset=["log_gdp"]),
            "ep_start ~ prior_qe + v2x_polyarchy + log_gdp",
        ),
        (
            "democracies with AGP",
            d.dropna(subset=["agp_score", "log_gdp"]),
            "ep_start ~ prior_qe + agp_score + v2x_polyarchy + log_gdp",
        ),
        (
            "democracies with AGP, no prior",
            d.dropna(subset=["agp_score", "log_gdp"]),
            "ep_start ~ agp_score + v2x_polyarchy + log_gdp",
        ),
        (
            "democracies, ongoing spell",
            d.dropna(subset=["log_gdp"]).assign(prior_qe=lambda x: x.ongoing_qe),
            "ep_start ~ prior_qe + v2x_polyarchy + log_gdp",
        ),
    ]:
        m = logit(f, s)
        r = dict(spec=lab, n=int(m.nobs), events=int(s.ep_start.sum()), countries=s.iso3.nunique())
        for v in ("prior_qe", "agp_score"):
            if v in m.params:
                r.update({v: m.params[v], f"{v}_se": m.bse[v], f"{v}_p": m.pvalues[v]})
        r["rate_prior1"] = s.loc[s.prior_qe == 1, "ep_start"].mean()
        r["rate_prior0"] = s.loc[s.prior_qe == 0, "ep_start"].mean()
        rows.append(r)
    return pd.DataFrame(rows)


def _power(n_dec: int, n_rise: int, sims: int = 4000) -> dict:
    """Power of the one-sided Fisher test at alpha = .05 for illustrative gaps."""
    out = {}
    for p_dec, p_rise in [(0.30, 0.20), (0.40, 0.15), (0.50, 0.15), (0.60, 0.10)]:
        hit = 0
        for _ in range(sims):
            a, b = RNG.binomial(n_dec, p_dec), RNG.binomial(n_rise, p_rise)
            _, p = stats.fisher_exact([[n_rise - b, b], [n_dec - a, a]], alternative="greater")
            hit += p < 0.05
        out[f"power_{p_dec:.2f}_vs_{p_rise:.2f}"] = hit / sims
    return out


def reversal() -> tuple[pd.DataFrame, pd.DataFrame]:
    sp = pd.read_csv(TABLES_DIR / "ch4_spells.csv")
    sp = sp[~sp.provisional].copy()
    link = pd.read_parquet(PROC_DIR / "agp_link.parquet")[
        ["country_text_id", "wave_year", "agp_score"]
    ]
    tr = []
    for r in sp.itertuples():
        w = link[
            (link.country_text_id == r.country_text_id) & (link.wave_year <= r.onset)
        ].sort_values("wave_year")
        tr.append(w.agp_score.iloc[-1] - w.agp_score.iloc[-2] if len(w) >= 2 else np.nan)
    sp["agp_trend"] = tr
    sp["recovered"] = (~sp.censored).astype(int)
    s = sp.dropna(subset=["agp_trend"]).copy()
    s["declining"] = (s.agp_trend < 0).astype(int)
    tab = pd.crosstab(s.declining, s.recovered)
    tab = tab.reindex(index=[0, 1], columns=[0, 1], fill_value=0)
    _, p_fisher = stats.fisher_exact(tab.values, alternative="greater")
    res = pd.DataFrame(
        [
            dict(
                spells_all=len(sp),
                recovered_all=int(sp.recovered.sum()),
                spells_with_trend=len(s),
                declining=int(s.declining.sum()),
                rising=int((1 - s.declining).sum()),
                rec_declining=int(s.loc[s.declining == 1, "recovered"].sum()),
                rec_rising=int(s.loc[s.declining == 0, "recovered"].sum()),
                p_fisher_one_sided=p_fisher,
                **_power(int(s.declining.sum()), int((1 - s.declining).sum())),
            )
        ]
    )
    return res, s


def main() -> None:
    e = escalation()
    e.to_csv(TABLES_DIR / "ch10_escalation.csv", index=False)
    print(e.round(3).T.to_string())
    c = escalation_cy()
    c.to_csv(TABLES_DIR / "ch10_escalation_cy.csv", index=False)
    print(c.round(3).T.to_string())
    r, s = reversal()
    r.to_csv(TABLES_DIR / "ch10_reversal.csv", index=False)
    s.to_csv(TABLES_DIR / "ch10_reversal_spells.csv", index=False)
    print(r.T.to_string())
    print(
        s[["country_text_id", "pathway", "onset", "agp_trend", "recovered", "recovery_year"]]
        .round(3)
        .to_string()
    )


if __name__ == "__main__":
    main()
