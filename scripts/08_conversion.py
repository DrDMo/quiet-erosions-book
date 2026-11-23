"""Chapter 9: conversion and gatekeeping (QE5).

QE5a (mediation). Mediator: two-year change in an index of executive conduct
toward the courts, the mean of standardized government attacks on the
judiciary (v2jupoatck) and court packing (v2jupack), both coded so that
higher values mean better conduct. Neither indicator enters any of the three
accountability indices (V-Dem v14 codebook, 5.2.2-5.2.4), so the indirect
path is not a mechanical overlap. A secondary mediator adds executive respect
for the constitution (v2exrescon), which enters v2x_horacc; for the
horizontal outcome that variant is descriptive only.
  a: dM(t, t+2) ~ AGP + level + log GDP + wave FE + country FE
  b: dY(t, t+4) ~ dM + AGP + same controls
  indirect = a * b; 95% CI from a country-cluster bootstrap (1,000 draws).

QE5b (moderation). dY(t, t+4) ~ AGP * Z + controls + FE for Z in party
institutionalization (v2xps_party), polarization (v2cacamps), presidential
format (head of state is head of government, or holds more cabinet power),
each standardized (format binary). Estimated per pathway and in a stacked
model with pathway-specific controls and a common AGP x Z term.

Outputs: results/tables/ch9_mediation.csv, ch9_moderation.csv
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from quiet_erosions import erosion as E
from quiet_erosions.paths import PROC_DIR, TABLES_DIR

PW = list(E.PATHWAYS)
RNG = np.random.default_rng(20261122)
B = 1000


def build() -> pd.DataFrame:
    d = pd.read_parquet(PROC_DIR / "agp_link.parquet")
    d = d[d.at_risk == 1].copy()
    d["agp_z"] = (d.agp_score - d.agp_score.mean()) / d.agp_score.std()
    d["log_gdp"] = np.log(d.gdppc.astype(float))
    d["wave"] = d.wave_chron.astype(int).astype(str)
    p = pd.read_parquet(PROC_DIR / "vdem_panel.parquet").set_index(["country_text_id", "year"])
    z = {c: (p[c] - p[c].mean()) / p[c].std() for c in ["v2jupoatck", "v2jupack", "v2exrescon"]}
    p["conduct"] = (z["v2jupoatck"] + z["v2jupack"]) / 2
    p["conduct_rc"] = (z["v2jupoatck"] + z["v2jupack"] + z["v2exrescon"]) / 3
    p["presidential"] = ((p.v2exhoshog == 1) | (p.v2ex_hosw > 0.5)).astype(float)

    def at(c, off=0):
        return np.array(
            [
                p[c].get((k, y + off), np.nan)
                for k, y in zip(d.country_text_id, d.wave_year, strict=True)
            ],
            dtype=float,
        )

    for m in ("conduct", "conduct_rc"):
        d[f"{m}_t0"] = at(m)
        d[f"d_{m}"] = at(m, 2) - d[f"{m}_t0"]
    for c, name in [
        ("v2xps_party", "party_inst"),
        ("v2cacamps", "polar"),
        ("presidential", "pres"),
    ]:
        d[name] = at(c)
    for c in ("party_inst", "polar"):
        d[f"{c}_z"] = (d[c] - d[c].mean()) / d[c].std()
    for pw in PW:
        d[f"d_{pw}_4"] = d[f"d_{pw}_4"].astype(float)
    return d


def _fit(f, s):
    return smf.ols(f, data=s).fit()


def mediation(d: pd.DataFrame, pw: str, med: str) -> dict:
    y = f"d_{pw}_4"
    cols = [y, f"d_{med}", f"{med}_t0", "agp_z", f"{pw}_t0", "log_gdp", "wave", "country_text_id"]
    s = (
        d[cols]
        .dropna()
        .rename(columns={y: "dy", f"d_{med}": "dm", f"{med}_t0": "m0", f"{pw}_t0": "y0"})
    )
    s["cid"] = s.country_text_id
    fa = "dm ~ agp_z + m0 + log_gdp + C(wave) + C(cid)"
    fb = "dy ~ dm + agp_z + y0 + log_gdp + C(wave) + C(cid)"
    a, b = _fit(fa, s), _fit(fb, s)
    grp = s.country_text_id.astype("category").cat.codes
    ca = smf.ols(fa, data=s).fit(cov_type="cluster", cov_kwds={"groups": grp})
    cb = smf.ols(fb, data=s).fit(cov_type="cluster", cov_kwds={"groups": grp})
    est = a.params.agp_z * b.params.dm
    countries = s.country_text_id.unique()
    by = {c: g for c, g in s.groupby("country_text_id")}
    draws = []
    for _ in range(B):
        pick = RNG.choice(countries, len(countries), replace=True)
        bs = pd.concat(
            [by[c].assign(cid=f"{c}_{i}") for i, c in enumerate(pick)], ignore_index=True
        )
        try:
            draws.append(_fit(fa, bs).params.agp_z * _fit(fb, bs).params.dm)
        except Exception:
            continue
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return dict(
        pathway=pw,
        mediator=med,
        a=ca.params.agp_z,
        a_se=ca.bse.agp_z,
        b=cb.params.dm,
        b_se=cb.bse.dm,
        direct=cb.params.agp_z,
        direct_se=cb.bse.agp_z,
        indirect=est,
        ci_lo=lo,
        ci_hi=hi,
        draws=len(draws),
        n=len(s),
        countries=len(countries),
    )


def moderation(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    mods = {
        "party_inst_z": "Party institutionalization",
        "polar_z": "Polarization",
        "pres": "Presidential format",
    }
    for z, lab in mods.items():
        for pw in PW:
            y = f"d_{pw}_4"
            s = d[[y, "agp_z", z, f"{pw}_t0", "log_gdp", "wave", "country_text_id"]].dropna()
            m = smf.ols(
                f"{y} ~ agp_z * {z} + {pw}_t0 + log_gdp + C(wave) + C(country_text_id)", data=s
            ).fit(
                cov_type="cluster",
                cov_kwds={"groups": s.country_text_id.astype("category").cat.codes},
            )
            t = f"agp_z:{z}"
            rows.append(
                dict(
                    moderator=lab,
                    model=pw,
                    coef=m.params[t],
                    se=m.bse[t],
                    p=m.pvalues[t],
                    agp_main=m.params.agp_z,
                    n=int(m.nobs),
                    z_sd=s[z].std(),
                )
            )
        st = []
        for pw in PW:
            s = d[[f"d_{pw}_4", "agp_z", z, f"{pw}_t0", "log_gdp", "wave", "country_text_id"]]
            st.append(s.rename(columns={f"d_{pw}_4": "dy", f"{pw}_t0": "lvl"}).assign(pw=pw))
        st = pd.concat(st).dropna()
        f = (
            f"dy ~ 0 + C(pw) + C(pw):agp_z + C(pw):{z} + agp_z:{z} + C(pw):lvl + C(pw):log_gdp"
            " + C(pw):C(wave) + C(country_text_id):C(pw)"
        )
        m = smf.ols(f, data=st).fit(
            cov_type="cluster", cov_kwds={"groups": st.country_text_id.astype("category").cat.codes}
        )
        t = f"agp_z:{z}"
        rows.append(
            dict(
                moderator=lab,
                model="stacked",
                coef=m.params[t],
                se=m.bse[t],
                p=m.pvalues[t],
                agp_main=np.nan,
                n=int(len(st) / 3),
                z_sd=np.nan,
            )
        )
    # joint: all three moderators together, stacked
    st = []
    for pw in PW:
        s = d[[f"d_{pw}_4", "agp_z", *mods, f"{pw}_t0", "log_gdp", "wave", "country_text_id"]]
        st.append(s.rename(columns={f"d_{pw}_4": "dy", f"{pw}_t0": "lvl"}).assign(pw=pw))
    st = pd.concat(st).dropna()
    inter = " + ".join(f"agp_z:{z} + C(pw):{z}" for z in mods)
    f = (
        f"dy ~ 0 + C(pw) + C(pw):agp_z + {inter} + C(pw):lvl + C(pw):log_gdp + C(pw):C(wave)"
        " + C(country_text_id):C(pw)"
    )
    m = smf.ols(f, data=st).fit(
        cov_type="cluster", cov_kwds={"groups": st.country_text_id.astype("category").cat.codes}
    )
    for z, lab in mods.items():
        t = f"agp_z:{z}"
        rows.append(
            dict(
                moderator=lab,
                model="stacked, joint",
                coef=m.params[t],
                se=m.bse[t],
                p=m.pvalues[t],
                agp_main=np.nan,
                n=int(len(st) / 3),
                z_sd=np.nan,
            )
        )
    return pd.DataFrame(rows)


def main() -> None:
    d = build()
    print(d[["party_inst", "polar", "pres", "d_conduct"]].describe().round(2))
    med = pd.DataFrame([mediation(d, pw, m) for m in ("conduct", "conduct_rc") for pw in PW])
    med.to_csv(TABLES_DIR / "ch9_mediation.csv", index=False)
    print(med.round(3).to_string())
    mod = moderation(d)
    mod.to_csv(TABLES_DIR / "ch9_moderation.csv", index=False)
    print(mod.round(3).to_string())


if __name__ == "__main__":
    main()
