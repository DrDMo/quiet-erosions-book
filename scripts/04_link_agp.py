"""Chapters 6-8: link country-wave AGP to the V-Dem/ERT panel.

Each country-wave AGP estimate from Volume 1 (agp_country_wave) is placed
at its wave year and matched to the accountability indices, the at-risk
flag, and quiet-erosion onsets (Chapter 4 primary rule). For horizons
h = 2, 4, 6 the file records the change in each latent index from the wave
year to wave year + h and whether a non-provisional onset on each pathway
began in (wave year, wave year + h].

Output
  QE_DATA_ROOT/processed/agp_link.parquet
  QE_DATA_ROOT/processed/agp_hazard.parquet  at-risk country-years carrying the most
                                         recent prior AGP wave (at most MAX_AGE years old)
  results/tables/ch6_link_coverage.csv   country-waves usable per wave and horizon
  results/tables/ch6_hazard_coverage.csv at-risk country-years and onsets with AGP
"""

from __future__ import annotations

import pandas as pd

from quiet_erosions import erosion as E
from quiet_erosions.paths import PROC_DIR, TABLES_DIR, ensure_dir

HORIZONS = (2, 4, 6)
MAX_AGE = 8  # years an AGP wave may be carried forward in the hazard frame

# The Volume 1 crosswalk was built from WVS country codes and omits ten
# countries surveyed only by the EVS. They are added here; Andorra, Macao,
# Northern Ireland and Puerto Rico have no V-Dem series and stay unmatched.
XWALK_SUPPLEMENT = {
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


def main() -> None:
    ensure_dir(TABLES_DIR)
    agp = pd.read_parquet(PROC_DIR / "agp_country_wave.parquet")
    xw = pd.read_parquet(PROC_DIR / "country_iso3_xwalk.parquet")
    extra = pd.DataFrame(list(XWALK_SUPPLEMENT.items()), columns=["country_num", "iso3"])
    xw = pd.concat([xw[~xw.country_num.isin(extra.country_num)], extra], ignore_index=True)
    panel = pd.read_parquet(PROC_DIR / "vdem_panel.parquet")
    onsets = pd.read_parquet(PROC_DIR / "erosion_onsets.parquet")

    agp = agp.merge(xw, on="country_num", how="left").rename(columns={"iso3": "country_text_id"})
    unmatched = agp[agp.country_text_id.isna() | ~agp.country_text_id.isin(panel.country_text_id)]
    print(f"country-waves without a V-Dem match: {len(unmatched)}")

    p = panel.merge(onsets, on=["country_text_id", "year"], how="left").set_index(
        ["country_text_id", "year"]
    )
    idx = {pw: E.PATHWAYS[pw].replace("_osp", "") for pw in E.PATHWAYS}
    rows = []
    for r in agp.dropna(subset=["country_text_id"]).itertuples():
        c, t = r.country_text_id, int(r.wave_year)
        if (c, t) not in p.index:
            continue
        base = p.loc[(c, t)]
        row = dict(
            country_text_id=c,
            wave_chron=r.wave_chron,
            wave_year=t,
            agp_score=r.agp_score,
            agp_score_se=r.agp_score_se_z,
            n=r.n,
            regime=base.v2x_regime,
            aut_ep=base.aut_ep,
            at_risk=int(base.v2x_regime >= 2 and base.aut_ep == 0),
            polyarchy=base.v2x_polyarchy,
            gdppc=base.e_gdppc,
            region=base.e_regionpol_6C,
        )
        for pw, v in idx.items():
            row[f"{pw}_t0"] = base[v]
            for h in HORIZONS:
                key = (c, t + h)
                row[f"d_{pw}_{h}"] = p.loc[key, v] - base[v] if key in p.index else pd.NA
                yrs = [(c, s) for s in range(t + 1, t + h + 1) if (c, s) in p.index]
                on = p.loc[yrs, f"onset_{pw}"].fillna(0) if yrs else pd.Series(dtype=float)
                row[f"onset_{pw}_{h}"] = int(on.max()) if len(on) else pd.NA
        rows.append(row)
    link = pd.DataFrame(rows)
    link.to_parquet(PROC_DIR / "agp_link.parquet", index=False)

    cov = []
    for w, g in link.groupby("wave_chron"):
        d = dict(wave=int(w), wave_year=int(g.wave_year.iloc[0]), country_waves=len(g))
        d["at_risk"] = int(g.at_risk.sum())
        for h in HORIZONS:
            d[f"at_risk_obs_h{h}"] = int((g.at_risk == 1).mul(g[f"d_horizontal_{h}"].notna()).sum())
            for pw in E.PATHWAYS:
                d[f"onsets_{pw}_h{h}"] = int(
                    g.loc[g.at_risk == 1, f"onset_{pw}_{h}"].fillna(0).sum()
                )
        cov.append(d)
    cov = pd.DataFrame(cov)
    cov.to_csv(TABLES_DIR / "ch6_link_coverage.csv", index=False)
    print(cov.T.to_string())

    # Country-year hazard frame: the most recent wave strictly before year t
    m = agp.dropna(subset=["country_text_id"])[
        ["country_text_id", "wave_year", "agp_score", "agp_score_se_z"]
    ].sort_values("wave_year")
    hz = panel.merge(onsets, on=["country_text_id", "year"], how="left").sort_values("year")
    hz = pd.merge_asof(
        hz,
        m.assign(agp_year=m.wave_year).rename(columns={"wave_year": "year_key"}),
        left_on="year",
        right_on="year_key",
        by="country_text_id",
        allow_exact_matches=False,
        direction="backward",
    )
    hz["agp_age"] = hz.year - hz.agp_year
    hz = hz[(hz.agp_age <= MAX_AGE) & (hz.year >= E.PRIMARY.min_year)]
    hz = hz.drop(columns="year_key")
    hz.to_parquet(PROC_DIR / "agp_hazard.parquet", index=False)
    hc = []
    for pw in E.PATHWAYS:
        r = hz[hz[f"risk_{pw}"] == 1]
        hc.append(
            dict(
                pathway=pw,
                at_risk_country_years=len(r),
                countries=r.country_text_id.nunique(),
                onsets=int(r[f"onset_{pw}"].sum()),
            )
        )
    hc = pd.DataFrame(hc)
    hc.to_csv(TABLES_DIR / "ch6_hazard_coverage.csv", index=False)
    print(hc.to_string())


if __name__ == "__main__":
    main()
