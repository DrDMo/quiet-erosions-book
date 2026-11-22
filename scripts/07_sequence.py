"""Chapter 8: order of pathway onsets in multi-pathway cases (QE4).

A multi-pathway case is a country with non-provisional quiet-erosion spells
on more than one pathway. QE4 concerns the position of vertical erosion, so
the test set is cases that include a vertical spell. For each, the first
onset year on each pathway is compared; ties count as neither preceding.
Cases are split at the median AGP of at-risk country-waves, using the most
recent wave at or before the first onset in the case.

QE4 is supported if horizontal or diagonal erosion precedes vertical in a
majority of upper-half cases, and that majority exceeds the lower-half share.

Output: results/tables/ch8_sequence.csv
"""

from __future__ import annotations

import pandas as pd

from quiet_erosions.paths import PROC_DIR, TABLES_DIR


def main() -> None:
    sp = pd.read_csv(TABLES_DIR / "ch4_spells.csv")
    sp = sp[~sp.provisional]
    first = sp.groupby(["country_text_id", "pathway"]).onset.min().unstack()
    multi = first[first.notna().sum(axis=1) > 1].copy()
    link = pd.read_parquet(PROC_DIR / "agp_link.parquet")
    med = link.loc[link.at_risk == 1, "agp_score"].median()
    rows = []
    for c, r in multi.iterrows():
        t0 = int(r.min())
        w = link[(link.country_text_id == c) & (link.wave_year <= t0)].sort_values("wave_year")
        agp = w.agp_score.iloc[-1] if len(w) else float("nan")
        v = r.get("vertical")
        hd = min(x for x in [r.get("horizontal"), r.get("diagonal")] if pd.notna(x))
        order = (
            "no vertical"
            if pd.isna(v)
            else ("H/D first" if hd < v else ("V first" if v < hd else "tie"))
        )
        rows.append(
            dict(
                country_text_id=c,
                vertical=v,
                horizontal=r.get("horizontal"),
                diagonal=r.get("diagonal"),
                first_onset=t0,
                agp_wave_year=int(w.wave_year.iloc[-1]) if len(w) else None,
                agp=agp,
                agp_half=("upper" if agp > med else "lower") if pd.notna(agp) else "no AGP",
                order=order,
            )
        )
    out = pd.DataFrame(rows).sort_values(["order", "first_onset"])
    out.to_csv(TABLES_DIR / "ch8_sequence.csv", index=False)
    print(f"median at-risk AGP: {med:.3f}")
    print(out.round(3).to_string())
    print(pd.crosstab(out.order, out.agp_half))


if __name__ == "__main__":
    main()
