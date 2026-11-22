"""Chapter 9, Section 9.6: high-demand at-risk country-waves and what followed.

Upper quartile of AGP among at-risk country-waves; counts those followed by a
four-year decline of at least delta (0.15 latent units) on any index, and by
a non-provisional quiet-erosion onset within four years.

Output: results/tables/ch9_high_demand.csv
"""

from __future__ import annotations

import pandas as pd

from quiet_erosions import plotstyle as S
from quiet_erosions.erosion import PRIMARY
from quiet_erosions.paths import PROC_DIR, TABLES_DIR


def main() -> None:
    d = pd.read_parquet(PROC_DIR / "agp_link.parquet")
    d = d[d.at_risk == 1]
    q = d.agp_score.quantile(0.75)
    h = d[d.agp_score > q].copy()
    ch = h[[f"d_{p}_4" for p in ("vertical", "horizontal", "diagonal")]].astype(float)
    h["any_decline_delta"] = (ch <= -PRIMARY.delta).any(axis=1)
    on = h[[f"onset_{p}_4" for p in ("vertical", "horizontal", "diagonal")]].fillna(0).astype(int)
    h["onset_4"] = on.max(axis=1).astype(bool)
    h["region_label"] = h.region.map(S.REGION_LABELS)
    h.sort_values("agp_score", ascending=False)[
        [
            "country_text_id",
            "wave_year",
            "agp_score",
            "region_label",
            "any_decline_delta",
            "onset_4",
        ]
    ].to_csv(TABLES_DIR / "ch9_high_demand.csv", index=False)
    print(f"upper quartile cut {q:.3f}; country-waves {len(h)}")
    print(f"followed by a decline >= delta on any index: {int(h.any_decline_delta.sum())}")
    print(f"followed by a quiet-erosion onset within 4 years: {int(h.onset_4.sum())}")
    print(h.region_label.value_counts().to_string())


if __name__ == "__main__":
    main()
