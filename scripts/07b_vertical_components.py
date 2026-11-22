"""Chapter 8: component breakdown of the four vertical-first spells.

Reports each v2x_veracc source indicator (V-Dem v14) in the year before
onset and four years after, for the spells classified "V first" in
ch8_sequence.csv. Output: results/tables/ch8_vertical_components.csv
"""

import os

import pandas as pd

ROOT = os.environ["QE_DATA_ROOT"]
VDEM = os.path.join(ROOT, "raw/unz/V-Dem-CY-Full+Others-v14.csv")
COLS = [
    "v2x_veracc", "v2elembaut", "v2elembcap", "v2elrgstry", "v2elirreg",
    "v2elintim", "v2elmulpar", "v2elfrfair", "v2psparban", "v2psbars",
    "v2psoppaut",
]
K = 4

seq = pd.read_csv("results/tables/ch8_sequence.csv")
cases = seq[seq["order"] == "V first"][["country_text_id", "vertical"]]
vd = pd.read_csv(VDEM, usecols=["country_text_id", "year"] + COLS)

rows = []
for c, onset in cases.itertuples(index=False):
    onset = int(onset)
    s = vd[vd.country_text_id == c].set_index("year")
    for v in COLS:
        pre, post = s.at[onset - 1, v], s.at[onset + K, v]
        rows.append({"country": c, "onset": onset, "indicator": v,
                     "pre": round(pre, 3), "post": round(post, 3),
                     "change": round(post - pre, 3)})
pd.DataFrame(rows).to_csv("results/tables/ch8_vertical_components.csv", index=False)
