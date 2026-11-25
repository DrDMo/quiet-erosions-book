# Quiet Erosions: Replication Archive

Code and output tables for **_Quiet Erosions: Authoritarian Demand and the Institutional Pathways of Democratic Backsliding_** (Darin R. Molnar, Shoreline Publishing Group, 2026; Volume 2 of the Shoreline Publishing Group Comparative Politics Series).

Every estimate in the book is produced by a script in `scripts/`. The tables they write are in `results/tables/`, and the figures are in `figures/`. Raw data are not redistributed: obtain them from the originating archives as described below.

## Repository layout

```text
.
├── scripts/                  # Analysis scripts, run in numerical order (see scripts/README.md)
├── src/quiet_erosions/       # Shared code: the quiet-erosion rule (erosion.py), paths, plot style
├── results/tables/           # Output tables (CSV), named by chapter
├── figures/                  # Book figures (PNG and PDF), named by chapter
├── tests/                    # Pytest checks for shared code
├── Data/                     # Local data folder (git-ignored)
├── pyproject.toml
├── CITATION.cff
└── LICENSE
```

## Data sources

Point the `QE_DATA_ROOT` environment variable at a local folder with this layout (without it, `Data/` in this repository is used):

| Path under `$QE_DATA_ROOT` | Source | Used by |
|---|---|---|
| `raw/unz/V-Dem-CY-Full+Others-v14.csv` | V-Dem Country-Year dataset v14 | `01_build_panel.py` |
| `raw/vdem_ert/ert.csv` | V-Dem Episodes of Regime Transformation (ERT) v16 | `01_build_panel.py` |
| `raw/wvs_trend/wvs_trend_full.parquet` | WVS Integrated Trend File v4.1, converted from SPSS | `03_build_individual.py` |
| `raw/evs_trend/evs_trend_full.parquet` | EVS Trend File v3.0.0 (ZA7503), converted from SPSS | `03_build_individual.py` |
| `processed/agp_country_wave.parquet` | Volume 1 country-wave AGP scores | `04_link_agp.py`, `09_escalation_reversal.py` |
| `processed/country_iso3_xwalk.parquet` | Volume 1 country crosswalk | `04`, `05`, `09`, `10` |

Download locations, versions, and citations for the four raw files are in the Volume 1 archive's [data sources documentation](https://github.com/DrDMo/demanding-strongmen-book/blob/main/docs/data_sources.md). The two survey files are converted to Parquet by Volume 1's `code/00_convert_spss.py`. The two Volume 1 files are produced by its `code/04_agp_scores.py` (AGP scores) and `code/07_ch6_hazard_models.py` (crosswalk). `04_link_agp.py` adds the ten EVS-only countries missing from the Volume 1 crosswalk (Chapter 3, Section 3.8.2).

## Reproducing the analyses

Requires Python 3.11 or newer.

```bash
git clone https://github.com/DrDMo/quiet-erosions-book.git
cd quiet-erosions-book
python -m venv .venv
# Windows:  .venv\Scripts\activate    macOS/Linux:  source .venv/bin/activate
pip install -e ".[analysis]"
```

Then run the scripts in order from the repository root:

| Script | Chapter | Writes |
|---|---|---|
| `01_build_panel.py` | all | Country-year panel, 1990–2023 (V-Dem v14 and ERT v16) |
| `02_ch4_quiet_erosion.py` | 4 | Quiet-erosion spells, incidence, the 81-combination sensitivity grid with the original rule (`ch4_sensitivity.csv`), Chapter 4 figures |
| `03_build_individual.py` | 5 | Respondent file with the confidence and essential-characteristics items |
| `04_link_agp.py` | 6–8 | AGP linked to the panel; link and hazard coverage tables |
| `05_ch5_targeting.py` | 5 | Institutional targeting (QE1), Figures 5.1–5.3 |
| `06_pathways.py` | 6–8 | Change and hazard models by pathway, Figures 6.1, 7.1, 8.1 |
| `07_sequence.py` | 8 | Order of pathway onsets (QE4) |
| `07b_vertical_components.py` | 8 | Components of the vertical-first spells |
| `08_conversion.py` | 9 | Mediation and moderation (QE5) |
| `08b_high_demand.py` | 9 | High-demand at-risk country-waves (Section 9.6) |
| `09_escalation_reversal.py` | 10 | Escalation (QE6) and reversal (QE7) |
| `10_robustness.py` | 11 | Robustness checks, Figure 11.1 |

`scripts/README.md` describes the data steps in more detail. The quiet-erosion rule, including Amendment 1, is documented in the docstring of `src/quiet_erosions/erosion.py`.

## License

Code is released under the MIT License (see [LICENSE](LICENSE)). The book's text and figures are not covered by this license; all rights to the book are reserved by the author. Data remain subject to the terms of their originating archives.

## Citation

Molnar, D. R. (2026). *Quiet Erosions: Authoritarian Demand and the Institutional Pathways of Democratic Backsliding*. Shoreline Publishing Group. See also [CITATION.cff](CITATION.cff).
