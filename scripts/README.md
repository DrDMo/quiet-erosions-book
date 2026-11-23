# Analysis scripts

Run in order from the repository root, with `QE_DATA_ROOT` pointing at a local
folder that holds the raw data (see the main README). Raw data never enters
this repository; scripts write intermediate files to `$QE_DATA_ROOT/processed`
and results to `results/tables/` and `figures/`.

| Script | Chapter | Does |
|---|---|---|
| `01_build_panel.py` | all | Builds the 1990–2023 country-year panel from V-Dem v14 and ERT v16 |
| `02_ch4_quiet_erosion.py` | 4 | Applies the quiet-erosion rule (`src/quiet_erosions/erosion.py`), writes spells, incidence, sensitivity grid, and Chapter 4 figures |

The quiet-erosion rule, including Amendment 1 (made before any link to
authoritarian governance preference was examined), is documented in the
docstring of `src/quiet_erosions/erosion.py`.

## 03_build_individual.py

Rebuilds the Volume 1 respondent file (WVS and EVS trend files, waves 3-7)
under the same rules and adds the confidence-in-institutions items (E069_*)
and the essential-characteristics-of-democracy items (E224-E229, E233) used
in Chapter 5. Respondent and valid-AGP counts match Volume 1 exactly.

## 04_link_agp.py

Links country-wave AGP to the V-Dem/ERT panel and the Chapter 4 onsets:
post-wave index changes and onsets at horizons of 2, 4 and 6 years, and a
country-year hazard frame carrying the latest prior wave forward (at most
8 years). Adds ten EVS-only countries missing from the Volume 1 crosswalk.
