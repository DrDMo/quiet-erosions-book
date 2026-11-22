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
