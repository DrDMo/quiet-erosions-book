# Quiet Erosions

Public code archive for the book **_Quiet Erosions: Authoritarian Demand and the Institutional Pathways of Democratic Backsliding_** by Darin Molnar.

## Purpose

This repository holds the code, analysis scripts, and replication material that support the empirical claims and figures in the book. It is intended as an open record so that readers, reviewers, and future researchers can:

- Inspect the data preparation steps behind every chart and statistic in the book.
- Re-run the analyses against the cited datasets.
- Extend or critique the methods on their own.

## Repository layout

```
.
├── Data/                        # Raw and intermediate datasets (most contents git-ignored)
├── notebooks/                   # Per-chapter analysis notebooks (.ipynb)
├── scripts/                     # One-shot data ingest / cleaning scripts
├── figures/                     # Rendered figure outputs (PNG/PDF/SVG)
├── src/quiet_erosions/          # Importable Python package: shared utilities
├── tests/                       # Pytest suite for shared utilities
├── pyproject.toml               # Project metadata + dependency groups
├── CITATION.cff                 # Machine-readable citation metadata
├── README.md
├── LICENSE
└── .gitignore
```

## Data

Most primary datasets used in the book are third-party and are not redistributed here. Where redistribution is permitted, files will live under `Data/`. Where it is not, the relevant scripts will document how to obtain the source data directly from its publisher.

## Setup

Requires Python 3.11 or newer.

```bash
# Clone
git clone https://github.com/DrDMo/quiet-erosions-book.git
cd quiet-erosions-book

# Create a virtual environment and install in editable mode with all extras
python -m venv .venv
# Windows:  .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate
pip install -e ".[analysis,notebook,dev]"
```

Then verify:

```bash
pytest -q
```

## Reproducing the analyses

Each chapter's analyses live in `notebooks/` (and any one-shot data preparation in `scripts/`). Reproduction instructions specific to a chapter will be documented alongside its notebook as those land.

## License

Code in this repository is released under the MIT License — see [LICENSE](LICENSE).

The book's prose, figures, and other narrative content are **not** covered by this license; all rights to the book text are reserved by the author.

## Citation

If you use code from this repository in academic work, please cite the book. A formal citation entry will be added when the book is published.

## Contact

Issues and pull requests are welcome via GitHub.
