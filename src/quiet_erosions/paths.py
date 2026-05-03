"""Repository path constants.

Importing from a single source keeps notebooks, scripts, and tests in
agreement about where data and figure outputs live, regardless of the
caller's working directory.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT: Path = Path(__file__).resolve().parents[2]
DATA_DIR: Path = REPO_ROOT / "Data"
FIGURES_DIR: Path = REPO_ROOT / "figures"
NOTEBOOKS_DIR: Path = REPO_ROOT / "notebooks"
SCRIPTS_DIR: Path = REPO_ROOT / "scripts"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
