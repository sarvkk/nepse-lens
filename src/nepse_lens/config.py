"""Paths and runtime settings.

Everything is overridable by environment variable so the same code runs in
local dev, CI, and Docker without edits.

Path resolution has to survive two very different installs:

* editable/source install — ``__file__`` lives under ``<repo>/src/nepse_lens``
* wheel install (Docker) — ``__file__`` lives under ``site-packages``

Deriving the data directory from ``__file__`` alone breaks the second case, so
we only trust it when a ``pyproject.toml`` sits two levels up. Otherwise we fall
back to the working directory, which is where the container is launched.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


def _project_root() -> Path:
    override = os.getenv("NEPSE_PROJECT_ROOT")
    if override:
        return Path(override)
    candidate = Path(__file__).resolve().parents[2]
    if (candidate / "pyproject.toml").exists():
        return candidate
    return Path.cwd()


REPO_ROOT = _project_root()

# Load `.env` before anything reads the environment. Real environment variables
# win over the file, so CI and Docker keep working without a `.env`.
load_dotenv(REPO_ROOT / ".env")

DATA_DIR = Path(os.getenv("NEPSE_DATA_DIR", REPO_ROOT / "data"))
FIXTURE_DIR = DATA_DIR / "fixtures"
RAW_DIR = DATA_DIR / "raw"
REPORTS_DIR = Path(os.getenv("NEPSE_REPORTS_DIR", REPO_ROOT / "reports"))
DB_PATH = Path(os.getenv("NEPSE_DB", DATA_DIR / "nepse.duckdb"))
if os.getenv("VERCEL") == "1" and "NEPSE_DB" not in os.environ:
    DB_PATH = DATA_DIR / "nepse-demo.duckdb"


@dataclass(frozen=True)
class Settings:
    base_url: str = os.getenv("NEPSE_BASE_URL", "https://www.nepalstock.com.np")
    token: str | None = os.getenv("NEPSE_TOKEN") or None
    request_timeout: float = float(os.getenv("NEPSE_TIMEOUT", "20"))
    rate_limit_seconds: float = float(os.getenv("NEPSE_RATE_LIMIT", "0.6"))
    user_agent: str = (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/131.0 Safari/537.36"
    )


SETTINGS = Settings()


def ensure_dirs() -> None:
    for path in (DATA_DIR, FIXTURE_DIR, RAW_DIR, REPORTS_DIR):
        path.mkdir(parents=True, exist_ok=True)
