"""Paths and runtime settings.

Everything is overridable by environment variable so the same code runs in
local dev, CI, and Docker without edits.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.getenv("NEPSE_DATA_DIR", REPO_ROOT / "data"))
FIXTURE_DIR = DATA_DIR / "fixtures"
RAW_DIR = DATA_DIR / "raw"
REPORTS_DIR = Path(os.getenv("NEPSE_REPORTS_DIR", REPO_ROOT / "reports"))
DB_PATH = Path(os.getenv("NEPSE_DB", DATA_DIR / "nepse.duckdb"))


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
