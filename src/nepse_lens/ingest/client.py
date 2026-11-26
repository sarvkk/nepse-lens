"""Live NEPSE client.

NEPSE's Angular app authenticates every API call with an ``Authorization:
Salter <token>`` header. The token is minted by the site and expires quickly, so
this client does not try to guess a login flow. Instead:

* set ``NEPSE_TOKEN`` to a token copied from the browser's network tab, or
* use the bundled fixture snapshot (``--source fixture``) for offline work.

Every successful response is archived under ``data/raw/<endpoint>/`` before
normalisation, so a bad parse never destroys the original payload.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from typing import Any

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from ..config import RAW_DIR, SETTINGS, Settings


class NepseError(RuntimeError):
    """Base error for ingestion failures."""


class NepseAuthError(NepseError):
    """Raised when NEPSE rejects the request for lack of a valid token."""


def _pick(row: dict[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in row and row[key] not in (None, ""):
            return row[key]
    return default


class NepseClient:
    def __init__(self, settings: Settings = SETTINGS, session: requests.Session | None = None):
        self.settings = settings
        self.session = session or self._build_session()

    def _build_session(self) -> requests.Session:
        session = requests.Session()
        retry = Retry(
            total=3,
            backoff_factor=0.6,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=("GET", "POST"),
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        session.headers.update(
            {
                "User-Agent": self.settings.user_agent,
                "Accept": "application/json, text/plain, */*",
                "Referer": f"{self.settings.base_url}/",
            }
        )
        return session

    def _headers(self) -> dict[str, str]:
        if not self.settings.token:
            raise NepseAuthError(
                "NEPSE requires a 'Salter' bearer token for its JSON API. "
                "Open nepalstock.com.np, copy the Authorization header from a "
                "market-data request in DevTools, and set NEPSE_TOKEN. "
                "For offline work use --source fixture."
            )
        return {"Authorization": f"Salter {self.settings.token}"}

    def _archive(self, name: str, payload: Any) -> None:
        folder = RAW_DIR / name
        folder.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        (folder / f"{stamp}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2))

    def _get(self, path: str) -> Any:
        url = f"{self.settings.base_url.rstrip('/')}/{path.lstrip('/')}"
        response = self.session.get(url, headers=self._headers(), timeout=self.settings.request_timeout)
        if response.status_code in (401, 403):
            raise NepseAuthError(f"NEPSE rejected the token for {path} (HTTP {response.status_code}).")
        response.raise_for_status()
        payload = response.json()
        self._archive(path.strip("/").replace("/", "_"), payload)
        time.sleep(self.settings.rate_limit_seconds)
        return payload

    # -- endpoints ---------------------------------------------------------
    def market_summary(self) -> dict[str, Any]:
        return self._get("api/nots/market-summary/")

    def today_share_price(self) -> list[dict[str, Any]]:
        payload = self._get("api/nots/todayshareprice/")
        return payload if isinstance(payload, list) else payload.get("content", [])

    def security_list(self) -> list[dict[str, Any]]:
        payload = self._get("api/nots/security/all")
        return payload if isinstance(payload, list) else payload.get("content", [])

    # -- normalisation -----------------------------------------------------
    def fetch_daily_snapshot(self, trade_date: str | None = None) -> pd.DataFrame:
        """Return one row per symbol in the same shape as the fixture data."""
        rows = self.today_share_price()
        if not rows:
            raise NepseError("NEPSE returned no rows for today's share price.")

        trade_date = trade_date or datetime.now(timezone.utc).date().isoformat()
        parsed: list[dict[str, Any]] = []
        for row in rows:
            symbol = _pick(row, "symbol", "symbolName", "securitySymbol")
            close = _pick(row, "closePrice", "lastTradedPrice", "closingPrice")
            if not symbol or close in (None, 0):
                continue
            volume = _pick(row, "totalTradedQuantity", "tradedQuantity", "volume", default=0)
            turnover = _pick(row, "totalTradedValue", "tradedValue", "turnover", default=0)
            parsed.append(
                {
                    "symbol": str(symbol).upper(),
                    "date": trade_date,
                    "open": _pick(row, "openPrice", "openingPrice", default=close),
                    "high": _pick(row, "highPrice", "dayHigh", default=close),
                    "low": _pick(row, "lowPrice", "dayLow", default=close),
                    "close": close,
                    "volume": int(float(volume or 0)),
                    "turnover": float(turnover or 0),
                    "sector": _pick(row, "sectorName", "sector", default="Unknown"),
                }
            )

        frame = pd.DataFrame(parsed)
        if frame.empty:
            raise NepseError("Could not normalise any rows; the NEPSE schema may have changed.")
        frame["date"] = pd.to_datetime(frame["date"])
        for col in ("open", "high", "low", "close", "turnover"):
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
        frame["volume"] = pd.to_numeric(frame["volume"], errors="coerce").fillna(0).astype("int64")
        return frame.dropna(subset=["close"]).reset_index(drop=True)
