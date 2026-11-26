from __future__ import annotations

import pytest

from nepse_lens.config import Settings
from nepse_lens.ingest.client import NepseAuthError, NepseClient, NepseError

ROWS = [
    {
        "symbol": "nabil",
        "closePrice": 720.5,
        "openPrice": 700.0,
        "highPrice": 730.0,
        "lowPrice": 695.0,
        "totalTradedQuantity": 1000,
        "totalTradedValue": 720500.0,
        "sectorName": "Commercial Banks",
    },
    {"symbol": "", "closePrice": 10.0},
    {"symbol": "SCB", "closePrice": None},
]


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class FakeSession:
    def __init__(self, payload, status_code=200):
        self.payload = payload
        self.status_code = status_code
        self.headers: dict[str, str] = {}
        self.calls: list[dict] = []

    def get(self, url, headers=None, timeout=None):
        self.calls.append({"url": url, "headers": headers, "timeout": timeout})
        return FakeResponse(self.payload, self.status_code)


@pytest.fixture(autouse=True)
def _no_sleep_and_tmp_archive(monkeypatch, tmp_path):
    monkeypatch.setattr("nepse_lens.ingest.client.time.sleep", lambda *_: None)
    monkeypatch.setattr("nepse_lens.ingest.client.RAW_DIR", tmp_path)


def make_client(payload=ROWS, token="test-token", status_code=200):
    return NepseClient(
        settings=Settings(token=token),
        session=FakeSession(payload, status_code),
    )


def test_missing_token_raises_actionable_error():
    client = NepseClient(settings=Settings(token=None), session=FakeSession(ROWS))
    with pytest.raises(NepseAuthError, match="NEPSE_TOKEN"):
        client.today_share_price()


def test_unauthorized_response_raises_auth_error():
    client = make_client(status_code=401)
    with pytest.raises(NepseAuthError):
        client.today_share_price()


def test_fetch_daily_snapshot_normalises_rows():
    client = make_client()
    frame = client.fetch_daily_snapshot(trade_date="2026-11-25")
    assert len(frame) == 1
    row = frame.iloc[0]
    assert row["symbol"] == "NABIL"
    assert row["close"] == 720.5
    assert row["sector"] == "Commercial Banks"
    assert row["volume"] == 1000
    assert str(row["date"].date()) == "2026-11-25"


def test_content_wrapper_is_unwrapped():
    client = make_client(payload={"content": ROWS})
    assert len(client.today_share_price()) == 3


def test_empty_payload_raises():
    client = make_client(payload=[])
    with pytest.raises(NepseError, match="no rows"):
        client.fetch_daily_snapshot()


def test_request_uses_authorization_header_and_timeout():
    client = make_client()
    client.market_summary()
    call = client.session.calls[0]
    assert call["headers"]["Authorization"] == "Salter test-token"
    assert call["timeout"] == Settings().request_timeout
    assert call["url"].endswith("/api/nots/market-summary/")
