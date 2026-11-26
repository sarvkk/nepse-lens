from __future__ import annotations

import importlib

from nepse_lens import config


def test_project_root_contains_pyproject():
    assert (config.REPO_ROOT / "pyproject.toml").exists()
    assert config.DB_PATH.parent == config.DATA_DIR


def test_vercel_uses_bundled_database(monkeypatch):
    monkeypatch.setenv("VERCEL", "1")
    monkeypatch.delenv("NEPSE_DB", raising=False)
    reloaded = importlib.reload(config)
    try:
        assert reloaded.DB_PATH == reloaded.DATA_DIR / "nepse-demo.duckdb"
        assert reloaded.DB_PATH.is_file()
    finally:
        monkeypatch.delenv("VERCEL")
        importlib.reload(config)


def test_data_dir_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("NEPSE_DATA_DIR", str(tmp_path))
    reloaded = importlib.reload(config)
    try:
        assert reloaded.DATA_DIR == tmp_path
        assert reloaded.FIXTURE_DIR == tmp_path / "fixtures"
    finally:
        monkeypatch.delenv("NEPSE_DATA_DIR")
        importlib.reload(config)


def test_dotenv_is_loaded_without_overriding_real_env(monkeypatch, tmp_path):
    (tmp_path / "pyproject.toml").write_text("", encoding="utf-8")
    (tmp_path / ".env").write_text("NEPSE_TOKEN=from-dotenv\nNEPSE_TIMEOUT=99\n", encoding="utf-8")
    monkeypatch.setenv("NEPSE_PROJECT_ROOT", str(tmp_path))
    monkeypatch.delenv("NEPSE_TOKEN", raising=False)
    monkeypatch.delenv("NEPSE_TIMEOUT", raising=False)
    monkeypatch.setenv("NEPSE_TIMEOUT", "7")
    reloaded = importlib.reload(config)
    try:
        assert reloaded.REPO_ROOT == tmp_path
        assert reloaded.SETTINGS.token == "from-dotenv"
        assert reloaded.SETTINGS.request_timeout == 7.0  # env beats .env
    finally:
        monkeypatch.delenv("NEPSE_PROJECT_ROOT")
        monkeypatch.delenv("NEPSE_TIMEOUT")
        monkeypatch.delenv("NEPSE_TOKEN", raising=False)
        importlib.reload(config)
