import importlib
import os
import sys
from pathlib import Path

# Allow `src` imports during pytest collection.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pytest
from fastapi.testclient import TestClient


def has_gpu() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


def has_env(key: str) -> bool:
    return bool(os.getenv(key))


@pytest.fixture()
def app_module(monkeypatch: pytest.MonkeyPatch):
    # Use the CI test preset but force providers that don't need external API keys
    monkeypatch.setenv("MONEYBOT_PRESET", "ci.test")
    # Env override system: MONEYBOT__section__key
    monkeypatch.setenv("MONEYBOT__models__llm__provider", "lm_studio")
    monkeypatch.setenv("MONEYBOT__models__embeddings__provider", "lm_studio")
    monkeypatch.setenv("MONEYBOT__models__llm__tool_agent_model", "qwen/qwen3-4b-2507")
    monkeypatch.setenv("MONEYBOT__models__llm__judge_model", "qwen/qwen3-8b")
    monkeypatch.setenv("MONEYBOT__models__embeddings__model", "text-embedding-nomic-embed-text-v1.5")
    module = importlib.import_module("src.langgraph_workflow.app")
    return module


@pytest.fixture()
def fastapi_app(app_module):
    return app_module.webapp


@pytest.fixture
def client(fastapi_app) -> TestClient:
    return TestClient(fastapi_app)


@pytest.fixture(autouse=True)
def reset_persistent_state(app_module):
    app_module.PERSISTENT_USER_STATE.clear()
    yield
    app_module.PERSISTENT_USER_STATE.clear()


@pytest.fixture
def tmp_data_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    base = tmp_path_factory.mktemp("data")
    for name in ("tickers", "portfolios", "market_databases", "sector_databases", "gkg_databases"):
        (base / name).mkdir(parents=True, exist_ok=True)
    return base


@pytest.fixture
def isolate_data_paths(tmp_data_dir: Path, monkeypatch: pytest.MonkeyPatch):
    """Redirect data directories to a temporary sandbox for file-heavy tests."""
    from src.data_pipeline import constants as c
    sandbox = tmp_data_dir

    monkeypatch.setattr(c, "DATA_PATH", sandbox)
    monkeypatch.setattr(c, "PORTFOLIOS_PATH", sandbox / "portfolios")
    monkeypatch.setattr(c, "TICKERS_PATH", sandbox / "tickers")
    monkeypatch.setattr(c, "SECTOR_DB_PATH", sandbox / "sector_databases")
    monkeypatch.setattr(c, "MARKET_DB_PATH", sandbox / "market_databases")
    monkeypatch.setattr(c, "GKG_DB_PATH", sandbox / "gkg_databases")

    # Update helper call sites imported elsewhere
    monkeypatch.setattr(c, "PORTFOLIO_PATH", lambda name: c.PORTFOLIOS_PATH / f"{name}.json")
    monkeypatch.setattr(c, "TICKER_PATH", lambda t: c.TICKERS_PATH / t)
    monkeypatch.setattr(c, "PREDICTIONS_PATH", lambda t: c.TICKER_PATH(t) / "predictions")

    # Sync modules that imported constants earlier
    targets = [
        "src.langgraph_workflow.app",
        "src.langgraph_workflow.utils",
        "src.data_pipeline.data_fetcher",
        "src.data_pipeline.data_collector",
    ]

    redirected_attrs = (
        ("DATA_PATH", c.DATA_PATH),
        ("PORTFOLIOS_PATH", c.PORTFOLIOS_PATH),
        ("PORTFOLIO_PATH", c.PORTFOLIO_PATH),
        ("TICKERS_PATH", c.TICKERS_PATH),
        ("TICKER_PATH", c.TICKER_PATH),
        ("PREDICTIONS_PATH", c.PREDICTIONS_PATH),
        ("SECTOR_DB_PATH", c.SECTOR_DB_PATH),
        ("MARKET_DB_PATH", c.MARKET_DB_PATH),
        ("GKG_DB_PATH", c.GKG_DB_PATH),
    )

    for target in targets:
        module = importlib.import_module(target)
        for attr, value in redirected_attrs:
            if hasattr(module, attr):
                monkeypatch.setattr(module, attr, value, raising=False)

    return sandbox


@pytest.fixture
def stub_sentence_transformer(monkeypatch: pytest.MonkeyPatch):
    class DummyModel:
        def __init__(self, *args, **kwargs):
            pass

        def encode(self, texts, convert_to_tensor=False, batch_size: int = 32, **kwargs):
            if isinstance(texts, str):
                texts = [texts]
            # Deterministic numeric signature for assertions
            return [[float(index + 1)] for index, _ in enumerate(texts)]

    dummy_factory = lambda *args, **kwargs: DummyModel()

    monkeypatch.setattr("src.data_pipeline.data_fetcher.SentenceTransformer", dummy_factory)
    monkeypatch.setattr("sentence_transformers.SentenceTransformer", dummy_factory)

    return DummyModel()
