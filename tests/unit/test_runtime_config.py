# Copyright 2025 unusedusername01
# SPDX-License-Identifier: Apache-2.0

import pytest


@pytest.mark.unit
def test_apply_runtime_config_updates_cors(app_module):
    config_override = {
        "server": {
            "host": "0.0.0.0",
            "port": 1234,
            "cors": {"allow_origins": ["http://example.com"]},
        },
        "workflow": {"top_k_investments": 3, "max_reruns": 1},
        "runtime": {"device": "cpu"},
        "models": {
            "llm": {"provider": app_module.LLM_PROVIDER},
            "embeddings": {"provider": app_module.EMBEDDINGS_PROVIDER},
        },
    }

    try:
        app_module._apply_runtime_config(config_override)
        assert app_module.SERVER_PORT == 1234
        assert "http://example.com" in app_module.ALLOW_ORIGINS
    finally:
        app_module._apply_runtime_config(app_module.CONFIG)


@pytest.mark.unit
def test_resolve_device_gpu_fallback(monkeypatch, app_module):
    monkeypatch.setattr(app_module, "cuda_available", lambda: False)
    assert app_module._resolve_device("cuda") == "cpu"


@pytest.mark.unit
def test_origin_from_url_extracts_origin(app_module):
    origin = app_module._origin_from_url("https://example.com/path?query=1")
    assert origin == "https://example.com"
