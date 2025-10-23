# MoneyBot Architecture Deep Dive

MoneyBot orchestrates data ingestion, orchestration, and presentation layers around a FastAPI backend and a planned web application. This document offers a systems-level view that complements the inline docstrings and configuration presets.

## System Overview

1. **Clients** – Terminal CLI, planned Vite/React web app, and any HTTP/WebSocket integrations.
2. **Application Server** – `src/langgraph_workflow/app.py` exposes REST + WebSocket APIs, loads presets from `config/`, and delegates orchestration to LangGraph state machines.
3. **Workflow Engine** – LangGraph nodes coordinate data fetching, scoring, synthesis, and user interaction loops.
4. **Data Layer** – Local JSON datasets under `data/` combined with real-time fetchers for news, fundamentals, and predictions.
5. **Model Providers** – Configurable LLM and embedding providers via `LangChainLLMProvider` with GPU fallbacks.
6. **Persistence** – Portfolio definitions (`data/portfolios/*.json`), cached market/news data (`data/market_databases/`, `data/gkg_databases/`, etc.), and generated vector stores.
7. **Monitoring & Logs** – Primarily standard output; planned integration hooks for systemd/NSSM and GitHub Actions artifacts.

The runtime relies heavily on environment-aware presets. Presets keep infrastructure concerns (ports, devices, providers) outside of code, enabling reproducible behavior in CI, staging, and production.

## Runtime Components

### FastAPI Application (`webapp`)

- Initializes after calling `_apply_runtime_config` to project values from the active preset.
- Exposes REST endpoints for portfolio CRUD, workflow execution (`/analysis/portfolio/...`), and utility endpoints.
- Hosts a WebSocket channel per portfolio to stream workflow progress and gather confirmations or edits.
- Applies CORS configuration derived from presets and runtime overridable CLI flags.

### LangGraph Workflow (`app`)

- Nodes: `init_agent_state`, `run_analysis`, `call_synthesizer`, `check_webapp`, `call_logger`, `call_cancel`.
- Conditional edges track user choices: confirm, rerun, edit, cancel.
- State is stored in `PERSISTENT_USER_STATE` (in-memory) keyed by portfolio id for multi-client flows.
- Durable logging and persistence are planned but currently delegated to JSON dumps and console logs.

### Data Intake

- **Market Fundamentals** – via `DataFetcher` calling Yahoo Finance (requires network access during real runs).
- **News Pipelines** – dual approach: `NewsFetcher` for API-backed news and `DataCollector` for cached local sources.
- **Prediction Engine** – `PredictionManager` orchestrates CPU/GPU prediction modes tuned by preset `runtime.device` and `workflow.prediction_strength`.
- **RAG Manager** – `RAGManager` loads embedding models from provider settings and seeds search indexes used in synthesis.

### Config Layers

1. **Base preset** (`config/base.yml`) – local defaults for developers.
2. **CI preset** (`config/ci.test.yml`) – light-weight CPU oriented settings so GitHub runners can operate within time limits.
3. **Deploy preset** (`config/ci.deploy.yml`) – GPU-centric options tuned for self-hosted runners and production scale.
4. **Ad hoc overrides** – CLI arguments (`--preset`, `--port`, `--host`) and environment variable `MONEYBOT_PRESET`.

Configuration is merged via `src/config/loader.py`, giving precedence to CLI > preset > base defaults. `_apply_runtime_config` then translates key paths into module-level globals (device, provider, CORS, port, etc.).

## Data Flow Narrative

1. **Session bootstrap** – A client selects a portfolio id. The backend loads `portfolio_data` from disk and synthesizes holdings, risk tolerance, and prediction strength.
2. **Data fetching** – For each ticker, the workflow fetches fundamentals, price histories, and optionally queries NewsData/GKG sources based on preset allowances.
3. **Feature assembly** – Results are grouped via criteria (sector/market/score) and ranked using heuristics and prediction scores.
4. **LLM synthesis** – LangChain LLMs and tools aggregate insights. The `judge` model scores candidate allocations before the tool agent renders final proposals.
5. **User decision loop** – Through WebSocket messages (`MessagePayload`), the client can confirm, edit allocations, request reruns, or cancel. Choices mutate the LangGraph state and may re-enter the workflow.
6. **Logging and finish** – Confirmed sessions trigger `call_logger` to produce artifacts (currently textual). Cancellations or reruns update the finite state machine accordingly.

## Module Responsibilities

| Module | Purpose |
| --- | --- |
| `src/langgraph_workflow/app.py` | Main entry point, FastAPI definitions, LangGraph orchestration. |
| `src/langgraph_workflow/custom_types.py` | Pydantic models, message schemas, and request/response DTOs. |
| `src/langgraph_workflow/utils.py` | Helper utilities for grouping tickers, currency conversion, share allocation, etc. |
| `src/langgraph_workflow/rag_manager.py` | Semantic retrieval helpers wrapping embedding store interactions. |
| `src/data_pipeline/data_fetcher.py` | Market data + news API fetchers. |
| `src/data_pipeline/data_collector.py` | Local cache readers for news datasets under `data/`. |
| `src/data_pipeline/prediction_model.py` | ML models and prediction pipelines keyed off `prediction_strength`. |
| `src/data_pipeline/llm_provider.py` | Provider registry for LangChain models and embeddings. |
| `src/config/loader.py` | Preset discovery, merge logic, CLI + env overrides. |
| `scripts/export_frontend_env.py` | Aligns backend preset values with frontend `.env` files. |

## Preset-driven Behavior

- **Device Selection** – `_resolve_device` inspects CUDA availability and preset hints. Deploy preset prefers GPU but safely falls back to CPU.
- **CORS Rules** – Derived from `server.cors.allow_origins`. Frontend base URLs automatically join the allow list to keep the Vite dev server in sync.
- **Workflow Tuning** – `top_k_investments`, `max_reruns`, news batching, and concurrency all originate in `workflow` + `runtime` sections.
- **Provider Swap** – Changing `models.llm.provider` or `models.embeddings.provider` selects different tool and judge models without modifying code.

## Observability Touch Points

- **Logging** – Python `logging` plus strategic `print` statements reflect major workflow events. CI captures stdout via pytest.
- **Metrics Hooks** – Not yet implemented; placeholders exist for asynchronous collectors.
- **Artifacts** – GitHub Actions CI uploads coverage XML and pytest caches. Deployment workflow leaves optional service logs (`moneybot.out`, `moneybot.err`).

## Typical Run Walkthrough

1. Developer launches FastAPI: `python -m src.langgraph_workflow.app --preset base --host 127.0.0.1 --port 8000`.
2. Web client connects to `ws://127.0.0.1:8000/ws/portfolio/<portfolio_id>`.
3. Client calls `/analysis/portfolio/<portfolio_id>/start`. Backend loads portfolio, fetches data, and streams updates via WebSocket.
4. When awaiting choice, user responds via `/analysis/portfolio/<portfolio_id>/respond` with `MessagePayload` JSON representing confirm, edit, deny, or cancel actions.
5. Confirm triggers final logging and ends the session; deny collects feedback and reruns; edit requests allocation adjustments; cancel stops the workflow.

Screenshots/logs for this flow are referenced in the README “Typical Run Narrative” section.

## Extensibility Guidelines

- **New Providers** – Extend `LangChainLLMProvider.PROVIDER_CONFIG`, ensure presets reference valid keys, and update docs.
- **Additional Endpoints** – Mirror DTOs across backend (`custom_types`) and frontend (`src/webapp/src/types/api.ts`) to keep both sides type safe.
- **Persistent State** – Swap `PERSISTENT_USER_STATE` with Redis or database when multi-instance concurrency is needed. Adjust documentation in `docs/deployment_guide.md` accordingly.
- **Observability** – Insert hooks before and after LangGraph node execution to emit metrics; describe new signals in `docs/testing_and_coverage.md`.

## Related Documentation

- [Backend API Reference](backend_api_reference.md)
- [Developer Guide](developer_guide.md)
- [Testing and Coverage](testing_and_coverage.md)
- [Deployment Guide](deployment_guide.md)
- [Webapp Guide](webapp_guide.md)

Keep this document updated whenever module boundaries or workflow nodes change. Long-form explanations here help new contributors understand why MoneyBot behaves the way it does, not just how to run it.
