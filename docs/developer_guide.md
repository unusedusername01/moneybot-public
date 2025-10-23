# Developer Guide

This guide walks through day-to-day tasks for MoneyBot contributors: environment setup, presets, data sources, and GPU/CPU fallbacks. It complements the README quick start and the architecture overview.

## Environments & Presets

MoneyBot leans on layered configuration files under `config/`:

| File | Purpose |
| --- | --- |
| `config/base.yml` | Default local developer experience. |
| `config/ci.test.yml` | Optimized for GitHub-hosted runners (CPU only). |
| `config/ci.deploy.yml` | GPU-first profile for deployment/self-hosted runners. |

Load presets by name:

```bash
python -m src.langgraph_workflow.app --preset base
python -m src.langgraph_workflow.app --preset ci.test --host 0.0.0.0 --port 9000
```

You may also set `MONEYBOT_PRESET=base` and omit the CLI flag.

### Runtime Overrides

- CLI flags supersede preset values for host/port.
- `.env` variables loaded via `dotenv` supply API keys (`NEWSDATA_API_KEY`, `TOGETHER_API_KEY`, etc.).
- Presets can reference values that depend on secrets; during CI/deploy workflows encoded secrets hydrate `.env.ci.*` and `config/ci.*.yml` automatically when missing from the repo.

## Local Development (Windows & Linux)

1. **Install Python 3.11** (align with GitHub Actions).
2. **Create a virtualenv**:
   - Windows (PowerShell): `python -m venv .venv; .\.venv\Scripts\Activate.ps1`
   - Linux/macOS: `python -m venv .venv && source .venv/bin/activate`
3. **Install dependencies**: `pip install -r requirements.txt`
4. **Copy environment file**: `cp env.example .env` (or author your own `.env`).
5. **Run backend**: `python -m src.langgraph_workflow.app --preset base`
6. **Launch webapp** (once the frontend is ready):
   ```bash
   cd src/webapp
   npm install
   npm run dev
   ```
7. **Sync frontend env**: `python scripts/export_frontend_env.py --preset base`

## Data Directories

- `data/portfolios/` – JSON files with portfolio metadata. Tests use these fixtures; add new ones cautiously.
- `data/gkg_databases/` and `data/market_databases/` – Cached news datasets. Heavy ingest jobs reside in `src/data_pipeline/`.
- `data/models/` – Place local model checkpoints if running air-gapped.

Tips:

- Avoid editing shared datasets without updating docs and notifying the team; tests assume certain fields exist in sample portfolios.
- Use git LFS if you introduce large artifacts.

## Device Selection & GPU Use

`_resolve_device` picks between CPU and CUDA depending on availability and preset hints.

- In CI (Ubuntu `ubuntu-latest`), `config/ci.test.yml` forces `runtime.device: cpu`.
- In deployments, `config/ci.deploy.yml` prefers GPU; the workflow checks `nvidia-smi` on Linux or CUDA presence on Windows before installing torch + CUDA wheels.
- You can override on CLI: `--device cpu` (extend CLI parser in app.py if explicit flag is needed).

## Avoiding Rate Limits

- Wrap API calls with caching layers—`DataCollector` already favors local caches.
- Use `prediction_strength: weak` in development to keep the ML pipeline lightweight.
- Skip or xfail tests that require external network access when keys are missing; see `docs/testing_and_coverage.md` for patterns.

## Adding Providers & Models

1. Extend `LangChainLLMProvider.PROVIDER_CONFIG` with provider metadata.
2. Update presets to reference new providers/models.
3. Document usage in this guide and the architecture overview.
4. Add smoke tests or mocks under `tests/unit`.

## Updating Configuration Schema

When adding new configuration keys:

1. Update `src/config/loader.py` to read the new paths.
2. Apply defaults in `_apply_runtime_config`.
3. Document the key in this guide and relevant docs.
4. Provide examples in README and `scripts/export_frontend_env.py` if necessary.

## Working with WebSockets

- Client must open `ws://<host>:<port>/ws/portfolio/<portfolio_id>` before calling `/analysis/portfolio/{portfolio_id}/start`.
- Messages follow `MessagePayload` schema from `custom_types.py`.
- When building new features, extend WebSocket messages carefully; add tests in `tests/e2e/test_analysis_flow.py` and update `docs/backend_api_reference.md`.

## Debugging Tips

- Use `uvicorn` logging verbosity: `uvicorn src.langgraph_workflow.app:webapp --reload --log-level debug` (for quick iteration; CLI invocation handles preset loading then hands to UVicorn).
- Set `LOG_LEVEL=DEBUG` in `.env` and adjust `logging` configuration to surface more detail.
- Inspect `PERSISTENT_USER_STATE` via Python shell to diagnose stuck sessions.

## Scripts & Automation

- `scripts/export_frontend_env.py` – Aligns backend preset base URL with frontend `.env` files.
- Add your own helper scripts under `scripts/` and reference them here.

## Contribution Workflow

1. Branch from `main` with meaningful name.
2. Implement features/tests. Keep README + docs synchronized.
3. Run `pytest --cov` locally (skip or mark tests that require secrets when absent).
4. Commit with conventional summary.
5. Push and open a pull request; GitHub Actions CI will run on `ubuntu-latest`.
6. Address code review feedback, update docs/tests accordingly.

## Checklist for New Developers

- [ ] Install required Python version and dependencies.
- [ ] Provision API keys where available; place them in `.env`.
- [ ] Review `docs/architecture.md` to understand the workflow graph.
- [ ] Run `pytest -m "unit"` to ensure baseline health.
- [ ] Experiment with sample portfolio via CLI or webapp once WebSocket integration is ready.

## Related Documents

- [Architecture Deep Dive](architecture.md)
- [Backend API Reference](backend_api_reference.md)
- [Testing & Coverage](testing_and_coverage.md)
- [Deployment Guide](deployment_guide.md)
- [Webapp Guide](webapp_guide.md)

Keep this guide updated with any new tooling, workflow tweaks, or onboarding gotchas.
