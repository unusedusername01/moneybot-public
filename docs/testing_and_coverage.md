# Testing and Coverage Guide

MoneyBot ships with a pytest-based test stack tailored for GitHub Actions (`ubuntu-latest`) and flexible enough for local runs. This guide explains the structure, markers, coverage expectations, and adaptive skipping strategy.

## Test Layout

```
tests/
  unit/
  integration/
  e2e/
  conftest.py
pytest.ini
```

### Unit Tests

- Cover pure functions and configuration helpers.
- Avoid network or filesystem writes beyond temporary directories.
- Run quickly (`pytest -m "unit"`).

### Integration Tests

- Exercise FastAPI endpoints via `TestClient`.
- Interact with local portfolio fixtures under `data/portfolios`.
- Expect deterministic responses when WebSocket is absent (e.g., start attempts return `400`).

### End-to-End Tests

- Focus on full workflow decision-making logic without invoking heavy external dependencies.
- Use `MessagePayload` objects to simulate user choices (confirm, deny, cancel, edit) and edge cases (invalid allocations).
- Stub LangGraph invocation and WebSocket push calls to keep runs deterministic and CPU-only.

## Markers

Defined in `pytest.ini`:

- `unit` – Fast, isolated tests.
- `integration` – API-level tests requiring FastAPI app imports.
- `e2e` – High-level workflow simulations.
- `gpu` – Require CUDA; automatically skipped when device unavailable.
- `external` – Depend on network/API keys; skipped when secrets absent.

Example usage:

```bash
pytest -m "unit"
pytest -m "integration"
pytest -m "e2e and not external"
pytest -m "not gpu"
```

## Adaptive Skipping

`tests/conftest.py` provides helpers:

- `has_gpu()` – Detects CUDA availability via `torch.cuda.is_available()`; GPU-marked tests call `pytest.skip` when false.
- `has_env(key)` – Utility to check for required secrets (`NEWSDATA_API_KEY`, etc.).
- Fixtures automatically clear `PERSISTENT_USER_STATE` between tests to avoid cross-test contamination.

When writing new tests:

1. Check for required env variables. Skip gracefully if missing with descriptive message.
2. Prefer mocking external API clients rather than hitting the network.
3. Keep timeouts low; GitHub runners have strict time budgets.

## Coverage

- CI workflow (`.github/workflows/ci.yml`) runs `pytest -m "unit" --cov=src` and then executes integration + e2e suites.
- Coverage report generated as `coverage.xml` and uploaded as artifact.
- Target thresholds (informal):
  - Unit layer ≥ 60% coverage of core orchestration helpers.
  - For regression-critical modules (config/loader, custom_types), aim for higher coverage.

You can explore coverage locally:

```bash
pytest --cov=src --cov-report=xml --cov-report=term
coverage html  # optional, requires coverage.py installed
```

## Adding New Tests

1. Pick appropriate directory (`unit`, `integration`, `e2e`).
2. Apply markers with `@pytest.mark.<marker>`.
3. Use fixtures from `tests/conftest.py` to get a configured FastAPI client or runtime module.
4. For asynchronous code, use `pytest.mark.asyncio` or rely on `TestClient` synchronous wrappers.
5. Update docs if you introduce new flows or markers.

## CI Interaction

- GitHub Actions step “Run unit tests with coverage” collects coverage.
- Integration and e2e steps reuse dependencies installed earlier.
- Failing tests produce artifacts (`pytest-logs`) for postmortem analysis.
- Secrets restoration steps hydrate `.env.ci.test` and `config/ci.test.yml` when absent.

## Troubleshooting

- **Import errors**: Ensure `MONEYBOT_PRESET` is set or rely on fixtures to import app module correctly.
- **Missing `.env`**: CI auto-creates `.env.ci.test` when secrets provided; locally ensure `.env` exists.
- **WebSocket errors**: End-to-end tests stub `push_message_to_client` to keep flows synchronous; verify stubs are in place when writing new tests.
- **Slow tests**: Reevaluate external dependencies and consider adding skip conditions or mocks.

## Related Docs

- [CI Workflow](../.github/workflows/ci.yml) – Implementation details for GitHub Actions.
- [Deployment Guide](deployment_guide.md) – Continuous deployment pipeline.
- [Architecture Deep Dive](architecture.md) – Context for LangGraph state transitions.

Well-curated tests improve confidence in MoneyBot’s sophisticated orchestration logic. Keep this guide updated as the suite evolves.
