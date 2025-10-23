# Webapp Guide

This document summarizes the planned MoneyBot web application architecture (see `src/webapp/plan.md`), how it interfaces with the backend, and how to run it locally once the frontend implementation is completed.

## Project Status

- The repository contains backend contracts and TypeScript DTO placeholders (`src/webapp/src/types/api.ts`).
- The `plan.md` file documents UI flows, data contracts, and pending tasks. Section 13 tracks backend ↔ frontend API parity.
- This guide ensures future frontend work aligns with the backend workflows described in `docs/architecture.md` and `docs/backend_api_reference.md`.

## Local Development Workflow

1. **Install dependencies**:
   ```bash
   cd src/webapp
   npm install
   ```

2. **Sync API base URL**:
   ```bash
   python ../../scripts/export_frontend_env.py --preset base
   ```
   This writes `.env.development.local` with `VITE_API_BASE_URL` derived from the active preset.

3. **Run backend**:
   ```bash
   python -m src.langgraph_workflow.app --preset base --host 127.0.0.1 --port 8000
   ```

4. **Start Vite dev server**:
   ```bash
   npm run dev
   ```

5. **Access app**: `http://127.0.0.1:5173` (default Vite port).

## API Contracts

- Keep `src/webapp/src/types/api.ts` synchronized with backend DTOs from `src/langgraph_workflow/custom_types.py`.
- Use TypeScript utility types to mirror `MessagePayload`, `PortfolioData`, and other Pydantic models.
- The frontend should treat `HttpResponse.err_code` as the canonical success indicator.

### Core Interactions

1. **Connect WebSocket** – `ws://<host>:<port>/ws/portfolio/{portfolio_id}`.
2. **Start Analysis** – `POST /analysis/portfolio/{portfolio_id}/start`.
3. **Handle Messages** – Render `state_update`, `awaiting_choice`, `awaiting_message`, and `awaiting_allocation` events.
4. **Send Responses** – `POST /analysis/portfolio/{portfolio_id}/respond` with user decisions serialized via `MessagePayload`.

### Decision Flows

- **Confirm** – Submit `ChoiceResponse(selection="confirm")`. Expect session to end after logging.
- **Edit** – Submit `ChoiceResponse(selection="edit")` followed by `AllocationResponse` payload once user adjusts allocations.
- **Deny/Rerun** – Submit deny choice, capture feedback via `MessageResponse`, then wait for rerun results.
- **Cancel** – Submit cancel choice to exit workflow immediately.

## Handling Edge Cases

- If backend signals rate limits or missing data via `state_update` prompts, surface clear UI messaging.
- Workflows may skip GPU-intensive prediction modes when runtime lacks resources; reflect this gracefully.
- Use exponential backoff when reconnecting WebSockets.

## Testing Strategy

- Frontend unit tests should mock backend responses using the documented DTO schemas.
- For integration tests, spin up the FastAPI app using the `ci.test` preset to leverage lightweight configuration.
- Coordinate with backend e2e tests to share payload fixtures.

## Deployment Considerations

- Production builds should extract the API base URL from deployment environment variables (mirroring `.env.ci.deploy`).
- Consider static hosting (e.g., Netlify, GitHub Pages) with reverse proxying to the FastAPI service or containerize combined backend/frontend.

## Updating the Plan

When backend endpoints or message schemas change:

1. Update `docs/backend_api_reference.md` and this guide.
2. Reconcile plan entries in `src/webapp/plan.md` (especially Section 13 mapping worksheet).
3. Adjust TypeScript types and frontend logic accordingly.
4. Extend tests (frontend and backend) to cover new paths.

## Related Documentation

- [Architecture Deep Dive](architecture.md)
- [Backend API Reference](backend_api_reference.md)
- [Developer Guide](developer_guide.md)
- [Testing & Coverage](testing_and_coverage.md)

This guide should evolve alongside frontend implementation. Keep it authoritative so new contributors understand how the webapp couples with MoneyBot’s backend workflows.
