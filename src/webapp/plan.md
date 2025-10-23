# MoneyBot Webapp Implementation Plan
React + TypeScript (typed API) + TailwindCSS. The UI must match or exceed the terminal flow in `src/full_example.py` and strictly use the API and DTOs in `webapp/src/types/api.ts` that mirror `langgraph_workflow/app.py`.

## 0) Objectives and Scope
- Provide a clean, pleasant UX to:
  - Manage a watchlist (add/edit/remove tickers; shares may be 0 to “track only”).
  - Define investment criteria and optional remarks.
  - Run an analysis and re-run it with additional constraints/remarks.
  - Receive a budget recommendation and supporting insights.
- Use only:
  - React for UI.
  - TypeScript for typed client requests (import DTOs from `webapp/src/types/api.ts`).
  - TailwindCSS for styling.

Notes:
- Treat `webapp/src/types/api.ts` as the contract. Do not retype responses client-side.
- Endpoints are defined in `src/langgraph_workflow/app.py`. Confirm routes/methods before wiring the client.
- The plan assumes a single-page React app with componentized flows.

---

## 1) Backend Alignment (Before Coding UI)
1. Open `src/langgraph_workflow/app.py` and inventory all HTTP endpoints:
   - Confirm base URL (likely http://localhost:8000 or similar).
   - Confirm CORS is enabled for the web client (localhost:5173 or 3000). If not, add it.
   - For each endpoint, note:
     - method (GET/POST)
     - path (e.g., /api/analyze)
     - request/response Pydantic models
     - pagination/streaming (if any)
2. Open `webapp/src/types/api.ts`:
   - Map exported types to backend endpoints:
     - Request/Response pairs (e.g., AnalyzeRequest ⇆ AnalyzeResponse).
     - Auxiliary DTOs (Ticker, Criteria, BudgetRecommendation, etc.).
   - Validate enums/union types match Python side.
3. Decide hosting approach for local dev:
   - Backend: `python -m src.langgraph_workflow.app` or `python src/langgraph_workflow/app.py` (confirm).
   - Frontend: Vite or CRA dev server (assume Vite). Ensure proxy or CORS is configured.

---

## 2) Frontend Architecture and UX
High-level IA
- Route: Single-page layout with panels/tabs:
  - Dashboard (default): Compose the primary flow.
    - Left column: Watchlist editor.
    - Middle: Criteria form.
    - Right: Analysis runner and latest results.
  - History tab (optional): View and re-run prior analyses.

Key Components
- Layout
  - AppShell: Header (branding), tab navigation, footer (disclaimer).
- Watchlist
  - WatchlistEditor: Add ticker (default shares=0), edit shares, remove ticker, validation, persist locally and/or via API if provided.
- Criteria
  - CriteriaForm: Risk tolerance, horizon, sector prefs, notes/remarks, budget constraints.
- Run and Results
  - AnalysisRunner: Run button, progress/skeleton, re-run with additional remarks.
  - ResultsDashboard: Signals, allocations, budget recommendation, confidence, rationale.
  - AnalysisHistory: List of past runs (if backend provides), diff between runs.

UX Principles
- Inline validation, clear errors.
- Skeleton loaders for analysis.
- Toasts/snackbars for success/error.
- Empty states and helpful placeholders.
- Persist last-used watchlist and criteria to localStorage.

---

## 3) Tooling and Project Setup
If the webapp is already scaffolded, skip initializations that already exist.

1) TailwindCSS
- Install and init:
  - PowerShell:
    - npm i -D tailwindcss postcss autoprefixer
    - npx tailwindcss init -p
- Configure content paths (Vite typical):
  - tailwind.config.js content: ["./index.html", "./src/**/*.{ts,tsx}"]
- Add base styles in index.css:
  - @tailwind base;
  - @tailwind components;
  - @tailwind utilities;

2) Optional (recommended) libs
- React Query for data fetching and caching:
  - npm i @tanstack/react-query
  - If you prefer plain fetch + custom hooks, skip this. This plan provides both paths.

3) Env configuration
- Frontend base URL for API:
  - Use Vite variables, e.g., VITE_API_BASE_URL
  - Ensure not to expose private keys in the frontend `.env`.
- Confirm CORS on backend.

---

## 4) API Client (Type-Safe)
Create a small API layer that:
- Uses `webapp/src/types/api.ts` DTOs.
- Provides functions per endpoint.
- Handles AbortController, timeouts, and error normalization.

```ts
// filepath: c:\Users\riri9\Desktop\repos\moneybot\webapp\src\lib\apiClient.ts
import type {
  // Import the exact DTOs that exist in your generated file:
  // Example placeholders — replace with the real exports:
  AnalyzeRequest,
  AnalyzeResponse,
  WatchlistItem,
  Criteria,
} from "../types/api";

const BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 60_000);

  try {
    const resp = await fetch(`${BASE_URL}${path}`, {
      ...init,
      headers: {
        "Content-Type": "application/json",
        ...(init?.headers || {}),
      },
      signal: controller.signal,
    });
    if (!resp.ok) {
      const text = await resp.text().catch(() => "");
      throw new Error(`HTTP ${resp.status}: ${text || resp.statusText}`);
    }
    return (await resp.json()) as T;
  } finally {
    clearTimeout(timeout);
  }
}

// Replace endpoint paths with the real ones from app.py:
export const Api = {
  analyze: (payload: AnalyzeRequest) =>
    request<AnalyzeResponse>("/api/analyze", {
      method: "POST",
      body: JSON.stringify(payload),
    }),

  // If backend supports watchlist persistence:
  // getWatchlist: () => request<WatchlistItem[]>("/api/watchlist", { method: "GET" }),
  // saveWatchlist: (items: WatchlistItem[]) =>
  //   request<void>("/api/watchlist", { method: "POST", body: JSON.stringify(items) }),
};
```

---

## 5) State Management
Two strategies:
- A) React Query (recommended): query/mutation hooks per endpoint, caching and retries.
- B) Lightweight custom hooks with useState/useEffect and the Api client.

```ts
// filepath: c:\Users\riri9\Desktop\repos\moneybot\webapp\src\hooks\useAnalysis.ts
import { useState } from "react";
import type { AnalyzeRequest, AnalyzeResponse } from "../types/api";
import { Api } from "../lib/apiClient";

export function useRunAnalysis() {
  const [loading, setLoading] = useState(false);
  const [data, setData] = useState<AnalyzeResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function run(payload: AnalyzeRequest) {
    setLoading(true);
    setError(null);
    try {
      const res = await Api.analyze(payload);
      setData(res);
      return res;
    } catch (e: any) {
      setError(e?.message || "Failed to run analysis");
      throw e;
    } finally {
      setLoading(false);
    }
  }

  return { run, loading, data, error, setData };
}
```

---

## 6) UI Composition
6.1 App Shell and Routing (single screen with sections)
```tsx
// filepath: c:\Users\riri9\Desktop\repos\moneybot\webapp\src\App.tsx
import { useState } from "react";
import WatchlistEditor from "./components/WatchlistEditor";
import CriteriaForm from "./components/CriteriaForm";
import AnalysisRunner from "./components/AnalysisRunner";
import ResultsDashboard from "./components/ResultsDashboard";
import AnalysisHistory from "./components/AnalysisHistory";

export default function App() {
  const [activeTab, setActiveTab] = useState<"dashboard" | "history">("dashboard");

  return (
    <div className="min-h-screen bg-gray-50 text-gray-900">
      <header className="border-b bg-white">
        <div className="mx-auto max-w-7xl px-6 py-4 flex items-center justify-between">
          <h1 className="text-xl font-semibold">MoneyBot</h1>
          <nav className="flex gap-4">
            <button className={`px-3 py-2 rounded ${activeTab === "dashboard" ? "bg-gray-900 text-white" : "hover:bg-gray-100"}`} onClick={() => setActiveTab("dashboard")}>Dashboard</button>
            <button className={`px-3 py-2 rounded ${activeTab === "history" ? "bg-gray-900 text-white" : "hover:bg-gray-100"}`} onClick={() => setActiveTab("history")}>History</button>
          </nav>
        </div>
      </header>

      {activeTab === "dashboard" ? (
        <main className="mx-auto max-w-7xl px-6 py-6 grid grid-cols-1 lg:grid-cols-3 gap-6">
          <section className="lg:col-span-1">
            <WatchlistEditor />
          </section>
          <section className="lg:col-span-1">
            <CriteriaForm />
          </section>
          <section className="lg:col-span-1">
            <AnalysisRunner />
            <div className="mt-6">
              <ResultsDashboard />
            </div>
          </section>
        </main>
      ) : (
        <main className="mx-auto max-w-7xl px-6 py-6">
          <AnalysisHistory />
        </main>
      )}

      <footer className="border-t bg-white">
        <div className="mx-auto max-w-7xl px-6 py-4 text-sm text-gray-500">
          Not financial advice. For educational purposes only.
        </div>
      </footer>
    </div>
  );
}
```

6.2 Watchlist Editor
- Add ticker input (uppercase), default shares=0.
- Editable list (ticker, shares).
- Remove item.
- Persist to localStorage; if backend supports persistence, sync.

```tsx
// filepath: c:\Users\riri9\Desktop\repos\moneybot\webapp\src\components\WatchlistEditor.tsx
import { useEffect, useState } from "react";
import type { WatchlistItem } from "../types/api";

const LS_KEY = "moneybot_watchlist";

export default function WatchlistEditor() {
  const [items, setItems] = useState<WatchlistItem[]>([]);
  const [ticker, setTicker] = useState("");

  useEffect(() => {
    const raw = localStorage.getItem(LS_KEY);
    if (raw) setItems(JSON.parse(raw));
  }, []);
  useEffect(() => {
    localStorage.setItem(LS_KEY, JSON.stringify(items));
  }, [items]);

  function addTicker() {
    const t = ticker.trim().toUpperCase();
    if (!t) return;
    if (items.some(i => i.ticker === t)) return;
    setItems(prev => [...prev, { ticker: t, shares: 0 } as WatchlistItem]);
    setTicker("");
  }

  function updateShares(idx: number, shares: number) {
    setItems(prev => prev.map((it, i) => (i === idx ? { ...it, shares } : it)));
  }

  function remove(idx: number) {
    setItems(prev => prev.filter((_, i) => i !== idx));
  }

  return (
    <div className="bg-white rounded-lg border p-4">
      <h2 className="text-lg font-semibold mb-3">Watchlist</h2>
      <div className="flex gap-2">
        <input
          value={ticker}
          onChange={(e) => setTicker(e.target.value)}
          placeholder="Ticker (e.g., AAPL)"
          className="flex-1 rounded border px-3 py-2"
        />
        <button onClick={addTicker} className="rounded bg-gray-900 text-white px-3 py-2 hover:bg-black">Add</button>
      </div>

      <ul className="mt-4 space-y-2">
        {items.map((it, idx) => (
          <li key={it.ticker} className="flex items-center gap-3 rounded border p-2">
            <span className="font-mono w-20">{it.ticker}</span>
            <input
              type="number"
              min={0}
              value={it.shares}
              onChange={(e) => updateShares(idx, Number(e.target.value))}
              className="w-28 rounded border px-2 py-1"
            />
            <span className="text-sm text-gray-500">shares</span>
            <div className="flex-1" />
            <button onClick={() => remove(idx)} className="text-red-600 hover:underline">Remove</button>
          </li>
        ))}
      </ul>
      <p className="mt-3 text-xs text-gray-500">Tip: Add tickers with 0 shares to track them without holdings.</p>
    </div>
  );
}
```

6.3 Criteria Form
- Collect inputs expected by `AnalyzeRequest` (mirror `api.ts`).
- Include remarks/notes for re-runs.

```tsx
// filepath: c:\Users\riri9\Desktop\repos\moneybot\webapp\src\components\CriteriaForm.tsx
import { useEffect, useState } from "react";
import type { Criteria } from "../types/api";

const LS_KEY = "moneybot_criteria";

export default function CriteriaForm() {
  const [criteria, setCriteria] = useState<Criteria>({} as Criteria);

  useEffect(() => {
    const raw = localStorage.getItem(LS_KEY);
    if (raw) setCriteria(JSON.parse(raw));
  }, []);
  useEffect(() => {
    localStorage.setItem(LS_KEY, JSON.stringify(criteria));
  }, [criteria]);

  function update<K extends keyof Criteria>(key: K, value: Criteria[K]) {
    setCriteria(prev => ({ ...prev, [key]: value }));
  }

  return (
    <div className="bg-white rounded-lg border p-4">
      <h2 className="text-lg font-semibold mb-3">Investment Criteria</h2>

      {/* Replace these with the real fields from Criteria type */}
      <div className="space-y-3">
        <div>
          <label className="block text-sm text-gray-600">Risk tolerance</label>
          <select className="mt-1 w-full rounded border px-3 py-2"
            value={(criteria as any).riskTolerance || ""}
            onChange={(e) => update("riskTolerance" as any, e.target.value as any)}>
            <option value="">Select</option>
            <option value="low">Low</option>
            <option value="moderate">Moderate</option>
            <option value="high">High</option>
          </select>
        </div>

        <div>
          <label className="block text-sm text-gray-600">Investment horizon (months)</label>
          <input type="number" className="mt-1 w-full rounded border px-3 py-2"
            value={(criteria as any).horizonMonths || 12}
            onChange={(e) => update("horizonMonths" as any, Number(e.target.value) as any)} />
        </div>

        <div>
          <label className="block text-sm text-gray-600">Remarks / Additional constraints</label>
          <textarea className="mt-1 w-full rounded border px-3 py-2"
            rows={4}
            value={(criteria as any).remarks || ""}
            onChange={(e) => update("remarks" as any, e.target.value as any)} />
        </div>
      </div>
      <p className="mt-3 text-xs text-gray-500">Adjust fields to match the Criteria interface in types/api.ts.</p>
    </div>
  );
}
```

6.4 Analysis Runner + Results
- Runner composes `AnalyzeRequest` from watchlist + criteria and calls POST /analyze.
- Results component displays signals and budget recommendation from `AnalyzeResponse`.
- Provide a “Refine and re-run” notes box to append remarks.

```tsx
// filepath: c:\Users\riri9\Desktop\repos\moneybot\webapp\src\components\AnalysisRunner.tsx
import { useEffect, useState } from "react";
import type { AnalyzeRequest, AnalyzeResponse, WatchlistItem, Criteria } from "../types/api";
import { useRunAnalysis } from "../hooks/useAnalysis";

export default function AnalysisRunner() {
  const [watchlist, setWatchlist] = useState<WatchlistItem[]>([]);
  const [criteria, setCriteria] = useState<Criteria | null>(null);
  const [extraRemarks, setExtraRemarks] = useState("");
  const { run, loading, data, error, setData } = useRunAnalysis();

  // Pull from localStorage (kept in the other components)
  useEffect(() => {
    const wl = localStorage.getItem("moneybot_watchlist");
    const cr = localStorage.getItem("moneybot_criteria");
    if (wl) setWatchlist(JSON.parse(wl));
    if (cr) setCriteria(JSON.parse(cr));
  }, []);

  async function onRun() {
    if (!criteria) return;
    const payload: AnalyzeRequest = {
      watchlist,
      criteria: {
        ...criteria,
        // merge in extra remarks if supported by the backend DTO:
        ...(extraRemarks ? { remarks: `${(criteria as any).remarks || ""}\n${extraRemarks}` } : {}),
      } as any,
    };
    await run(payload);
  }

  function onRerunWithRemarks() {
    if (!data) return;
    onRun();
  }

  return (
    <div className="bg-white rounded-lg border p-4">
      <h2 className="text-lg font-semibold mb-3">Run Analysis</h2>
      <button
        disabled={loading}
        onClick={onRun}
        className={`rounded px-4 py-2 text-white ${loading ? "bg-gray-400" : "bg-indigo-600 hover:bg-indigo-700"}`}
      >
        {loading ? "Running..." : "Run Analysis"}
      </button>

      <div className="mt-4">
        <label className="block text-sm text-gray-600">Additional remarks (optional)</label>
        <textarea
          className="mt-1 w-full rounded border px-3 py-2"
          rows={3}
          value={extraRemarks}
          onChange={(e) => setExtraRemarks(e.target.value)}
          placeholder="E.g., favor tech growth; exclude energy; budget cap $5k..."
        />
        <div className="mt-2 flex justify-end">
          <button
            onClick={onRerunWithRemarks}
            disabled={loading}
            className="rounded px-3 py-2 border hover:bg-gray-50"
          >
            Refine & Re-run
          </button>
        </div>
      </div>

      {error && <p className="mt-3 text-sm text-red-600">{error}</p>}

      {!loading && !error && data == null && (
        <p className="mt-3 text-sm text-gray-500">Run an analysis to see results here.</p>
      )}
    </div>
  );
}
```

```tsx
// filepath: c:\Users\riri9\Desktop\repos\moneybot\webapp\src\components\ResultsDashboard.tsx
import { useEffect, useState } from "react";
import type { AnalyzeResponse } from "../types/api";

export default function ResultsDashboard() {
  const [result, setResult] = useState<AnalyzeResponse | null>(null);

  // Pull latest from hook state via localStorage or a simple event bus; here simplified:
  useEffect(() => {
    // Optionally, sync with AnalysisRunner via a global store or context.
  }, []);

  if (!result) {
    return (
      <div className="bg-white rounded-lg border p-4">
        <h2 className="text-lg font-semibold mb-3">Results</h2>
        <div className="animate-pulse space-y-2">
          <div className="h-4 bg-gray-200 rounded" />
          <div className="h-4 bg-gray-200 rounded w-5/6" />
          <div className="h-4 bg-gray-200 rounded w-4/6" />
        </div>
        <p className="mt-3 text-sm text-gray-500">Results will appear after running an analysis.</p>
      </div>
    );
  }

  // Replace with actual fields from AnalyzeResponse
  const signals = (result as any).signals || [];
  const recommendation = (result as any).budgetRecommendation;
  const rationale = (result as any).explanation || (result as any).rationale;

  return (
    <div className="bg-white rounded-lg border p-4">
      <h2 className="text-lg font-semibold mb-3">Results</h2>

      {recommendation && (
        <div className="mb-4 rounded border p-3 bg-emerald-50 border-emerald-200">
          <h3 className="font-semibold text-emerald-900">Budget Recommendation</h3>
          <pre className="text-sm text-emerald-800 mt-2 whitespace-pre-wrap">
            {JSON.stringify(recommendation, null, 2)}
          </pre>
        </div>
      )}

      <div className="mb-4">
        <h3 className="font-semibold">Signals</h3>
        <ul className="mt-2 space-y-2">
          {signals.map((s: any, idx: number) => (
            <li key={idx} className="rounded border p-2">
              <div className="flex items-center justify-between">
                <span className="font-mono">{s.ticker}</span>
                <span className="text-sm rounded px-2 py-1 bg-gray-900 text-white">{s.action}</span>
              </div>
              <p className="text-sm text-gray-600 mt-1">Confidence: {s.confidence}</p>
            </li>
          ))}
        </ul>
      </div>

      {rationale && (
        <div>
          <h3 className="font-semibold">Explanation</h3>
          <p className="text-sm text-gray-700 mt-1 whitespace-pre-wrap">{rationale}</p>
        </div>
      )}
    </div>
  );
}
```

6.5 Analysis History (optional if backend supports it)
```tsx
// filepath: c:\Users\riri9\Desktop\repos\moneybot\webapp\src\components\AnalysisHistory.tsx
export default function AnalysisHistory() {
  return (
    <div className="bg-white rounded-lg border p-4">
      <h2 className="text-lg font-semibold mb-3">Analysis History</h2>
      <p className="text-sm text-gray-500">Hook up to a backend endpoint if available; otherwise omit this tab.</p>
    </div>
  );
}
```

---

## 7) Data Wiring Checklist
- Replace placeholder fields with real DTO properties from `types/api.ts`.
- Confirm `AnalyzeRequest` shape:
  - watchlist: WatchlistItem[]
  - criteria: Criteria
  - optional fields: remarks, constraints, budget, etc.
- Confirm `AnalyzeResponse` shape:
  - recommendation/budgetRecommendation
  - signals array (ticker, action, confidence)
  - explanation/rationale text
  - any IDs for history.

---

## 8) Styling (Tailwind)
- Use rounded cards, neutral grays, and consistent spacing (p-4, gap-6).
- Use utility classes only; avoid custom CSS where possible.
- Provide accessible color contrast and focus states.

---

## 9) End-to-End Flow Parity with Terminal
- The UI must replicate the CLI mockup in `full_example.py`:
  1) Create watchlist (tickers with shares; allow 0).
  2) Set criteria (risk, horizon, constraints).
  3) Run analysis.
  4) Re-run with additional remarks.
  5) Show budget recommendation.
- Validate any domain-specific prompts/fields present in the CLI script are available in the form.

---

## 10) Testing and QA
- Manual QA:
  - Add/remove/edit tickers; shares 0 allowed.
  - Invalid ticker validation (optional).
  - Criteria persistence to localStorage.
  - Successful analyze call: loading state, success UI.
  - Error path: simulated 500, network timeout → user sees actionable message.
  - Re-run with extra remarks modifies payload as expected.
- Unit tests (optional if setup allows):
  - Pure functions (payload builders).
  - Watchlist reducer logic (if added).

---

## 11) Dev and Run Commands (Windows)
- Backend (example):
  - py -m venv .venv && .\.venv\Scripts\Activate.ps1
  - pip install -r requirements.txt
  - python -m src.langgraph_workflow.app
- Frontend:
  - cd webapp
  - npm i
  - npm run dev
  - Ensure VITE_API_BASE_URL points to the backend (e.g., http://127.0.0.1:8000)

---

## 12) Deployment Notes
- Build frontend: npm run build → static assets.
- Serve behind a reverse proxy; set CORS on backend to allow the deployed origin.
- Use environment variables for base URL; do not hardcode.

---

## 13) Mapping Worksheet (fill from app.py and api.ts)
Populate this before coding:

| Feature | Method | Path | Request Type | Response Type | Notes |
|--------|--------|------|--------------|---------------|-------|
| Run analysis | POST | /api/analyze | AnalyzeRequest | AnalyzeResponse | Main endpoint |
| Get history | GET | /api/analyses | — | AnalysisSummary[] | Optional |
| Get analysis by id | GET | /api/analyses/:id | — | AnalyzeResponse | Optional |
| Watchlist get/save | GET/POST | /api/watchlist | —/WatchlistItem[] | WatchlistItem[]/void | Optional |

Adjust paths/types to exactly match your backend.

---

## 14) Implementation Order (Milestones)
1. Align endpoints and types (Sections 1 and 13).
2. Tailwind setup and base AppShell.
3. WatchlistEditor (local persistence).
4. CriteriaForm (local persistence).
5. API client and Analyze flow.
6. AnalysisRunner + loading/error UX.
7. ResultsDashboard with real fields.
8. Refine & Re-run remarks flow.
9. Optional: History.
10. Polish: accessibility, keyboard nav, empty states, responsive tweaks.

---

## 15) Risks and Mitigations
- DTO mismatch → Always import types from `types/api.ts`.
- CORS issues → Enable CORS in backend with allowed origin.
- Long-running analysis → Add higher timeout, show progress; consider server-sent events if backend supports.
- Validation gaps → Add minimal front-end constraints; trust server as source of truth.

---

## 16) Done Criteria
- Users can manage a watchlist with 0-share tickers.
- Users can set criteria and remarks.
- Users can run analysis and re-run with additional remarks.
- Users receive a budget recommendation and signals with explanations.
- UI is responsive, accessible, and pleasant with Tailwind.
- All network code is type-safe and consistent with `types/api.ts`.