import { useCallback, useEffect, useMemo, useState } from "react";
import AnalysisHistory from "./components/AnalysisHistory";
import AnalysisRunner from "./components/AnalysisRunner";
import CriteriaForm from "./components/CriteriaForm";
import PortfolioSelector from "./components/PortfolioSelector";
import WatchlistEditor from "./components/WatchlistEditor";
import { Api, BASE_URL } from "./lib/apiClient";
import { useAnalysisSession } from "./hooks/useAnalysisSession";
import type { HoldingEntry, PortfolioData, PortfolioId } from "./types/portfolio";
import type { FieldStatus } from "./types/validation";
import type { GetSharesPriceRequest } from "./types/api";

function createDefaultPortfolio(): PortfolioData {
  const nextYear = new Date();
  nextYear.setFullYear(nextYear.getFullYear() + 1);
  return {
    budget: 10000,
    target_date: nextYear.toISOString().slice(0, 10),
    holdings: {},
    currency: "USD",
    risk_tolerance: "medium",
    criteria: "market_cap",
    prediction_strength: "medium",
  };
}

function holdingsToEntries(holdings: Record<string, number>): HoldingEntry[] {
  return Object.entries(holdings).map(([ticker, shares]) => ({ ticker, shares }));
}

function sanitizeHoldings(entries: HoldingEntry[]): Record<string, number> {
  return entries.reduce<Record<string, number>>((acc, entry) => {
    const ticker = entry.ticker.trim().toUpperCase();
    if (!ticker) {
      return acc;
    }
    const shares = Number(entry.shares);
    if (Number.isNaN(shares) || shares < 0) {
      return acc;
    }
    acc[ticker] = shares;
    return acc;
  }, {});
}

export default function App() {
  const [portfolios, setPortfolios] = useState<PortfolioId[]>([]);
  const [selectedPortfolio, setSelectedPortfolio] = useState<PortfolioId | null>(null);
  const [portfolioData, setPortfolioData] = useState<PortfolioData>(createDefaultPortfolio());
  const [loadingPortfolio, setLoadingPortfolio] = useState(false);
  const [portfolioError, setPortfolioError] = useState<string | null>(null);
  const [feedback, setFeedback] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  const [dirty, setDirty] = useState(false);
  const [pendingCreate, setPendingCreate] = useState(false);
  const [tickerStatus, setTickerStatus] = useState<Record<string, FieldStatus>>({});
  const [tickerErrors, setTickerErrors] = useState<Record<string, string>>({});
  const [currencyStatus, setCurrencyStatus] = useState<FieldStatus>("idle");
  const [currencyError, setCurrencyError] = useState<string | null>(null);

  const {
    state: analysisState,
    start: startAnalysis,
    sendChoice,
    sendMessage,
    sendAllocation,
    reset: resetAnalysis,
  } = useAnalysisSession(selectedPortfolio, BASE_URL);

  const holdingsEntries = useMemo(() => holdingsToEntries(portfolioData.holdings), [portfolioData.holdings]);
  const currencyCode = useMemo(() => (portfolioData.currency || "").trim().toUpperCase(), [portfolioData.currency]);

  const loadPortfolios = useCallback(async () => {
    const result = await Api.listPortfolios();
    if (!result.success) {
      setPortfolioError(result.message);
      return;
    }
    setPortfolioError(null);
    const fetched = result.data;
    const withDraft = pendingCreate && selectedPortfolio && !fetched.includes(selectedPortfolio)
      ? [...fetched, selectedPortfolio]
      : fetched;
    setPortfolios(withDraft);

    if (!selectedPortfolio && withDraft.length > 0) {
      setSelectedPortfolio(withDraft[0]);
    }
    if (withDraft.length === 0 && !pendingCreate) {
      setSelectedPortfolio(null);
      setPortfolioData(createDefaultPortfolio());
      setDirty(false);
    }
  }, [pendingCreate, selectedPortfolio]);

  const loadPortfolioData = useCallback(async (portfolioId: PortfolioId) => {
    setLoadingPortfolio(true);
    setPortfolioError(null);
    const result = await Api.loadPortfolioData(portfolioId);
    setLoadingPortfolio(false);
    if (!result.success) {
      setPortfolioError(result.message);
      return;
    }

    const normalized: PortfolioData = {
      ...createDefaultPortfolio(),
      ...result.data,
      holdings: result.data.holdings ?? {},
    };

    setPortfolioData(normalized);
    setDirty(false);
    setPendingCreate(false);
    setFeedback(null);
    setTickerStatus({});
    setTickerErrors({});
    setCurrencyStatus("valid");
    setCurrencyError(null);
  }, []);

  useEffect(() => {
    loadPortfolios().catch((error: unknown) => {
      console.error("Failed to load portfolios", error);
      setPortfolioError("Failed to load portfolios");
    });
  }, [loadPortfolios]);

  useEffect(() => {
    if (!selectedPortfolio) {
      return;
    }
    if (pendingCreate) {
      resetAnalysis();
      return;
    }
    loadPortfolioData(selectedPortfolio).catch((error: unknown) => {
      console.error("Failed to load portfolio", error);
      setPortfolioError("Failed to load portfolio data");
    });
    resetAnalysis();
  }, [loadPortfolioData, pendingCreate, resetAnalysis, selectedPortfolio]);

  const handleHoldingsChange = useCallback((entries: HoldingEntry[]) => {
    const holdings = sanitizeHoldings(entries);
    setPortfolioData((prev) => ({ ...prev, holdings }));
    setDirty(true);
    setFeedback(null);

    setTickerStatus((prev) => {
      const next: Record<string, FieldStatus> = {};
      Object.keys(holdings).forEach((ticker) => {
        next[ticker] = prev[ticker] ?? "idle";
      });
      return next;
    });
    setTickerErrors((prev) => {
      const next: Record<string, string> = {};
      Object.keys(holdings).forEach((ticker) => {
        if (prev[ticker]) {
          next[ticker] = prev[ticker];
        }
      });
      return next;
    });
  }, []);

  const handleCriteriaChange = useCallback((changes: Partial<PortfolioData>) => {
    setPortfolioData((prev) => {
      const next: PortfolioData = { ...prev, ...changes };
      if (typeof changes.currency === "string") {
        next.currency = changes.currency.toUpperCase();
      }
      return next;
    });
    setDirty(true);
    setFeedback(null);
  }, []);

  const validateCurrency = useCallback(async (code: string) => {
    const normalized = code.trim().toUpperCase();
    if (!normalized) {
      setCurrencyStatus("invalid");
      setCurrencyError("Currency code is required.");
      return false;
    }
    if (normalized.length !== 3) {
      setCurrencyStatus("invalid");
      setCurrencyError("Currency codes use three letters (e.g. USD).");
      return false;
    }
    setCurrencyStatus("checking");
    setCurrencyError(null);
    const result = await Api.validateCurrency(normalized);
    if (result.success) {
      setCurrencyStatus("valid");
      setCurrencyError(null);
      return true;
    }
    setCurrencyStatus("invalid");
    setCurrencyError(result.message ?? "Invalid currency code");
    return false;
  }, []);

  const validateTicker = useCallback(async (rawTicker: string) => {
    const ticker = rawTicker.trim().toUpperCase();
    if (!ticker) {
      return false;
    }

    if (currencyCode.length !== 3) {
      setTickerStatus((prev) => ({ ...prev, [ticker]: "idle" }));
      setTickerErrors((prev) => {
        if (!(ticker in prev)) {
          return prev;
        }
        const { [ticker]: _removed, ...rest } = prev;
        return rest;
      });
      return false;
    }

    setTickerStatus((prev) => ({ ...prev, [ticker]: "checking" }));
    const request: GetSharesPriceRequest = {
      type: "http_request",
      ticker,
      amount: null,
      currency: currencyCode as GetSharesPriceRequest["currency"],
      allow_fractional: true,
    };
    try {
      const result = await Api.getSharesPrice(request);
      if (result.success) {
        setTickerStatus((prev) => ({ ...prev, [ticker]: "valid" }));
        setTickerErrors((prev) => {
          if (!(ticker in prev)) {
            return prev;
          }
          const { [ticker]: _removed, ...rest } = prev;
          return rest;
        });
        return true;
      }
      setTickerStatus((prev) => ({ ...prev, [ticker]: "invalid" }));
      setTickerErrors((prev) => ({ ...prev, [ticker]: result.message ?? "Ticker validation failed" }));
      return false;
    } catch (error) {
      console.error("Ticker validation failed", error);
      setTickerStatus((prev) => ({ ...prev, [ticker]: "invalid" }));
      setTickerErrors((prev) => ({ ...prev, [ticker]: "Unable to validate ticker" }));
      return false;
    }
  }, [currencyCode]);

  useEffect(() => {
    if (!currencyCode) {
      setCurrencyStatus("invalid");
      setCurrencyError("Currency code is required.");
      return;
    }
    const timeout = window.setTimeout(() => {
      void validateCurrency(currencyCode);
    }, 400);
    return () => window.clearTimeout(timeout);
  }, [currencyCode, validateCurrency]);

  useEffect(() => {
    if (holdingsEntries.length === 0) {
      setTickerStatus({});
      setTickerErrors({});
      return;
    }
    if (currencyStatus !== "valid" || currencyCode.length !== 3) {
      return;
    }
    holdingsEntries.forEach(({ ticker }) => {
      void validateTicker(ticker);
    });
  }, [currencyCode, currencyStatus, holdingsEntries, validateTicker]);

  const handleClearFields = useCallback(() => {
    setPortfolioData(createDefaultPortfolio());
    setDirty(true);
    setFeedback(null);
    setTickerStatus({});
    setTickerErrors({});
    setCurrencyStatus("idle");
    setCurrencyError(null);
  }, []);

  const handleNewPortfolio = useCallback(() => {
    const timestamp = new Date().toISOString().slice(0, 19).replace(/[-:T]/g, "");
    const suggested = `portfolio_${timestamp}`;
    const input = window.prompt("Create new portfolio", suggested);
    if (!input) {
      return;
    }
    const normalized = input.trim().replace(/\s+/g, "_");
    if (!normalized) {
      return;
    }
    setPortfolios((prev) => (prev.includes(normalized) ? prev : [...prev, normalized]));
    setSelectedPortfolio(normalized);
    setPortfolioData(createDefaultPortfolio());
    setDirty(true);
    setPendingCreate(true);
    setTickerStatus({});
    setTickerErrors({});
    setCurrencyStatus("idle");
    setCurrencyError(null);
    setFeedback(null);
  }, []);

  const handleDeleteSelected = useCallback(async () => {
    if (!selectedPortfolio) {
      return;
    }
    if (pendingCreate) {
      setPortfolios((prev) => prev.filter((id) => id !== selectedPortfolio));
      setSelectedPortfolio(null);
      setPortfolioData(createDefaultPortfolio());
      setPendingCreate(false);
      setDirty(false);
      setFeedback(null);
      return;
    }
    const confirmed = window.confirm(`Delete portfolio ${selectedPortfolio}?`);
    if (!confirmed) {
      return;
    }
    const result = await Api.deletePortfolio(selectedPortfolio);
    if (!result.success) {
      setPortfolioError(result.message);
      return;
    }
    setPortfolioError(null);
    setPortfolios((prev) => prev.filter((id) => id !== selectedPortfolio));
    setSelectedPortfolio(null);
    setPortfolioData(createDefaultPortfolio());
    setDirty(false);
    setFeedback(null);
    await loadPortfolios();
  }, [loadPortfolios, pendingCreate, selectedPortfolio]);

  const handleSave = useCallback(async () => {
    if (!selectedPortfolio) {
      setPortfolioError("Select or create a portfolio first.");
      return false;
    }

    setSaving(true);
    setPortfolioError(null);
    setFeedback(null);

    const holdings = sanitizeHoldings(holdingsEntries);
    if (Object.keys(holdings).length === 0) {
      setSaving(false);
      setPortfolioError("Add at least one ticker before saving.");
      return false;
    }

  const currencyValid = await validateCurrency(currencyCode);
    const tickerResults = await Promise.all(Object.keys(holdings).map((ticker) => validateTicker(ticker)));
    const hasInvalidTicker = tickerResults.some((isValid) => !isValid);

    if (!currencyValid || hasInvalidTicker) {
      setSaving(false);
      setPortfolioError("Fix validation errors before saving.");
      return false;
    }

    if (!portfolioData.target_date) {
      setSaving(false);
      setPortfolioError("Set a target date before saving.");
      return false;
    }

    const payload: PortfolioData = {
      ...portfolioData,
  currency: currencyCode,
      holdings,
    };

    const response = pendingCreate
      ? await Api.createPortfolio(selectedPortfolio, payload)
      : await Api.savePortfolio(selectedPortfolio, payload);

    setSaving(false);

    if (!response.success) {
      setPortfolioError(response.message);
      return false;
    }

    setPortfolioError(null);
    setPortfolioData(payload);
    setDirty(false);
    setPendingCreate(false);
    setFeedback("Portfolio saved successfully.");
    await loadPortfolios();
    return true;
  }, [currencyCode, holdingsEntries, loadPortfolios, pendingCreate, portfolioData, selectedPortfolio, validateCurrency, validateTicker]);

  const handleStart = useCallback(async () => {
    if (!selectedPortfolio) {
      setPortfolioError("Select or create a portfolio first.");
      return;
    }
    try {
      if (pendingCreate || dirty) {
        const saved = await handleSave();
        if (!saved) {
          return;
        }
      }
      await startAnalysis();
    } catch (error) {
      console.error("Unable to start analysis", error);
    }
  }, [dirty, handleSave, pendingCreate, selectedPortfolio, startAnalysis]);

  const handleRestartAnalysis = useCallback(async () => {
    if (!selectedPortfolio) {
      setPortfolioError("Select or create a portfolio first.");
      return;
    }
    try {
      if (pendingCreate || dirty) {
        const saved = await handleSave();
        if (!saved) {
          return;
        }
      }
      resetAnalysis();
      const started = await startAnalysis();
      if (!started) {
        setPortfolioError((prev) => prev ?? "Unable to restart analysis.");
      }
    } catch (error) {
      console.error("Unable to restart analysis", error);
      setPortfolioError("Unable to restart analysis.");
    }
  }, [dirty, handleSave, pendingCreate, resetAnalysis, selectedPortfolio, startAnalysis]);

  const handleReturnToPortfolioPicker = useCallback(() => {
    resetAnalysis();
    const selectElement = document.getElementById("portfolio-select") as HTMLSelectElement | null;
    if (selectElement) {
      selectElement.focus();
    }
    if (typeof window !== "undefined") {
      window.scrollTo({ top: 0, behavior: "smooth" });
    }
  }, [resetAnalysis]);

  const hasInvalidTickers = useMemo(
    () => Object.values(tickerStatus).some((status) => status === "invalid"),
    [tickerStatus]
  );
  const isTickerChecking = useMemo(
    () => Object.values(tickerStatus).some((status) => status === "checking"),
    [tickerStatus]
  );

  const canStart = Boolean(
    selectedPortfolio &&
      !pendingCreate &&
      !dirty &&
      !saving &&
      !hasInvalidTickers &&
      !isTickerChecking &&
      currencyStatus === "valid" &&
      portfolioData.target_date &&
      Object.keys(portfolioData.holdings).length > 0 &&
      !["running", "awaiting_input", "connecting"].includes(analysisState.status) &&
      analysisState.messages.every((message) => !message.pending)
  );

  return (
    <div className="min-h-screen bg-gray-50 text-gray-900 transition-colors dark:bg-gray-950 dark:text-gray-100">
      <div className="mx-auto flex max-w-6xl flex-col gap-8 px-6 py-8">
        <header className="text-center">
          <p className="text-xs uppercase tracking-[0.35em] text-indigo-500 dark:text-indigo-300">MoneyBot</p>
          <h1 className="mt-2 text-3xl font-semibold">AI Portfolio Copilot</h1>
          <p className="mt-2 text-sm text-gray-600 dark:text-gray-400">
            Curate your watchlist, set your investment profile, and let MoneyBot walk you through the allocation conversation.
          </p>
        </header>

        <section className="rounded-2xl border border-gray-200 bg-white/90 p-6 shadow-sm backdrop-blur dark:border-gray-800 dark:bg-gray-900/80">
          <div className="mb-4 text-left">
            <h2 className="text-lg font-semibold">Portfolio manager</h2>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              The watchlist and investment profile together define the portfolio under review.
            </p>
          </div>
          <PortfolioSelector
            portfolios={portfolios}
            selected={selectedPortfolio}
            dirty={dirty}
            saving={saving}
            busy={loadingPortfolio || saving}
            pendingCreate={pendingCreate}
            onSelect={(id) => {
              setSelectedPortfolio(id);
              setPendingCreate(false);
              setFeedback(null);
            }}
            onSave={handleSave}
            onClear={handleClearFields}
            onCreateNew={handleNewPortfolio}
            onDelete={handleDeleteSelected}
          />
          {(portfolioError || feedback) && (
            <div className="mt-4 space-y-1">
              {portfolioError && <p className="text-sm text-red-500">{portfolioError}</p>}
              {feedback && <p className="text-sm text-emerald-500">{feedback}</p>}
            </div>
          )}
        </section>

        <div className="grid gap-6 lg:grid-cols-[minmax(0,2fr)_minmax(0,3fr)]">
          <div className="space-y-6">
            <WatchlistEditor
              holdings={holdingsEntries}
              onChange={handleHoldingsChange}
              onValidateTicker={validateTicker}
              tickerStatus={tickerStatus}
              tickerErrors={tickerErrors}
            />
            <CriteriaForm
              value={portfolioData}
              onChange={handleCriteriaChange}
              currencyStatus={currencyStatus}
              currencyError={currencyError}
              onValidateCurrency={validateCurrency}
            />
          </div>
          <AnalysisRunner
            portfolioId={selectedPortfolio}
            currency={portfolioData.currency}
            canStart={canStart}
            onStart={handleStart}
            onRestart={handleRestartAnalysis}
            onReturnToPortfolio={handleReturnToPortfolioPicker}
            sendChoice={sendChoice}
            sendMessage={sendMessage}
            sendAllocation={sendAllocation}
            analysis={analysisState}
          />
        </div>

        <AnalysisHistory />
      </div>

      <footer className="border-t border-gray-200 bg-white/70 py-4 text-xs text-gray-500 dark:border-gray-800 dark:bg-gray-900/80 dark:text-gray-400">
        <div className="mx-auto max-w-6xl px-6">
          MoneyBot provides educational guidance only. Always perform your own due diligence before making investment decisions.
        </div>
      </footer>
    </div>
  );
}
