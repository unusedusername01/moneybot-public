import { ChangeEvent, useMemo, useState } from "react";
import type { HoldingEntry } from "../types/portfolio";
import type { FieldStatus } from "../types/validation";

interface WatchlistEditorProps {
  holdings: HoldingEntry[];
  onChange: (entries: HoldingEntry[]) => void;
  onValidateTicker?: (ticker: string) => Promise<boolean>;
  tickerStatus?: Record<string, FieldStatus>;
  tickerErrors?: Record<string, string>;
}

type ValidationState = "idle" | "valid" | "invalid" | "checking";

export default function WatchlistEditor({
  holdings,
  onChange,
  onValidateTicker,
  tickerStatus = {},
  tickerErrors = {},
}: WatchlistEditorProps) {
  const [tickerInput, setTickerInput] = useState("");
  const [sharesInput, setSharesInput] = useState("0");
  const [validation, setValidation] = useState<ValidationState>("idle");
  const [error, setError] = useState<string | null>(null);

  const sortedHoldings = useMemo(
    () => [...holdings].sort((a, b) => a.ticker.localeCompare(b.ticker)),
    [holdings]
  );

  function handleSharesChange(ticker: string, event: ChangeEvent<HTMLInputElement>) {
    const value = event.target.value;
    const numeric = Number(value);
    if (Number.isNaN(numeric) || numeric < 0) {
      return;
    }
    const updated = holdings.map((entry) =>
      entry.ticker === ticker ? { ...entry, shares: numeric } : entry
    );
    onChange(updated);
  }

  async function handleAddTicker() {
    setError(null);
    const ticker = tickerInput.trim().toUpperCase();
    if (!ticker) {
      setError("Enter a ticker symbol");
      return;
    }
    if (holdings.some((entry) => entry.ticker === ticker)) {
      setError("Ticker already in watchlist");
      return;
    }

    const shares = Number(sharesInput);
    if (Number.isNaN(shares) || shares < 0) {
      setError("Shares must be a positive number or zero");
      return;
    }

    if (onValidateTicker) {
      try {
        setValidation("checking");
        const isValid = await onValidateTicker(ticker);
        setValidation(isValid ? "valid" : "invalid");
        if (!isValid) {
          setError("Ticker could not be validated");
          return;
        }
      } catch (validationError) {
        console.error("Ticker validation failed", validationError);
        setValidation("invalid");
        setError("Validation failed");
        return;
      }
    }

    const updated = [...holdings, { ticker, shares }];
    onChange(updated);
    setTickerInput("");
    setSharesInput("0");
    setValidation("idle");
  }

  function handleRemoveTicker(ticker: string) {
    const updated = holdings.filter((entry) => entry.ticker !== ticker);
    onChange(updated);
  }

  return (
    <div className="rounded-2xl border border-gray-200 bg-white/90 p-6 shadow-sm backdrop-blur dark:border-gray-800 dark:bg-gray-900/80">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-semibold text-gray-800 dark:text-gray-100">Watchlist</h2>
        <span className="text-sm text-gray-500 dark:text-gray-400">{holdings.length} tickers</span>
      </div>

      <p className="mt-2 text-sm text-gray-600 dark:text-gray-400">
        Add tickers to track. Shares can be set to zero for watch-only entries.
      </p>

      <div className="mt-4 flex flex-col gap-3">
        <div className="flex flex-col gap-2 sm:flex-row sm:items-end">
          <div className="flex-1">
            <label className="block text-xs font-medium uppercase tracking-wide text-gray-500 dark:text-gray-400">
              Ticker
            </label>
            <input
              value={tickerInput}
              onChange={(event: ChangeEvent<HTMLInputElement>) => setTickerInput(event.target.value.toUpperCase())}
              placeholder="AAPL"
              maxLength={8}
              className="mt-1 w-full rounded border border-gray-300 bg-white px-3 py-2 text-sm shadow-sm focus:border-indigo-500 focus:outline-none focus:ring-1 focus:ring-indigo-500 dark:border-gray-700 dark:bg-gray-900 dark:text-gray-100"
            />
          </div>
          <div>
            <label className="block text-xs font-medium uppercase tracking-wide text-gray-500 dark:text-gray-400">
              Shares
            </label>
            <input
              type="number"
              min={0}
              step={1}
              value={sharesInput}
              onChange={(event: ChangeEvent<HTMLInputElement>) => setSharesInput(event.target.value)}
              className="mt-1 w-full rounded border border-gray-300 bg-white px-3 py-2 text-sm shadow-sm focus:border-indigo-500 focus:outline-none focus:ring-1 focus:ring-indigo-500 dark:border-gray-700 dark:bg-gray-900 dark:text-gray-100"
            />
          </div>
          <button
            type="button"
            onClick={handleAddTicker}
            className="rounded bg-indigo-600 px-4 py-2 text-sm font-medium text-white shadow transition hover:bg-indigo-700 dark:bg-indigo-500 dark:hover:bg-indigo-400"
          >
            Add
          </button>
        </div>
        {error && <p className="text-sm text-red-500 dark:text-red-300">{error}</p>}
        {validation === "checking" && <p className="text-xs text-gray-500 dark:text-gray-400">Validating ticker...</p>}
      </div>

      <ul className="mt-4 space-y-2">
        {sortedHoldings.map((entry) => (
          <li
            key={entry.ticker}
            className="rounded border border-gray-200 px-3 py-2 text-sm shadow-sm dark:border-gray-700"
          >
            <div className="flex flex-wrap items-center gap-3">
              <span className="font-mono text-base text-gray-800 dark:text-gray-100">{entry.ticker}</span>
              <input
                type="number"
                min={0}
                step={0.01}
                value={entry.shares}
                onChange={(event: ChangeEvent<HTMLInputElement>) => handleSharesChange(entry.ticker, event)}
                className="w-28 rounded border border-gray-300 bg-white px-2 py-1 text-sm focus:border-indigo-500 focus:outline-none focus:ring-1 focus:ring-indigo-500 dark:border-gray-700 dark:bg-gray-900 dark:text-gray-100"
              />
              <span className="text-xs uppercase tracking-wide text-gray-500 dark:text-gray-400">shares</span>
              <div className="flex-1" />
              {tickerStatus[entry.ticker] === "checking" && (
                <span className="text-xs text-gray-500 dark:text-gray-400">Checking…</span>
              )}
              {tickerStatus[entry.ticker] === "valid" && (
                <span className="text-xs text-emerald-600 dark:text-emerald-300">Valid</span>
              )}
              {tickerStatus[entry.ticker] === "invalid" && (
                <span className="text-xs text-red-500 dark:text-red-300">Invalid</span>
              )}
              <button
                type="button"
                onClick={() => handleRemoveTicker(entry.ticker)}
                className="rounded border border-red-200 px-2 py-1 text-xs font-medium text-red-600 transition hover:bg-red-50 dark:border-red-400/40 dark:text-red-300 dark:hover:bg-red-500/10"
              >
                Remove
              </button>
            </div>
            {tickerErrors[entry.ticker] && (
              <p className="mt-2 text-xs text-red-500 dark:text-red-300">{tickerErrors[entry.ticker]}</p>
            )}
          </li>
        ))}
        {sortedHoldings.length === 0 && (
          <li className="rounded border border-dashed border-gray-300 px-3 py-4 text-center text-sm text-gray-500 dark:border-gray-700 dark:text-gray-400">
            No tickers yet. Add one above to get started.
          </li>
        )}
      </ul>
    </div>
  );
}
