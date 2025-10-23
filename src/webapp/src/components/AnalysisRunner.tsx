import { ChangeEvent, FormEvent, useEffect, useMemo, useRef, useState } from "react";
import type {
  AnalysisSessionState,
  AwaitingState,
  ChatMessage,
  CurrencyCode,
} from "../hooks/useAnalysisSession";

interface AnalysisRunnerProps {
  portfolioId: string | null;
  currency: string;
  canStart: boolean;
  onStart: () => Promise<void>;
  onRestart: () => Promise<void>;
  onReturnToPortfolio: () => void;
  sendChoice: (choice: string) => Promise<boolean>;
  sendMessage: (message: string) => Promise<boolean>;
  sendAllocation: (
    allocation: Record<string, number>,
    currency?: CurrencyCode,
  ) => Promise<boolean>;
  analysis: AnalysisSessionState;
}

const LOADING_FRAMES = ["", ".", "..", "..."] as const;

function formatShares(value: number): string {
  if (!Number.isFinite(value)) {
    return "0";
  }
  if (Number.isInteger(value)) {
    return value.toLocaleString(undefined, { maximumFractionDigits: 0 });
  }
  return value.toLocaleString(undefined, {
    minimumFractionDigits: 2,
    maximumFractionDigits: 4,
  });
}

function formatCurrency(value: number, currency: string): string {
  const safeCurrency = currency && currency.length === 3 ? currency.toUpperCase() : "USD";
  return new Intl.NumberFormat(undefined, {
    style: "currency",
    currency: safeCurrency,
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  }).format(value);
}

function LiveStatusCard({
  message,
  animate,
}: {
  message: string | null;
  animate: boolean;
}) {
  const [frame, setFrame] = useState(0);

  useEffect(() => {
    if (!message || !animate) {
      setFrame(0);
      return;
    }
    const timer = window.setInterval(() => {
      setFrame((prev) => (prev + 1) % LOADING_FRAMES.length);
    }, 420);
    return () => window.clearInterval(timer);
  }, [animate, message]);

  if (!message) {
    return null;
  }

  const suffix = animate ? LOADING_FRAMES[frame] : "";

  return (
    <div className="flex justify-start">
      <div className="max-w-xl rounded-lg border border-indigo-200 bg-indigo-50 px-4 py-3 text-sm text-indigo-900 shadow dark:border-indigo-400/40 dark:bg-indigo-500/10 dark:text-indigo-100">
        <div className="mb-1 text-xs uppercase tracking-wide text-indigo-500 dark:text-indigo-300">Status</div>
        <p>{`${message}${suffix}`}</p>
      </div>
    </div>
  );
}

function ChoiceActions({
  prompt,
  choices,
  onSelect,
  disabled,
}: {
  prompt: string;
  choices: string[];
  onSelect: (value: string) => Promise<boolean>;
  disabled: boolean;
}) {
  const [busyChoice, setBusyChoice] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function handleSelect(choice: string) {
    if (disabled || busyChoice) {
      return;
    }
    setError(null);
    setBusyChoice(choice);
    const ok = await onSelect(choice);
    if (!ok) {
      setError("Unable to send your selection. Try again.");
    }
    setBusyChoice(null);
  }

  return (
    <div className="space-y-3">
      <p className="text-sm text-gray-600 dark:text-gray-300">{prompt}</p>
      <div className="flex flex-wrap gap-2">
        {choices.map((choice) => (
          <button
            key={choice}
            type="button"
            onClick={() => handleSelect(choice)}
            disabled={disabled || busyChoice !== null}
            className="rounded border border-indigo-200 px-3 py-2 text-sm font-medium capitalize text-indigo-700 transition hover:bg-indigo-50 disabled:cursor-not-allowed disabled:opacity-60 dark:border-indigo-400/40 dark:text-indigo-100 dark:hover:bg-indigo-500/10"
          >
            {busyChoice === choice ? "Sending..." : choice}
          </button>
        ))}
      </div>
      {error && <p className="text-sm text-red-600 dark:text-red-300">{error}</p>}
    </div>
  );
}

function MessageComposer({
  prompt,
  onSubmit,
  disabled,
}: {
  prompt: string;
  onSubmit: (value: string) => Promise<boolean>;
  disabled: boolean;
}) {
  const [draft, setDraft] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (disabled || busy) {
      return;
    }
    const trimmed = draft.trim();
    if (!trimmed) {
      setError("Enter a message first.");
      return;
    }
    setBusy(true);
    setError(null);
    const ok = await onSubmit(trimmed);
    if (ok) {
      setDraft("");
    } else {
      setError("Message failed to send. Please retry.");
    }
    setBusy(false);
  }

  return (
    <form onSubmit={handleSubmit} className="space-y-3">
      <p className="text-sm text-gray-600 dark:text-gray-300">{prompt}</p>
      <input
        type="text"
        value={draft}
        disabled={disabled || busy}
        onChange={(event: ChangeEvent<HTMLInputElement>) => setDraft(event.target.value)}
        className="w-full rounded border border-gray-300 bg-white px-3 py-2 text-sm shadow-sm focus:border-indigo-500 focus:outline-none focus:ring-1 focus:ring-indigo-500 disabled:cursor-not-allowed disabled:opacity-60 dark:border-gray-700 dark:bg-gray-900 dark:text-gray-100"
        placeholder="Tell MoneyBot how to adapt the analysis"
      />
      <div className="flex justify-end">
        <button
          type="submit"
          disabled={disabled || busy}
          className="rounded bg-indigo-600 px-3 py-2 text-sm font-medium text-white transition hover:bg-indigo-700 disabled:cursor-not-allowed disabled:opacity-60 dark:bg-indigo-500 dark:hover:bg-indigo-400"
        >
          {busy ? "Sending..." : "Send message"}
        </button>
      </div>
      {error && <p className="text-sm text-red-600 dark:text-red-300">{error}</p>}
    </form>
  );
}

function AllocationAdjuster({
  prompt,
  tickers,
  base,
  currency,
  disabled,
  onSubmit,
}: {
  prompt: string;
  tickers: string[];
  base: Record<string, number>;
  currency: string;
  disabled: boolean;
  onSubmit: (draft: Record<string, number>) => Promise<boolean>;
}) {
  const sortedTickers = useMemo(() => [...tickers].sort(), [tickers]);
  const [draft, setDraft] = useState<Record<string, number>>(() => {
    const initial: Record<string, number> = {};
    sortedTickers.forEach((ticker) => {
      initial[ticker] = Number(base[ticker] ?? 0);
    });
    return initial;
  });
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setDraft(() => {
      const next: Record<string, number> = {};
      sortedTickers.forEach((ticker) => {
        next[ticker] = Number(base[ticker] ?? 0);
      });
      return next;
    });
  }, [base, sortedTickers]);

  function handleChange(ticker: string, event: ChangeEvent<HTMLInputElement>) {
    const value = Number(event.target.value);
    if (!Number.isFinite(value) || value < 0) {
      return;
    }
    setDraft((prev) => ({ ...prev, [ticker]: value }));
  }

  function handleClear(ticker: string) {
    setDraft((prev) => ({ ...prev, [ticker]: 0 }));
  }

  function handleReset(ticker: string) {
    setDraft((prev) => ({ ...prev, [ticker]: Number(base[ticker] ?? 0) }));
  }

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (disabled || busy) {
      return;
    }
    setBusy(true);
    setError(null);
    const ok = await onSubmit(draft);
    if (!ok) {
      setError("Allocation update failed. Please retry.");
    }
    setBusy(false);
  }

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      <p className="text-sm text-gray-600 dark:text-gray-300">{prompt}</p>
      <div className="overflow-hidden rounded-lg border border-gray-200 shadow-sm dark:border-gray-700">
        <table className="min-w-full divide-y divide-gray-200 text-sm dark:divide-gray-700">
          <thead className="bg-gray-50 text-xs uppercase text-gray-500 dark:bg-gray-800 dark:text-gray-400">
            <tr>
              <th className="px-3 py-2 text-left font-medium">Ticker</th>
              <th className="px-3 py-2 text-right font-medium">Shares</th>
              <th className="px-3 py-2 text-right font-medium">Actions</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-200 dark:divide-gray-800">
            {sortedTickers.map((ticker) => (
              <tr key={ticker}>
                <td className="px-3 py-2 font-mono text-sm text-gray-800 dark:text-gray-100">{ticker}</td>
                <td className="px-3 py-2 text-right">
                  <input
                    type="number"
                    min={0}
                    step={0.0001}
                    value={draft[ticker] ?? 0}
                    onChange={(event: ChangeEvent<HTMLInputElement>) => handleChange(ticker, event)}
                    disabled={disabled || busy}
                    className="w-28 rounded border border-gray-300 bg-white px-2 py-1 text-right focus:border-indigo-500 focus:outline-none focus:ring-1 focus:ring-indigo-500 disabled:cursor-not-allowed disabled:opacity-60 dark:border-gray-700 dark:bg-gray-900 dark:text-gray-100"
                  />
                </td>
                <td className="px-3 py-2 text-right text-xs">
                  <div className="flex justify-end gap-2">
                    <button
                      type="button"
                      onClick={() => handleClear(ticker)}
                      disabled={disabled || busy}
                      className="rounded border border-gray-300 px-2 py-1 text-[11px] uppercase tracking-wide text-gray-600 transition hover:bg-gray-100 disabled:cursor-not-allowed disabled:opacity-60 dark:border-gray-600 dark:text-gray-300 dark:hover:bg-gray-800"
                    >
                      Clear
                    </button>
                    <button
                      type="button"
                      onClick={() => handleReset(ticker)}
                      disabled={disabled || busy}
                      className="rounded border border-indigo-300 px-2 py-1 text-[11px] uppercase tracking-wide text-indigo-600 transition hover:bg-indigo-50 disabled:cursor-not-allowed disabled:opacity-60 dark:border-indigo-400/40 dark:text-indigo-200 dark:hover:bg-indigo-500/10"
                    >
                      Reset
                    </button>
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="flex items-center justify-between text-xs text-gray-500 dark:text-gray-400">
        <span>Shares priced in {currency}</span>
        <button
          type="submit"
          disabled={disabled || busy}
          className="rounded bg-indigo-600 px-3 py-2 text-xs font-semibold uppercase tracking-wide text-white transition hover:bg-indigo-700 disabled:cursor-not-allowed disabled:opacity-60 dark:bg-indigo-500 dark:hover:bg-indigo-400"
        >
          {busy ? "Submitting..." : "Submit allocation"}
        </button>
      </div>
      {error && <p className="text-sm text-red-600 dark:text-red-300">{error}</p>}
    </form>
  );
}

function renderPromptResolution(message: Extract<ChatMessage, { role: "ai"; variant: "prompt" }>): JSX.Element {
  if (!message.resolved) {
    return <p className="text-sm text-gray-500 dark:text-gray-400">Awaiting your response…</p>;
  }

  if (message.promptType === "allocation" && message.responseAllocation) {
    const tickers = Object.keys(message.responseAllocation);
    return (
      <div className="space-y-3">
        <p className="text-sm text-gray-600 dark:text-gray-300">You submitted this allocation:</p>
        <div className="overflow-hidden rounded-lg border border-gray-200 shadow-sm dark:border-gray-700">
          <table className="min-w-full divide-y divide-gray-200 text-sm dark:divide-gray-700">
            <thead className="bg-gray-50 text-xs uppercase text-gray-500 dark:bg-gray-800 dark:text-gray-400">
              <tr>
                <th className="px-3 py-2 text-left font-medium">Ticker</th>
                <th className="px-3 py-2 text-right font-medium">Shares</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-200 dark:divide-gray-800">
              {tickers.map((ticker) => (
                <tr key={ticker}>
                  <td className="px-3 py-2 font-mono text-sm text-gray-800 dark:text-gray-100">{ticker}</td>
                  <td className="px-3 py-2 text-right text-sm text-gray-700 dark:text-gray-200">
                    {formatShares(message.responseAllocation?.[ticker] ?? 0)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    );
  }

  if (message.responseText) {
    return (
      <p className="text-sm text-gray-600 dark:text-gray-300">
        You responded: <span className="font-medium text-gray-900 dark:text-gray-100">{message.responseText}</span>
      </p>
    );
  }

  return <p className="text-sm text-gray-500 dark:text-gray-400">Response recorded.</p>;
}

function renderChatMessage(message: ChatMessage, activeAwaiting: AwaitingState | null): JSX.Element {
  if (message.role === "user") {
    return (
      <div key={message.id} className="flex justify-end">
        <div className={`max-w-xl rounded-lg bg-indigo-600 px-4 py-3 text-sm text-white shadow dark:bg-indigo-500 ${message.pending ? "opacity-90" : ""}`}>
          <p className="whitespace-pre-wrap leading-relaxed">{message.text}</p>
          {message.pending && (
            <span className="mt-2 block text-[11px] uppercase tracking-wide text-indigo-100/80">Sending…</span>
          )}
        </div>
      </div>
    );
  }

  if (message.variant === "text") {
    return (
      <div key={message.id} className="flex justify-start">
        <div className="max-w-xl rounded-lg border border-gray-200 bg-white px-4 py-3 text-sm text-gray-800 shadow dark:border-gray-700 dark:bg-gray-900/80 dark:text-gray-100">
          <p className="whitespace-pre-wrap leading-relaxed">{message.text}</p>
        </div>
      </div>
    );
  }

  if (message.variant === "allocation") {
    const tickers = Object.keys(message.allocation);
    return (
      <div key={message.id} className="flex justify-start">
        <div className="max-w-2xl space-y-4 rounded-lg border border-emerald-300 bg-emerald-50 px-4 py-3 text-sm text-emerald-900 shadow dark:border-emerald-400/50 dark:bg-emerald-500/10 dark:text-emerald-100">
          <div>
            <p className="text-sm font-semibold uppercase tracking-wide text-emerald-600 dark:text-emerald-300">Allocation Proposal</p>
            <p className="text-sm text-emerald-800 dark:text-emerald-100">Suggested allocation in shares ({message.currency}).</p>
          </div>
          <div className="overflow-hidden rounded-lg border border-emerald-200 bg-white shadow-sm dark:border-emerald-400/30 dark:bg-gray-900/80">
            <table className="min-w-full divide-y divide-emerald-100 text-sm text-emerald-900 dark:divide-emerald-400/30 dark:text-emerald-100">
              <thead className="bg-emerald-100 text-xs uppercase tracking-wide text-emerald-600 dark:bg-emerald-500/20 dark:text-emerald-200">
                <tr>
                  <th className="px-3 py-2 text-left font-semibold">Ticker</th>
                  <th className="px-3 py-2 text-right font-semibold">Shares</th>
                  {message.totals && <th className="px-3 py-2 text-right font-semibold">Value</th>}
                </tr>
              </thead>
              <tbody className="divide-y divide-emerald-100 dark:divide-emerald-400/30">
                {tickers.map((ticker) => (
                  <tr key={ticker}>
                    <td className="px-3 py-2 font-mono text-sm">{ticker}</td>
                    <td className="px-3 py-2 text-right">{formatShares(message.allocation[ticker] ?? 0)}</td>
                    {message.totals && (
                      <td className="px-3 py-2 text-right">
                        {formatCurrency(message.totals[ticker] ?? 0, message.currency)}
                      </td>
                    )}
                  </tr>
                ))}
              </tbody>
              {message.grandTotal !== undefined && (
                <tfoot>
                  <tr>
                    <td className="px-3 py-2 text-right font-semibold" colSpan={message.totals ? 2 : 1}>
                      Total
                    </td>
                    <td className="px-3 py-2 text-right font-semibold">
                      {formatCurrency(message.grandTotal ?? 0, message.currency)}
                    </td>
                  </tr>
                </tfoot>
              )}
            </table>
          </div>
        </div>
      </div>
    );
  }

  if (message.variant === "prompt") {
    const isActive = activeAwaiting?.messageId === message.id && !message.resolved;
    return (
      <div key={message.id} className="flex justify-start">
        <div
          className={`max-w-xl space-y-4 rounded-lg border px-4 py-3 text-sm shadow transition dark:bg-gray-900/80 ${
            isActive
              ? "border-indigo-300 bg-indigo-50 text-indigo-900 dark:border-indigo-400/50 dark:text-indigo-100"
              : "border-gray-200 bg-white text-gray-800 dark:border-gray-700 dark:text-gray-100"
          }`}
        >
          <div>
            <p className="text-xs font-semibold uppercase tracking-wide">Action Required</p>
            <p className="mt-1 whitespace-pre-wrap leading-relaxed">{message.prompt}</p>
          </div>
          {renderPromptResolution(message)}
        </div>
      </div>
    );
  }

  const unreachable: never = message;
  console.warn("Unhandled chat message", unreachable);
  return (
    <div className="flex justify-start">
      <div className="max-w-xl rounded-lg border border-gray-200 bg-white px-4 py-3 text-sm text-gray-800 shadow dark:border-gray-700 dark:bg-gray-900/80 dark:text-gray-100">
        <p className="whitespace-pre-wrap leading-relaxed">Unsupported message type.</p>
      </div>
    </div>
  );
}

export default function AnalysisRunner({
  portfolioId,
  currency,
  canStart,
  onStart,
  onRestart,
  onReturnToPortfolio,
  sendChoice,
  sendMessage,
  sendAllocation,
  analysis,
}: AnalysisRunnerProps) {
  const [starting, setStarting] = useState(false);
  const [restarting, setRestarting] = useState(false);
  const [restartError, setRestartError] = useState<string | null>(null);
  const conversationRef = useRef<HTMLDivElement | null>(null);

  // Auto-scroll when new messages arrive.
  useEffect(() => {
    const container = conversationRef.current;
    if (!container) {
      return;
    }
    container.scrollTo({ top: container.scrollHeight, behavior: "smooth" });
  }, [analysis.messages.length, analysis.statusMessage?.text]);

  async function handleStart() {
    if (!canStart || starting || restarting) {
      return;
    }
    setStarting(true);
    try {
      await onStart();
    } finally {
      setStarting(false);
    }
  }

  async function handleRestart() {
    if (restarting) {
      return;
    }
    setRestartError(null);
    setRestarting(true);
    try {
      await onRestart();
    } catch (error) {
      console.error("Failed to restart analysis", error);
      setRestartError("Unable to restart analysis. Try again in a moment.");
    } finally {
      setRestarting(false);
    }
  }

  function handleReturnToPortfolio() {
    onReturnToPortfolio();
  }

  const awaiting = analysis.awaiting;
  const orderedMessages = useMemo(
    () => [...analysis.messages].sort((a, b) => (a.clientSequence ?? 0) - (b.clientSequence ?? 0)),
    [analysis.messages]
  );
  const allocationCurrency = useMemo(
    () => (analysis.allocationCurrency || currency || "USD").toUpperCase(),
    [analysis.allocationCurrency, currency],
  );

  function renderAwaitingPanel() {
    if (!awaiting) {
      return null;
    }
    const disabled = analysis.status === "completed" || analysis.status === "error";

    switch (awaiting.kind) {
      case "choice":
        return (
          <ChoiceActions
            prompt={awaiting.prompt}
            choices={awaiting.choices}
            onSelect={sendChoice}
            disabled={disabled}
          />
        );
      case "message":
        return <MessageComposer prompt={awaiting.prompt} onSubmit={sendMessage} disabled={disabled} />;
      case "allocation":
        return (
          <AllocationAdjuster
            prompt={awaiting.prompt}
            tickers={
              awaiting.expectedKeys && awaiting.expectedKeys.length > 0
                ? awaiting.expectedKeys
                : Object.keys(analysis.allocation ?? {})
            }
            base={analysis.allocation ?? {}}
            currency={allocationCurrency}
            disabled={disabled}
            onSubmit={(draft) => sendAllocation(draft, allocationCurrency as CurrencyCode)}
          />
        );
      default:
        return null;
    }
  }

  const awaitingPanel = renderAwaitingPanel();
  const statusMessage = analysis.status === "completed" ? null : analysis.statusMessage?.text ?? null;
  const showStatusAnimation =
    analysis.status !== "completed" && (analysis.status === "running" || analysis.status === "awaiting_input");

  return (
    <section className="flex h-full flex-col gap-5 rounded-2xl border border-gray-200 bg-white/90 p-6 shadow-sm backdrop-blur dark:border-gray-800 dark:bg-gray-900/80">
      <header className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h2 className="text-lg font-semibold text-gray-800 dark:text-gray-100">Analysis Conversation</h2>
          <p className="text-sm text-gray-500 dark:text-gray-400">
            {portfolioId ? `Portfolio ${portfolioId}` : "Select a portfolio to begin."}
          </p>
        </div>
        <button
          type="button"
          onClick={handleStart}
          disabled={!canStart || starting || restarting}
          className="rounded bg-indigo-600 px-4 py-2 text-sm font-medium text-white transition hover:bg-indigo-700 disabled:cursor-not-allowed disabled:opacity-60 dark:bg-indigo-500 dark:hover:bg-indigo-400"
        >
          {starting ? "Starting..." : "Run analysis"}
        </button>
      </header>

      <div className="space-y-3">
        <div className="flex items-center gap-3 text-xs uppercase tracking-wide text-gray-500 dark:text-gray-400">
          <span className="rounded border border-gray-300 px-2 py-1 dark:border-gray-700">Status: {analysis.status}</span>
          <span className="text-[11px] lowercase text-gray-400 dark:text-gray-500">
            Allocations priced in {allocationCurrency}
          </span>
        </div>
        {analysis.error && (
          <div className="rounded-lg border border-red-300 bg-red-50 px-4 py-3 text-sm text-red-700 dark:border-red-500/50 dark:bg-red-500/10 dark:text-red-200">
            {analysis.error}
          </div>
        )}
        <LiveStatusCard message={statusMessage} animate={showStatusAnimation} />
      </div>

      <div
        ref={conversationRef}
        className="flex-1 space-y-4 overflow-y-auto rounded-xl border border-gray-200 bg-gray-50/60 p-4 dark:border-gray-800 dark:bg-gray-900/40"
      >
        {analysis.messages.length === 0 ? (
          <div className="flex h-full items-center justify-center">
            <p className="text-sm text-gray-500 dark:text-gray-400">Start an analysis to receive MoneyBot updates.</p>
          </div>
        ) : (
          orderedMessages.map((message) => renderChatMessage(message, awaiting ?? null))
        )}
      </div>

      {analysis.conversationClosed && (
        <div className="space-y-3 rounded-xl border border-emerald-300 bg-emerald-50 px-5 py-4 text-sm text-emerald-900 shadow dark:border-emerald-400/40 dark:bg-emerald-500/10 dark:text-emerald-100">
          <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
            <div>
              <p className="text-xs font-semibold uppercase tracking-wide text-emerald-600 dark:text-emerald-300">
                Analysis Complete
              </p>
              <p className="mt-1 text-sm text-emerald-900 dark:text-emerald-100">
                MoneyBot finished this review. Run it again or switch portfolios to explore another scenario.
              </p>
            </div>
            <div className="flex flex-wrap gap-2">
              <button
                type="button"
                onClick={handleRestart}
                disabled={restarting}
                className="rounded bg-emerald-600 px-4 py-2 text-sm font-medium text-white transition hover:bg-emerald-700 disabled:cursor-not-allowed disabled:opacity-60 dark:bg-emerald-500 dark:hover:bg-emerald-400"
              >
                {restarting ? "Restarting..." : "Run again for this portfolio"}
              </button>
              <button
                type="button"
                onClick={handleReturnToPortfolio}
                className="rounded border border-emerald-300 px-4 py-2 text-sm font-medium text-emerald-700 transition hover:bg-emerald-100 dark:border-emerald-400/40 dark:text-emerald-200 dark:hover:bg-emerald-500/10"
              >
                Choose different portfolio
              </button>
            </div>
          </div>
          {restartError && <p className="text-sm text-emerald-800/80 dark:text-emerald-200/80">{restartError}</p>}
        </div>
      )}

      {awaitingPanel && (
        <div className="rounded-2xl border border-indigo-200 bg-white/90 p-5 shadow-sm dark:border-indigo-400/40 dark:bg-gray-900/90">
          {awaitingPanel}
        </div>
      )}
    </section>
  );
}
