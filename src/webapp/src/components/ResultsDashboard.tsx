import type { AnalysisSessionState, ChatMessage } from "../hooks/useAnalysisSession";

interface ResultsDashboardProps {
  analysis: AnalysisSessionState;
}

export default function ResultsDashboard({ analysis }: ResultsDashboardProps) {
  const allocationEntries = analysis.allocation ? Object.entries(analysis.allocation) : [];

  return (
    <div className="bg-white rounded-lg border shadow-sm p-4 space-y-4">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-semibold text-gray-800">Results</h2>
        <span className="text-xs uppercase tracking-wide text-gray-500">
          {analysis.status === "completed" ? "Completed" : "In progress"}
        </span>
      </div>

      {(() => {
        const lastState = [...analysis.messages].reverse().find((m) => m.role === "ai" && m.variant === "state") as (ChatMessage & { variant: "state" }) | undefined;
        if (!lastState) return null;
        return (
          <div className="rounded border border-dashed px-3 py-2 shadow-sm">
            <div className="text-xs uppercase tracking-wide text-gray-400">Last update</div>
            <p className="text-sm text-gray-700">{lastState.text}</p>
          </div>
        );
      })()}

      {allocationEntries.length > 0 ? (
        <div>
          <h3 className="text-sm font-semibold text-gray-700">Recommended allocation</h3>
          <div className="mt-2 grid grid-cols-1 gap-2">
            {allocationEntries.map(([ticker, shares]) => (
              <div key={ticker} className="flex items-center justify-between rounded border px-3 py-2 text-sm shadow-sm">
                <span className="font-mono text-base text-gray-800">{ticker}</span>
                <span className="text-gray-700">{shares}</span>
              </div>
            ))}
          </div>
        </div>
      ) : (
        <p className="text-sm text-gray-500">Run an analysis to view recommended allocations.</p>
      )}

      {analysis.messages.length > 0 && (
        <div className="space-y-2 text-xs text-gray-500">
          <div className="uppercase tracking-wide">Recent messages</div>
          <ul className="space-y-1">
            {[...analysis.messages].slice(-5).map((m) => (
              <li key={m.id} className="rounded border px-2 py-1">
                <span className="font-semibold text-gray-700">{m.role === "user" ? "You" : (m.variant === "state" ? "Update" : m.variant === "allocation" ? "Allocation" : "MoneyBot")}</span>
                {"text" in m && m.text ? (
                  <span className="ml-2 text-gray-600">{m.text}</span>
                ) : null}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
