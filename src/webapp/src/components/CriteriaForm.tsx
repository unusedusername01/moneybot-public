import { ChangeEvent } from "react";
import type { PortfolioData } from "../types/portfolio";
import type { FieldStatus } from "../types/validation";
import { CRITERIA_OPTIONS, PREDICTION_STRENGTH_OPTIONS, RISK_TOLERANCE_OPTIONS } from "../constants/options";

interface CriteriaFormProps {
  value: PortfolioData;
  onChange: (changes: Partial<PortfolioData>) => void;
  currencyStatus: FieldStatus;
  currencyError: string | null;
  onValidateCurrency?: (currency: string) => Promise<boolean> | boolean;
}

export default function CriteriaForm({ value, onChange, currencyStatus, currencyError, onValidateCurrency }: CriteriaFormProps) {
  function handleChange<K extends keyof PortfolioData>(key: K, event: ChangeEvent<HTMLInputElement | HTMLSelectElement>) {
    const rawValue = event.target.value;
    let computed: PortfolioData[K];

    if (key === "budget") {
      computed = Number(rawValue) as PortfolioData[K];
    } else {
      computed = rawValue as PortfolioData[K];
    }

    onChange({ [key]: computed } as Partial<PortfolioData>);
  }

  return (
    <div className="rounded-2xl border border-gray-200 bg-white/90 p-6 shadow-sm backdrop-blur dark:border-gray-800 dark:bg-gray-900/80">
      <div className="flex items-start justify-between gap-3">
        <div>
          <h2 className="text-lg font-semibold text-gray-800 dark:text-gray-100">Investment profile</h2>
          <p className="mt-1 text-sm text-gray-600 dark:text-gray-400">
            Configure the capital, timeline, and model preferences MoneyBot uses while reasoning about allocations.
          </p>
        </div>
      </div>

      <div className="mt-6 grid grid-cols-1 gap-4">
        <label className="flex flex-col gap-1 text-sm">
          <span className="text-xs font-medium uppercase tracking-wide text-gray-500 dark:text-gray-400">Budget (in {value.currency || "—"})</span>
          <input
            type="number"
            min={0}
            step={1}
            value={value.budget ?? 0}
            onChange={(event: ChangeEvent<HTMLInputElement>) => handleChange("budget", event)}
            className="rounded border border-gray-300 bg-white px-3 py-2 shadow-sm focus:border-indigo-500 focus:outline-none focus:ring-1 focus:ring-indigo-500 dark:border-gray-700 dark:bg-gray-900"
          />
        </label>

        <label className="flex flex-col gap-1 text-sm">
          <span className="text-xs font-medium uppercase tracking-wide text-gray-500 dark:text-gray-400">Currency</span>
          <div className="flex items-center gap-2">
            <input
              value={value.currency ?? ""}
              onChange={(event: ChangeEvent<HTMLInputElement>) => handleChange("currency", event)}
              placeholder="USD"
              maxLength={3}
              className="flex-1 rounded border border-gray-300 bg-white px-3 py-2 uppercase shadow-sm focus:border-indigo-500 focus:outline-none focus:ring-1 focus:ring-indigo-500 dark:border-gray-700 dark:bg-gray-900"
            />
            <button
              type="button"
              onClick={() => onValidateCurrency && onValidateCurrency(value.currency ?? "")}
              className="rounded border border-gray-300 px-3 py-2 text-xs font-medium text-gray-700 transition hover:bg-gray-100 disabled:cursor-not-allowed disabled:opacity-50 dark:border-gray-700 dark:text-gray-200 dark:hover:bg-gray-700/40"
            >
              Validate
            </button>
          </div>
          {currencyStatus === "checking" && <span className="text-xs text-gray-500 dark:text-gray-400">Validating currency…</span>}
          {currencyStatus === "valid" && <span className="text-xs text-emerald-600 dark:text-emerald-300">Currency code confirmed</span>}
          {currencyStatus === "invalid" && (
            <span className="text-xs text-red-500 dark:text-red-300">{currencyError ?? "Invalid currency code"}</span>
          )}
          {currencyStatus === "idle" && !currencyError && (
            <span className="text-xs text-gray-500 dark:text-gray-400">ISO-4217 currency, e.g. USD, EUR, GBP.</span>
          )}
          {currencyError && currencyStatus !== "invalid" && (
            <span className="text-xs text-red-500 dark:text-red-300">{currencyError}</span>
          )}
        </label>

        <label className="flex flex-col gap-1 text-sm">
          <span className="text-xs font-medium uppercase tracking-wide text-gray-500 dark:text-gray-400">Target date</span>
          <input
            type="date"
            value={value.target_date ?? ""}
            onChange={(event: ChangeEvent<HTMLInputElement>) => handleChange("target_date", event)}
            className="rounded border border-gray-300 bg-white px-3 py-2 shadow-sm focus:border-indigo-500 focus:outline-none focus:ring-1 focus:ring-indigo-500 dark:border-gray-700 dark:bg-gray-900"
          />
        </label>

        <label className="flex flex-col gap-1 text-sm">
          <span className="text-xs font-medium uppercase tracking-wide text-gray-500 dark:text-gray-400">Risk tolerance</span>
          <select
            value={value.risk_tolerance ?? ""}
            onChange={(event: ChangeEvent<HTMLSelectElement>) => handleChange("risk_tolerance", event)}
            className="rounded border border-gray-300 bg-white px-3 py-2 shadow-sm focus:border-indigo-500 focus:outline-none focus:ring-1 focus:ring-indigo-500 dark:border-gray-700 dark:bg-gray-900"
          >
            <option value="">Select…</option>
            {RISK_TOLERANCE_OPTIONS.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
        </label>

        <label className="flex flex-col gap-1 text-sm">
          <span className="text-xs font-medium uppercase tracking-wide text-gray-500 dark:text-gray-400">Grouping criteria</span>
          <select
            value={value.criteria ?? ""}
            onChange={(event: ChangeEvent<HTMLSelectElement>) => handleChange("criteria", event)}
            className="rounded border border-gray-300 bg-white px-3 py-2 shadow-sm focus:border-indigo-500 focus:outline-none focus:ring-1 focus:ring-indigo-500 dark:border-gray-700 dark:bg-gray-900"
          >
            <option value="">Let MoneyBot decide</option>
            {CRITERIA_OPTIONS.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
        </label>

        <label className="flex flex-col gap-1 text-sm">
          <span className="text-xs font-medium uppercase tracking-wide text-gray-500 dark:text-gray-400">Prediction strength</span>
          <select
            value={value.prediction_strength ?? ""}
            onChange={(event: ChangeEvent<HTMLSelectElement>) => handleChange("prediction_strength", event)}
            className="rounded border border-gray-300 bg-white px-3 py-2 shadow-sm focus:border-indigo-500 focus:outline-none focus:ring-1 focus:ring-indigo-500 dark:border-gray-700 dark:bg-gray-900"
          >
            <option value="">Balanced</option>
            {PREDICTION_STRENGTH_OPTIONS.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
        </label>
      </div>
    </div>
  );
}
