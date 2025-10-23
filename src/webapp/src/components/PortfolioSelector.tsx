import type { PortfolioId } from "../types/portfolio";

interface PortfolioSelectorProps {
  portfolios: PortfolioId[];
  selected: PortfolioId | null;
  dirty: boolean;
  saving: boolean;
  busy: boolean;
  pendingCreate: boolean;
  onSelect: (portfolioId: PortfolioId) => void;
  onSave: () => Promise<boolean> | boolean;
  onClear: () => void;
  onCreateNew: () => void;
  onDelete: (portfolioId: PortfolioId) => Promise<void> | void;
}

export default function PortfolioSelector({
  portfolios,
  selected,
  dirty,
  saving,
  busy,
  pendingCreate,
  onSelect,
  onSave,
  onClear,
  onCreateNew,
  onDelete,
}: PortfolioSelectorProps) {
  async function handleSave() {
    try {
      await onSave();
    } catch (error) {
      console.error("Failed to save portfolio", error);
    }
  }

  async function handleDelete() {
    if (!selected) {
      return;
    }
    await onDelete(selected);
  }

  return (
    <div className="flex flex-col gap-4">
      <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
        <div className="flex flex-wrap items-center gap-3">
          <label className="text-sm font-medium text-gray-700 dark:text-gray-200" htmlFor="portfolio-select">
            Portfolio
          </label>
          <select
            id="portfolio-select"
            value={selected ?? ""}
            onChange={(event) => {
              const value = event.target.value as PortfolioId;
              if (!value) {
                return;
              }
              onSelect(value);
            }}
            disabled={busy}
            className="rounded border border-gray-300 bg-white px-3 py-2 text-sm shadow-sm focus:border-indigo-500 focus:outline-none focus:ring-1 focus:ring-indigo-500 dark:border-gray-700 dark:bg-gray-900"
          >
            <option value="" disabled>
              Select a portfolio
            </option>
            {portfolios.map((portfolio) => (
              <option key={portfolio} value={portfolio}>
                {portfolio}
              </option>
            ))}
          </select>
          <button
            type="button"
            onClick={onCreateNew}
            disabled={busy}
            className="rounded border border-indigo-200 px-3 py-2 text-sm font-medium text-indigo-600 transition hover:bg-indigo-50 disabled:cursor-not-allowed disabled:opacity-50 dark:border-indigo-400/40 dark:text-indigo-300 dark:hover:bg-indigo-500/10"
          >
            New portfolio
          </button>
          <button
            type="button"
            onClick={handleDelete}
            disabled={!selected || busy}
            className="rounded border border-red-200 px-3 py-2 text-sm font-medium text-red-600 transition hover:bg-red-50 disabled:cursor-not-allowed disabled:opacity-50 dark:border-red-400/40 dark:text-red-300 dark:hover:bg-red-500/10"
          >
            Delete
          </button>
        </div>

        <div className="flex items-center gap-3 text-xs">
          {pendingCreate && (
            <span className="rounded-full border border-amber-300 bg-amber-50 px-3 py-1 font-medium text-amber-700 dark:border-amber-400/40 dark:bg-amber-500/10 dark:text-amber-200">
              Draft portfolio — save to keep
            </span>
          )}
          {!pendingCreate && dirty && (
            <span className="rounded-full border border-amber-300 bg-amber-50 px-3 py-1 font-medium text-amber-700 dark:border-amber-400/40 dark:bg-amber-500/10 dark:text-amber-200">
              Unsaved changes
            </span>
          )}
          {!pendingCreate && !dirty && selected && (
            <span className="rounded-full border border-emerald-200 bg-emerald-50 px-3 py-1 font-medium text-emerald-600 dark:border-emerald-400/40 dark:bg-emerald-500/10 dark:text-emerald-200">
              Saved
            </span>
          )}
        </div>
      </div>

      <div className="flex flex-wrap items-center gap-2">
        <button
          type="button"
          onClick={handleSave}
          disabled={busy || (!dirty && !pendingCreate) || saving || !selected}
          className="rounded bg-emerald-600 px-4 py-2 text-sm font-medium text-white shadow transition hover:bg-emerald-700 disabled:cursor-not-allowed disabled:opacity-60 dark:bg-emerald-500 dark:hover:bg-emerald-400"
        >
          {saving ? "Saving..." : pendingCreate ? "Save new portfolio" : dirty ? "Save changes" : "Saved"}
        </button>
        <button
          type="button"
          onClick={onClear}
          disabled={busy}
          className="rounded border border-gray-300 px-4 py-2 text-sm font-medium text-gray-700 transition hover:bg-gray-100 disabled:cursor-not-allowed disabled:opacity-50 dark:border-gray-600 dark:text-gray-200 dark:hover:bg-gray-700/40"
        >
          Clear fields
        </button>
      </div>
    </div>
  );
}
