export default function AnalysisHistory() {
  return (
    <div className="rounded-2xl border border-gray-200 bg-white/90 p-6 text-sm text-gray-600 shadow-sm backdrop-blur dark:border-gray-800 dark:bg-gray-900/80 dark:text-gray-300">
      <h2 className="text-lg font-semibold text-gray-800 dark:text-gray-100">History</h2>
      <p className="mt-2">
        Historical runs will appear here when the backend exposes an endpoint for session archives.
      </p>
    </div>
  );
}
