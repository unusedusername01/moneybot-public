export const RISK_TOLERANCE_OPTIONS = [
  { value: "low", label: "Low" },
  { value: "medium", label: "Medium" },
  { value: "high", label: "High" },
] as const;

export const CRITERIA_OPTIONS = [
  { value: "sector", label: "Sector" },
  { value: "market", label: "Market" },
  { value: "score", label: "Score" },
  { value: "market_cap", label: "Market Cap" },
] as const;

export const PREDICTION_STRENGTH_OPTIONS = [
  { value: "weak", label: "Weak" },
  { value: "medium", label: "Medium" },
  { value: "strong", label: "Strong" },
] as const;
