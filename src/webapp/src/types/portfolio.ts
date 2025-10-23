export type PortfolioId = string;

export interface PortfolioData {
  budget: number;
  target_date: string;
  holdings: Record<string, number>;
  currency: string;
  risk_tolerance?: string | null;
  criteria?: string | null;
  prediction_strength?: string | null;
}

export interface HoldingEntry {
  ticker: string;
  shares: number;
}
