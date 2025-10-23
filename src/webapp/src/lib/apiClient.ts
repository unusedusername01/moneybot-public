import type {
  AllocationResponse,
  ChoiceResponse,
  CurrencyValidationRequest,
  DeletePortfolioRequest,
  EditPortfolioRequest,
  GetSharesPriceRequest,
  GetSharesPriceResponse,
  HttpResponse,
  ListPortfoliosResponse,
  LoadPortfolioDataResponse,
  MessagePayload,
  MessageResponse,
  MessageType,
  PortfolioDataRequest,
  StateUpdate,
} from "../types/api";
import type { PortfolioData, PortfolioId } from "../types/portfolio";

const DEFAULT_BASE_URL = "http://127.0.0.1:8000";

const BASE_URL = ((): string => {
  const url = (import.meta as ImportMeta & { env?: Record<string, string> }).env?.VITE_API_BASE_URL;
  if (typeof url === "string" && url.trim().length > 0) {
    return url.trim().replace(/\/$/, "");
  }
  return DEFAULT_BASE_URL;
})();

type ApiResult<T> = {
  success: true;
  data: T;
} | {
  success: false;
  status: number;
  message: string;
};

async function parseJson<T>(response: Response): Promise<T | null> {
  const text = await response.text();
  if (!text) {
    return null;
  }
  try {
    return JSON.parse(text) as T;
  } catch (error) {
    console.error("Failed to parse JSON", error);
    return null;
  }
}

function toPortfolioData(raw: LoadPortfolioDataResponse["portfolio_data"]): PortfolioData {
  const {
    budget,
    target_date,
    holdings,
    currency,
    risk_tolerance,
    criteria,
    prediction_strength,
  } = raw;

  return {
    budget: typeof budget === "number" ? budget : 0,
    target_date: typeof target_date === "string" ? target_date : "",
    holdings: (typeof holdings === "object" && holdings !== null) ? holdings as Record<string, number> : {},
    currency: typeof currency === "string" ? currency : "USD",
    risk_tolerance: typeof risk_tolerance === "string" ? risk_tolerance : null,
    criteria: typeof criteria === "string" ? criteria : null,
    prediction_strength: typeof prediction_strength === "string" ? prediction_strength : null,
  };
}

async function handleHttpResponse(response: Response): Promise<ApiResult<HttpResponse>> {
  const body = await parseJson<HttpResponse>(response);
  if (!body) {
    return {
      success: false,
      status: response.status,
      message: `${response.status} ${response.statusText}`,
    };
  }
  if (body.err_code && body.err_code !== 200) {
    return {
      success: false,
      status: body.err_code,
      message: body.details ?? "Unknown server error",
    };
  }
  return { success: true, data: body };
}

async function postJson(path: string, payload: unknown, init?: RequestInit): Promise<Response> {
  return fetch(`${BASE_URL}${path}`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
    ...init,
  });
}

export const Api = {
  async listPortfolios(): Promise<ApiResult<string[]>> {
    const response = await fetch(`${BASE_URL}/utils/list_portfolios`, {
      method: "GET",
    });
    const body = await parseJson<ListPortfoliosResponse | HttpResponse>(response);
    if (!body) {
      return { success: false, status: response.status, message: "Empty response" };
    }
    if ("portfolios" in body) {
      return { success: true, data: body.portfolios };
    }
    const errCode = body.err_code ?? response.status;
    return {
      success: false,
      status: errCode,
      message: body.details ?? "Failed to list portfolios",
    };
  },

  async loadPortfolioData(portfolioId: PortfolioId): Promise<ApiResult<PortfolioData>> {
    const request: PortfolioDataRequest = {
      type: "http_request",
      portfolio_id: portfolioId,
    };
    const response = await postJson("/utils/load_portfolio_data", request);
    const body = await parseJson<LoadPortfolioDataResponse | HttpResponse>(response);
    if (!body) {
      return { success: false, status: response.status, message: "Empty response" };
    }
    if ("portfolio_data" in body) {
      return { success: true, data: toPortfolioData(body.portfolio_data) };
    }
    const errCode = body.err_code ?? response.status;
    return {
      success: false,
      status: errCode,
      message: body.details ?? "Failed to load portfolio",
    };
  },

  async savePortfolio(portfolioId: PortfolioId, data: PortfolioData): Promise<ApiResult<HttpResponse>> {
    const request: EditPortfolioRequest = {
      type: "http_request",
      portfolio_id: portfolioId,
      budget: data.budget,
      target_date: data.target_date,
      holdings: data.holdings,
      currency: data.currency,
      risk_tolerance: data.risk_tolerance ?? null,
      criteria: data.criteria ?? null,
      prediction_strength: data.prediction_strength ?? null,
    };
    const response = await postJson("/utils/edit_portfolio", request);
    return handleHttpResponse(response);
  },

  async createPortfolio(portfolioId: PortfolioId, data: PortfolioData): Promise<ApiResult<HttpResponse>> {
    const request: EditPortfolioRequest = {
      type: "http_request",
      portfolio_id: portfolioId,
      budget: data.budget,
      target_date: data.target_date,
      holdings: data.holdings,
      currency: data.currency,
      risk_tolerance: data.risk_tolerance ?? null,
      criteria: data.criteria ?? null,
      prediction_strength: data.prediction_strength ?? null,
    };
    const response = await postJson("/utils/create_portfolio", request);
    return handleHttpResponse(response);
  },

  async deletePortfolio(portfolioId: PortfolioId): Promise<ApiResult<HttpResponse>> {
    const request: DeletePortfolioRequest = {
      type: "http_request",
      portfolio_id: portfolioId,
    };
    const response = await postJson("/utils/delete_portfolio", request);
    return handleHttpResponse(response);
  },

  async validateCurrency(currency: string): Promise<ApiResult<HttpResponse>> {
    const request: CurrencyValidationRequest = {
      type: "http_request",
      currency,
    };
    const response = await postJson("/utils/validate_currency", request);
    return handleHttpResponse(response);
  },

  async getSharesPrice(request: GetSharesPriceRequest): Promise<ApiResult<GetSharesPriceResponse>> {
    const response = await postJson("/utils/get_shares_price", request);
    const body = await parseJson<GetSharesPriceResponse | HttpResponse>(response);
    if (!body) {
      return { success: false, status: response.status, message: "Empty response" };
    }
    if ("price" in body) {
      return { success: true, data: body };
    }
    const errCode = body.err_code ?? response.status;
    return {
      success: false,
      status: errCode,
      message: body.details ?? "Failed to fetch share price",
    };
  },

  async startAnalysis(portfolioId: PortfolioId): Promise<ApiResult<HttpResponse>> {
    const response = await postJson(`/analysis/portfolio/${encodeURIComponent(portfolioId)}/start`, {});
    if (response.status === 405) {
      // fallback to GET for compatibility with older clients
      const fallback = await fetch(`${BASE_URL}/analysis/portfolio/${encodeURIComponent(portfolioId)}/start`, {
        method: "GET",
      });
      return handleHttpResponse(fallback);
    }
    return handleHttpResponse(response);
  },

  async sendAnalysisResponse(portfolioId: PortfolioId, payload: MessagePayload): Promise<ApiResult<HttpResponse>> {
    const payloadJson = JSON.stringify(payload);
    const response = await postJson(`/analysis/portfolio/${encodeURIComponent(portfolioId)}/respond`, payloadJson);
    return handleHttpResponse(response);
  },
};

export { BASE_URL };
export type { MessagePayload, MessageType, MessageResponse, ChoiceResponse, AllocationResponse, StateUpdate };
