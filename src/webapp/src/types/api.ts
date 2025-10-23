/**
 * This file was automatically generated from pydantic models by running pydantic2ts.
 * Do not modify it by hand - update the pydantic models and re-run the script.
 */

export type ValuesType = "shares" | "cash";
export type MessageType =
  | "state_update"
  | "awaiting_choice"
  | "awaiting_message"
  | "awaiting_allocation"
  | "choice_response"
  | "message_response"
  | "allocation_response"
  | "end_analysis"
  | "http_request"
  | "http_response";

export interface AllocationResponse {
  type?: "allocation_response";
  content: {
    [k: string]: number;
  };
  values?: ValuesType;
  currency?:
    | (
        | "AED"
        | "AFN"
        | "ALL"
        | "AMD"
        | "ANG"
        | "AOA"
        | "ARS"
        | "AUD"
        | "AWG"
        | "AZN"
        | "BAM"
        | "BBD"
        | "BDT"
        | "BGN"
        | "BHD"
        | "BIF"
        | "BMD"
        | "BND"
        | "BOB"
        | "BOV"
        | "BRL"
        | "BSD"
        | "BTN"
        | "BWP"
        | "BYN"
        | "BZD"
        | "CAD"
        | "CDF"
        | "CHE"
        | "CHF"
        | "CHW"
        | "CLF"
        | "CLP"
        | "CNY"
        | "COP"
        | "COU"
        | "CRC"
        | "CUC"
        | "CUP"
        | "CVE"
        | "CZK"
        | "DJF"
        | "DKK"
        | "DOP"
        | "DZD"
        | "EGP"
        | "ERN"
        | "ETB"
        | "EUR"
        | "FJD"
        | "FKP"
        | "GBP"
        | "GEL"
        | "GHS"
        | "GIP"
        | "GMD"
        | "GNF"
        | "GTQ"
        | "GYD"
        | "HKD"
        | "HNL"
        | "HRK"
        | "HTG"
        | "HUF"
        | "IDR"
        | "ILS"
        | "INR"
        | "IQD"
        | "IRR"
        | "ISK"
        | "JMD"
        | "JOD"
        | "JPY"
        | "KES"
        | "KGS"
        | "KHR"
        | "KMF"
        | "KPW"
        | "KRW"
        | "KWD"
        | "KYD"
        | "KZT"
        | "LAK"
        | "LBP"
        | "LKR"
        | "LRD"
        | "LSL"
        | "LYD"
        | "MAD"
        | "MDL"
        | "MGA"
        | "MKD"
        | "MMK"
        | "MNT"
        | "MOP"
        | "MRU"
        | "MUR"
        | "MVR"
        | "MWK"
        | "MXN"
        | "MXV"
        | "MYR"
        | "MZN"
        | "NAD"
        | "NGN"
        | "NIO"
        | "NOK"
        | "NPR"
        | "NZD"
        | "OMR"
        | "PAB"
        | "PEN"
        | "PGK"
        | "PHP"
        | "PKR"
        | "PLN"
        | "PYG"
        | "QAR"
        | "RON"
        | "RSD"
        | "RUB"
        | "RWF"
        | "SAR"
        | "SBD"
        | "SCR"
        | "SDG"
        | "SEK"
        | "SGD"
        | "SHP"
        | "SLE"
        | "SLL"
        | "SOS"
        | "SRD"
        | "SSP"
        | "STN"
        | "SVC"
        | "SYP"
        | "SZL"
        | "THB"
        | "TJS"
        | "TMT"
        | "TND"
        | "TOP"
        | "TRY"
        | "TTD"
        | "TWD"
        | "TZS"
        | "UAH"
        | "UGX"
        | "USD"
        | "USN"
        | "UYI"
        | "UYU"
        | "UYW"
        | "UZS"
        | "VED"
        | "VES"
        | "VND"
        | "VUV"
        | "WST"
        | "XAF"
        | "XCD"
        | "XOF"
        | "XPF"
        | "XSU"
        | "XUA"
        | "YER"
        | "ZAR"
        | "ZMW"
        | "ZWL"
      )
    | null;
}
export interface AwaitingAllocationRequest {
  type?: "awaiting_allocation";
  prompt: string;
  expected_keys?: string[] | null;
}
export interface AwaitingChoiceRequest {
  type?: "awaiting_choice";
  choices?: string[];
  prompt: string;
}
export interface AwaitingMessageRequest {
  type?: "awaiting_message";
  prompt: string;
}
export interface AwaitingSchema {
  type: MessageType;
}
/**
 * Base abstract message class.
 *
 * Messages are the inputs and outputs of ChatModels.
 */
export interface BaseMessage {
  content:
    | string
    | (
        | string
        | {
            [k: string]: unknown;
          }
      )[];
  additional_kwargs?: {
    [k: string]: unknown;
  };
  response_metadata?: {
    [k: string]: unknown;
  };
  type: string;
  name?: string | null;
  id?: string | null;
  [k: string]: unknown;
}
export interface ChoiceResponse {
  type?: "choice_response";
  selection: string;
}
export interface CurrencyValidationRequest {
  type?: "http_request";
  currency: string;
}
export interface DeletePortfolioRequest {
  type?: "http_request";
  portfolio_id: string;
}
export interface EditPortfolioRequest {
  type?: "http_request";
  portfolio_id: string;
  budget: number;
  target_date: string;
  holdings: {
    [k: string]: number;
  };
  currency: string;
  risk_tolerance?: string | null;
  criteria?: string | null;
  prediction_strength?: string | null;
}
export interface EndAnalysis {
  type?: "end_analysis";
}
export interface GetSharesPriceRequest {
  type?: "http_request";
  ticker: string | string[];
  amount: number | number[] | null;
  currency?:
    | (
        | "AED"
        | "AFN"
        | "ALL"
        | "AMD"
        | "ANG"
        | "AOA"
        | "ARS"
        | "AUD"
        | "AWG"
        | "AZN"
        | "BAM"
        | "BBD"
        | "BDT"
        | "BGN"
        | "BHD"
        | "BIF"
        | "BMD"
        | "BND"
        | "BOB"
        | "BOV"
        | "BRL"
        | "BSD"
        | "BTN"
        | "BWP"
        | "BYN"
        | "BZD"
        | "CAD"
        | "CDF"
        | "CHE"
        | "CHF"
        | "CHW"
        | "CLF"
        | "CLP"
        | "CNY"
        | "COP"
        | "COU"
        | "CRC"
        | "CUC"
        | "CUP"
        | "CVE"
        | "CZK"
        | "DJF"
        | "DKK"
        | "DOP"
        | "DZD"
        | "EGP"
        | "ERN"
        | "ETB"
        | "EUR"
        | "FJD"
        | "FKP"
        | "GBP"
        | "GEL"
        | "GHS"
        | "GIP"
        | "GMD"
        | "GNF"
        | "GTQ"
        | "GYD"
        | "HKD"
        | "HNL"
        | "HRK"
        | "HTG"
        | "HUF"
        | "IDR"
        | "ILS"
        | "INR"
        | "IQD"
        | "IRR"
        | "ISK"
        | "JMD"
        | "JOD"
        | "JPY"
        | "KES"
        | "KGS"
        | "KHR"
        | "KMF"
        | "KPW"
        | "KRW"
        | "KWD"
        | "KYD"
        | "KZT"
        | "LAK"
        | "LBP"
        | "LKR"
        | "LRD"
        | "LSL"
        | "LYD"
        | "MAD"
        | "MDL"
        | "MGA"
        | "MKD"
        | "MMK"
        | "MNT"
        | "MOP"
        | "MRU"
        | "MUR"
        | "MVR"
        | "MWK"
        | "MXN"
        | "MXV"
        | "MYR"
        | "MZN"
        | "NAD"
        | "NGN"
        | "NIO"
        | "NOK"
        | "NPR"
        | "NZD"
        | "OMR"
        | "PAB"
        | "PEN"
        | "PGK"
        | "PHP"
        | "PKR"
        | "PLN"
        | "PYG"
        | "QAR"
        | "RON"
        | "RSD"
        | "RUB"
        | "RWF"
        | "SAR"
        | "SBD"
        | "SCR"
        | "SDG"
        | "SEK"
        | "SGD"
        | "SHP"
        | "SLE"
        | "SLL"
        | "SOS"
        | "SRD"
        | "SSP"
        | "STN"
        | "SVC"
        | "SYP"
        | "SZL"
        | "THB"
        | "TJS"
        | "TMT"
        | "TND"
        | "TOP"
        | "TRY"
        | "TTD"
        | "TWD"
        | "TZS"
        | "UAH"
        | "UGX"
        | "USD"
        | "USN"
        | "UYI"
        | "UYU"
        | "UYW"
        | "UZS"
        | "VED"
        | "VES"
        | "VND"
        | "VUV"
        | "WST"
        | "XAF"
        | "XCD"
        | "XOF"
        | "XPF"
        | "XSU"
        | "XUA"
        | "YER"
        | "ZAR"
        | "ZMW"
        | "ZWL"
      )
    | null;
  allow_fractional?: boolean;
}
export interface GetSharesPriceResponse {
  type?: "http_response";
  err_code?: number;
  details?: string | null;
  ticker: string | string[];
  price?: number | number[] | null;
}
export interface HttpRequest {
  type?: "http_request";
}
export interface HttpResponse {
  type?: "http_response";
  err_code?: number;
  details?: string | null;
}
export interface ListPortfoliosResponse {
  type?: "http_response";
  err_code?: number;
  details?: string | null;
  portfolios: string[];
}
export interface LoadPortfolioDataResponse {
  type?: "http_response";
  err_code?: number;
  details?: string | null;
  portfolio_data: {
    [k: string]:
      | string
      | number
      | {
          [k: string]: number;
        }
      | null;
  };
}
export interface MessageData {
  type: MessageType;
}
export interface MessagePayload {
  data:
    | StateUpdate
    | AwaitingChoiceRequest
    | AwaitingMessageRequest
    | AwaitingAllocationRequest
    | ChoiceResponse
    | MessageResponse
    | AllocationResponse
    | EndAnalysis;
  timeout?: number | null;
}
export interface StateUpdate {
  type?: "state_update";
  prompt: string;
}
export interface MessageResponse {
  type?: "message_response";
  content: string;
}
export interface PortfolioDataRequest {
  type?: "http_request";
  portfolio_id: string;
}
