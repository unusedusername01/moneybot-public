import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  Api,
  type AllocationResponse,
  type ChoiceResponse,
  type MessagePayload,
  type MessageResponse,
  type MessageType,
  type StateUpdate,
} from "../lib/apiClient";
import type {
  AwaitingAllocationRequest,
  AwaitingChoiceRequest,
  AwaitingMessageRequest,
  HttpResponse,
} from "../types/api";
import type { PortfolioId } from "../types/portfolio";

export type AnalysisStatus =
  | "idle"
  | "connecting"
  | "running"
  | "awaiting_input"
  | "completed"
  | "error";

export type AwaitingState =
  | { kind: "choice"; messageId: string; prompt: string; choices: string[] }
  | { kind: "message"; messageId: string; prompt: string }
  | { kind: "allocation"; messageId: string; prompt: string; expectedKeys?: string[] | null }
  | null;

export type CurrencyCode = NonNullable<AllocationResponse["currency"]>;

export interface StatusMessage {
  text: string;
  timestamp: number;
}

type BaseChatMessage = {
  id: string;
  timestamp: number;
  clientSequence: number;
  pending?: boolean;
};

export type ChatMessage =
  | (BaseChatMessage & {
      role: "ai";
      variant: "text";
      text: string;
    })
  | (BaseChatMessage & {
      role: "ai";
      variant: "allocation";
      allocation: AllocationResponse["content"];
      currency: string;
      totals?: Record<string, number>;
      grandTotal?: number;
    })
  | (BaseChatMessage & {
      role: "ai";
      variant: "prompt";
      promptType: "choice" | "message" | "allocation";
      prompt: string;
      choices?: string[];
      expectedKeys?: string[] | null;
      resolved: boolean;
      responseText?: string;
      responseAllocation?: Record<string, number>;
    })
  | (BaseChatMessage & {
      role: "user";
      variant: "text";
      text: string;
    });

export interface AnalysisSessionState {
  status: AnalysisStatus;
  awaiting: AwaitingState;
  statusMessage: StatusMessage | null;
  messages: ChatMessage[];
  allocation: AllocationResponse["content"] | null;
  allocationCurrency: CurrencyCode;
  error: string | null;
  conversationClosed: boolean;
}

type AllocationEnrichmentTask = {
  allocation: AllocationResponse["content"];
  currency: CurrencyCode;
  messageId: string;
};

const WS_RECONNECT_INTERVAL = 8000;

function createWebSocketUrl(baseUrl: string, portfolioId: PortfolioId): string {
  const normalized = baseUrl.replace(/\/$/, "");
  const protocol = normalized.startsWith("https") ? "wss" : "ws";
  const remainder = normalized.replace(/^https?:\/\//, "");
  return `${protocol}://${remainder}/ws/portfolio/${encodeURIComponent(portfolioId)}`;
}

function parseWebSocketData(message: unknown): MessagePayload | null {
  if (typeof message !== "string") {
    return null;
  }
  try {
    const firstPass = JSON.parse(message);
    if (typeof firstPass === "string") {
      return JSON.parse(firstPass) as MessagePayload;
    }
    if (firstPass && typeof firstPass === "object" && "data" in firstPass) {
      return firstPass as MessagePayload;
    }
  } catch (error) {
    console.error("Failed to parse WebSocket payload", error);
  }
  return null;
}

function createMessageId(prefix: string) {
  return `${prefix}-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

const initialState: AnalysisSessionState = {
  status: "idle",
  awaiting: null,
  statusMessage: null,
  messages: [],
  allocation: null,
  allocationCurrency: "USD",
  error: null,
  conversationClosed: false,
};

export function useAnalysisSession(portfolioId: PortfolioId | null, apiBaseUrl: string): {
  state: AnalysisSessionState;
  start: () => Promise<boolean>;
  sendChoice: (choice: string) => Promise<boolean>;
  sendMessage: (message: string) => Promise<boolean>;
  sendAllocation: (allocation: AllocationResponse["content"], currency?: CurrencyCode) => Promise<boolean>;
  reset: () => void;
} {
  const [state, setState] = useState<AnalysisSessionState>(initialState);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const stateRef = useRef(state);
  const messageSequenceRef = useRef(0);

  useEffect(() => {
    stateRef.current = state;
  }, [state]);

  const nextClientSequence = useCallback(() => {
    messageSequenceRef.current += 1;
    return messageSequenceRef.current;
  }, []);

  const clearReconnectTimer = useCallback(() => {
    if (reconnectRef.current) {
      clearTimeout(reconnectRef.current);
      reconnectRef.current = null;
    }
  }, []);

  const closeWebSocket = useCallback(() => {
    clearReconnectTimer();
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
  }, [clearReconnectTimer]);

  useEffect(() => closeWebSocket, [closeWebSocket]);

  const applyAllocationEnrichment = useCallback((messageId: string, totals: Record<string, number>) => {
    setState((prev) => {
      const messages = prev.messages.map((message) => {
        if (message.id === messageId && message.role === "ai" && message.variant === "allocation") {
          const grandTotal = Object.values(totals).reduce((acc, value) => acc + value, 0);
          return {
            ...message,
            totals,
            grandTotal,
          };
        }
        return message;
      });
      return {
        ...prev,
        messages,
      };
    });
  }, []);

  const handlePriceEnrichment = useCallback(
    async (allocation: AllocationResponse["content"], currency: CurrencyCode, messageId: string) => {
      const tickers = Object.keys(allocation);
      const amounts = Object.values(allocation);
      if (tickers.length === 0) {
        return;
      }
      const result = await Api.getSharesPrice({
        type: "http_request",
        ticker: tickers,
        amount: amounts,
        currency,
        allow_fractional: true,
      });
      if (!result.success) {
        console.warn("Unable to enrich allocation with prices", result.message);
        return;
      }
      const data = result.data.price;
      const totals: Record<string, number> = {};
      if (Array.isArray(data)) {
        data.forEach((value, index) => {
          totals[tickers[index]] = typeof value === "number" ? value : 0;
        });
      } else if (typeof data === "number") {
        totals[tickers[0]] = data;
      }
      applyAllocationEnrichment(messageId, totals);
    },
    [applyAllocationEnrichment]
  );

  const handlePayload = useCallback(
    (payload: MessagePayload) => {
      const type = payload.data.type as MessageType;
      let allocationForEnrichment: {
        allocation: AllocationResponse["content"];
        currency: CurrencyCode;
        messageId: string;
      } | null = null;

      setState((prev) => {
        const now = Date.now();
        const defaultStatus = prev.status === "idle" ? "running" : prev.status;
        let status = defaultStatus;
        let awaiting = prev.awaiting;
        let allocation = prev.allocation;
        let allocationCurrency = prev.allocationCurrency;
        let error = prev.error;
        let statusMessage = prev.statusMessage;
        let messages = prev.messages;
        let conversationClosed = prev.conversationClosed;

        switch (type) {
          case "state_update": {
            const prompt = (payload.data as StateUpdate).prompt;
            status = "running";
            awaiting = null;
            conversationClosed = false;
            statusMessage = {
              text: prompt,
              timestamp: now,
            };

            const emphasizePrompts = new Set([
              "Analysis started",
              "Analysis restarted",
              "Logging interaction into the database",
            ]);
            if (emphasizePrompts.has(prompt)) {
              const alreadyLogged = prev.messages.some(
                (message) => message.role === "ai" && message.variant === "text" && message.text === prompt
              );
              if (!alreadyLogged) {
                messages = [
                  ...messages,
                  {
                    id: createMessageId("status"),
                    role: "ai",
                    variant: "text",
                    text: prompt,
                    timestamp: now,
                    clientSequence: nextClientSequence(),
                  },
                ];
              }
            }
            break;
          }
          case "awaiting_choice": {
            const data = payload.data as AwaitingChoiceRequest;
            const messageId = createMessageId("prompt-choice");
            const messageSequence = nextClientSequence();
            messages = [
              ...messages,
              {
                id: messageId,
                role: "ai",
                variant: "prompt",
                promptType: "choice",
                prompt: data.prompt,
                choices: data.choices ?? ["confirm", "edit", "deny", "cancel"],
                resolved: false,
                timestamp: now,
                clientSequence: messageSequence,
              },
            ];
            awaiting = {
              kind: "choice",
              messageId,
              prompt: data.prompt,
              choices: data.choices ?? ["confirm", "edit", "deny", "cancel"],
            };
            status = "awaiting_input";
            statusMessage = {
              text: data.prompt,
              timestamp: now,
            };
            break;
          }
          case "awaiting_message": {
            const data = payload.data as AwaitingMessageRequest;
            const messageId = createMessageId("prompt-message");
            const messageSequence = nextClientSequence();
            messages = [
              ...messages,
              {
                id: messageId,
                role: "ai",
                variant: "prompt",
                promptType: "message",
                prompt: data.prompt,
                resolved: false,
                timestamp: now,
                clientSequence: messageSequence,
              },
            ];
            awaiting = {
              kind: "message",
              messageId,
              prompt: data.prompt,
            };
            status = "awaiting_input";
            statusMessage = {
              text: data.prompt,
              timestamp: now,
            };
            break;
          }
          case "awaiting_allocation": {
            const data = payload.data as AwaitingAllocationRequest;
            const messageId = createMessageId("prompt-allocation");
            const messageSequence = nextClientSequence();
            messages = [
              ...messages,
              {
                id: messageId,
                role: "ai",
                variant: "prompt",
                promptType: "allocation",
                prompt: data.prompt,
                expectedKeys: data.expected_keys,
                resolved: false,
                timestamp: now,
                clientSequence: messageSequence,
              },
            ];
            awaiting = {
              kind: "allocation",
              messageId,
              prompt: data.prompt,
              expectedKeys: data.expected_keys,
            };
            status = "awaiting_input";
            statusMessage = {
              text: data.prompt,
              timestamp: now,
            };
            break;
          }
          case "allocation_response": {
            const data = payload.data as AllocationResponse;
            const messageId = createMessageId("allocation");
            const messageSequence = nextClientSequence();
            allocation = data.content;
            allocationCurrency = data.currency ?? prev.allocationCurrency ?? "USD";
            status = "running";
            awaiting = null;
            conversationClosed = false;
            messages = [
              ...messages,
              {
                id: messageId,
                role: "ai",
                variant: "allocation",
                allocation: data.content,
                currency: allocationCurrency,
                timestamp: now,
                clientSequence: messageSequence,
              },
            ];
            statusMessage = {
              text: "Allocation proposal ready for review.",
              timestamp: now,
            };
            allocationForEnrichment = {
              allocation: data.content,
              currency: allocationCurrency,
              messageId,
            };
            break;
          }
          case "message_response": {
            const data = payload.data as MessageResponse;
            messages = [
              ...messages,
              {
                id: createMessageId("ai-text"),
                role: "ai",
                variant: "text",
                text: data.content,
                timestamp: now,
                clientSequence: nextClientSequence(),
              },
            ];
            status = defaultStatus;
            awaiting = null;
            conversationClosed = false;
            break;
          }
          case "end_analysis": {
            status = "completed";
            awaiting = null;
            conversationClosed = true;
            messages = [
              ...messages,
              {
                id: createMessageId("end"),
                role: "ai",
                variant: "text",
                text: "Analysis complete. Review the recommendations or restart when ready.",
                timestamp: now,
                clientSequence: nextClientSequence(),
              },
            ];
            statusMessage = {
              text: "Analysis complete.",
              timestamp: now,
            };
            break;
          }
          case "http_response": {
            const http = payload.data as HttpResponse;
            if (http.err_code && http.err_code !== 200) {
              status = "error";
              error = http.details ?? `Server returned ${http.err_code}`;
            }
            break;
          }
          default:
            break;
        }

        return {
          ...prev,
          status,
          awaiting,
          allocation,
          allocationCurrency,
          statusMessage,
          error,
          messages,
          conversationClosed,
        };
      });

      const task = allocationForEnrichment as AllocationEnrichmentTask | null;
      if (task) {
        void handlePriceEnrichment(task.allocation, task.currency, task.messageId);
      }
    },
    [handlePriceEnrichment, nextClientSequence]
  );

  const openWebSocket = useCallback(async () => {
    if (!portfolioId) {
      throw new Error("No portfolio selected");
    }
    closeWebSocket();
    const wsUrl = createWebSocketUrl(apiBaseUrl, portfolioId);
    const ws = new WebSocket(wsUrl);
    wsRef.current = ws;

    messageSequenceRef.current = 0;

    setState((prev) => ({
      ...prev,
      status: "connecting",
      awaiting: null,
      allocation: null,
      allocationCurrency: "USD",
      statusMessage: null,
      error: null,
      messages: [],
      conversationClosed: false,
    }));

    ws.onerror = () => {
      setState((prev) => ({ ...prev, status: "error", error: "WebSocket error" }));
    };

    ws.onclose = () => {
      wsRef.current = null;
      setState((prev) => {
        if (prev.status === "completed" || prev.status === "error") {
          return prev;
        }
        // If we were running an analysis and it closed unexpectedly, treat it as an error
        if (prev.status === "running" || prev.status === "awaiting_input") {
          return {
            ...prev,
            status: "error",
            error: "Connection closed unexpectedly. The analysis may have encountered an error.",
          };
        }
        // Otherwise attempt reconnection
        reconnectRef.current = setTimeout(() => {
          openWebSocket().catch((error: unknown) => {
            console.error("Failed to reopen WebSocket", error);
          });
        }, WS_RECONNECT_INTERVAL);
        return { ...prev, status: "idle" };
      });
    };

    ws.onmessage = (event) => {
      const payload = parseWebSocketData(event.data);
      if (payload) {
        handlePayload(payload);
      }
    };

    await new Promise<void>((resolve, reject) => {
      ws.onopen = () => {
        setState((prev) => ({ ...prev, status: "running" }));
        resolve();
      };
      ws.onerror = (event) => {
        reject(event);
      };
    });
  }, [apiBaseUrl, closeWebSocket, handlePayload, portfolioId]);

  const start = useCallback(async () => {
    if (!portfolioId) {
      setState((prev) => ({ ...prev, error: "Select or create a portfolio first" }));
      return false;
    }
    try {
      await openWebSocket();
      const response = await Api.startAnalysis(portfolioId);
      if (!response.success) {
        setState((prev) => ({
          ...prev,
          status: "error",
          error: response.message,
        }));
        return false;
      }
      return true;
    } catch (error) {
      console.error("Failed to start analysis", error);
      setState((prev) => ({ ...prev, status: "error", error: "Failed to start analysis" }));
      return false;
    }
  }, [openWebSocket, portfolioId]);

  const sendChoice = useCallback(
    async (choice: string) => {
      if (!portfolioId) {
        throw new Error("No portfolio selected");
      }

      const currentState = stateRef.current;
      if (!currentState.awaiting || currentState.awaiting.kind !== "choice") {
        return false;
      }
      const awaitingPrompt = currentState.awaiting;
      if (currentState.messages.some((message) => message.pending)) {
        return false;
      }

      const timestamp = Date.now();
      const userMessageId = createMessageId("user");
      const optimisticMessage: ChatMessage = {
        id: userMessageId,
        role: "user",
        variant: "text",
        text: choice,
        timestamp,
        clientSequence: nextClientSequence(),
        pending: true,
      };

      setState((prev) => {
        const awaiting = prev.awaiting;
        const updatedMessages = prev.messages.map((message) => {
          if (
            awaiting &&
            awaiting.kind === "choice" &&
            message.id === awaiting.messageId &&
            message.role === "ai" &&
            message.variant === "prompt" &&
            message.promptType === "choice"
          ) {
            return {
              ...message,
              resolved: true,
              responseText: choice,
            };
          }
          return message;
        });

        return {
          ...prev,
          status: "running",
          awaiting: null,
          error: null,
          messages: [...updatedMessages, optimisticMessage],
          conversationClosed: false,
        };
      });

      const payload: MessagePayload = {
        data: {
          type: "choice_response",
          selection: choice,
        } as ChoiceResponse,
      };

      const result = await Api.sendAnalysisResponse(portfolioId, payload);
      if (!result.success) {
        setState((prev) => {
          const revertedMessages = prev.messages
            .filter((message) => message.id !== userMessageId)
            .map((message) => {
              if (
                message.id === awaitingPrompt.messageId &&
                message.role === "ai" &&
                message.variant === "prompt" &&
                message.promptType === "choice"
              ) {
                return {
                  ...message,
                  resolved: false,
                  responseText: undefined,
                };
              }
              return message;
            });

          return {
            ...prev,
            status: "awaiting_input",
            awaiting: awaitingPrompt,
            error: result.message,
            messages: revertedMessages,
            statusMessage: currentState.statusMessage,
          };
        });
        return false;
      }

      setState((prev) => ({
        ...prev,
        messages: prev.messages.map((message) =>
          message.id === userMessageId
            ? {
                ...message,
                pending: false,
              }
            : message
        ),
      }));

      return true;
    },
    [nextClientSequence, portfolioId]
  );

  const sendMessage = useCallback(
    async (message: string) => {
      if (!portfolioId) {
        throw new Error("No portfolio selected");
      }
      const currentState = stateRef.current;
      if (!currentState.awaiting || currentState.awaiting.kind !== "message") {
        return false;
      }
      const awaitingPrompt = currentState.awaiting;
      if (currentState.messages.some((entry) => entry.pending)) {
        return false;
      }

      const timestamp = Date.now();
      const userMessageId = createMessageId("user");
      const optimisticMessage: ChatMessage = {
        id: userMessageId,
        role: "user",
        variant: "text",
        text: message,
        timestamp,
        clientSequence: nextClientSequence(),
        pending: true,
      };

      setState((prev) => {
        const awaiting = prev.awaiting;
        const updatedMessages = prev.messages.map((entry) => {
          if (
            awaiting &&
            awaiting.kind === "message" &&
            entry.id === awaiting.messageId &&
            entry.role === "ai" &&
            entry.variant === "prompt" &&
            entry.promptType === "message"
          ) {
            return {
              ...entry,
              resolved: true,
              responseText: message,
            };
          }
          return entry;
        });

        return {
          ...prev,
          status: "running",
          awaiting: null,
          error: null,
          messages: [...updatedMessages, optimisticMessage],
          conversationClosed: false,
        };
      });

      const payload: MessagePayload = {
        data: {
          type: "message_response",
          content: message,
        } as MessageResponse,
      };

      const result = await Api.sendAnalysisResponse(portfolioId, payload);
      if (!result.success) {
        setState((prev) => {
          const revertedMessages = prev.messages
            .filter((entry) => entry.id !== userMessageId)
            .map((entry) => {
              if (
                entry.id === awaitingPrompt.messageId &&
                entry.role === "ai" &&
                entry.variant === "prompt" &&
                entry.promptType === "message"
              ) {
                return {
                  ...entry,
                  resolved: false,
                  responseText: undefined,
                };
              }
              return entry;
            });

          return {
            ...prev,
            status: "awaiting_input",
            awaiting: awaitingPrompt,
            error: result.message,
            messages: revertedMessages,
            statusMessage: currentState.statusMessage,
          };
        });
        return false;
      }

      setState((prev) => ({
        ...prev,
        messages: prev.messages.map((entry) =>
          entry.id === userMessageId
            ? {
                ...entry,
                pending: false,
              }
            : entry
        ),
      }));

      return true;
    },
    [nextClientSequence, portfolioId]
  );

  const sendAllocation = useCallback(
    async (allocation: AllocationResponse["content"], currency?: CurrencyCode) => {
      if (!portfolioId) {
        throw new Error("No portfolio selected");
      }
      const currentState = stateRef.current;
      if (!currentState.awaiting || currentState.awaiting.kind !== "allocation") {
        return false;
      }
      const awaitingPrompt = currentState.awaiting;
      if (currentState.messages.some((entry) => entry.pending)) {
        return false;
      }

      const previousAllocation = currentState.allocation;
      const timestamp = Date.now();
      const userMessageId = createMessageId("user");
      const optimisticMessage: ChatMessage = {
        id: userMessageId,
        role: "user",
        variant: "text",
        text: "Submitted revised allocation",
        timestamp,
        clientSequence: nextClientSequence(),
        pending: true,
      };

      setState((prev) => {
        const awaiting = prev.awaiting;
        const updatedMessages = prev.messages.map((entry) => {
          if (
            awaiting &&
            awaiting.kind === "allocation" &&
            entry.id === awaiting.messageId &&
            entry.role === "ai" &&
            entry.variant === "prompt" &&
            entry.promptType === "allocation"
          ) {
            return {
              ...entry,
              resolved: true,
              responseAllocation: allocation,
            };
          }
          return entry;
        });

        return {
          ...prev,
          status: "running",
          awaiting: null,
          allocation,
          error: null,
          messages: [...updatedMessages, optimisticMessage],
          conversationClosed: false,
        };
      });

      const payload: MessagePayload = {
        data: {
          type: "allocation_response",
          content: allocation,
          values: "shares",
          currency: currency ?? currentState.allocationCurrency ?? "USD",
        } as AllocationResponse,
      };

      const result = await Api.sendAnalysisResponse(portfolioId, payload);
      if (!result.success) {
        setState((prev) => {
          const revertedMessages = prev.messages
            .filter((entry) => entry.id !== userMessageId)
            .map((entry) => {
              if (
                entry.id === awaitingPrompt.messageId &&
                entry.role === "ai" &&
                entry.variant === "prompt" &&
                entry.promptType === "allocation"
              ) {
                return {
                  ...entry,
                  resolved: false,
                  responseAllocation: undefined,
                };
              }
              return entry;
            });

          return {
            ...prev,
            status: "awaiting_input",
            awaiting: awaitingPrompt,
            error: result.message,
            allocation: previousAllocation,
            messages: revertedMessages,
            statusMessage: currentState.statusMessage,
          };
        });
        return false;
      }

      setState((prev) => ({
        ...prev,
        messages: prev.messages.map((entry) =>
          entry.id === userMessageId
            ? {
                ...entry,
                pending: false,
              }
            : entry
        ),
      }));

      return true;
    },
    [nextClientSequence, portfolioId]
  );

  const reset = useCallback(() => {
    closeWebSocket();
    messageSequenceRef.current = 0;
    stateRef.current = initialState;
    setState(initialState);
  }, [closeWebSocket]);

  return useMemo(
    () => ({
      state,
      start,
      sendChoice,
      sendMessage,
      sendAllocation,
      reset,
    }),
    [reset, sendAllocation, sendChoice, sendMessage, start, state]
  );
}
