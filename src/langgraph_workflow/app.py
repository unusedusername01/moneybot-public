# Copyright 2025 unusedusername01
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import os
import sys
from typing import Any, Dict, TypedDict, List, Optional, Union, Set, Sequence, Annotated, Callable, Tuple
from urllib.parse import urlparse
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, START, END

from langchain_core.tools import tool
from dotenv import load_dotenv
from datetime import datetime

from pydantic import TypeAdapter, ValidationError
import multiprocessing as mp
try:
    from torch.cuda import is_available as cuda_available
    import torch.multiprocessing as torch_mp
except Exception:
    # Fallbacks when PyTorch is not available (for lightweight CI/local runs)
    def cuda_available() -> bool:  # type: ignore
        return False

    torch_mp = mp  # type: ignore

from fastapi import Body, FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from argparse import ArgumentParser
import asyncio
import uvicorn
from concurrent.futures import ThreadPoolExecutor
import traceback

from src.data_pipeline.data_fetcher import DataFetcher, NewsFetcher
from src.data_pipeline.llm_provider import LangChainLLMProvider
from src.data_pipeline.prediction_model import PredictionManager
from src.data_pipeline.constants import PORTFOLIO_PATH, PORTFOLIOS_PATH
from src.config.loader import load_config, get as get_config

from src.langgraph_workflow.custom_types import _TYPE_TO_MODEL
from src.langgraph_workflow.utils import get_shares_price, load_portfolio_data, select_prediction_horizons, load_ticker_data, group_by_criteria, order_ticker_by_score, render_batches, typed_dict_repr, get_shares_number
from src.langgraph_workflow.rag_manager import RAGManager
from src.langgraph_workflow.custom_types import *

load_dotenv()

logger = logging.getLogger(__name__)

RUN_PARALLEL = False  # Set to False for single-threaded execution
if RUN_PARALLEL:
    torch_mp.set_start_method('spawn', force=True)
LLM_PROVIDER = 'lm_studio'
EMBEDDINGS_PROVIDER = 'lm_studio'
DEVICE = 'cuda' if cuda_available() else 'cpu'
CRITERIAS = ['sector', 'market', 'score', 'market_cap']
TOP_K_INVESTMENTS = 5  # Number of top investments to consider in the final report
MAX_RERUNS = 2  # Maximum number of reruns allowed per analysis session (MAX_RERUNS + 1 total runs allowed)

PERSISTENT_USER_STATE = {}  # NOTE: Global state for the webapp users ()
SERVER_PORT = 8000
IPV4 = '127.0.0.1'
ALLOW_ORIGINS = [
    "http://localhost:3000",  # React dev server
    "http://localhost:8080",  # Vue dev server
    "http://127.0.0.1:3000",
    "http://127.0.0.1:8080",
    "http://localhost:5173",  # Vite dev server
    "http://127.0.0.1:5173",
]
WS_PING_INTERVAL = 30
WS_PING_TIMEOUT = 60
FRONTEND_BASE_URL = "http://127.0.0.1:8000"

executor: Optional[ThreadPoolExecutor] = None

tool_agent_model: Optional[str] = None
judge_model: Optional[str] = None
embeddings_model: Optional[str] = None

llm: Optional[LangChainLLMProvider] = None
embeddings: Optional[LangChainLLMProvider] = None

manager: Optional[RAGManager] = None
news_fetcher: Optional[NewsFetcher] = None

_MESSAGE_PAYLOAD_ADAPTER = TypeAdapter(MessagePayload)


def _extract_cli_preset(argv: Sequence[str]) -> Optional[str]:
    """Return the preset passed via command-line arguments, if any."""
    for index, argument in enumerate(argv):
        if argument == "--preset":
            if index + 1 < len(argv):
                return argv[index + 1]
            return None
        if argument.startswith("--preset="):
            return argument.split("=", 1)[1]
    return None


def _normalize_provider(provider: Optional[str], *, default: str) -> str:
    if not provider:
        return default
    if provider not in LangChainLLMProvider.PROVIDER_CONFIG:
        logger.warning("Unsupported provider '%s'. Falling back to '%s'.", provider, default)
        return default
    return provider


def _resolve_device(requested: Optional[str]) -> str:
    if not requested or requested.lower() == "auto":
        return 'cuda' if cuda_available() else 'cpu'

    normalized = requested.lower()
    if normalized.startswith('cuda'):
        if not cuda_available():
            logger.warning("CUDA requested but not available. Falling back to CPU.")
            return 'cpu'
        return requested
    if normalized == 'cpu':
        return 'cpu'
    return requested


def _origin_from_url(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    parsed = urlparse(value)
    if parsed.scheme and parsed.netloc:
        return f"{parsed.scheme}://{parsed.netloc}"
    return None

def _configure_cors(app: FastAPI, allow_origins: List[str]) -> None:
    """Apply CORS middleware with the provided allow list."""
    app.user_middleware = [mw for mw in app.user_middleware if mw.cls is not CORSMiddleware]
    app.middleware_stack = None  # Force rebuild with new middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allow_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["*"],
    )


def _apply_runtime_config(config: Dict[str, Any]) -> None:
    """Apply configuration overrides to runtime globals."""
    global RUN_PARALLEL, LLM_PROVIDER, EMBEDDINGS_PROVIDER, DEVICE
    global CRITERIAS, TOP_K_INVESTMENTS, MAX_RERUNS
    global SERVER_PORT, IPV4, ALLOW_ORIGINS, FRONTEND_BASE_URL
    global WS_PING_INTERVAL, WS_PING_TIMEOUT, executor
    global tool_agent_model, judge_model, embeddings_model
    global llm, embeddings, manager, news_fetcher

    existing_run_parallel = RUN_PARALLEL
    RUN_PARALLEL = bool(get_config(config, "runtime.run_parallel", existing_run_parallel))
    if RUN_PARALLEL:
        torch_mp.set_start_method('spawn', force=True)

    # Allow env overrides to take precedence during tests/CI without editing config files
    previous_llm_provider = LLM_PROVIDER
    previous_embeddings_provider = EMBEDDINGS_PROVIDER

    configured_llm_provider = os.getenv("MONEYBOT__models__llm__provider") or get_config(
        config,
        "models.llm.provider",
        previous_llm_provider,
    )
    configured_embeddings_provider = os.getenv("MONEYBOT__models__embeddings__provider") or get_config(
        config,
        "models.embeddings.provider",
        previous_embeddings_provider,
    )

    LLM_PROVIDER = _normalize_provider(configured_llm_provider, default=previous_llm_provider)
    EMBEDDINGS_PROVIDER = _normalize_provider(configured_embeddings_provider, default=previous_embeddings_provider)

    DEVICE = _resolve_device(get_config(config, "runtime.device", DEVICE))

    TOP_K_INVESTMENTS = int(get_config(config, "workflow.top_k_investments", TOP_K_INVESTMENTS))
    MAX_RERUNS = int(get_config(config, "workflow.max_reruns", MAX_RERUNS))

    SERVER_PORT = int(get_config(config, "server.port", SERVER_PORT))
    IPV4 = get_config(config, "server.host", IPV4)

    existing_allowed_origins = ALLOW_ORIGINS
    configured_origins = get_config(config, "server.cors.allow_origins", existing_allowed_origins)
    if isinstance(configured_origins, (list, tuple, set)):
        ALLOW_ORIGINS = list(configured_origins)
    elif isinstance(configured_origins, str):
        ALLOW_ORIGINS = [configured_origins]
    else:
        ALLOW_ORIGINS = list(existing_allowed_origins)

    FRONTEND_BASE_URL = get_config(config, "frontend.base_url", FRONTEND_BASE_URL)
    frontend_origin = _origin_from_url(FRONTEND_BASE_URL)
    if frontend_origin and frontend_origin not in ALLOW_ORIGINS:
        ALLOW_ORIGINS.append(frontend_origin)

    WS_PING_INTERVAL = int(get_config(config, "server.websocket.ping_interval", WS_PING_INTERVAL))
    WS_PING_TIMEOUT = int(get_config(config, "server.websocket.ping_timeout", WS_PING_TIMEOUT))

    max_workers = int(get_config(config, "runtime.max_workers", getattr(executor, "_max_workers", 8)))
    if executor is not None:
        executor.shutdown(wait=False)
    executor = ThreadPoolExecutor(max_workers=max_workers)

    previous_tool_agent_model = tool_agent_model
    previous_judge_model = judge_model
    previous_embeddings_model = embeddings_model

    configured_tool_agent = os.getenv("MONEYBOT__models__llm__tool_agent_model") or get_config(
        config,
        "models.llm.tool_agent_model",
        previous_tool_agent_model,
    )
    configured_judge = os.getenv("MONEYBOT__models__llm__judge_model") or get_config(
        config,
        "models.llm.judge_model",
        previous_judge_model,
    )
    configured_embeddings_model = os.getenv("MONEYBOT__models__embeddings__model") or get_config(
        config,
        "models.embeddings.model",
        previous_embeddings_model,
    )

    tool_agent_model = configured_tool_agent
    embeddings_model = configured_embeddings_model

    if not tool_agent_model:
        raise ValueError(f"No tool_agent_model configured for LLM provider '{LLM_PROVIDER}'.")
    judge_model = configured_judge or tool_agent_model
    if not embeddings_model:
        raise ValueError(f"No embeddings model configured for provider '{EMBEDDINGS_PROVIDER}'.")

    llm = LangChainLLMProvider(
        provider=LLM_PROVIDER,
        default_llm_model=tool_agent_model,
    )
    embeddings = LangChainLLMProvider(
        provider=EMBEDDINGS_PROVIDER,
        default_embeddings_model=embeddings_model,
    )

    resolved_embeddings = embeddings.get_embeddings(embeddings_model)
    manager = RAGManager(resolved_embeddings)
    news_fetcher = NewsFetcher(llm_provider=llm)

CLI_PRESET = _extract_cli_preset(sys.argv[1:])
ENV_PRESET = os.getenv("MONEYBOT_PRESET")
CONFIG: Dict[str, Any] = load_config(
    preset=CLI_PRESET or ENV_PRESET,
    fail_fast=bool(CLI_PRESET or ENV_PRESET),
)

webapp = FastAPI()
_apply_runtime_config(CONFIG)
_configure_cors(webapp, ALLOW_ORIGINS)

@tool
def sector_news_retriever_tool(sector: str, query: str) -> str:
    """
    Retrieve news articles for a specific sector.
    The sector must be one specified by a ticker's fundamental data.
    Common sectors include: 'technology', 'consumer_cyclical', 'healthcare', 'finance', 'energy' etc.
    Args:
        sector (str): The sector for which to retrieve news articles.
        query (str): The query to search within the sector news.
    Returns:
        str: A string containing the news articles for the specified sector.
    """
    if manager is None:
        raise RuntimeError("RAG manager is not initialized.")
    return manager._sector_news_retriever_tool(sector, query)

@tool
def market_news_retriever_tool(market: str, query: str) -> str:
    """
    Retrieve news articles for a specific market.
    The market must be one specified by a ticker's fundamental data.
    Common markets include 'us_market', 'fr_market', 'de_market', 'cn_market', etc.
    Args:
        market (str): The market for which to retrieve news articles.
        query (str): The query to search within the market news.
    Returns:
        str: A string containing the news articles for the specified market.
    """
    if manager is None:
        raise RuntimeError("RAG manager is not initialized.")
    return manager._market_news_retriever_tool(market, query)

judge_tools = [
    sector_news_retriever_tool, market_news_retriever_tool
]
connections: Dict[str, WebSocket] = {} # portfolio_id -> WebSocket

def allocate_budget(ticker_scores: Dict[str, Tuple[int, str]], budget: int, currency: str, allow_fractional = True) -> Dict[str, float]:
    """
    Allocate the budget based on the scores of the tickers.
    Returns a dictionary with ticker symbols as keys and allocated shares number as values.
    Args:
        ticker_scores (Dict[str, Tuple[int, str]]): A dictionary with ticker symbols as keys and a tuple of (score, metadata) as values.
        budget (int): The total budget to allocate.
        currency (str): The currency in which the budget is allocated (ISO 4217 format, e.g: USD, EUR, CHF, ...)
        allow_fractional (bool): Whether to allow fractional shares. (True by default)
    Returns:
        Union[Dict[str, float], Dict[str, int]]: A dictionary with ticker symbols as keys and allocated shares number as values.
        The values are float if allow_fractional is True, otherwise they are int.
    """
    total_score = sum(score for score, _ in ticker_scores.values())
    if total_score == 0:
        return {ticker: 0.0 for ticker in ticker_scores.keys()}

    allocation = {ticker: (score / total_score) * budget for ticker, (score, _) in ticker_scores.items()}

    shares = {ticker: get_shares_number(ticker, amount, currency, allow_fractional) for ticker, amount in allocation.items()}

    print("Allocated shares:", shares)

    return shares

@webapp.get("/")
async def root():
    return {"status": "running"}

# One way websocket endpoint
@webapp.websocket("/ws/portfolio/{portfolio_id}")
async def websocket_endpoint(websocket: WebSocket, portfolio_id: str):
    await websocket.accept()
    connections[portfolio_id] = websocket
    print("connection open")
    
    try:
        # Keep connection alive with better error handling
        while True:
            try:
                # Check if connection is still alive
                await asyncio.wait_for(websocket.receive_text(), timeout=0.1)
            except asyncio.TimeoutError:
                # No message received, continue
                pass
            except Exception:
                # Connection closed
                break
            
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        connections.pop(portfolio_id, None)
        print("connection closed")



class AgentState(TypedDict):
    """
    State for the full graph
    """
    # Exposed data (from the webapp)
    app_data: AppData

    # Internal data
    portfolio_data: PortfolioData  # Data for the selected portfolio
    batches: BatchesData  # Data for the batches to process
    ranked_batches: List[RankedBatchData]  # Ranked batches after the judge evaluation

def get_state(portfolio_id: str) -> AgentState:
    """
    Get the current state for a specific portfolio.
    """
    state = PERSISTENT_USER_STATE.get(portfolio_id, {})

    if not state:
        raise ValueError(f"No state found for portfolio ID: {portfolio_id}")

    return state

def decide_next_node(state: AgentState) -> str:
    # If the version is 1 or higher, resume the state by rerouting directly to next node
    if state['app_data']['version'] >= 1:
        next_node = state['app_data']['next_node']
        # Directly reroute to next node if not None else pause/stop (END)
        return next_node if next_node else 'stop'
    
    return 'default'


async def init_agent_state(state: AgentState) -> AgentState:
    """
    Initialize the agent state with default values.
    """
    if news_fetcher is None:
        raise RuntimeError("News fetcher is not initialized.")
    if executor is None:
        raise RuntimeError("Thread executor is not initialized.")

    if not state['app_data'] or not state['app_data']['selected_portfolio']:
        raise ValueError("AppData and selected portfolio must be provided in the initial state.")

    # Step 1: Load the portfolio data
    state['portfolio_data'] = load_portfolio_data(state['app_data']['selected_portfolio'])
    if not state['portfolio_data']:
        raise ValueError("Portfolio data could not be loaded. Ensure the portfolio exists and is correctly formatted.")

    # Extract portfolio details
    budget, target_date, holdings, currency, risk_tolerance, criteria, prediction_strength = (
        state['portfolio_data']['budget'],
        state['portfolio_data']['target_date'],
        state['portfolio_data']['holdings'],
        state['portfolio_data']['currency'],
        state['portfolio_data']['risk_tolerance'],
        state['portfolio_data']['criteria'],
        state['portfolio_data']['prediction_strength']
    )

    # Send initial message to client
    message = MessagePayload(
        data=StateUpdate(prompt="Fetching latest data for tickers"),
        timeout=10
    )

    await push_message_to_client(
        portfolio_id=state['app_data']['selected_portfolio'],
        message=message
    )
    await asyncio.sleep(0)

    # Value checks on indispensable fields
    if not budget or not target_date or not holdings:
        raise ValueError("Portfolio data must contain budget, target_date, and tickers.")
    
    # Convert target_date to datetime format
    target_date = datetime.strptime(target_date, "%Y-%m-%d")

    fundamentals_timeout = 5 # Timeout for fetching fundamentals
    historical_prices_timeout = 30 # Timeout for fetching historical prices
    news_timeout = 90 # Timeout for fetching news
    prediction_strength_timeout = { # Timeout per predictions based on strength
        'weak': 20,
        'medium': 30,
        'strong': 50
    }

    # Step 2: Fetch the latest data for each ticker (offloaded to threads)
    for ticker in holdings.keys():
        message = MessagePayload(
            data=StateUpdate(prompt=f'Fetching latest data for {ticker}'),
            timeout=fundamentals_timeout + historical_prices_timeout + news_timeout
        )
        await push_message_to_client(
            portfolio_id=state['app_data']['selected_portfolio'],
            message=message
        )
        
        # Offload blocking operations to threads
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            executor, 
            DataFetcher.fetch_fundamentals, 
            ticker
        )
        await loop.run_in_executor(
            executor, 
            DataFetcher.fetch_historical_prices, 
            ticker
        )
        await loop.run_in_executor(
            executor, 
            news_fetcher.fetch_news, 
            ticker
        )
        
        # Yield control to event loop
        await asyncio.sleep(0)

    # Step 3: Compute predictions for each ticker
    # Determine the prediction days based on target_date and risk_tolerance
    days_ahead = select_prediction_horizons(target_date, risk_tolerance)
    prediction_strength = prediction_strength if prediction_strength in ['weak', 'medium', 'strong'] else 'medium'

    predictions_timeout = prediction_strength_timeout[prediction_strength] * len(days_ahead) * len(holdings)
    data_loading_timeout = 10  # Timeout for data loading

    message = MessagePayload(
        data=StateUpdate(prompt=f'Fetching {prediction_strength} predictions'),
        timeout=predictions_timeout + data_loading_timeout
    )

    await push_message_to_client(
        portfolio_id=state['app_data']['selected_portfolio'],
        message=message
    )

    # NOTE: Run parallely on tickers instead of days_ahead to avoid concurrent file access issues
    # Step 3: Compute predictions for each ticker
    for days in days_ahead:
        if RUN_PARALLEL:
            # Use asyncio for parallel execution instead of multiprocessing
            tasks = [
                asyncio.to_thread(
                    PredictionManager.run_prediction_for_ticker,
                    ticker, days, prediction_strength
                )
                for ticker in holdings.keys()
            ]
            await asyncio.gather(*tasks)
        else:
            # Sequential execution with thread offloading
            for ticker in holdings.keys():
                await asyncio.to_thread(
                    PredictionManager.run_prediction_for_ticker,
                    ticker, days, prediction_strength
                )
                # Yield control frequently
                await asyncio.sleep(0)


    # Step 4: Collect the data and prepare batches
    ticker_data = [load_ticker_data(ticker) for ticker in holdings.keys()]
    if not ticker_data or None in ticker_data:
        raise ValueError("At least one ticker data is missing or None. Check the data collection process.")
    
    # Split the tickers into batches based on the criteria
    criteria = criteria if criteria in CRITERIAS else 'market_cap'

    batches = group_by_criteria(ticker_data, criteria, max_batch_size=5, split_absolute=True)
    print("\n\nBatches created:\n" + "\n".join(
        str([tickerdata['ticker'] for tickerdata in batch]) for batch in batches
    ))
    if not batches:
        raise ValueError("No valid batches were created. Check the grouping criteria and ticker data.")
    
    batches_data = BatchesData(
        data=batches,
        criteria=criteria,
        ranking=order_ticker_by_score if criteria == 'score' else None
    )

    state['batches'] = batches_data

    # Save the state to persistent storage
    PERSISTENT_USER_STATE[state['app_data']['selected_portfolio']] = state

    # Reroute to run_analysis
    return state

async def run_analysis(state: AgentState) -> AgentState:
    """
    Run the analysis on the batches of data.
    """
    if llm is None:
        raise RuntimeError("LLM provider is not initialized.")
    if executor is None:
        raise RuntimeError("Thread executor is not initialized.")
    if judge_model is None or tool_agent_model is None:
        raise RuntimeError("LLM models are not configured for analysis.")

    # Prepare the batches for the judge
    batches = state['batches']['data']
    ranked_batches = []

    formatted_batches = render_batches(batches)

    # Send a message to the client indicating the analysis has started
    portfolio_id = state['app_data']['selected_portfolio']

    message = MessagePayload(
        data=StateUpdate(prompt='Starting the ticker analysis'),
        timeout=(len(formatted_batches) + 3) * 60 # ~1min per batch with some buffer
    )

    await push_message_to_client(
        portfolio_id=portfolio_id,
        message=message
    )

    portfolio_data = state['portfolio_data']

    user_preferences = ['budget', 'holdings', 'target_date', 'risk_tolerance']

    user_preferences = ', '.join([f"{k}: {portfolio_data[k]}" for k in user_preferences])

    print(f"\n\nUser preferences: {user_preferences}")
    print(f"\n\nN batches: ", len(formatted_batches))

    for tickers, batch in formatted_batches:
        try:
            # Offload blocking operations to thread
            loop = asyncio.get_running_loop()
            scores = await loop.run_in_executor(
                executor,
                llm.evaluate_batch,
                batch,
                user_preferences,
                judge_model,
                judge_tools,
                tool_agent_model,
            )
            await asyncio.sleep(0) # Yield control to the event loop

            if not scores or not isinstance(scores, dict):
                print(f"Warning: No scores returned for batch {tickers}")
                # Send error message to client
                error_message = MessagePayload(
                    data=StateUpdate(prompt=f'Warning: Failed to evaluate batch {tickers}'),
                    timeout=5
                )
                await push_message_to_client(portfolio_id=portfolio_id, message=error_message)
                continue

            ranked_batch = RankedBatchData(
                tickers=tickers,
                scores=scores
            )

            ranked_batches.append(ranked_batch)
        except Exception as e:
            print(f"Error evaluating batch {tickers}: {e}")
            import traceback
            traceback.print_exc()
            # Send error message to client
            error_message = MessagePayload(
                data=StateUpdate(prompt=f'Error evaluating batch {tickers}: {str(e)}'),
                timeout=10
            )
            try:
                await push_message_to_client(portfolio_id=portfolio_id, message=error_message)
            except Exception as msg_error:
                print(f"Failed to send error message: {msg_error}")
            # Continue to next batch instead of failing completely
            continue

    if not ranked_batches:
        error_message = MessagePayload(
            data=StateUpdate(prompt='Error: Failed to evaluate any batches. Analysis cannot continue.'),
            timeout=10
        )
        try:
            await push_message_to_client(portfolio_id=portfolio_id, message=error_message)
        except Exception as msg_error:
            print(f"Failed to send error message: {msg_error}")
        raise RuntimeError("No batches were successfully evaluated.")

    state['ranked_batches'] = ranked_batches

    # Update persistent storage
    PERSISTENT_USER_STATE[portfolio_id] = state

    return state

async def call_synthesizer(state: AgentState) -> AgentState:
    """
    Call the synthesizer LLM to generate a final report.
    """
    if llm is None:
        raise RuntimeError("LLM provider is not initialized.")

    batches = state['ranked_batches']
    portfolio_id = state['app_data']['selected_portfolio']

    message = MessagePayload(
        data=StateUpdate(prompt='Finalizing the ticker analysis'),
        timeout=90 # ~1m30 for the model to respond
    )

    await push_message_to_client(
        portfolio_id=portfolio_id,
        message=message
    )

    # Trim to keep only the N last human messages
    N = 2 # NOTE: Adjust this value to change the number of messages considered
    additional_queries = []
    for i, message in enumerate([human_msg for human_msg in state['app_data']['messages'] if isinstance(human_msg, HumanMessage)][-N:]):
        additional_queries.append(f"Query {i}: {message.content}")

    portfolio_data = state['portfolio_data']

    holdings, target_date, risk_tolerance = (
        portfolio_data['holdings'],
        portfolio_data['target_date'],
        portfolio_data['risk_tolerance']
    )

    # Format past interactions as a system prompt
    base_prompt = f"""\n
    Finally, here are some additional queries/requests from the user:
    {additional_queries}
    """ if additional_queries else "" + f""" and here are some of the user's preferences:
    - Current holdings: {holdings}
    - Target date: {target_date}
    - Risk tolerance (low, medium, high): {risk_tolerance}
    """

    def sub_merge_batches(b1: RankedBatchData, b2: RankedBatchData, final_batch_size: int) -> RankedBatchData:
        doubles = b1['scores'].keys() & b2['scores'].keys()
        # Remove duplicates while maintaining similar lengths
        while doubles:
            ticker = doubles.pop()
            if len(b1['tickers']) > len(b2['tickers']):
                b1['tickers'].remove(ticker)
                b1['scores'].pop(ticker)
            else:
                b2['tickers'].remove(ticker)
                b2['scores'].pop(ticker)

        if len(b1['tickers']) + len(b2['tickers']) <= final_batch_size:
            return RankedBatchData(
                tickers=b1['tickers'] + b2['tickers'],
                scores={**b1['scores'], **b2['scores']}
            )

        prompt = base_prompt + f"""
        Here are the batches to merge:
        Batch 1:
        {str(b1['scores'])}
        Batch 2:
        {str(b2['scores'])}\n"""

        updated_scores = llm.merge_batches(prompt)

        # Pick the top scores
        updated_scores = dict(sorted(updated_scores.items(), key=lambda item: item[1][0], reverse=True)[:final_batch_size])

        return RankedBatchData(
            tickers=list(updated_scores.keys()),
            scores=updated_scores
        )

    def merge_batches(list_data: List[RankedBatchData], final_batch_size: int = TOP_K_INVESTMENTS) -> RankedBatchData:
        if not list_data:
            raise ValueError("Batch list is empty.")
        elif len(list_data) == 1:
            return list_data[0]
        elif len(list_data) == 2:
            return sub_merge_batches(list_data[0], list_data[1], final_batch_size)
        else:
            left, right = list_data[:len(list_data)//2], list_data[len(list_data)//2:]
            merged_left = merge_batches(left, final_batch_size)
            merged_right = merge_batches(right, final_batch_size)

            return sub_merge_batches(merged_left, merged_right, final_batch_size)

    loop = asyncio.get_event_loop()

    result = await loop.run_in_executor(executor, merge_batches, batches, TOP_K_INVESTMENTS)

    budget, currency = state['portfolio_data']['budget'], state['portfolio_data']['currency']

    budget_allocation = allocate_budget(result['scores'], budget, currency)

    state['app_data']['budget_allocation'] = budget_allocation

    allocation_message = MessagePayload(
        data=AllocationResponse(
            content=budget_allocation,
            currency=currency,
        ),
    )

    await push_message_to_client(portfolio_id, allocation_message)

    # Share the judge rationale so the frontend can display decision context
    rationale_lines = []
    for ticker, (score, review) in result['scores'].items():
        summary = review.strip() if isinstance(review, str) else ""
        if summary:
            rationale_lines.append(f"{ticker} (score {score}): {summary}")
        else:
            rationale_lines.append(f"{ticker} (score {score})")

    if rationale_lines:
        rationale_text = "Rationale behind the allocation:\n" + "\n".join(
            f"- {line}" for line in rationale_lines
        )
    else:
        rationale_text = "Rationale behind the allocation is currently unavailable."

    review_message = MessagePayload(
        data=MessageResponse(content=rationale_text),
    )

    await push_message_to_client(portfolio_id, review_message)

    return state

async def check_webapp(state: AgentState) -> AgentState:
    """
    Check the status of the web application.
    This runs in background to update the persistent user state.
    """
    # Set awaiting metadata
    state['app_data']['awaiting'] = True
    state['app_data']['await_schema'] = MessageType.CHOICE_RESPONSE
    state['app_data']['version'] = 1

    # Extract user's portfolio's ID and the latest response
    portfolio_id = state['app_data']['selected_portfolio']

    # Update persistent storage
    PERSISTENT_USER_STATE[portfolio_id] = state

    message = MessagePayload(
        data=AwaitingChoiceRequest(
            prompt="Please choose one of the following options"
        )
    )

    await push_message_to_client(portfolio_id, message)

    return state

async def call_logger(state: AgentState) -> AgentState:
    """
    Call the log the whole interaction process to the database.
    """
    portfolio_id = state['app_data']['selected_portfolio']
    await push_message_to_client(
        portfolio_id,
        MessagePayload(
            data=StateUpdate(prompt="Logging interaction into the database"),
        )
    )

    # Here we:
    # 1. Log the conversation and additional data to the database
    # 2. Clean up the persistent state
    # 3. Update the current portfolio assets based on the allocation
    # Optionals / To be added later:
    # # Right now:
        # - Take action directly onto the user's brokerage account (e.g., place orders)
        # - Send a summary report to the user via email or other means
        # - In the background
    # # In the background:
        # - Monitor the market for significant changes that might affect the recommendations / Send alerts if necessary
        # - Analyze the effectiveness of the recommendations over time
        # - Add a task to a scheduler to train any kind of model that could benefit from the collected data (e.g., a recommendation model, a prediction model, etc.)

    # 1. Log the conversation and additional data to the database
    # ...

    # 2. Clean up the persistent state
    if portfolio_id in PERSISTENT_USER_STATE:
        del PERSISTENT_USER_STATE[portfolio_id]

    # 3. Update the current portfolio assets based on the allocation
    allocation = state['app_data'].get('budget_allocation', {})
    portfolio_data = state['portfolio_data']
    current_holdings = portfolio_data.get('holdings', {})

    updated_holdings = current_holdings.copy()
    for ticker, shares in allocation.items():
        updated_holdings[ticker] = updated_holdings.get(ticker, 0) + shares

    portfolio_data['holdings'] = updated_holdings

    write_path = PORTFOLIO_PATH(portfolio_id)

    # Dump and overwrite if necessary
    with open(write_path, 'w') as f:
        json.dump(portfolio_data, f, indent=2)

    # Here do any additional steps if necessary ...


    # Notify the client that the analysis has ended
    message = MessagePayload(
        data=EndAnalysis()
    )

    await push_message_to_client(portfolio_id, message)

    return state

async def call_cancel(state: AgentState) -> AgentState:
    """
    Cancel the graph execution and __end__ the current session.
    """
    # Find the state in persistent storage and delete it
    portfolio_id = state['app_data']['selected_portfolio']
    if portfolio_id in PERSISTENT_USER_STATE:
        del PERSISTENT_USER_STATE[portfolio_id]

    await push_message_to_client(
        portfolio_id,
        MessagePayload(
            data=EndAnalysis()
        )
    )

    return state

# Sender
async def push_message_to_client(portfolio_id: str, message: MessagePayload):
    """
    Push a message to the client for a specific portfolio.
    """
    websocket = connections.get(portfolio_id)
    if websocket:
        try:
            print("Sending message:", message.to_json())
            await websocket.send_json(message.to_json())
            # Yield control after sending
            await asyncio.sleep(0)
        except Exception as e:
            print(f"Error sending message to {portfolio_id}: {e}")
            # Remove broken connection
            connections.pop(portfolio_id, None)
            raise ValueError(f"WebSocket connection lost for portfolio ID: {portfolio_id}")
    else:
        raise ValueError(f"No WebSocket connection found for portfolio ID: {portfolio_id}")


graph = StateGraph(AgentState)
graph.add_node('init_agent_state', init_agent_state)
graph.add_node('run_analysis', run_analysis)
graph.add_node('call_synthesizer', call_synthesizer)
graph.add_node('check_webapp', check_webapp)
graph.add_node('call_logger', call_logger)
graph.add_node('call_cancel', call_cancel)

graph.add_conditional_edges(
    START,
    decide_next_node,
    {
        'confirm': 'call_logger',
        'cancel': 'call_cancel',
        'default': 'init_agent_state',
        'rerun': 'call_synthesizer',
        'stop': END
    }
)

graph.add_edge('init_agent_state', 'run_analysis')
graph.add_edge('run_analysis', 'call_synthesizer')
graph.add_edge('call_synthesizer', 'check_webapp')
graph.add_edge('check_webapp', END)
graph.add_edge('call_logger', END)
graph.add_edge('call_cancel', END)

app = graph.compile()

# POST request handler for user responses
@webapp.post("/analysis/portfolio/{portfolio_id}/respond", response_model=HttpResponse)
async def respond(portfolio_id: str, response: Any = Body(...)):
    """
    Respond to a user's query for a specific portfolio.
    """
    try:
        if isinstance(response, (bytes, bytearray)):
            payload = MessagePayload.from_json(response.decode("utf-8"))
        elif isinstance(response, str):
            payload = MessagePayload.from_json(response)
        else:
            payload = _MESSAGE_PAYLOAD_ADAPTER.validate_python(response)
    except (ValidationError, ValueError, TypeError) as exc:
        return HttpResponse(err_code=422, details=f"Invalid payload: {exc}")

    state = get_state(portfolio_id)
    
    assert state['app_data']['awaiting'], "The application is not awaiting a response."

    expected_schema = state['app_data']['await_schema']

    schema_to_allowed = {
        MessageType.AWAITING_CHOICE: {MessageType.CHOICE_RESPONSE},
        MessageType.AWAITING_MESSAGE: {MessageType.MESSAGE_RESPONSE},
        MessageType.AWAITING_ALLOCATION: {MessageType.ALLOCATION_RESPONSE, MessageType.MESSAGE_RESPONSE},
    }

    if expected_schema in schema_to_allowed:
        allowed_payloads = schema_to_allowed[expected_schema]
    elif expected_schema == MessageType.ALLOCATION_RESPONSE:
        allowed_payloads = {MessageType.ALLOCATION_RESPONSE, MessageType.MESSAGE_RESPONSE}
    elif isinstance(expected_schema, MessageType):
        allowed_payloads = {expected_schema}
    else:
        allowed_payloads = set()

    if payload.data.type not in allowed_payloads:
        return HttpResponse(
            err_code=400,
            details=(
                f"Invalid payload type: {payload.data.type}. "
                f"Expected one of {[t.value for t in sorted(allowed_payloads, key=lambda t: t.value)]} "
                f"while awaiting {expected_schema.value if isinstance(expected_schema, MessageType) else expected_schema}."
            ),
        )
    if payload.data.type == MessageType.CHOICE_RESPONSE:
        valid_choices = AwaitingChoiceRequest(prompt='').choices
        if not isinstance(payload.data, ChoiceResponse) or payload.data.selection not in valid_choices:
            return HttpResponse(
                err_code=400,
                details=f"Invalid response: {payload.data.selection}. Expected one of {valid_choices}.",
            )
        choice = payload.data.selection
        if choice == 'confirm':
            state['app_data']['awaiting'] = False
            state['app_data']['await_schema'] = None
            state['app_data']['next_node'] = 'confirm'
        if choice == 'edit':
            state['app_data']['awaiting'] = True # Safeguard
            state['app_data']['await_schema'] = MessageType.ALLOCATION_RESPONSE
            state['app_data']['next_node'] = None # Reset next node (safeguard)
            message = MessagePayload(
                data=AwaitingAllocationRequest(
                    prompt="Edit the allocation for the following tickers: "
                )
            )
            await push_message_to_client(portfolio_id, message)
        if choice == 'deny':
            if state['app_data']['version'] >= MAX_RERUNS:
                state['app_data']['awaiting'] = False
                state['app_data']['await_schema'] = None
                state['app_data']['next_node'] = 'cancel'
                
                message = MessagePayload(
                    data=StateUpdate(
                        prompt="Maximum number of reruns reached. Cancelling the analysis."
                    )
                ) # For now, we cancel the analysis after max reruns
                await push_message_to_client(portfolio_id, message)

                return HttpResponse(err_code=200)

            state['app_data']['awaiting'] = True # Safeguard
            state['app_data']['await_schema'] = MessageType.MESSAGE_RESPONSE
            state['app_data']['next_node'] = None # Reset next node (safeguard)
            message = MessagePayload(
                data=AwaitingMessageRequest(
                    prompt="How can we adapt the analysis to your needs?"
                )
            )
            await push_message_to_client(portfolio_id, message)
        if choice == 'cancel':
            state['app_data']['awaiting'] = False
            state['app_data']['await_schema'] = None
            state['app_data']['next_node'] = 'cancel'

    elif payload.data.type == MessageType.ALLOCATION_RESPONSE:
        new_budget_allocation = payload.data.content

        if not isinstance(new_budget_allocation, dict):
            return HttpResponse(err_code=400, details="Invalid budget allocation format.")
        diff_tickers = set(new_budget_allocation.keys()) - set(state['app_data']['budget_allocation'].keys())
        if diff_tickers:
            return HttpResponse(err_code=400, details=f"New budget allocation contains unknown tickers: {diff_tickers}.")
        if any(amount < 0 or not isinstance(amount, float) for amount in new_budget_allocation.values()):
            return HttpResponse(err_code=400, details="All allocation values must be non-negative numbers.")            

        state['app_data']['budget_allocation'] = new_budget_allocation
        state['app_data']['awaiting'] = False
        state['app_data']['await_schema'] = None
        state['app_data']['next_node'] = 'confirm'

    elif payload.data.type == MessageType.MESSAGE_RESPONSE:
        if state['app_data']['await_schema'] in {MessageType.MESSAGE_RESPONSE, MessageType.ALLOCATION_RESPONSE}:
            message = HumanMessage(content=payload.data.content)
            state['app_data']['messages'] += [message]
            state['app_data']['awaiting'] = False
            state['app_data']['await_schema'] = None
            state['app_data']['version'] += 1 # Fix: Increment version on user input
            state['app_data']['next_node'] = 'rerun'
            state['app_data']['budget_allocation'] = {}
            await push_message_to_client(
                portfolio_id,
                MessagePayload(
                    data=StateUpdate(prompt="Analysis restarted")
                )
            )
        else:
            return HttpResponse(err_code=400, details="Unexpected message response state.")

    # Reinvoke the app with the updated state, the first node will reroute appropriately
    try:
        await app.ainvoke(state)
    except Exception as e:
        print(f"Error invoking app: {e}")
        return HttpResponse(err_code=500, details=str(e))

    return HttpResponse(err_code=200)

async def run(portfolio_id: str):
    """
    Run the application with the given data.
    """
    if portfolio_id in PERSISTENT_USER_STATE:
        state = PERSISTENT_USER_STATE[portfolio_id]
        if 'budget_allocation' not in state['app_data']:
            state['app_data']['budget_allocation'] = {}
    else:
        app_data = AppData(
            selected_portfolio=portfolio_id,
            messages=[],
            budget_allocation={},
            awaiting=False,
            await_schema=None,
            version=0,
            next_node=None
        )

        state = {'app_data': app_data}

    await app.ainvoke(state)

@webapp.post("/analysis/portfolio/{portfolio_id}/start", response_model=HttpResponse, summary="Start the analysis for a specific portfolio.")
async def start_analysis(portfolio_id: str):
    """
    Start the analysis for a specific portfolio.
    """
    try:
        try:
            await push_message_to_client(
                portfolio_id,
                MessagePayload(
                    data=StateUpdate(prompt="Analysis started"),
                )
            )
        except ValueError as e:
            print(f"Unable to send initial status for {portfolio_id}: {e}")
            return HttpResponse(err_code=400, details=f"No WebSocket connection for portfolio {portfolio_id}")
        
        await run(portfolio_id)
    except ValueError as ve:
        print(f"ValueError in analysis: {ve}")
        # Try to send error to client
        try:
            await push_message_to_client(
                portfolio_id,
                MessagePayload(data=StateUpdate(prompt=f"Error: {str(ve)}"))
            )
        except Exception:
            pass
        return HttpResponse(err_code=400, details=str(ve))
    except RuntimeError as re:
        print(f"RuntimeError in analysis: {re}")
        # Try to send error to client
        try:
            await push_message_to_client(
                portfolio_id,
                MessagePayload(data=StateUpdate(prompt=f"Error: {str(re)}"))
            )
        except Exception:
            pass
        return HttpResponse(err_code=500, details=str(re))
    except WebSocketDisconnect:
        print(f"WebSocket disconnected for {portfolio_id}")
        return HttpResponse(err_code=408, details="WebSocket connection closed.")
    except Exception as e:
        print(f"Unexpected error in analysis: {e}")
        import traceback
        traceback.print_exc()
        # Try to send error to client
        try:
            await push_message_to_client(
                portfolio_id,
                MessagePayload(data=StateUpdate(prompt=f"Error: {str(e)}"))
            )
        except Exception:
            pass
        return HttpResponse(err_code=500, details=str(e))

    return HttpResponse(err_code=200)

@webapp.post("/utils/get_shares_price", response_model=Union[HttpResponse, GetSharesPriceResponse], summary="Get the current price of a specific number of shares for a given ticker.")
async def http_get_shares_price(request: GetSharesPriceRequest):
    """Get the latest price or notional value for one or more tickers."""

    try:
        raw_tickers = request.ticker
        raw_amounts = request.amount

        if isinstance(raw_tickers, str):
            tickers = [raw_tickers]
        elif isinstance(raw_tickers, list) and all(isinstance(t, str) for t in raw_tickers):
            if not raw_tickers:
                return HttpResponse(err_code=400, details="Ticker list cannot be empty.")
            tickers = raw_tickers
        else:
            return HttpResponse(err_code=400, details="Invalid ticker payload.")

        if isinstance(raw_amounts, list):
            if len(raw_amounts) != len(tickers):
                return HttpResponse(err_code=400, details="Tickers and amounts lists must have the same length.")
            if not all(isinstance(amount, (int, float, type(None))) for amount in raw_amounts):
                return HttpResponse(err_code=400, details="Amounts list must contain numbers or null values.")
            amounts = [float(amount) if amount is not None else None for amount in raw_amounts]
        elif isinstance(raw_amounts, (int, float)):
            amounts = [float(raw_amounts)] * len(tickers)
        elif raw_amounts is None:
            amounts = [None] * len(tickers)
        else:
            return HttpResponse(err_code=400, details="Invalid amount payload.")

        currency = request.currency
        allow_fractional = True if request.allow_fractional is None else bool(request.allow_fractional)

        prices: List[float] = []
        for ticker, amount in zip(tickers, amounts):
            if amount is not None and amount < 0:
                return HttpResponse(err_code=400, details="Amounts must be non-negative numbers or null.")

            price = get_shares_price(
                ticker=ticker,
                number_of_shares=amount,
                currency=currency,
                allow_fractional=allow_fractional,
            )
            prices.append(price)

        response = GetSharesPriceResponse(
            err_code=200,
            ticker=tickers if len(tickers) > 1 else tickers[0],
            price=prices if len(prices) > 1 else prices[0],
        )

        return response
    except ValueError as exc:
        return HttpResponse(err_code=400, details=str(exc))
    except Exception as exc:
        print(f"[WARN] get_shares_price failed: {exc}")
        return HttpResponse(err_code=500, details="Failed to fetch share price.")

@webapp.post("/utils/load_portfolio_data", response_model=Union[HttpResponse, LoadPortfolioDataResponse], summary="Load the data for a specific portfolio.")
async def http_load_portfolio_data(request: PortfolioDataRequest):
    data = load_portfolio_data(request.portfolio_id)

    if not data:
        return HttpResponse(err_code=404, details=f"Portfolio {request.portfolio_id} not found.")

    response = LoadPortfolioDataResponse(
        err_code=200,
        portfolio_data={**data}
    )

    return response

@webapp.post("/utils/edit_portfolio", response_model=HttpResponse, summary="Edit a specific portfolio.")
async def http_edit_portfolio(request: EditPortfolioRequest):
    try:
        portfolio_data = PortfolioData(
            budget=request.budget,
            target_date=request.target_date,
            holdings=request.holdings,
            currency=request.currency,
            risk_tolerance=request.risk_tolerance,
            criteria=request.criteria,
            prediction_strength=request.prediction_strength
        )
    except ValidationError as ve:
        return HttpResponse(err_code=400, details=str(ve))

    write_path = PORTFOLIO_PATH(request.portfolio_id)

    assert write_path.resolve().parent == PORTFOLIOS_PATH.resolve(), "Invalid portfolio path. (Don't try and be sneaky like that bruh)"

    # Dump and overwrite if necessary
    with open(write_path, 'w') as f:
        json.dump(portfolio_data, f, indent=2)

    return HttpResponse(err_code=200)

@webapp.post("/utils/create_portfolio", response_model=HttpResponse, summary="Create a new portfolio.")
async def http_create_portfolio(request: EditPortfolioRequest):
    if PORTFOLIO_PATH(request.portfolio_id).resolve().exists():
        return HttpResponse(err_code=400, details="Portfolio already exists.")
    return await http_edit_portfolio(request)


@webapp.delete("/utils/delete_portfolio", response_model=HttpResponse, summary="Delete a specific portfolio.")
async def http_delete_portfolio(request: DeletePortfolioRequest):
    try:
        portfolio_path = PORTFOLIO_PATH(request.portfolio_id)
        assert portfolio_path.exists(), "Portfolio does not exist."
        assert portfolio_path.resolve().parent == PORTFOLIOS_PATH.resolve(), "Invalid portfolio path. (Don't try and be sneaky like that bruh)"
        portfolio_path.unlink()  # Delete the file
    except FileNotFoundError:
        return HttpResponse(err_code=404, details="Portfolio not found.")
    except Exception as e:
        return HttpResponse(err_code=500, details=str(e))

    return HttpResponse(err_code=200)

@webapp.get("/utils/list_portfolios", response_model=Union[HttpResponse, ListPortfoliosResponse], summary="List all portfolios.")
async def http_list_portfolios():
    try:
        portfolios = [p.stem for p in PORTFOLIOS_PATH.glob("*.json")]
    except ValueError as ve:
        return HttpResponse(err_code=400, details=str(ve))
    except Exception as e:
        return HttpResponse(err_code=500, details=str(e))
    if not portfolios:
        return HttpResponse(err_code=404, details="No portfolios found.")

    return ListPortfoliosResponse(err_code=200, portfolios=portfolios)

@webapp.post("/utils/validate_currency", response_model=HttpResponse, summary="Validate a currency code.")
async def http_validate_currency(request: CurrencyValidationRequest):
    """
    Validate a currency code.
    """
    currency_adapter = TypeAdapter(Currency)
    try:
        validated_currency = currency_adapter.validate_python(request.currency)
    except ValidationError as e:
        return HttpResponse(err_code=400, details=str(e))
    except Exception as e:
        return HttpResponse(err_code=500, details=str(e))

    return HttpResponse(err_code=200)

if __name__ == "__main__":
    parser = ArgumentParser(description="Run the MoneyBot application.")
    parser.add_argument("--preset", type=str, help="Configuration preset to load (expects config/<preset>.yml).")
    parser.add_argument("--port", type=int, default=SERVER_PORT, help="The port to run the application on.")
    parser.add_argument("--host", type=str, default=IPV4, help="The host to run the application on.")
    args = parser.parse_args()

    preset_to_use = args.preset or os.getenv("MONEYBOT_PRESET")
    if preset_to_use:
        CONFIG = load_config(preset=preset_to_use, fail_fast=bool(args.preset))
        _apply_runtime_config(CONFIG)
        _configure_cors(webapp, ALLOW_ORIGINS)

    uvicorn.run(
        webapp,
        host=args.host,
        port=args.port,
        loop="asyncio",
        ws_ping_interval=WS_PING_INTERVAL,
        ws_ping_timeout=WS_PING_TIMEOUT,
    )
