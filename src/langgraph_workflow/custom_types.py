# Copyright 2025 unusedusername01
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime
from enum import Enum
import json
from turtle import st
from typing import Any, Type, Union, Optional, Dict, List, Callable, TypedDict, Annotated, Sequence, Tuple, Literal
from langchain_core.messages import BaseMessage
from operator import add as add_messages
from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_extra_types.currency_code import Currency

# --- SCHEMAS ---
class MessageType(str, Enum):
    STATE_UPDATE = "state_update"
    AWAITING_CHOICE = "awaiting_choice"
    AWAITING_MESSAGE = "awaiting_message"
    AWAITING_ALLOCATION = "awaiting_allocation"
    CHOICE_RESPONSE = "choice_response"
    MESSAGE_RESPONSE = "message_response"
    ALLOCATION_RESPONSE = "allocation_response"
    END_ANALYSIS = "end_analysis"
    HTTP_REQUEST = "http_request"
    HTTP_RESPONSE = "http_response"

class MessageData(BaseModel):
    type: MessageType

# --- SERVER SIDE ---
class StateUpdate(MessageData):
    type: Literal[MessageType.STATE_UPDATE] = MessageType.STATE_UPDATE
    prompt: str

class AwaitingSchema(MessageData):
    pass

class AwaitingChoiceRequest(AwaitingSchema):
    type: Literal[MessageType.AWAITING_CHOICE] = MessageType.AWAITING_CHOICE
    choices: list[str] = Field(default_factory=lambda: ["confirm", "edit", "deny", "cancel"])
    prompt: str

class AwaitingMessageRequest(AwaitingSchema):
    type: Literal[MessageType.AWAITING_MESSAGE] = MessageType.AWAITING_MESSAGE
    prompt: str

class AwaitingAllocationRequest(AwaitingSchema):
    type: Literal[MessageType.AWAITING_ALLOCATION] = MessageType.AWAITING_ALLOCATION
    prompt: str
    expected_keys: list[str] | None = None

class HttpResponse(MessageData):
    type: Literal[MessageType.HTTP_RESPONSE] = MessageType.HTTP_RESPONSE
    err_code: int = 0
    details: Optional[str] = None

class GetSharesPriceResponse(HttpResponse):
    ticker: Union[str, List[str]]
    price: Optional[Union[float, List[float]]] = None # Can be None if we want to use get_shares_price to validate a ticker's existence

class LoadPortfolioDataResponse(HttpResponse):
    portfolio_data: Dict[str, Union[str, int, float, Dict[str, float], None]]

class ListPortfoliosResponse(HttpResponse):
    portfolios: List[str]

# --- CLIENT SIDE ---
class ChoiceResponse(MessageData):
    type: Literal[MessageType.CHOICE_RESPONSE] = MessageType.CHOICE_RESPONSE
    selection: str

class MessageResponse(MessageData):
    type: Literal[MessageType.MESSAGE_RESPONSE] = MessageType.MESSAGE_RESPONSE
    content: str

class AllocationResponse(MessageData):
    class ValuesType(str, Enum):
        SHARES = "shares"
        CASH = "cash"
    type: Literal[MessageType.ALLOCATION_RESPONSE] = MessageType.ALLOCATION_RESPONSE
    content: Dict[str, float]
    values: ValuesType = ValuesType.SHARES
    currency: Optional[Currency] = None

class EndAnalysis(MessageData):
    type: Literal[MessageType.END_ANALYSIS] = MessageType.END_ANALYSIS

class HttpRequest(MessageData):
    type: Literal[MessageType.HTTP_REQUEST] = MessageType.HTTP_REQUEST

class GetSharesPriceRequest(HttpRequest):
    ticker: Union[str, List[str]]
    amount: Optional[Union[float, List[float]]]
    currency: Optional[Currency] = Currency("USD")
    allow_fractional: bool = True

class PortfolioDataRequest(HttpRequest):
    portfolio_id: str
    # NOTE: Maybe add a user_id: str ?

class CurrencyValidationRequest(HttpRequest):
    currency: str

class EditPortfolioRequest(HttpRequest):
    portfolio_id: str
    budget: int
    target_date: str               # expected format dd-mm-YYYY converted to YYYY-MM-DD
    holdings: Dict[str, float]
    currency: str                  # ISO 4217 code
    risk_tolerance: Optional[str] = None
    criteria: Optional[str] = None
    prediction_strength: Optional[str] = None

    @model_validator(mode='after')
    def validate_all_fields(self) -> dict:
        # 1. budget must be non-negative integer
        budget = self.budget
        if not isinstance(budget, int) or budget < 0:
            raise ValueError('budget must be a non-negative integer')

        # 2. target_date must parse with dd-mm-YYYY
        td = self.target_date
        if not isinstance(td, str):
            raise TypeError('target_date must be a string')
        try:
            datetime.strptime(td, '%Y-%m-%d')
        except ValueError:
            raise ValueError('target_date must be in YYYY-MM-DD format')

        # 3. holdings must be dict[str, float]
        holdings = self.holdings
        if not isinstance(holdings, dict):
            raise TypeError('holdings must be a dictionary')
        for ticker, qty in holdings.items():
            if not isinstance(ticker, str):
                raise TypeError('holdings keys must be strings')
            if not isinstance(qty, float):
                raise TypeError('holdings values must be floats')

        # 4. currency must be a non-empty string
        currency = self.currency
        if not isinstance(currency, str) or not currency:
            raise ValueError('currency must be a non-empty string')

        # 5. optional fields must be one of the allowed choices or None
        choices_map = {
            'risk_tolerance': {'low', 'medium', 'high'},
            'criteria': {'sector', 'market', 'score', 'market_cap'},
            'prediction_strength': {'weak', 'medium', 'strong'},
        }
        for field, allowed in choices_map.items():
            val = getattr(self, field)
            if val is not None and val not in allowed:
                raise ValueError(f"{field} must be one of {allowed} or None")

        return self

class DeletePortfolioRequest(HttpRequest):
    portfolio_id: str

# --- CONSTANTS ---
# Mapping from the `type` (string) to the concrete pydantic model
_TYPE_TO_MODEL: Dict[str, Type[MessageData]] = {
    MessageType.STATE_UPDATE.value: StateUpdate,
    MessageType.AWAITING_CHOICE.value: AwaitingChoiceRequest,
    MessageType.AWAITING_MESSAGE.value: AwaitingMessageRequest,
    MessageType.AWAITING_ALLOCATION.value: AwaitingAllocationRequest,
    MessageType.CHOICE_RESPONSE.value: ChoiceResponse,
    MessageType.MESSAGE_RESPONSE.value: MessageResponse,
    MessageType.ALLOCATION_RESPONSE.value: AllocationResponse,
    MessageType.END_ANALYSIS.value: EndAnalysis,
    }

# --- MESSAGE PAYLOAD ---
class MessagePayload(BaseModel):
    data: Annotated[
        Union[
            StateUpdate,
            AwaitingChoiceRequest, 
            AwaitingMessageRequest,
            AwaitingAllocationRequest,
            ChoiceResponse,
            MessageResponse,
            AllocationResponse,
            EndAnalysis
        ], 
        Field(discriminator='type')
    ]
    timeout: Optional[int] = None  # in seconds
    
    def to_dict(self) -> dict:
        """Convert the MessagePayload instance to a dictionary."""
        return self.model_dump()
    
    def to_json(self) -> str:
        """Convert the MessagePayload instance to a JSON string."""
        return self.model_dump_json()
    
    @classmethod
    def export_json_schema(cls) -> str:
        """Export the JSON schema for MessagePayload as a string."""
        return json.dumps(cls.model_json_schema(), indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'MessagePayload':
        """Create a MessagePayload instance from a JSON string."""
        return cls.model_validate_json(json_str)

# --- Langgraph types ---

class AppData(TypedDict):
    """ Data exposed to the webapp """
    selected_portfolio: Optional[str]
    messages: Annotated[Sequence[BaseMessage], add_messages] # Synthetizer-User messages
    budget_allocation: Dict[str, float]  # Allocated budget for each ticker
    awaiting: bool = False     # is waiting for user
    await_schema: AwaitingSchema
    version: int               # optimistic locking
    next_node: Optional[str] = None

class FundamentalData(TypedDict):
    trailing_pe: Optional[float]
    peg_ratio: Optional[float]
    price_to_book: Optional[float]
    return_on_equity: Optional[float]
    debt_to_equity: Optional[float]
    profit_margin: Optional[float]
    revenue_growth: Optional[float]
    sector: Optional[str]
    short_name: Optional[str]
    market: Optional[str]

class NewsData(TypedDict):
    ticker: str
    timestamp: str
    news: List[Dict[str, str]]
    total_articles: int

class PredictionData(TypedDict):
    ticker: str
    timestamp: str
    prediction_mode: str
    expected_price: float
    current_price: float
    prediction_score: float
    days_ahead: int
    prediction_date: str
    model_details: Dict[str, Union[str, float, List[float]]]

class TickerData(TypedDict):
    ticker: str
    fundamentals: FundamentalData
    news: NewsData
    predictions: List[PredictionData]

TickerDataList = List[TickerData]

class RankedBatchData(TypedDict):
    tickers: List[str]
    scores: Dict[str, Tuple[int, str]]  # Dict of ticker symbol to (score, review) 0 <= score <= 100

class BatchesData(TypedDict):
    data: List[TickerDataList]  # List of lists of TickerData
    criteria: str # Can be 'sector', 'market', or 'score'. Used to group tickers in batches for processing.
    ranking: Optional[Callable[[List[TickerData]], List[TickerData]]] = None # Optional ranking to sort tickers if score is the criteria.

class PortfolioData(TypedDict):
    """Portfolio data to be used by the webapp"""
    # Primary data (needed for portfolio management)
    budget: int
    target_date: str # yyyy-mm-dd
    holdings: Dict[str, float]
    currency: str # ISO 4217 currency code

    # Optional data (used for better results)
    risk_tolerance: Optional[str]  # e.g., 'low', 'medium', 'high'
    criteria: Optional[str]  # e.g., 'sector', 'market', 'score', 'market_cap'
    prediction_strength: Optional[str]  # e.g., 'weak', 'medium', 'strong'