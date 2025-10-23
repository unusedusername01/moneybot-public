import json
import time
from typing import Optional, List, Dict, Any, Tuple, TypedDict, get_type_hints, Callable
from datetime import datetime
from langgraph.graph.state import CompiledStateGraph
from PIL import Image as PILImage
import io
import numpy as np
import yfinance as yf

from src.data_pipeline.constants import MODELS_PATH, PORTFOLIO_PATH, PORTFOLIOS_PATH
from src.data_pipeline.data_collector import DataCollector
from src.langgraph_workflow.custom_types import TickerData, FundamentalData, NewsData, PredictionData, PortfolioData, TickerDataList


_FX_CACHE: Dict[str, Tuple[float, float]] = {}
_FX_CACHE_TTL = 300.0  # seconds

def load_ticker_data(ticker: str) -> Optional[TickerData]:
    """
    Load ticker data from the data collector.
    Returns None if no data is found or truncated.
    """
    predictions: List[PredictionData] = DataCollector.collect_predictions(ticker)
    fundamentals: FundamentalData = DataCollector.collect_fundamentals(ticker)
    news: NewsData = DataCollector.collect_news(ticker)

    if not predictions or not fundamentals or not news:
        return None
    
    return TickerData(
        ticker=ticker,
        fundamentals=fundamentals,
        news=news,
        predictions=predictions
    )

def order_ticker_by_score(ticker_data: List[TickerData]) -> List[TickerData]:
    """
    Rank tickers by a composite score computed via NumPy:
      - sum of prediction_scores
      - trailing P/E
      - return on equity
      - debt to equity
      - profit margin
    Each metric is min–max normalized across the universe,
    weighted equally, then aggregated into a final score.
    """
    # Define equal weights for the five metrics
    weights = np.full(5, 0.5, dtype=np.float64)
    total_weight = weights.sum()

    n = len(ticker_data)
    # Preallocate raw metric array (5 metrics × n tickers)
    raw = np.zeros((5, n), dtype=np.float64)

    # 1) Populate raw metrics
    for i, t in enumerate(ticker_data):
        # Sum of prediction_scores
        psum = sum(p['prediction_score'] for p in t.get('predictions', []))
        f = t.get('fundamentals', {})
        raw[0, i] = psum
        raw[1, i] = f.get('trailing_pe', 0.0) or 0.0
        raw[2, i] = f.get('return_on_equity', 0.0) or 0.0
        raw[3, i] = f.get('debt_to_equity', 0.0) or 0.0
        raw[4, i] = f.get('profit_margin', 0.0) or 0.0

    # 2) Compute min and max for each metric (row)
    vmin = raw.min(axis=1, keepdims=True)
    vmax = raw.max(axis=1, keepdims=True)

    # 3) Min–max normalize: (x - min) / (max - min), clamp to [0,1]
    denom = np.clip(vmax - vmin, a_min=1e-6, a_max=None)  # avoid division by zero
    normed = (raw - vmin) / denom
    normed = np.clip(normed, 0.0, 1.0)

    # 4) Compute weighted score for each ticker
    #    `weights[:, None]` broadcasts to shape (5, n)
    scores = (weights[:, None] * normed).sum(axis=0) / total_weight

    # 5) Sort descending and reorder the input list
    sorted_indices = np.argsort(-scores)  # negative for descending order
    ordered = [ticker_data[i] for i in sorted_indices]

    return ordered


def group_by_criteria(ticker_data: List[TickerData], criteria: str, max_batch_size: int = 5, split_absolute: bool = False) -> List[TickerDataList]:
    """
    Group ticker data by a specified criteria (sector, market, score or market_cap).
    Args:
        ticker_data: List of TickerData to group.
        criteria: The criteria to group by ('sector', 'market', 'score', or 'market_cap').
        max_batch_size: Maximum size of each batch.
        split_absolute: If True, max_batch_size is treated as an absolute size.
    Returns:
        A list of lists, where each inner list contains TickerData objects that share the same criteria value.
    """
    if max_batch_size <= 0:
        raise ValueError("max_batch_size must be greater than 0")

    def split_balanced_groups(group, max_batch_size):
        n = len(group)
        if n <= max_batch_size:
            return [group]

        k = (n + max_batch_size - 1) // max_batch_size  # Minimum number of groups
        base = n // k
        extra = n % k

        result = []
        start = 0
        for i in range(k):
            size = base + (1 if i < extra else 0)
            result.append(group[start:start + size])
            start += size
        return result
    
    def split_absolute_groups(group, max_batch_size):
        n = len(group)
        if n <= max_batch_size:
            return [group]

        result = []
        for i in range(0, n - (n % max_batch_size), max_batch_size):
            result.append(group[i:i + max_batch_size])

        if n % max_batch_size != 0:
            # Add one last group of size equal to max_batch_size (overlapping on previous group)
            last_group = group[-max_batch_size:]
            result.append(last_group)

        return result
    
    splitter = split_balanced_groups if not split_absolute else split_absolute_groups
    
    if criteria in ['sector', 'market']:
        # Group by sector or market
        groups: Dict[str, List[TickerData]] = {}
        for ticker in ticker_data:
            key = ticker['fundamentals'].get(criteria, 'Unknown')
            groups.setdefault(key, []).append(ticker)

        result = []

        for key, group in groups.items():
            result.extend(splitter(group, max_batch_size))
            
    elif criteria in ['score', 'market_cap']:
        if criteria == 'score':
            ordered_tickers = order_ticker_by_score(ticker_data)
        elif criteria == 'market_cap':
            ordered_tickers = sorted(ticker_data, key=lambda x: x['fundamentals'].get('market_cap', 0), reverse=True)

        # Split into batches of max_batch_size
        result = splitter(ordered_tickers, max_batch_size)

    else:
        # Split the list directly
        result = splitter(ticker_data, max_batch_size)
    
    return result

def load_portfolio_data(portfolio_name: str) -> Optional[PortfolioData]:
    """
    Load portfolio from name
    """

    filepath = PORTFOLIO_PATH(portfolio_name)

    if not filepath.exists():
        return None

    assert filepath.resolve().parent == PORTFOLIOS_PATH.resolve(), "Invalid portfolio path. (Don't try and be sneaky like that bruh)"

    with open(filepath, "r") as f:
        data = json.load(f)

    return PortfolioData(
        budget=data.get("budget", 0.0),
        target_date=data.get("target_date", datetime.now().isoformat()),
        holdings=data.get("holdings", {}),
        currency=data.get("currency", "USD"),
        risk_tolerance=data.get("risk_tolerance", None),
        criteria=data.get("criteria", None),
        prediction_strength=data.get("prediction_strength", None)
    )

def select_prediction_horizons(target_date: datetime,
                               risk_tolerance: str) -> list[int]:
    """
    Returns a sorted list of integer day-horizons for prediction,
    adjusted for how far away `target_date` is and the investor's risk profile.
    """
    # 1. Compute days until the target (minimum 1) using date precision to avoid off-by-one
    today = datetime.now().date()
    target_day = target_date.date()
    days_until_target = max((target_day - today).days, 1)

    # 2. Base horizon buckets
    base_horizons = [1, 7, 30, 90, 180, 365]

    # 3. Risk‐based selection from the base buckets
    risk = risk_tolerance.lower()
    if risk == "low":
        # Focus on short‐ to medium‐term only
        allowed = base_horizons[:3]        # [1, 7, 30]
    elif risk == "medium":
        # Include up to 3‑month outlook
        allowed = base_horizons[:4]        # [1, 7, 30, 90]
    elif risk == "high":
        # Use full spectrum plus optional target‐date strike
        allowed = base_horizons[:]         # [1, 7, 30, 90, 180, 365]
    else:
        raise ValueError("risk_tolerance must be one of 'low', 'medium', 'high'")

    # 4. Clamp to the actual time‐to‐target
    horizons = [d for d in allowed if d <= days_until_target]

    # 5. Ensure we include an exact strike at target_date if nothing matches
    if days_until_target not in horizons:
        horizons.append(days_until_target)

    # 6. Return sorted, unique list
    return sorted(set(horizons))

def load_ticker_data(ticker: str, limit_days: int = 7) -> Optional[TickerData]:
    """
    Collect all relevant data for a ticker.
    Returns a dictionary with fundamentals, predictions, and news.
    If `field` is specified, return only its value; else return the full dict.
    Returns None if no data is found or if data is older than `limit_days`.
    """
    fundamentals = DataCollector.collect_fundamentals(ticker, limit_days=limit_days)
    predictions = DataCollector.collect_predictions(ticker, limit_days=limit_days)
    news = DataCollector.collect_news(ticker, limit_days=limit_days)
    
    result = TickerData(
        ticker=ticker,
        fundamentals=fundamentals,
        news=news,
        predictions=predictions
    )
    
    return result

def serialize_for_llm(obj: Dict[str, Any]) -> str:
    """
    – Drops all None values
    – Sorts keys
    – Emits the leanest legal JSON
    """
    import json
    cleaned = {k: v for k, v in obj.items() if v is not None}
    return json.dumps(cleaned, separators=(",", ":"), sort_keys=True)

def typed_dict_repr(
    td: Dict[str, Any],
    indent: int = 2,
    level: int = 0,
    max_list: int = 5,
    max_depth: int = 3
) -> str:
    """
    Concise repr of nested dicts/lists, with optional depth and list limits.
    """
    if level > max_depth:
        return '...'
    pad = ' ' * (indent * level)
    lines = []
    for key in sorted(td):
        val = td[key]
        if isinstance(val, dict):
            rep = typed_dict_repr(val, indent, level+1, max_list, max_depth)
        elif isinstance(val, list):
            elems = []
            for i, item in enumerate(val):
                if i >= max_list:
                    elems.append('...')
                    break
                elems.append(
                    typed_dict_repr(item, indent, level+2, max_list, max_depth)
                    if isinstance(item, dict) else repr(item)
                )
            rep = '[' + ', '.join(elems) + ']'
        else:
            rep = repr(val)
        lines.append(f"{pad}{' ' * indent}{repr(key)}: {rep}")
    if not lines:
        return '{}'
    return '{\n' + ',\n'.join(lines) + '\n' + pad + '}'

def render_batches(batches: List[TickerDataList]) -> List[Tuple[List[str], str]]:
    """
    Render a list of batches as a list of strings.
    """
    reprs = []

    for i, batch in enumerate(batches):
        ticker_names = [ticker['ticker'] for ticker in batch]
        str = f"Batch {i+1}: [\n"
        for ticker in batch:
            str += typed_dict_repr(ticker)
            str += ",\n"
        str += "]"
        reprs.append((ticker_names, str))

    return reprs

def show_graph(app: CompiledStateGraph) -> None:
    png_bytes = app.get_graph().draw_mermaid_png()
    image = PILImage.open(io.BytesIO(png_bytes))
    image.show()

def convert_currency(amount: float, from_currency: str, to_currency: str) -> float:
    """Convert ``amount`` between currencies using Yahoo Finance FX pairs.

    Yahoo exposes FX rates as synthetic tickers in the form ``{FROM}{TO}=X``.
    We first try the direct pair (e.g. ``USDCHF=X``). If no data is available we
    fall back to the inverse pair and divide instead. This avoids relying on
    external APIs that frequently reject anonymous traffic, which caused the
    previous ConnectionReset errors whenever the user selected a non-USD
    portfolio currency.
    """

    from_currency = (from_currency or "").upper()
    to_currency = (to_currency or "").upper()

    if not from_currency or not to_currency:
        raise ValueError("Both from_currency and to_currency must be provided")

    if from_currency == to_currency:
        return amount

    def _cached_rate(key: str, fetch: Callable[[], Optional[float]]) -> Optional[float]:
        now = time.time()
        cached = _FX_CACHE.get(key)
        if cached and now - cached[1] <= _FX_CACHE_TTL:
            return cached[0]
        rate = fetch()
        if rate is not None:
            _FX_CACHE[key] = (rate, now)
        return rate

    def _fetch_rate(pair: str) -> Optional[float]:
        ticker = yf.Ticker(pair)
        history = ticker.history(period="1d")
        if history.empty:
            return None
        return float(history["Close"].iloc[-1])

    direct_pair = f"{from_currency}{to_currency}=X"
    rate = _cached_rate(direct_pair, lambda: _fetch_rate(direct_pair))
    if rate is not None and rate > 0:
        return amount * rate

    inverse_pair = f"{to_currency}{from_currency}=X"
    inverse_rate = _cached_rate(inverse_pair, lambda: _fetch_rate(inverse_pair))
    if inverse_rate is not None and inverse_rate > 0:
        return amount / inverse_rate

    raise ValueError(f"Unable to find FX rate for {from_currency}->{to_currency}")

def get_shares_number(ticker: str, amount: float, currency: str, allow_fractional: bool = True) -> float:
    # Compute how many shares can be bought with a given amount of money
    stock = yf.Ticker(ticker)
    stock_currency = stock.info.get('currency', 'USD')

    if stock_currency != currency:
        amount = convert_currency(amount, currency, stock_currency)

    shares = amount / stock.history(period='1d')['Close'].iloc[-1]

    return round(shares, 2) if allow_fractional else int(shares)

def get_shares_price(
    ticker: str,
    number_of_shares: Optional[float] = None,
    currency: Optional[str] = None,
    allow_fractional: bool = True,
) -> float:
    """Return the latest price (or notional value) for the given ticker.

    If ``number_of_shares`` is provided we return the notional value for that
    many shares. When ``currency`` is set and differs from the ticker's native
    currency we convert the latest close before applying the share count.
    """

    stock = yf.Ticker(ticker)
    history = stock.history(period="1d")
    if history.empty:
        raise ValueError(f"No pricing data available for ticker {ticker}")

    latest_close = float(history["Close"].iloc[-1])
    stock_currency = stock.info.get("currency", "USD")

    if currency and currency != stock_currency:
        latest_close = convert_currency(latest_close, stock_currency, currency)

    if number_of_shares is None:
        return latest_close

    shares = float(number_of_shares)
    if not allow_fractional:
        shares = int(shares)

    return latest_close * shares