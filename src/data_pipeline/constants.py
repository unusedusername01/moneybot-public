# Copyright 2025 unusedusername01
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from datetime import datetime

ROOT_PATH = Path(__file__).resolve().parent.parent.parent
DATA_PATH = ROOT_PATH / "data"
PORTFOLIOS_PATH = DATA_PATH / "portfolios"
TICKERS_PATH = DATA_PATH / "tickers"
SECTOR_DB_PATH = DATA_PATH / "sector_databases"
MARKET_DB_PATH = DATA_PATH / "market_databases"
MODELS_PATH = DATA_PATH / "models"
GKG_DB_PATH = DATA_PATH / "gkg_databases"

def TICKER_PATH(ticker: str) -> Path:
    return TICKERS_PATH / ticker

def PREDICTIONS_PATH(ticker: str) -> Path:
    return TICKER_PATH(ticker) / "predictions"

def PORTFOLIO_PATH(portfolio_name: str) -> Path:
    return PORTFOLIOS_PATH / (portfolio_name + ".json")

def get_today_str() -> str:
    return datetime.now().strftime('%Y-%m-%d')

def get_file_prefix(data_type: str, field_value: str, timestamp: bool = True) -> str:
    """
    Generate a file prefix based on the data type and field value.
    """
    suffix = f"{data_type}_"
    prefix = f"{field_value.lower().replace(' ', '_')}_" if data_type == 'news' and field_value else ''
    today_str = get_today_str()

    return prefix + suffix + (today_str if timestamp else '')

def get_filename(data_type: str, field_value: str = None) -> str:
    """
    Add current date and file extension to the file prefix.
    """
    prefix = get_file_prefix(data_type, field_value)

    return f"{prefix}.json"