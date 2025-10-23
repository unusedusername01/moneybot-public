import json
from datetime import datetime, timedelta
from .constants import *
from typing import Dict, List, Tuple, Union, Any

class DataCollector:
    @staticmethod
    def get_all_tickers() -> List[str]:
        """
        Get all tickers from the data directory.
        Returns a list of ticker strings.
        """
        ticker_dir = TICKERS_PATH
        if not ticker_dir.exists():
            return []
        
        return [
            d.name for d in ticker_dir.iterdir()
            if d.is_dir() and all(c.isupper() or c == '.' for c in d.name) # Safe check for ticker format
        ]

    @staticmethod
    def extract_date(file: Path):
        try:
            return datetime.strptime(file.stem.split('_')[-1], '%Y-%m-%d')
        except Exception:
            return datetime.min
    
    @staticmethod
    def extract_predictions_info(file: Path):
        # Define strength order (strong > medium > weak)
        strength_order = {'strong': 3, 'medium': 2, 'weak': 1}
        try:
            # Extract mode and date from filename (e.g., predictions_medium_2025-07-09.json)
            parts = file.stem.split('_')
            if len(parts) >= 3:
                mode = parts[1]
                date_str = parts[2]
                date = datetime.strptime(date_str, '%Y-%m-%d')
                strength = strength_order.get(mode, 0)
                return file, date, strength
            return file, datetime.min, 0
        except Exception:
            return file, datetime.min, 0


    @staticmethod
    def collect_fundamentals(ticker: str, field: str = None, limit_days: int = 7) -> Union[Dict, float, str, None]: 
        """
        Load the latest fundamental data for a given ticker.
        If `field` is specified, return only its value; else return the full dict.
        Returns None if no data, field is found or if data is older than `limit_days`.
        """
        data_dir = TICKER_PATH(ticker)
        fundamental_files = list(data_dir.glob('fundamentals_*.json'))
        if not fundamental_files:
            return None

        latest_file = max(fundamental_files, key=DataCollector.extract_date)
        if (datetime.now() - DataCollector.extract_date(latest_file)).days > limit_days:
            return None
        
        with open(latest_file, 'r') as f:
            fundamental_data = json.load(f)

        if field:
            return fundamental_data.get(field, None)
        
        return fundamental_data

    @staticmethod
    def collect_predictions(ticker: str, field: str = None, limit_days: int = 7) -> Union[Dict, List, None]:
        """
        Load the latest prediction data for a given ticker.
        Prioritizes the strongest prediction mode (strong > medium > weak) and most recent data.
        If `field` is specified, return only its value; else return the full dict.
        Returns None if no data, field is found or if data is older than `limit_days`.
        """
        predictions_dir = PREDICTIONS_PATH(ticker)
        if not predictions_dir.exists():
            return None

        # Find all prediction files
        prediction_files = list(predictions_dir.glob('predictions_*.json'))
        if not prediction_files:
            return None

        # Sort files by date (most recent first) and strength (strong > medium > weak)
        file_info = [DataCollector.extract_predictions_info(f) for f in prediction_files]
        file_info.sort(key=lambda x: (x[1], x[2]), reverse=True)

        # Get the best file (highest strength, most recent)
        best_file, date, _ = file_info[0]

        if (datetime.now() - date).days > limit_days:
            return None
        
        with open(best_file, 'r') as f:
            predictions_data = json.load(f)
        
        return [prediction['prediction'] for prediction in predictions_data]

    @staticmethod
    def collect_news(ticker: str, field: str = None, limit_days: int = 7) -> Union[Dict, None]:
        """
        Load the latest combined news data for a given ticker.
        Prioritizes the most recent data.
        If `field` is specified, return only its value; else return the full dict.
        Returns None if no data, field is found or if data is older than `limit_days`.
        """
        data_dir = TICKER_PATH(ticker)
        news_files = list(data_dir.glob(f"{get_file_prefix('news', ticker, timestamp=False)}*.json"))
        if not news_files:
            return None

        latest_file = max(news_files, key=DataCollector.extract_date)
        if (datetime.now() - DataCollector.extract_date(latest_file)).days > limit_days:
            return None
        
        with open(latest_file, 'r') as f:
            news_data = json.load(f)

        if field:
            return {field: news_data.get(field)} if field in news_data else None
        
        return news_data
    
    @staticmethod
    def collect_sector_news(sector: str, limit_days: int = 7) -> Union[Dict, None]:
        """
        Load the latest sector news data.
        If `field` is specified, return only its value; else return the full dict.
        Returns None if no data, field is found or if data is older than `limit_days`.
        """
        data_dir = SECTOR_DB_PATH
        sector_files = list(data_dir.glob(f"{get_file_prefix('news', sector, timestamp=False)}*.json"))
        if not sector_files:
            return None
        
        latest_file = max(sector_files, key=DataCollector.extract_date)
        if (datetime.now() - DataCollector.extract_date(latest_file)).days > limit_days:
            return None
        
        with open(latest_file, 'r') as f:
            sector_news_data = json.load(f)

        return sector_news_data
    
    @staticmethod
    def collect_market_news(market: str, limit_days: int = 7) -> Union[Dict, None]:
        """
        Load the latest market news data.
        If `field` is specified, return only its value; else return the full dict.
        Returns None if no data, field is found or if data is older than `limit_days`.
        """
        data_dir = MARKET_DB_PATH
        market_files = list(data_dir.glob(f"{get_file_prefix('news', market, timestamp=False)}*.json"))
        if not market_files:
            return None
        
        latest_file = max(market_files, key=DataCollector.extract_date)
        if (datetime.now() - DataCollector.extract_date(latest_file)).days > limit_days:
            return None
        
        with open(latest_file, 'r') as f:
            market_news_data = json.load(f)

        return market_news_data