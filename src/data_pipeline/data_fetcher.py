import json
from datetime import datetime, timedelta
import torch
import yfinance as yf
import requests
from urllib.parse import urlparse
from typing import Dict, List, Tuple, Union, Any, Optional
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity
import time
from dotenv import load_dotenv
import os


from src.data_pipeline.llm_provider import BaseLLMProvider
from src.data_pipeline.constants import *
from src.data_pipeline.data_collector import DataCollector

class DataFetcher:
    FUNDAMENTALS_FIELDS = [
        'trailingPE', 'pegRatio', 'priceToBook', 'returnOnEquity',
        'debtToEquity', 'profitMargins', 'revenueGrowth', 'sector',
        'shortName', 'market', 
    ]

    @staticmethod
    def ensure_dir(field_name: str, field_value: str = None) -> Path:
        match field_name:
            case 'ticker':
                path = TICKER_PATH(field_value)
            case 'sector':
                path = SECTOR_DB_PATH
            case 'market':
                path = MARKET_DB_PATH
            case 'gkg':
                path = GKG_DB_PATH
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
        
        return path

    @staticmethod
    def save_daily_log(data: Dict, field_name: str, field_value: str, data_type: str) -> Path:
        if not field_name or not field_value or field_name not in ['ticker', 'sector', 'market']:
            raise ValueError("Field name and value must be provided and field name must be one of: 'ticker', 'sector', 'market'")
        
        save_dir = DataFetcher.ensure_dir(field_name, field_value)

        filepath = save_dir / get_filename(data_type, field_value)

        # If not ticker (sector or market), check if file exists and update
        if field_name != 'ticker':
            if filepath.exists():
                with open(filepath, 'r') as f:
                    existing_data = json.load(f)

                if field_name == 'sector' and 'news' in existing_data and 'news' in data:
                    existing_urls = {article.get('url') for article in existing_data['news']}

                    accumulated_news = existing_data['news'].copy()
                    for article in data['news']:
                        if article.get('url') not in existing_urls:
                            accumulated_news.append(article)

                    merged = existing_data.copy()
                    merged.update({k: v for k, v in data.items() if k != 'news'})
                    merged['news'] = accumulated_news
                    merged['total_articles'] = len(accumulated_news)
                    data = merged
                else:
                    existing_data.update(data)
                    data = existing_data
        
        # Save the data to the file
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        return filepath


    @staticmethod
    def fetch_historical_prices(ticker, period='5y', interval='1d'):
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period, interval=interval)
        if not hist.empty:
            hist = hist.reset_index()
            for col in hist.columns:
                if str(hist[col].dtype).startswith('datetime'):
                    hist[col] = hist[col].astype(str)
            data = hist.to_dict(orient='records')
        else:
            data = []
        # Save the historical prices
        DataFetcher.save_daily_log(data, 'ticker', ticker, 'historical_prices')

        return data

    @staticmethod
    def fetch_fundamentals(ticker: str) -> Dict[str, Union[None, float, str]]:
        stock = yf.Ticker(ticker)
        info = stock.info
        fundamentals = {k: info.get(k) for k in DataFetcher.FUNDAMENTALS_FIELDS}

        # Save the fundamentals data
        DataFetcher.save_daily_log(fundamentals, 'ticker', ticker, 'fundamentals')
        return fundamentals

class NewsFetcher:
    # Constants for maximum articles to fetch
    MAX_NEWS_PER_SECTOR = 20
    MAX_NEWS_PER_MARKET = 20

    @staticmethod
    def format_field_value(field_value: str) -> str:
        """
        Format the field value to be used in filenames.
        """
        return field_value.replace(' ', '_').lower()

    def __init__(
            self, llm_provider: BaseLLMProvider, newsdata_api_key: Optional[str] = None,
            k_ticker=5, k_sector=5, k_market=5, use_themes=True
        ):
        load_dotenv()

        self.newsdata_api_key = newsdata_api_key or os.getenv('NEWSDATA_API_KEY')
        
        if llm_provider is None:
            raise ValueError("Invalid keys")
        self.llm_provider = llm_provider
        self.k_ticker = k_ticker
        self.k_sector = k_sector
        self.k_market = k_market

        self.k_ticker_dict = {}
        self.k_sector_dict = {}
        self.k_market_dict = {}

        self.use_themes = use_themes

        # Initialize sentence transformer model
        self.model: SentenceTransformer = SentenceTransformer('all-MiniLM-L6-v2')

        if use_themes:
            raw_db = self._get_gkg_database()
            keys_list = list(raw_db.keys())
            embeddings = self.model.encode(keys_list, convert_to_tensor=False, batch_size=32)
            
            self.gkg_vector_db = {torch.Tensor(emb): raw_db[key] for emb, key in zip(embeddings, keys_list)}

    def _check_curr_news_count(self, field_name: str, field_value: str) -> int:
        if field_name not in ['ticker', 'sector', 'market']:
            raise ValueError("Field name must be one of: 'ticker', 'sector', 'market'")
        
        match field_name:
            case 'ticker':
                path = TICKER_PATH(field_value)
            case 'sector':
                path = SECTOR_DB_PATH
            case 'market':
                path = MARKET_DB_PATH

        filepath = path / get_filename('news', field_value)

        # Check if the path exists
        if not path.exists() or not filepath.exists():
            match field_name:
                case 'ticker':
                    return 0
                case 'sector':
                    self.k_sector_dict[field_value] = self.MAX_NEWS_PER_SECTOR
                    return 0
                case 'market':
                    self.k_market_dict[field_value] = self.MAX_NEWS_PER_MARKET
                    return 0
        else:
            with open(filepath, 'r') as f:
                data = json.load(f)

            count = data.get('total_articles', 0)

            if field_name == 'sector':
                self.k_sector_dict[field_value] = max(self.MAX_NEWS_PER_SECTOR - count, 0)
            elif field_name == 'market':
                self.k_market_dict[field_value] = max(self.MAX_NEWS_PER_MARKET - count, 0)

            return count

    def _compute_similarity_scores(self, articles: List[Dict], keywords: List[str]) -> List[Tuple[Dict, float]]:
        """
        Compute similarity scores between articles and keywords using vector similarity.
        
        Args:
            articles: List of article dictionaries with 'title' and 'summary'
            keywords: List of keywords to compare against
            
        Returns:
            List of tuples (article, similarity_score) sorted by score
        """
        if not articles or not keywords:
            return [(article, 0.0) for article in articles]
            
        # Combine title and summary for each article
        article_texts = [f"{a.get('title', '')} {a.get('summary', '')}" for a in articles]
        
        # Encode articles and keywords
        article_embeddings = self.model.encode(article_texts, convert_to_tensor=True)
        keyword_embeddings = self.model.encode(keywords, convert_to_tensor=True)
        
        # Compute cosine similarity between each article and all keywords
        # Shape: (n_articles, n_keywords)
        similarities = util.pytorch_cos_sim(article_embeddings, keyword_embeddings)
        
        # Get max similarity score for each article (best matching keyword)
        max_similarities = similarities.max(dim=1)[0].cpu().numpy()
        
        # Pair articles with their scores and sort by score
        scored_articles = list(zip(articles, max_similarities))
        return sorted(scored_articles, key=lambda x: x[1], reverse=True)


    def fetch_news(self, ticker, rate_limit=True):
        """
        Main method to fetch news from all three sources: ticker, sector, and market.
        
        Args:
            ticker (str): The stock ticker symbol
            
        Returns:
            Dict: Combined news from all sources
        """
        print(f"Fetching comprehensive news for {ticker}...")
        
        save_format = lambda field_name, field_value, news: {
            field_name: field_value,
            'timestamp': datetime.now().isoformat(),
            'news': news,
            'total_articles': len(news)
        }

        # Fetch ticker specific news
        ticker_news = self._fetch_ticker_news(ticker)

        # Save ticker news (Erasing current news if exists)
        DataFetcher.save_daily_log(save_format('ticker', ticker, ticker_news), 'ticker', ticker, 'news')
        
        # Check and update needed news counts
        sector = DataCollector.collect_fundamentals(ticker, field='sector')
        market = DataCollector.collect_fundamentals(ticker, field='market')
        self._check_curr_news_count('sector', sector)
        self._check_curr_news_count('market', market)
        
        # Fetch sector and market news
        sector_news = self._fetch_sector_news(ticker)
        if rate_limit:
            time.sleep(10)
        market_news = self._fetch_market_news(ticker)
        if rate_limit:
            time.sleep(10)
        
        # Save sector and market news (Augmenting current news)
        sector_data = save_format('sector', sector, sector_news)
        market_data = save_format('market', market, market_news)

        DataFetcher.save_daily_log(sector_data, 'sector', sector, 'news')
        DataFetcher.save_daily_log(market_data, 'market', market, 'news')

    def _fetch_ticker_news(self, ticker):
        """Fetch news for a specific ticker using Yahoo Finance's native summary data."""        
        try:
            print(f"Fetching {self.k_ticker} news articles for {ticker}...")
            
            # Get fresh news data using get_news() method
            stock = yf.Ticker(ticker)
            news_items = stock.get_news()
            
            if not news_items:
                print(f"No news found for {ticker}")
                return []
            
            processed_articles = []
            
            for item in news_items:
                if len(processed_articles) >= self.k_ticker:
                    break
                
                # Extract from nested content structure
                content = item.get("content", {})
                if not content:
                    continue
                
                # Get essential fields
                title = content.get("title", "").strip()
                summary = content.get("summary", "").strip()
                
                # Handle URL extraction from nested structure
                url_info = content.get("canonicalUrl") or content.get("clickThroughUrl", {})
                url = url_info.get("url", "").strip() if isinstance(url_info, dict) else ""
                
                # Skip if missing essential data
                if not title or not url:
                    continue
                
                # Get publisher info
                provider = content.get("provider", {})
                source = provider.get("displayName", "Yahoo Finance") if isinstance(provider, dict) else "Yahoo Finance"
                
                # Build article object
                processed_articles.append({
                    'title': title,
                    'summary': summary,
                    'content': summary,  # Use summary as content instead of scraping
                    'url': url,
                    'published': content.get("pubDate", ""),
                    'source': source,
                    'category': 'ticker_news'
                })
                
                # Rate limiting
                time.sleep(1)
            
            print(f"Successfully fetched {len(processed_articles)} articles for {ticker}")
            return processed_articles
            
        except Exception as e:
            print(f"Error fetching news for {ticker}: {str(e)}")
            return []

    def _fetch_sector_news(self, ticker):
        """Fetch news related to the ticker's sector."""
        sector = DataCollector.collect_fundamentals(ticker, field='sector')
        if self.k_sector_dict.get(sector, 0) <= 0:
            print(f"No more sector news to fetch for {ticker}, max_limit={self.MAX_NEWS_PER_SECTOR} reached")
            return []
        try:
            to_fetch = min(self.k_sector_dict.get(sector, 0), self.k_sector)
            print(f"Fetching {to_fetch} sector news for {ticker}...")

            # Get or generate sector keywords
            sector_keywords = self._get_cached_keywords(ticker, 'sector')
            
            if not sector_keywords:
                print(f"Generating sector keywords for {ticker}...")
                sector_keywords = self.llm_provider.generate_sector_keywords(ticker)
                print(f"Generated sector keywords: {sector_keywords}")
                self._cache_keywords(ticker, sector_keywords, 'sector')
            
            # Fetch news using sector keywords
            all_articles = []
            for keyword in sector_keywords:
                if self.newsdata_api_key:
                    articles = self._fetch_news_by_keyword_newsdata(keyword)
                else:
                    articles = self._fetch_news_by_keyword_gdelt(
                        keyword,
                        timespan='1d'
                    )
                all_articles.extend(articles)
                time.sleep(1)  # Rate limiting
            
            # Filter and select top articles
            if all_articles:
                filtered_articles = self._filter_by_relevance(all_articles, sector_keywords, to_fetch)
                # Add category
                for article in filtered_articles:
                    article['category'] = 'sector'
                print(f"Successfully fetched {len(filtered_articles)} sector news articles")
                return filtered_articles
            else:
                print("No sector articles found")
                return []
                
        except Exception as e:
            print(f"Error in _fetch_sector_news: {str(e)}")
            return []


    def _fetch_market_news(self, ticker):
        """Fetch news related to the ticker's market (global/geopolitical)."""
        market = DataCollector.collect_fundamentals(ticker, field='market')
        if self.k_market_dict.get(market, 0) <= 0:
            print(f"No more market news to fetch for {ticker}, max_limit={self.MAX_NEWS_PER_MARKET} reached")
            return []
        try:
            to_fetch = min(self.k_market_dict.get(market, 0), self.k_market)
            print(f"Fetching {to_fetch} market news for {ticker}...")

            # Get or generate market keywords
            market_keywords = self._get_cached_keywords(ticker, 'market')
            
            if not market_keywords:
                print(f"Generating market keywords for {ticker}...")
                market_keywords = self.llm_provider.generate_market_keywords(ticker)
                print(f"Generated market keywords: {market_keywords}")
                self._cache_keywords(ticker, market_keywords, 'market')
            
            # Fetch news using market keywords
            all_articles = []
            for keyword in market_keywords:
                if self.newsdata_api_key:
                    articles = self._fetch_news_by_keyword_newsdata(keyword, category='world')
                else:
                    articles = self._fetch_news_by_keyword_gdelt(
                        keyword,
                        timespan='1d'
                    )
                all_articles.extend(articles)
                time.sleep(1)  # Rate limiting
            
            # Filter and select top articles
            if all_articles:
                filtered_articles = self._filter_by_relevance(all_articles, market_keywords, to_fetch)
                # Add category
                for article in filtered_articles:
                    article['category'] = 'market'
                print(f"Successfully fetched {len(filtered_articles)} market news articles")
                return filtered_articles
            else:
                print("No market articles found")
                return []
                
        except Exception as e:
            print(f"Error in _fetch_market_news: {e}")
            return []


    def _fetch_news_by_keyword_newsdata(self, keyword, category='business'):
        """Fetch news articles for a specific keyword using newsdata.io API."""
        try:
            if not self.newsdata_api_key:
                print("No newsdata API key provided")
                return []
                
            response = requests.get(
                'https://newsdata.io/api/1/news',
                params={
                    'apikey': self.newsdata_api_key,
                    'q': keyword,
                    'language': 'en',
                    'category': category
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 'success':
                    articles = data.get('results', [])
                    
                    # Process articles
                    processed_articles = []
                    for article in articles:
                        processed_articles.append({
                            'title': article.get('title', ''),
                            'summary': article.get('description', ''),
                            'url': article.get('link', ''),
                            'published': article.get('pubDate', ''),
                            'source': article.get('source_id', ''),
                            'keyword': keyword
                        })
                    
                    return processed_articles
                else:
                    print(f"API error for keyword {keyword}: {data.get('message', 'Unknown error')}")
                    return []
            else:
                print(f"HTTP error for keyword {keyword}: {response.status_code}")
                return []
                
        except Exception as e:
            print(f"Error fetching news for keyword {keyword}: {str(e)}")
            return []

    def _fetch_news_by_keyword_gdelt(self, keyword: str, category: str = None,
                                 timespan: str = '24h', maxrecords: int = 200, top_k: int = 3):
        """
        Fetch news articles for a specific keyword using GDELT DOC 2.0 API,
        dynamically mapping any category string via embedding similarity.
        Returns the same JSON structure as the newsdata.io function.
        """
        LANGUAGE = 'sourcelang:english'
        QUERY_LIMIT_SIZE = 250 # 250 characters max per query

        # Remove size 2 or less word from keyword for it to be valid
        keyword = ' '.join([kwd for kwd in keyword.split(' ') if len(kwd) > 2])

        # A full query looks like: <keyword1> <keyword2> ... <keywordN> sourcelang:english (<category1> <category2> ... <categoryN>)
        def _map_category_to_query(cat: str) -> str:
            if not cat:
                return ""
            # Canonical GDELT themes (expandable)
            themes = [
                'business', 'economy', 'finance', 'market', 'company',
                'technology', 'innovation', 'software', 'startups', 'digital',
                'politics', 'government', 'election', 'policy', 'diplomacy',
                'health', 'medical', 'healthcare', 'medicine', 'public health',
                'sports', 'football', 'basketball', 'soccer', 'athletics',
                'entertainment', 'movie', 'music', 'celebrity', 'film',
                'science', 'research', 'study', 'discovery', 'environment',
                'climate', 'sustainability', 'ecology', 'wildlife', 'world'
            ]
            # Embed category + themes
            cat_emb = self.model.encode(cat, convert_to_tensor=True)
            theme_embs = self.model.encode(themes, convert_to_tensor=True)
            # Compute similarities and pick top_k
            scores = util.cos_sim(cat_emb, theme_embs)[0]
            top_idxs = scores.topk(k=top_k).indices.tolist()
            chosen = [themes[i] for i in top_idxs]
            # Build OR fragment
            return " (" + " OR ".join(f"{t}" for t in chosen) + ")"
        
        def _remove_element(type: str, text: str, min_elements: int = 1) -> str:
            """
            Remove the last element of a specific type from the text.
            Robust version that handles all edge cases properly.
            """
            if type not in ['keyword', 'category', 'theme']:
                raise ValueError("Type must be 'keyword', 'category', or 'theme'.")
            if min_elements < 1:
                raise ValueError("min_elements must be at least 1.")
            
            # Validate input format
            if LANGUAGE not in text:
                raise ValueError(f"Text must contain '{LANGUAGE}' separator")
            
            # Parse text structure safely
            parts = text.split(LANGUAGE, 1)  # Split only on first occurrence
            keywords_section = parts[0].strip()
            after_language = parts[1] if len(parts) > 1 else ""
            
            # Extract keywords (filter empty strings from splitting)
            keywords = [k for k in keywords_section.split() if k] if keywords_section else []
            
            # Handle themes section (marked by parentheses)
            paren_start = after_language.find('(')
            
            if paren_start >= 0:
                # Has parentheses - split categories and themes
                categories_section = after_language[:paren_start].strip()
                themes_section = after_language[paren_start+1:].rstrip(')')
                
                categories = [c for c in categories_section.split() if c] if categories_section else []
                themes_tokens = themes_section.split() if themes_section.strip() else []
                themes = [t for t in themes_tokens if t != 'OR']
                has_parentheses = True
            else:
                # No parentheses - everything is categories
                categories = [c for c in after_language.split() if c] if after_language.strip() else []
                themes = []
                has_parentheses = False
            
            # Apply removal logic (only if above minimum threshold)
            if type == 'keyword' and len(keywords) > min_elements:
                keywords = keywords[:-1]
            elif type == 'category' and len(categories) > min_elements:
                categories = categories[:-1]  
            elif type == 'theme' and len(themes) > min_elements:
                themes = themes[:-1]
            
            # Reconstruct text maintaining original format
            result_parts = []
            
            # Keywords section
            if keywords:
                result_parts.append(' '.join(keywords))
            
            # Language separator
            result_parts.append(LANGUAGE)
            
            # Categories section  
            if categories:
                result_parts.append(' '.join(categories))
            
            # Themes section
            if has_parentheses:
                if themes:
                    themes_part = '(' + ' OR '.join(themes) + ')'
                else:
                    themes_part = '()'
                result_parts.append(themes_part)
            
            return ' '.join(result_parts)

        # Build GDELT query
        fragments = [keyword, LANGUAGE]
        if category:
            fragments.append(_map_category_to_query(category))
        if self.use_themes:
            top_K_themes = 5

            keyword_vector = self.model.encode(keyword, convert_to_tensor=True)
            device = keyword_vector.device

            # Build parallel structures from the dict
            gkg_vectors = torch.stack(list(self.gkg_vector_db.keys())).to(device)  # [N, d]
            gkg_themes = list(self.gkg_vector_db.values())  # [N]

            # Compute similarities
            theme_similarities = util.cos_sim(keyword_vector, gkg_vectors)  # [1, N]
            top_theme_indices = theme_similarities.topk(k=top_K_themes).indices[0].tolist()

            # Index into themes, not the dict
            gkg_themes = [f"theme:{gkg_themes[i]}" for i in top_theme_indices]

            if gkg_themes:
                gkg_themes_query = "(" + " OR ".join(gkg_themes) + ")"
                fragments.append(gkg_themes_query)


        full_query = " ".join(fragments)

        priority = ['keyword', 'category', 'theme']
        i = 0
        while len(full_query) > QUERY_LIMIT_SIZE:
            full_query = _remove_element(priority[i], full_query)
            i += 1
            i %= len(priority)

        params = {
            'query': full_query,
            'mode': 'artlist',
            'format': 'json',
            'maxrecords': maxrecords,
            'timespan': timespan,
            'sort': 'datedesc',
        }

        # Rate-limit per GDELT policy
        time.sleep(5)
        response = requests.get('https://api.gdeltproject.org/api/v2/doc/doc',
                                params=params, timeout=30)

        if response.status_code == 429:
            print(f"Rate limit exceeded for keyword {keyword}.")
            return []
        if response.status_code != 200:
            print(f"HTTP error {response.status_code} for keyword {keyword}")
            return []

        try:
            data = response.json()
        except ValueError:
            print(f"Error decoding JSON response for keyword {keyword}.", response.text)
            return []
        
        articles = data.get('articles', [])

        # Helpers for formatting
        def _format_date(gdelt_date):
            if len(gdelt_date or "") >= 14:
                y, mo, d, h, mi, s = (gdelt_date[:4], gdelt_date[4:6],
                                    gdelt_date[6:8], gdelt_date[8:10],
                                    gdelt_date[10:12], gdelt_date[12:14])
                return f"{y}-{mo}-{d} {h}:{mi}:{s}"
            return gdelt_date or ""

        def _extract_source(url):
            try:
                dom = urlparse(url).netloc.lower()
                return dom[4:] if dom.startswith('www.') else dom
            except:
                return ''


        # Normalize format
        processed = []
        try:
            for art in articles:
                processed.append({
                    'title': art.get('title', ''),
                    'summary': art.get('title', ''),
                    'url': art.get('url', ''),
                    'published': _format_date(art.get('seendate', '')),
                    'source': _extract_source(art.get('url', '')),
                    'keyword': keyword
                })
        except Exception as e:
            print(f"Error processing GDELT articles: {str(e)}")
            return []


        return processed


    def _get_cached_keywords(self, ticker, keyword_type):
        """Get cached keywords for a ticker if they exist and are recent."""
        try:
            # Create ticker directory if it doesn't exist
            ticker_dir = DataFetcher.ensure_dir('ticker', ticker)
                
            # Path to keywords cache file
            cache_file = ticker_dir / f"{keyword_type}_keywords.json"
            
            if cache_file.exists():
                print(f"Found cached {keyword_type} keywords for {ticker}")
                with open(cache_file, 'r') as f:
                    cache_data = json.load(f)
                    
                return cache_data['keywords']
                    
            return None
            
        except Exception as e:
            print(f"Error reading {keyword_type} keyword cache: {str(e)}")
            return None


    def _cache_keywords(self, ticker, keywords, keyword_type):
        """Cache generated keywords for a ticker."""
        try:
            ticker_dir = DataFetcher.ensure_dir('ticker', ticker)
            
            # Path to keywords cache file
            cache_file = ticker_dir / f'{keyword_type}_keywords.json'
            
            # Save keywords with timestamp
            cache_data = {
                'timestamp': datetime.now().isoformat(),
                'keywords': keywords,
                'ticker': ticker,
                'type': keyword_type
            }
            
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
                
            print(f"Cached {keyword_type} keywords for {ticker}")
            
        except Exception as e:
            print(f"Error caching {keyword_type} keywords: {str(e)}")


    def _filter_by_relevance(self, articles, keywords, top_k, duplicate_threshold=0.8):
        """
        Filter articles by relevance and remove duplicates.
        
        Args:
            articles: List of article dictionaries
            keywords: List of keywords for relevance scoring
            top_k: Number of top articles to return
            duplicate_threshold: Similarity threshold for duplicate detection
            
        Returns:
            List of filtered articles
        """
        if not articles or top_k <= 0:
            return []

        try:
            # Compute similarity scores
            scored_articles = self._compute_similarity_scores(articles, keywords)
            
            # Remove duplicates based on content similarity
            unique_articles = []
            seen_vectors = []
            
            for article, score in scored_articles:
                if len(unique_articles) >= top_k:
                    break
                    
                # Create article text for similarity check
                article_text = f"{article.get('title', '')} {article.get('summary', '')}"
                article_vector = self.model.encode(article_text)
                
                # Check for duplicates
                is_duplicate = False
                for seen_vector in seen_vectors:
                    similarity = cosine_similarity([article_vector], [seen_vector])[0][0]
                    if similarity > duplicate_threshold:
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    unique_articles.append(article)
                    seen_vectors.append(article_vector)
            
            return unique_articles

        except Exception as e:
            print(f"Error in _filter_by_relevance: {str(e)}")
            return articles[:top_k]

    def _get_gkg_database(self, limit=timedelta(days=7)) -> dict[str, str]:
        """
        Fetches the latest GDELT GKG theme lookup table and builds
        a mapping from human-readable labels to raw theme codes.

        Returns:
            dict[str, str]: { "Readable Label": "RAW_CODE" }
        """
        db_filename = "gkg_themes"

        all_existing_db = list(GKG_DB_PATH.glob(db_filename + "_*.json"))
        current_date = datetime.now()

        if all_existing_db:
            latest_file = max(all_existing_db, key=lambda f: f.stem.split('_')[-1]) if all_existing_db else None

            if latest_file and (current_date - datetime.strptime(latest_file.stem.split('_')[-1], '%Y-%m-%d')) < limit:
                try:
                    with open(latest_file, 'r') as f:
                        return json.load(f)
                except Exception as e:
                    print(f"Error loading {latest_file}: {str(e)}")

        url = "http://data.gdeltproject.org/api/v2/guides/LOOKUP-GKGTHEMES.TXT"
        resp = requests.get(url)
        resp.raise_for_status()

        mapping = {}

        for line in resp.text.splitlines():
            # Each line looks like: WB_663_GEOGRAPHIC_INFORMATION_SYSTEMS<TAB>49922
            parts = line.strip().split("\t")
            if len(parts) < 1:
                continue
            code = parts[0]

            # Split the code into tokens
            tokens = code.split("_")
            if len(tokens) <= 2:
                label = " ".join(tokens).title()
            else:
                if tokens[0] not in ['WB', 'TAX']:
                    if len(tokens[1]) <= 4:
                        # Both first tokens are codes
                        label = " ".join(tokens[2:]).title()
                    else:
                        label = " ".join(tokens[1:]).title()
                else:
                    label = " ".join(tokens).title()

            mapping[label] = code

        dir = DataFetcher.ensure_dir('gkg')

        with open(dir / f"{db_filename}_{current_date.strftime('%Y-%m-%d')}.json", 'w') as f:
            json.dump(mapping, f, indent=2)

        return mapping