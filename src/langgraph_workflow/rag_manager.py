# Copyright 2025 unusedusername01
# SPDX-License-Identifier: Apache-2.0

from typing import Union, Dict, List
from langchain.tools import tool
from langchain_openai import OpenAIEmbeddings
from langchain_together import TogetherEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.vectorstores.base import VectorStoreRetriever
from sentence_transformers import SentenceTransformer
from numpy import dot, transpose

import re

from src.data_pipeline.data_collector import DataCollector
from src.data_pipeline.constants import MODELS_PATH, SECTOR_DB_PATH, MARKET_DB_PATH, get_file_prefix

class RAGManager:
    """
    Manages the retrieval-augmented generation (RAG) workflow.
    """
    SECTOR_COLLECTION_NAME = "sector_news"
    MARKET_COLLECTION_NAME = "market_news"

    def __init__(self, embeddings: Union[TogetherEmbeddings, HuggingFaceEmbeddings, OpenAIEmbeddings, GoogleGenerativeAIEmbeddings]):
        self.embeddings = embeddings
        self.text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n"],  # To split by article
            chunk_size=400,       
            chunk_overlap=50,     # Minimal overlap for metadata preservation
            length_function=len,
            is_separator_regex=False
        )
        self.sector_retrievers: Dict[str, VectorStoreRetriever] = {}
        self.market_retrievers: Dict[str, VectorStoreRetriever] = {}
        self.k = 5  # Number of top results to return

    @staticmethod
    def format_articles(articles: List[Dict[str, str]]) -> str:
        """
        Formats a list of articles into a list of strings.
        Each article is separated by a double newline to facilitate splitting.
        Only include title, summary, source and keyword in the result.
        Args:
            articles (List[Dict[str, str]]): List of articles where each article is a dictionary.
        Returns:
            List[str]: List of formatted article strings.
        """
        formatted_articles = ""  # Initialize with an empty string
        format = lambda ttl, smy, src, kw: f"""\n[TITLE] {ttl} | [SUMMARY] {smy} | [SOURCE] {src} | [KEYWORD] {kw}\n"""

        for article in articles:
            title = article.get('title', 'No Title')
            summary = article.get('summary', 'No Summary')
            source = article.get('source', 'Unknown Source')
            keyword = article.get('keyword', 'No Keyword')
            formatted_articles += format(title, summary, source, keyword)

        # Remove the last newline character for cleaner output
        if formatted_articles.endswith('\n'):
            formatted_articles = formatted_articles[:-1]

        return formatted_articles
    
    @staticmethod
    def extract_field(text: str, field: str) -> str:
        """
        Extracts a specific field from the text.
        Args:
            text (str): The text to extract the field from.
            field (str): The field to extract.
        Returns:
            str: The extracted field value or None if not found.
        """
        field = field.upper()
        match = re.search(rf"\[{field}\](.*?)\|", text)

        if match:
            return match.group(1).strip()
        return None

    @staticmethod
    def greedy_reorder(distance_matrix):
        n = distance_matrix.shape[0]
        unvisited = set(range(n))
        order = [unvisited.pop()]  # start at arbitrary index
        while unvisited:
            last = order[-1]
            # find nearest neighbor among unvisited
            next_idx = min(unvisited, key=lambda j: distance_matrix[last, j])
            unvisited.remove(next_idx)
            order.append(next_idx)
        return order
    
    def remove_duplicates(self, results, threshold=10):
        print("Size of results before removing duplicates:", len(results))
        print([result.page_content for result in results])
        def hamming_distance(a, b):
            diff = 0
            if len(a) != len(b):
                diff = abs(len(a) - len(b))
                a, b = a.ljust(len(b)), b.ljust(len(a))  # Pad shorter string
            return sum(el1 != el2 for el1, el2 in zip(a, b)) + diff

        seen = set()
        unique_contents = []
        for doc in results:
            content = doc.page_content
            title = self.extract_field(content, 'TITLE')
            if not title:
                continue
            if all(hamming_distance(title, seen_title) > threshold for seen_title in seen):
                unique_contents.append(content)
                seen.add(title)
            if len(unique_contents) == self.k:
                break

        print(len(unique_contents), "unique contents found after removing duplicates")
        return unique_contents

    def vectorize_market_news(self, market: str) -> VectorStoreRetriever:
        """
        Vectorizes news articles for a specific market and returns a Chroma vector store.
        Args:
            market (str): The market for which to vectorize news articles.
        Returns:
            VectorStoreRetriever: A retriever for the market news.
        """
        # Step 1: Find the latest market news file
        market_news_data = DataCollector.collect_market_news(market)
        if not market_news_data:
            raise ValueError(f"No news data found for market: {market}")

        # Step 2: Reorganize the data by regrouping articles form given by keyword similarity
        get_keyword = lambda article: article.get('keyword', 'NaN')

        articles = market_news_data.get('news', [])
        if not articles:
            raise ValueError(f"No articles found in market news data for market: {market}")
        
        all_keywords = set(map(get_keyword, articles))
        valid_keywords = [k for k in all_keywords if k != 'NaN']
        invalid_keywords = [k for k in all_keywords if k == 'NaN']

        model = SentenceTransformer('all-MiniLM-L6-v2')
        keyword_embeddings = model.encode(valid_keywords, normalize_embeddings=True)

        similarity = dot(keyword_embeddings, transpose(keyword_embeddings))
        distance = 1 - similarity
        
        ordered_indices = self.greedy_reorder(distance)

        # Result is a list of articles ordered by keyword similarity
        sorted_keywords = [valid_keywords[i] for i in ordered_indices] + invalid_keywords
        # Sort articles according to their position in sorted_keywords
        keyword_to_index = {k: i for i, k in enumerate(sorted_keywords)}
        articles = sorted(
            articles,
            key=lambda article: keyword_to_index.get(article.get('keyword', 'NaN'), len(sorted_keywords))
        )

        # Step 3: Format the articles for the splitter
        formatted_articles = self.format_articles(articles)

        # Step 4: Split the formatted articles into chunks
        texts = self.text_splitter.split_text(formatted_articles)
        if not texts:
            raise ValueError(f"No text chunks created for market: {market}")
        
        # Step 5: Create a Chroma vector store
        db = Chroma.from_texts(
            texts,
            self.embeddings,
            persist_directory=str(MARKET_DB_PATH / get_file_prefix('market', market)),
            collection_name=self.MARKET_COLLECTION_NAME
        )

        # Step 6: Create a retriever from the vector store
        retriever = db.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self.k * 3}  # Get more results to facilitate dupplicate removal
        )

        # Step 7: Store the retriever in the market_retriever dictionary
        self.market_retrievers[market] = retriever

        return retriever
        

    def vectorize_sector_news(self, sector: str) -> VectorStoreRetriever:
        """
        Vectorizes news articles for a specific sector and returns a Chroma vector store.
        Args:
            sector (str): The sector for which to vectorize news articles.
        Returns:
            VectorStoreRetriever: A retriever for the sector news.
        """
        # Step 1: Find the latest sector news file
        sector_news_data = DataCollector.collect_sector_news(sector)
        if not sector_news_data:
            raise ValueError(f"No news data found for sector: {sector}")

        # Step 2: Reorganize the data by regrouping articles form given by keyword similarity
        get_keyword = lambda article: article.get('keyword', 'NaN')

        articles = sector_news_data.get('news', [])
        if not articles:
            raise ValueError(f"No articles found in sector news data for sector: {sector}")
        
        all_keywords = set(map(get_keyword, articles))
        valid_keywords = [k for k in all_keywords if k != 'NaN']
        invalid_keywords = [k for k in all_keywords if k == 'NaN']

        model = SentenceTransformer('all-MiniLM-L6-v2')
        keyword_embeddings = model.encode(valid_keywords, normalize_embeddings=True)

        similarity = dot(keyword_embeddings, transpose(keyword_embeddings))
        distance = 1 - similarity

        ordered_indices = self.greedy_reorder(distance)

        # Result is a list of articles ordered by keyword similarity
        sorted_keywords = [valid_keywords[i] for i in ordered_indices] + invalid_keywords
        # Sort articles according to their position in sorted_keywords
        keyword_to_index = {k: i for i, k in enumerate(sorted_keywords)}
        articles = sorted(
            articles,
            key=lambda article: keyword_to_index.get(article.get('keyword', 'NaN'), len(sorted_keywords))
        )

        # Step 3: Format the articles for the splitter
        formatted_articles = self.format_articles(articles)

        # Step 4: Split the formatted articles into chunks
        texts = self.text_splitter.split_text(formatted_articles)
        if not texts:
            raise ValueError(f"No text chunks created for sector: {sector}")
        
        print(texts)
        
        # Step 5: Create a Chroma vector store
        db = Chroma.from_texts(
            texts,
            self.embeddings,
            persist_directory=str(SECTOR_DB_PATH / get_file_prefix('news', sector)),
            collection_name=self.SECTOR_COLLECTION_NAME
        )

        # Step 6: Create a retriever from the vector store
        retriever = db.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self.k * 3} # Get more results to facilitate dupplicate removal
        )

        # Step 7: Store the retriever in the sector_retrievers dictionary
        self.sector_retrievers[sector] = retriever

        return retriever

    def _sector_news_retriever_tool(self, sector: str, query: str) -> str:
        """
        Retrieve relevant news articles for a given sector.
        Args:
            sector (str): The sector for which to retrieve news articles.
            query (str): The query to search within the sector news.
        Returns:
            str: A string containing the relevant news articles for the specified sector.
        """
        if sector not in self.sector_retrievers:
            self.sector_retrievers[sector] = self.vectorize_sector_news(sector)
        retriever = self.sector_retrievers[sector]

        results = retriever.invoke(query)
        results = self.remove_duplicates(results)

        return "\n".join(results)
        
    def _market_news_retriever_tool(self, market: str, query: str) -> str:
        """
        Retrieve relevant news articles for a given market.
        Args:
            market (str): The market for which to retrieve news articles.
            query (str): The query to search within the market news.
        Returns:
            str: A string containing the relevant news articles for the specified market.
        """
        if market not in self.market_retrievers:
            self.market_retrievers[market] = self.vectorize_market_news(market)
        retriever = self.market_retrievers[market]

        results = retriever.invoke(query)
        results = self.remove_duplicates(results)

        return "\n".join(results)
    
    # @tool
    # def sector_news_retriever(self, sector: str, query: str) -> str:
    #     """
    #     Retrieve news articles for a specific sector.
    #     /!\ The sector must be one specified by a ticker's fundamental data.
    #     Common sectors include: 'technology', 'consumer_cyclical', 'healthcare', 'finance', 'energy' etc.
    #     Args:
    #         sector (str): The sector for which to retrieve news articles.
    #         query (str): The query to search within the sector news.
    #     Returns:
    #         str: A string containing the news articles for the specified sector.
    #     """
    #     return self._sector_news_retriever_tool(sector, query)
    
    # @tool
    # def market_news_retriever(self, market: str, query: str) -> str:
    #     """
    #     Retrieve news articles for a specific market.
    #     /!\ The market must be one specified by a ticker's fundamental data.
    #     Common markets include 'us_market', 'fr_market', 'de_market', 'cn_market', etc.
    #     Args:
    #         market (str): The market for which to retrieve news articles.
    #         query (str): The query to search within the market news.
    #     Returns:
    #         str: A string containing the news articles for the specified market.
    #     """
    #     return self._market_news_retriever_tool(market, query)
