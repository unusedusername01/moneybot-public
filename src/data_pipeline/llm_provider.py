# Copyright 2025 unusedusername01
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime
import os
import re
import json
import ast
from typing import Annotated, Optional, List, Dict, Any, Sequence, Tuple, Union
from chromadb import Embeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_together import ChatTogether, TogetherEmbeddings
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline, HuggingFaceEmbeddings
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.embeddings.base import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import Tool, tool, InjectedToolArg
from dotenv import load_dotenv
from operator import add as add_messages
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage, ToolMessage
from abc import ABC, abstractmethod
import openai
import requests
from src.data_pipeline.constants import MODELS_PATH
from src.data_pipeline.data_collector import DataCollector

class BaseLLMProvider(ABC):
    """Abstract base for any LLM provider implementation."""
    # Provider configurations
    PROVIDER_CONFIG = {
        'openai': {
            'llm_call_function': '_call_openai_llm',
            'embeddings_call_function': '_call_openai_embeddings',
            'llm_get_function': '_get_openai_llm',
            'embeddings_get_function': '_get_openai_embeddings',
            'api_key_env': 'OPENAI_API_KEY',
        },
        'together': {
            'llm_call_function': '_call_together_llm',
            'embeddings_call_function': '_call_together_embeddings',
            'llm_get_function': '_get_together_llm',
            'embeddings_get_function': '_get_together_embeddings',
            'api_key_env': 'TOGETHER_API_KEY',
        },
        'google': {
            'llm_call_function': '_call_gemini_llm',
            'embeddings_call_function': '_call_google_embeddings',
            'llm_get_function': '_get_google_llm',
            'embeddings_get_function': '_get_google_embeddings',
            'api_key_env': 'GOOGLE_API_KEY',
        },
        'anthropic': {
            'llm_call_function': '_call_anthropic_llm',
            'embeddings_call_function': '_call_anthropic_embeddings',
            'llm_get_function': '_get_anthropic_llm',
            'embeddings_get_function': '_get_anthropic_embeddings',
            'api_key_env': 'ANTHROPIC_API_KEY',
        },
        'huggingface': {
            'llm_call_function': '_call_huggingface_llm',
            'embeddings_call_function': '_call_huggingface_embeddings',
            'llm_get_function': '_get_huggingface_llm',
            'embeddings_get_function': '_get_huggingface_embeddings',
            'api_key_env': 'HUGGINGFACE_API_KEY',
        },
    # 'openrouter': {
    #     'llm_call_function': '_call_openrouter_llm',
    #     'embeddings_call_function': '_call_openrouter_embeddings',
    #     'llm_get_function': '_get_openrouter_llm',
    #     'embeddings_get_function': '_get_openrouter_embeddings',
    #     'api_key_env': 'OPENROUTER_API_KEY',
    # }, NOTE: No support for OpenRouter yet
        'local': {
            'llm_call_function': '_call_local_llm',
            'embeddings_call_function': '_call_local_embeddings',
            'llm_get_function': '_get_local_llm',
            'embeddings_get_function': '_get_local_embeddings',
            'model_directory': MODELS_PATH,
        },
        'lm_studio': {
            'llm_call_function': '_call_lm_studio_llm',
            'embeddings_call_function': '_call_lm_studio_embeddings',
            'llm_get_function': '_get_lm_studio_llm',
            'embeddings_get_function': '_get_lm_studio_embeddings',
            'base_url': 'http://127.0.0.1:1234/v1',
        }
    }

    def __init__(self, api_key: Optional[str], provider: str):
        load_dotenv()
        # Check validity of the provider
        if provider not in self.PROVIDER_CONFIG:
            raise ValueError(f"Unsupported provider: {provider}. Supported providers are: {', '.join(self.PROVIDER_CONFIG.keys())}.")
        self.provider = provider

        # Defaults can be set by subclasses to provide implicit model selection.
        self.default_llm_model: Optional[str] = None
        self.default_embeddings_model: Optional[str] = None

        # Check api_key is loaded for providers that require it
        # NOTE: load_dotenv() impacts langchain's provider directly, so we don't need to pass api_key explicitly,
        # but here we use it for validation and debugging purposes
        if provider in self.PROVIDER_CONFIG and provider not in ['local', 'lm_studio']:
            api_key = api_key or os.getenv(self.PROVIDER_CONFIG[provider]['api_key_env'])
            if not api_key:
                raise ValueError(f"API key for {provider} not found. Please set the environment variable {self.PROVIDER_CONFIG[provider]['api_key_env']}.")
        elif provider == 'lm_studio':
            api_key = 'dummy_key'

        self.api_key = api_key

    def _resolve_llm_model(self, model: Optional[str]) -> str:
        """Return the requested LLM model or fall back to the configured default."""
        candidate = model or self.default_llm_model
        if not candidate:
            raise ValueError(
                "No LLM model specified and no default_llm_model is configured for "
                f"provider '{self.provider}'."
            )
        return candidate

    def _resolve_embeddings_model(self, model: Optional[str]) -> str:
        """Return the requested embeddings model or use the configured default."""
        candidate = model or self.default_embeddings_model
        if not candidate:
            raise ValueError(
                "No embeddings model specified and no default_embeddings_model is "
                f"configured for provider '{self.provider}'."
            )
        return candidate

    @abstractmethod
    def call_llm(
        self,
        model: Optional[str],
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 1024, 
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """
        Make a call to the LLM provider.
        Args:
            model (Optional[str]): The model to use for the LLM call
            prompt (str): The prompt to send to the LLM
            system_prompt (Optional[str]): Additional system prompt
            max_tokens (int): Maximum number of tokens to generate
            temperature (float): Sampling temperature (0.0 to 1.0)
            **kwargs: Additional keyword arguments for provider-specific configurations
        Returns:
            str: The raw text response from the LLM
        """
        pass

    @abstractmethod
    def call_embeddings(
        self,
        model: Optional[str],
        texts: Union[str, List[str]],
        **kwargs
    ) -> List[float] | List[List[float]]:
        """
        Make a call to the embeddings provider.
        Args:
            model (Optional[str]): The model to use for embeddings
            texts (Union[str, List[str]]): Text or list of texts to embed
            **kwargs: Additional keyword arguments for provider-specific configurations
        Returns:
            Any: The embeddings for the provided texts
        """
        pass

    @abstractmethod
    def get_llm(
        self, 
        model: Optional[str],
        max_tokens: int = 1024,
        temperature: float = 0.7,
        **kwargs
        ) -> Any:
        """
        Get the LLM instance for the specified model.
        Args:
            model (Optional[str]): The model name
        Returns:
            Any: The LLM instance
        """
        pass

    @abstractmethod
    def get_embeddings(
        self,
        model: Optional[str],
        **kwargs
        ) -> Any:
        """
        Get the embeddings instance for the specified model.
        Args:
            model (Optional[str]): The model name
        Returns:
            Any: The embeddings instance
        """
        pass

    @staticmethod
    def list_local_models() -> List[str]:
        """
        List all local models available in the configured models directory.
        Returns:
            List[str]: A list of model names available locally.
        """
        if not MODELS_PATH.exists():
            return []
        return [model.name for model in MODELS_PATH.iterdir() if model.is_dir() and not model.name.startswith('.')]
    
    @staticmethod
    def get_model_path(model_name: str) -> str:
        """
        Get the full path for a local model.
        Args:
            model_name (str): The name of the model
        Returns:
            str: The full path to the model directory
        """
        path = MODELS_PATH / model_name
        if not path.exists():
            raise ValueError(f"Model {model_name} does not exist in the local models directory.")
        
        return str(path)
    
    @staticmethod
    def get_messages(prompt: str, system_prompt: Optional[str] = None, response_example: Optional[str] = None) -> Annotated[Sequence[BaseMessage], add_messages]:
        """
        Prepare the messages for the LLM call.
        Args:
            prompt (str): The user prompt
            system_prompt (Optional[str]): Additional system prompt
        Returns:
            List[Union[SystemMessage, HumanMessage]]: The formatted messages for the LLM
        """
        messages: Annotated[Sequence[BaseMessage], add_messages] = []

        if system_prompt:
            messages += [SystemMessage(content=system_prompt)]

        messages += [HumanMessage(content=prompt)]

        if response_example:
            messages += [AIMessage(content=response_example)]

        return messages
        
    def generate_sector_keywords(self, ticker: str, system_prompt: Optional[str] = None) -> List[str]:
        """
        Generate relevant sector keywords for a given ticker.
        
        Args:
            ticker (str): The stock ticker symbol
            system_prompt (Optional[str]): Additional system_prompt for keyword generation
            
        Returns:
            List[str]: A list of relevant sector keywords
        """
        # Load fundamental data
        fundamental_data = DataCollector.collect_fundamentals(ticker)
        
        # Format fundamental data for the prompt
        fundamental_info = ""
        if fundamental_data:
            fundamental_info = f"""
            Company Information:
            - Sector: {fundamental_data.get('sector', 'Unknown')}
            - Company Name: {fundamental_data.get('shortName', ticker)}
            """

        system_prompt = f"""
        You are an expert at generating targeted search keywords for sector-specific news.
        For a given ticker, generate a list of 3-5 search phrases that will help find relevant news articles about the company's sector.

        IMPORTANT: Your response must be ONLY a Python list of strings, like this:
        ["phrase 1", "phrase 2", "phrase 3"]

        Do not include any explanations, code, or other text. JUST THE LIST OF SEARCH PHRASES.

        Focus on these sector-specific aspects:
        1. The company's sector/industry name and trends
        2. Major competitors in the same sector
        3. Regulatory developments affecting the sector
        4. Technology trends in the sector
        5. Market dynamics and growth prospects for the sector

        Example for AAPL (Technology sector):
        ["technology sector trends", "smartphone industry competition", "consumer electronics market", "tech stock performance", "Apple competitors Samsung Google"]
        """

        prompt = f"""
        Generate a list of 3-5 search phrases for the ticker {ticker} that will help find relevant news articles about the company's sector.
        Here is the company information:
        {fundamental_info}
        """
        
        # Get raw response from LLM
        resolved_model = self._resolve_llm_model(None)

        response = self.call_llm(resolved_model, prompt, system_prompt, max_tokens=256)
        print("Sector keywords response:", response)
        
        # Parse the response
        keywords = self._parse_keyword_response(response)
        return keywords if keywords else [f"{ticker} sector"]

    def generate_market_keywords(self, ticker: str, system_prompt: Optional[str] = None) -> List[str]:
        """
        Generate relevant market keywords for a given ticker.
        
        Args:
            ticker (str): The stock ticker symbol
            system_prompt (Optional[str]): Additional system_prompt for keyword generation
            
        Returns:
            List[str]: A list of relevant market keywords
        """
        # Load fundamental data
        fundamental_data = DataCollector.collect_fundamentals(ticker)
        
        # Format fundamental data for the prompt
        fundamental_info = ""
        if fundamental_data:
            fundamental_info = f"""
            Company Information:
            - Market: {fundamental_data.get('market', 'Unknown')}
            - Company Name: {fundamental_data.get('shortName', ticker)}
            - Sector: {fundamental_data.get('sector', 'Unknown')}
            """

        system_prompt = f"""You are an expert at generating targeted search keywords for market-wide and geopolitical news.

        IMPORTANT: Your response must be ONLY a Python list of strings, like this:
        ["phrase 1", "phrase 2", "phrase 3"]

        Do not include any explanations, code, or other text. Just the list.

        Focus on these market-wide aspects:
        1. Global economic trends and indicators
        2. Geopolitical events affecting markets
        3. Currency and trade dynamics
        4. Central bank policies and interest rates
        5. International trade relationships
        6. Market volatility and investor sentiment

        Example for AAPL (US market):
        ["US China trade relations", "global supply chain disruption", "Federal Reserve interest rates", "global semiconductor shortage", "international market volatility"]
        """

        prompt = f"""
        Generate a list of 3-5 search phrases for the ticker {ticker} that will help find relevant news articles about the market.
        Here is the company information:
        {fundamental_info}
        """
        
        # Get raw response from LLM
        resolved_model = self._resolve_llm_model(None)
        response = self.call_llm(resolved_model, prompt, system_prompt, max_tokens=256)
        print("Market keywords response:", response)
        
        # Parse the response
        keywords = self._parse_keyword_response(response)
        return keywords if keywords else [f"{ticker} market"]

    def evaluate_batch(self, batch: str, user_preferences: str, judge_model: Optional[str] = None, tools: Optional[List[Tool]] = None, tool_agent_model: Optional[str] = None) -> Dict[str, Tuple[int, str]]:
        """
        Evaluate a batch of tickers using the judge LLM.
        
        Args:
            batch (str): The batch of tickers to evaluate
            user_preferences (str): User preferences for evaluation
            tools (List[Tool]): List of tools to use for evaluation

        Returns:
            Dict[str, Tuple[int, str]]: A dictionary mapping ticker symbols to their scores and reviews
        """
        tool_agent = self.get_llm(tool_agent_model, max_tokens=1024, temperature=0.7)

        prompt = batch
        
        tool_agent_system_prompt = f"""
        You are a **financial news RAG agent**. Current date: {datetime.now().strftime('%Y-%m-%d')}.

        Your task: Fetch necessary data in order for a judge agent to evaluate batches of ticker data.
        The judge will have access to the same data as you, including:
        - Ticker specific news articles
        - Fundamental company data
        - Price predictions data
        
        And you will have access to two databases more indirect to tickers:
        - Market news articles
        - Sector news articles

        =======================
        NEWS HANDLING
        ======================
        - Call tools when you judge of:
        (a) A lack of contextual information (concerning the market / sector of a ticker)
        (b) Available ticker news are irrelevant or insufficient
        (c) A ticker is particularly sensitive to market or sector events (e.g. Trade wars, sectorial crises, ...)
        """
            
        judge_system_prompt = f"""
        You are a **financial data judge**. Current date: {datetime.now().strftime('%Y-%m-%d')}.

        Your task: evaluate batches of ticker data and return ONLY a Python dictionary of type Dict[str, Tuple[int, str]] where:
        - Key = ticker symbol (string, uppercase)
        - Value = (score: int, review: str)

        STRICT OUTPUT RULE:
        - Return ONLY the dictionary literal, no extra text, markdown, or commentary.
        - Example of valid output:
        {{"AAPL": (55, "Concise review..."), "GOOGL": (92, "Concise review..."), "INTC": (6, "Concise review...")}}

        ======================
        PREDICTIONS HANDLING
        ======================
        - Predictions are supportive, not primary.
        - Prediction mode determines how relevant and strong the prediction model was ['weak', 'medium', 'strong']

        ======================
        SCORING FRAMEWORK
        ======================
        Score each ticker 0–100 using weighted components:

        1. Fundamentals (50%)
        - Lower debt_to_equity → better
        - Lower trailing_pe relative to sector → better
        - Higher profit_margin, return_on_equity, revenue_growth → better
        - Extreme or distorted price_to_book → penalize
        - Average across available metrics if some missing

        2. News Sentiment (20%)
        - Negative recent material events (lawsuit, ban, fraud) → strong penalty
        - Positive analyst upgrades, product wins → reward
        - Map polarity into 0–100 band, weight accordingly

        3. Portfolio Fit & Preferences (20%)
        - Inputs: {user_preferences}
        - Medium tolerance → avoid extremes, penalize volatility
        - If user already holds large % in ticker, subtract ≤5 points (encourage diversification especially in case of low risk tolerance)

        4. Predictions (10%)
        - Usable predictions only (per rules above)
        - Directional positive → +up to 5
        - Directional negative → -up to 5
        - NOTE: Directional negative generally indicates a turbulent stock history

        Final score = weighted sum, clamped to [0,100], rounded to int.

        NOTE: The final score will determine proportionally an allocation so if ticker is not promising, just rate it a 0.

        ======================
        REVIEW FORMAT
        ======================
        - Each review: MAX 4 sentences, 30-60 words total
        - Must include:
        (1) One fundamentals statement
        (2) One short note on news / prediction / portfolio fit
        - Article references ≤10 words, cite source in parentheses

        ======================
        DELIVERABLE
        ======================
        For your final output, return ONLY the dictionary: Dict[str, Tuple[int, str]] as described above, nothing else.
        """

        if isinstance(tool_agent, BaseChatModel) and tools:
            tool_dict = {tool.name: tool for tool in tools}
            tool_agent = tool_agent.bind_tools(tools)
            current_query, max_queries = 1, 5

            messages = self.get_messages(prompt=batch, system_prompt=tool_agent_system_prompt)

            while current_query <= max_queries:
                # Invoke LLM
                messages += [tool_agent.invoke(messages)]

                # If no tool calls, stop early
                tool_calls = getattr(messages[-1], "tool_calls", None)
                if not tool_calls:
                    break

                # For each tool call, execute and append a ToolMessage
                for call in tool_calls:
                    if current_query > max_queries:
                        break
                    args: Dict[str, Any] = call["args"]
                    result = tool_dict[call["name"]].invoke(args)

                    messages += [
                        ToolMessage(
                            content=result,
                            tool_call_id=call["id"],
                            name=call["name"],      # optional, but useful for logging/debug
                        )
                    ]
                    current_query += 1

            prompt += "\n\nAdditional news data about the tickers sectors / market:" + '\n'.join([message.content for message in messages if isinstance(message, ToolMessage)])

        resolved_judge_model = self._resolve_llm_model(judge_model)

        result = self.call_llm(
            model=resolved_judge_model,
            system_prompt=judge_system_prompt,
            prompt=prompt,
            max_tokens=16384,
            temperature=0.3,
        )

        # If used a thinking model (very likely) remove using markers
        think_start, think_end = '<think>', '</think>'
        if think_start in result and think_end in result:
            result = result.split(think_start)[1].split(think_end)[1]

        print("\n\nResult:\n\n", result)

        try:
            parsed_result = ast.literal_eval(result)
            if not isinstance(parsed_result, dict):
                print(f"Warning: evaluate_batch expected dict, got {type(parsed_result)}")
                return {}
            return parsed_result
        except (ValueError, SyntaxError) as e:
            print(f"Error parsing LLM result: {e}")
            print(f"Raw result was: {result}")
            return {}

    def synthetize_ranked_batches(self, ranked_batches: str, synthetizer_context: str, model: Optional[str] = None) -> str:
        """
        Synthetize the ranked batches into a final report.
        
        Args:
            ranked_batches (str): The ranked batches to synthetize
            synthetizer_context (str): Context for the synthetizer LLM
            
        Returns:
            str: The final report as a string
        """
        resolved_model = self._resolve_llm_model(model)

        response = self.call_llm(
            model=resolved_model,
            prompt=ranked_batches,
            system_prompt=synthetizer_context,
            max_tokens=4096,
            temperature=0.2,
        )

        print("\n\nSynthetized response:\n", response)
        
        return response

    def merge_batches(self, batches: str) -> Dict[str, Tuple[int]]:
        """
        Merge multiple batches into a single batch.

        Args:
            batches (str): The batches to merge
            final_batch_size (int): The desired size of the final batch

        Returns:
            str: The merged batch as a string
        """
        system_prompt = """
        Your goal is to merge two batches of financial data into a single batch.
        A batch of financial data is of type Dict[str, Tuple[int, str]], and consists of multiple entries where
        keys are ticker symbols and values are tuples containing the score and review.
        As you may have guessed now, your task is to merge these batches into a single batch BUT doing so you must pay attention to the followings:
        - Since each batch has been graded INDEPENDENTLY, the tricky part is to ensure that the final merged batch maintains a coherent narrative and does not simply concatenate the inputs.
        This involves rescaling scores if needed, reflecting on all the reviews and ensuring that the final output is a cohesive and comprehensive batch.
        Your final response should be a single python Dict[str, Tuple[int, str]] identical in structure to the input batches.
        """

        resolved_model = self._resolve_llm_model(None)

        response = self.call_llm(
            model=resolved_model,
            prompt=batches,
            system_prompt=system_prompt,
            max_tokens=8192,
            temperature=0.2,
        )

        return self._parse_merged_batch_response(response)

    def _parse_keyword_response(self, response: str) -> List[str]:
        """Parse the LLM response to extract keywords."""
        try:
            # Clean up the response
            response = response.strip()
            
            # Try to parse as JSON list first
            if response.startswith('[') and response.endswith(']'):
                import ast
                return ast.literal_eval(response)
            
            # Try to extract from bracket notation
            if '[' in response and ']' in response:
                start = response.find('[')
                end = response.rfind(']') + 1
                list_str = response[start:end]
                import ast
                return ast.literal_eval(list_str)
            
            # Fall back to comma-separated parsing
            if ',' in response:
                items = [item.strip().strip('"\'') for item in response.split(',')]
                return [item for item in items if item]
            
            # Return as single item if no other parsing works
            return [response.strip().strip('"\'')]
            
        except Exception as e:
            print(f"Error parsing keyword response: {e}")
            return []
    
    # @staticmethod
    # def _parse_evaluation_response(response: str) -> Dict[str, Tuple[int, str]]:
    #     def grab(s: str) -> str:
    #         i = s.find("{")
    #         if i < 0: raise ValueError("no dict")
    #         d = 0
    #         for j,ch in enumerate(s[i:], start=i):
    #             d += (ch == "{") - (ch == "}")
    #             if d == 0: return s[i:j+1]
    #         raise ValueError("unbalanced braces")

    #     def to_int(x: Any) -> int:
    #         if isinstance(x, bool): raise ValueError("bad score")
    #         if isinstance(x, (int, float)): v = int(round(x))
    #         else:
    #             m = re.search(r"-?\d+", str(x or ""))
    #             if not m: raise ValueError("bad score")
    #             v = int(m.group())
    #         return max(0, min(100, v))

    #     block = grab(response)
    #     try:
    #         obj = ast.literal_eval(block)
    #     except Exception:
    #         obj = json.loads(block.replace("'", '"').replace("None", "null").replace("True", "true").replace("False", "false"))

    #     if not isinstance(obj, dict): raise ValueError("parsed non-dict")
    #     out: Dict[str, Tuple[int, str]] = {}
    #     for k,v in obj.items():
    #         if isinstance(v, (list, tuple)):
    #             score, review = (v[0], v[1] if len(v) > 1 else "")
    #         elif isinstance(v, dict):
    #             score = v.get("score") or v.get("rating")
    #             review = v.get("review") or v.get("reason") or ""
    #         else:
    #             score, review = v, ""
    #         out[str(k).strip()] = (to_int(score), str(review).strip())
    #     return out
    
    @staticmethod
    def _parse_synthesis_response(response: str) -> Dict[str, Union[str, Dict[str, float]]]:
        """
        Parse the synthesis response from the LLM.
        
        Args:
            response (str): The raw response from the LLM
            
        Returns:
            Dict[str, Union[str, Dict[str, float]]]: Parsed response containing review and budget allocation
        """
        try:
            # Clean up the response
            response = response.strip()
            
            # Try to parse as JSON
            if response.startswith('{') and response.endswith('}'):
                return json.loads(response)
            
            # Fallback to manual parsing
            pattern = r'\{([^}]+)\}'
            match = re.search(pattern, response)
            if match:
                content = match.group(1)
                items = content.split(',')
                result = {}
                for item in items:
                    key, value = item.split(':', 1)
                    result[key.strip()] = value.strip().strip('"')
                return result
            
            raise ValueError("Response format not recognized")
        
        except Exception as e:
            print(f"Error parsing synthesis response: {e}")
            return {}
        

    def _parse_merged_batch_response(response: str) -> Dict[str, Tuple[int, str]]:
        """
        Parse the merged batch response from the LLM.
        
        Args:
            response (str): The raw response from the LLM
            
        Returns:
            Dict[str, Tuple[int, str]]: Parsed response containing ticker symbols, scores, and reviews
        """
        try:
            # Clean up the response
            response = response.strip()
            
            # Try to parse as JSON
            if response.startswith('{') and response.endswith('}'):
                return json.loads(response)
            
            # Fallback to manual parsing
            pattern = r'\{([^}]+)\}'
            match = re.search(pattern, response)
            if match:
                content = match.group(1)
                items = content.split(',')
                result = {}
                for item in items:
                    key, value = item.split(':', 1)
                    score_review = value.strip().strip('"').split(',')
                    score = int(score_review[0].strip())
                    review = score_review[1].strip() if len(score_review) > 1 else ""
                    result[key.strip()] = (score, review)
                return result
            
            raise ValueError("Response format not recognized")
        
        except Exception as e:
            print(f"Error parsing merged batch response: {e}")
            return {}

class LMStudioEmbeddings(Embeddings):
    def __init__(self, model: str, api_key: str = "sk-test_dummyapikey1234567890", url: str = "http://localhost:1234/v1/embeddings"):
        self.model = model
        self.api_key = api_key
        self.url = url
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

    def _embed(self, inputs: List[str]) -> List[List[float]]:
        payload = {
            "model": self.model,
            "input": inputs
        }
        response = requests.post(self.url, headers=self.headers, json=payload)
        response.raise_for_status()
        result = response.json()
        # LM Studio embeddings are typically returned under result["data"][i]["embedding"]
        return [item["embedding"] for item in result.get("data", [])]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._embed(texts)

    def embed_query(self, text: str) -> List[float]:
        embeddings = self._embed([text])
        return embeddings[0] if embeddings else []
    
class LangChainLLMProvider(BaseLLMProvider):
    """A centralized class for handling LLM API calls across different providers using LangChain."""
    def __init__(
        self,
        api_key: Optional[str] = None,
        provider: str = 'together',
        *,
        default_llm_model: Optional[str] = None,
        default_embeddings_model: Optional[str] = None,
    ):
        """
        Initialize the LLM provider.
        Args:
            provider (str): The LLM provider to use ('openai', 'together', 'google', 'anthropic', 'huggingface', 'local', 'lm_studio')
        """
        super().__init__(api_key, provider)

        self.default_llm_model = default_llm_model
        self.default_embeddings_model = default_embeddings_model

    def clone(self):
        return LangChainLLMProvider(
            api_key=self.api_key,
            provider=self.provider,
            default_llm_model=self.default_llm_model,
            default_embeddings_model=self.default_embeddings_model,
        )

    # =========================
    # ===== High level API =====
    # =========================

    def call_llm(
        self,
        model: Optional[str],
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 200,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """
        Make a call to the LLM provider.
        Args:
            prompt (str): The prompt to send to the LLM
            system_prompt (Optional[str]): Additional system_prompt for the prompt
            max_tokens (int): Maximum number of tokens to generate
            temperature (float): Sampling temperature (0.0 to 1.0)
            **kwargs: Additional keyword arguments for provider-specific configurations
        Returns:
            Any: The raw text response from the LLM
        """
        resolved_model = self._resolve_llm_model(model)

        call_fn_name = self.PROVIDER_CONFIG[self.provider]['llm_call_function']
        call_fn = getattr(self, call_fn_name)
        return call_fn(resolved_model, prompt, system_prompt, max_tokens, temperature, **kwargs)
    
    def call_embeddings(
        self,
        model: Optional[str],
        texts: Union[str, List[str]],
        **kwargs
        ) -> List[float] | List[List[float]]:
        """
        Make a call to the embeddings provider.
        Args:
            model (Optional[str]): The model to use for embeddings
            texts (Union[str, List[str]]): Text or collection of texts to embed
            **kwargs: Additional keyword arguments for provider-specific configurations
        Returns:
            Any: The embeddings for the provided texts
        """
        resolved_model = self._resolve_embeddings_model(model)

        call_fn_name = self.PROVIDER_CONFIG[self.provider]['embeddings_call_function']
        call_fn = getattr(self, call_fn_name)
        return call_fn(resolved_model, texts, **kwargs)
    
    def get_llm(
            self,
            model: Optional[str],
            max_tokens: int = 1024,
            temperature: float = 0.7,
            **kwargs
        ) -> Union[
        ChatOpenAI,
        ChatTogether,
        ChatGoogleGenerativeAI,
        ChatAnthropic,
        ChatHuggingFace,
        ]:
        """
        Get the LLM instance for the specified model.
        Args:
            model (Optional[str]): The model name
        Returns:
            Any: The LLM instance
        """
        resolved_model = self._resolve_llm_model(model)

        get_fn_name = self.PROVIDER_CONFIG[self.provider]['llm_get_function']
        get_fn = getattr(self, get_fn_name)
        return get_fn(resolved_model, max_tokens=max_tokens, temperature=temperature, **kwargs)
    
    def get_embeddings(self, model: Optional[str], **kwargs) -> Union[
        OpenAIEmbeddings,
        TogetherEmbeddings,
        GoogleGenerativeAIEmbeddings,
        HuggingFaceEmbeddings,
        ]:
        """
        Get the embeddings instance for the specified model.
        Args:
            model (Optional[str]): The model name
        Returns:
            Any: The embeddings instance
        """
        resolved_model = self._resolve_embeddings_model(model)

        get_fn_name = self.PROVIDER_CONFIG[self.provider]['embeddings_get_function']
        get_fn = getattr(self, get_fn_name)

        return get_fn(resolved_model, **kwargs)

    # =========================
    # ===== Low level API =====
    # =========================

    def _get_openai_llm(self, model, max_tokens, temperature, **kwargs) -> ChatOpenAI:
        return ChatOpenAI(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            api_key=self.api_key,
            model_kwargs={**kwargs}
        )
        
    def _get_together_llm(self, model, max_tokens, temperature, **kwargs) -> ChatTogether:
        return ChatTogether(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            api_key=self.api_key,
            model_kwargs={**kwargs}
        )
    
    def _get_google_llm(self, model, max_tokens, temperature, **kwargs) -> ChatGoogleGenerativeAI:
        raise NotImplementedError("Google LLM is not implemented yet. Please use another provider.")
    
    def _get_anthropic_llm(self, model, max_tokens, temperature, **kwargs) -> ChatAnthropic:
        raise NotImplementedError("Anthropic LLM is not implemented yet. Please use another provider.")

    def _get_huggingface_llm(self, model, max_tokens, temperature, **kwargs) -> ChatHuggingFace:
        raise NotImplementedError("HuggingFace LLM is not implemented yet. Please use another provider.")

    def _get_local_llm(self, model, max_tokens, temperature, **kwargs) -> ChatHuggingFace:
        from torch.cuda import is_available as cuda_available
        model_path = self.get_model_path(model)

        # Set device to cuda if available, otherwise cpu
        device = kwargs.get('device', 0 if cuda_available() else -1)

        pipeline = HuggingFacePipeline.from_model_id(
            model_id=model_path,
            task="text-generation",
            device=device,
            model_kwargs={
                "local_files_only": True,
            },
            pipeline_kwargs={
                "max_new_tokens": max_tokens,
                "temperature": temperature,
            },
        )

        return ChatHuggingFace(llm=pipeline)
    
    def _get_lm_studio_llm(self, model: str, max_tokens: int = 1024, temperature: float = 0.7, **kwargs) -> ChatOpenAI:
        return ChatOpenAI(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            api_key=self.api_key,
            base_url=self.PROVIDER_CONFIG['lm_studio'].get('base_url', 'http://localhost:1234')
        )

    def _get_openai_embeddings(self, model: str, **kwargs) -> OpenAIEmbeddings:
        return OpenAIEmbeddings(
            model=model,
            api_key=self.api_key,
            model_kwargs={**kwargs}
        )
    
    def _get_together_embeddings(self, model: str, **kwargs) -> TogetherEmbeddings:
        return TogetherEmbeddings(
            model=model,
            api_key=self.api_key,
            model_kwargs={**kwargs}
        )

    def _get_google_embeddings(self, model: str, **kwargs) -> GoogleGenerativeAIEmbeddings:
        raise NotImplementedError("Google embeddings are not implemented yet. Please use another provider.")

    def _get_huggingface_embeddings(self, model: str, **kwargs) -> HuggingFaceEmbeddings:
        model_path = self.get_model_path(model)

        return HuggingFaceEmbeddings(
            model_name=model_path,
            model_kwargs={
                "local_files_only": True,
                **kwargs
            },
        )

    def _get_local_embeddings(self, model: str, **kwargs) -> HuggingFaceEmbeddings:
        from torch.cuda import is_available as cuda_available

        model_path = self.get_model_path(model)

        # Set device to cuda if available, otherwise cpu
        device = kwargs.get('device', 'cuda' if cuda_available() else 'cpu')
        kwargs['device'] = device

        return HuggingFaceEmbeddings(
            model_name=model_path,
            model_kwargs={
                "local_files_only": True,
                **kwargs
            },
        )

    def _get_lm_studio_embeddings(self, model: str, **kwargs) -> LMStudioEmbeddings:
        return LMStudioEmbeddings(
            model=model,
            api_key=self.api_key,
        )

    def _call_together_llm(
            self,
            model: str,
            prompt: str,
            system_prompt: Optional[str] = None,
            max_tokens: int = 2048,
            temperature: float = 0.7
        ):
        llm = self._get_together_llm(model, max_tokens=max_tokens, temperature=temperature)

        messages = self.get_messages(prompt, system_prompt)
        if not messages:
            raise ValueError("No messages provided for LLM call.")
        
        response = llm.invoke(messages)

        return response.content if isinstance(response, AIMessage) else str(response)

    def _call_openai_llm(self, *args, **kwargs):
        raise NotImplementedError("OpenAI API not implemented yet")

    def _call_google_llm(self, *args, **kwargs):
        raise NotImplementedError("Gemini API not implemented yet")

    def _call_anthropic_llm(self, *args, **kwargs):
        raise NotImplementedError("Anthropic API not implementation yet")

    def _call_huggingface_llm(self, *args, **kwargs):
        raise NotImplementedError("HuggingFace API not implemented yet")

    def _call_local_llm(self, model: str, prompt: str, system_prompt: Optional[str] = None, max_tokens: int = 200, temperature: float = 0.7) -> Any:
        messages = self.get_messages(prompt, system_prompt)
        if not messages:
            raise ValueError("No messages provided for LLM call.")
        
        llm = self._get_local_llm(model, max_tokens=max_tokens, temperature=temperature)
        
        response = llm.invoke(messages)

        return response.content if isinstance(response, AIMessage) else str(response)
    
    def _call_lm_studio_llm(self, model: str, prompt: str, system_prompt: Optional[str] = None, max_tokens: int = 200, temperature: float = 0.7) -> Any:
        llm = self._get_lm_studio_llm(model, max_tokens=max_tokens, temperature=temperature)

        messages = self.get_messages(prompt, system_prompt)
        if not messages:
            raise ValueError("No messages provided for LLM call.")

        response = llm.invoke(messages)

        return response.content if isinstance(response, AIMessage) else str(response)

    def _call_openai_embeddings(self, model: str, texts: List[str], **kwargs):
        raise NotImplementedError("OpenAI embeddings not implemented yet")
    
    def _call_google_embeddings(self, model: str, texts: List[str], **kwargs):
        raise NotImplementedError("Google embeddings not implemented yet")
    
    def _call_anthropic_embeddings(self, model: str, texts: List[str], **kwargs):
        raise NotImplementedError("Anthropic embeddings not implemented yet")
    
    def _call_huggingface_embeddings(self, model: str, texts: List[str], **kwargs):
        raise NotImplementedError("HuggingFace embeddings not implemented yet")
    
    def _call_together_embeddings(self, model: str, texts: str | List[str], **kwargs):
        embeddings = self._get_together_embeddings(model, **kwargs)
        
        if isinstance(texts, str):
            texts = [texts]
            return embeddings.embed_documents(texts)[0]
        elif isinstance(texts, list):
            return embeddings.embed_documents(texts)
        else:
            raise ValueError("texts must be a string or a list of strings.")
    
    def _call_local_embeddings(self, model: str, texts: List[str], **kwargs):
        embeddings = self._get_local_embeddings(model, **kwargs)

        if isinstance(texts, str):
            texts = [texts]
            return embeddings.embed_documents(texts)[0]
        elif isinstance(texts, list):
            return embeddings.embed_documents(texts)

    def _call_lm_studio_embeddings(self, model: str, texts: Union[str, List[str]], **kwargs):
        embeddings = self._get_lm_studio_embeddings(model, **kwargs)

        if isinstance(texts, str):
            texts = [texts]
            return embeddings.embed_documents(texts)[0]
        elif isinstance(texts, list):
            return embeddings.embed_documents(texts)
        else:
            raise ValueError("texts must be a string or a list of strings.")
