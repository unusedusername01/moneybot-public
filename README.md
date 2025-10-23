# MoneyBot - AI-Powered Financial Analysis & Portfolio Management

MoneyBot is an intelligent financial analysis system that combines real-time market data, news sentiment analysis, and ML-based predictions to provide comprehensive portfolio recommendations. 
The typical use case involves:
1. Creating a portfolio with a certain budget and target date, and a watchlist of stocks to invest in
2. Run an analysis to get recommendations for the portfolio
3. (Optional) Re-run the analysis with different parameters to see how the recommendations change
4. Once satisfied, allocate the portfolio to the stocks in the watchlist

## � Documentation Hub

Long-form documentation now lives under `docs/`:

- [Architecture Deep Dive](docs/architecture.md) – end-to-end data flow, LangGraph orchestration, and configuration layering
- [Backend API Reference](docs/backend_api_reference.md) – REST/WebSocket contracts and DTO schemas
- [Developer Guide](docs/developer_guide.md) – environment setup, presets, and contributor workflow
- [Testing & Coverage](docs/testing_and_coverage.md) – pytest layout, markers, and coverage expectations
- [Deployment Guide](docs/deployment_guide.md) – GitHub Actions deploy workflow and self-hosted runner setup
- [Webapp Guide](docs/webapp_guide.md) – frontend integration plan, API parity, and local dev flow

## �🚀 Features

### Core Capabilities
- **Multi-Source Data Integration**: Real-time financial data from Yahoo Finance, market news, and sector-specific information
- **Dual News Collection System**: 
  - **Advanced NewsFetcher**: NewsData API integration with sentence transformers and GKG themes
  - **Default DataCollector**: Efficient local news data retrieval and caching
- **AI-Powered Analysis**: Advanced LLM integration for financial analysis and sentiment evaluation
- **Portfolio Management**: Complete portfolio creation, editing, and optimization workflows
- **Predictive Modeling**: Machine learning-based price predictions with configurable strength levels
- **Real-time Web Interface**: WebSocket-based real-time communication for interactive analysis
- **Multi-Currency Support**: Support for 150+ international currencies
- **Risk Assessment**: Configurable risk tolerance levels and portfolio diversification analysis

### Technical Features
- **LangGraph Workflows**: Sophisticated state management and workflow orchestration
- **RAG (Retrieval-Augmented Generation)**: Context-aware analysis using vector databases
- **Parallel Processing**: GPU-accelerated predictions and concurrent data processing
- **Modular Architecture**: Extensible design supporting multiple LLM providers
- **Type-Safe APIs**: Full TypeScript integration with Pydantic model validation

## 🏗️ Architecture

![MoneyBot architecture overview](assets/images/architecture_diagram.svg)

The diagram above captures how market data, LangGraph workflows, and the React webapp interact across ingestion, orchestration, and presentation layers.

### Key Modules

- **`src/langgraph_workflow/`**: Core workflow orchestration and state management
  - **`app.py`**: Main FastAPI application with WebSocket support
  - **`rag_manager.py`**: Vector database management and news retrieval
  - **`utils.py`**: Data loading and processing utilities
- **`src/data_pipeline/`**: Data collection, processing, and LLM integration
  - **`data_fetcher.py`**: Yahoo Finance integration and NewsFetcher class
  - **`data_collector.py`**: Default local data collection and caching
  - **`llm_provider.py`**: Multi-provider LLM integration
  - **`prediction_model.py`**: ML-based price prediction models
- **`src/webapp/`**: Frontend TypeScript interfaces and API types (In development)
- **`data/`**: Temporary and persistent storage for portfolios, market data, and vector databases

## 📋 Prerequisites

- Python 3.11+
- CUDA-compatible GPU (optional, for local model inference)
- API keys for external services (if not self-hosted models)

## 🛠️ Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd moneybot
```

### 2. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 3. Set Up Environment Variables
```bash
cp env.example .env
```

Edit `.env` with your API keys:
```env
NEWSDATA_API_KEY=your_newsdata_key_here
TOGETHER_API_KEY=your_together_key_here
OPENAI_API_KEY=your_openai_key_here
GOOGLE_API_KEY=your_google_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here
HUGGINGFACE_API_KEY=your_huggingface_key_here
```

### 4. Initialize Data Directories
The system will automatically create necessary data directories on first run.

## 🚀 Quick Start

### Terminal Interface

1. **Start the backend using a preset**
```bash
python -m src.langgraph_workflow.app --preset base --host 127.0.0.1 --port 8000
```

2. **Open a WebSocket session**
```bash
# Example using websocat (or your preferred client)
websocat ws://127.0.0.1:8000/ws/portfolio/portfolio1
```

3. **Trigger an analysis run**
```bash
curl -X POST http://127.0.0.1:8000/analysis/portfolio/portfolio1/start
```

4. **Respond to MoneyBot prompts**
```bash
curl -X POST http://127.0.0.1:8000/analysis/portfolio/portfolio1/respond \
   -H "Content-Type: application/json" \
   -d '{"data":{"type":"choice_response","selection":"confirm"}}'
```

During a typical run you will see:

- `state_update` messages streamed over WebSocket as data fetching and synthesis progress
- `awaiting_choice` prompts requesting confirm/edit/deny/cancel decisions
- `awaiting_allocation` prompts when the workflow needs revised allocations
- `end_analysis` once the workflow completes or is cancelled

Screenshots and log snippets for the full narrative are available in [docs/architecture.md](docs/architecture.md) and the new [Testing & Coverage](docs/testing_and_coverage.md) guide.

### Web Interface

1. **Start the backend API**
```bash
python -m src.langgraph_workflow.app --port 8000 --host 127.0.0.1
```

2. **Install frontend dependencies (first run)**
```bash
cd src/webapp
npm install
```

3. **Launch the Vite dev server**
```bash
npm run dev
```

The frontend is served at `http://127.0.0.1:5173` and communicates with the FastAPI backend at `http://127.0.0.1:8000`.

Refer to the [Webapp Guide](docs/webapp_guide.md) for environment export scripts and API contract details.

## 🧭 Webapp Walkthrough

1. **Connect the services**: Ensure the FastAPI backend and the Vite dev server are both running using the commands above.
2. **Select or create a portfolio**: Use the portfolio selector in the left sidebar to load existing configurations or create a new one with budget, horizon, and tickers.
3. **Start an analysis**: Click **Run analysis**. The chat stream shows live status updates as data is fetched, predictions are generated, and allocations are synthesized.
4. **Monitor progress in the chat**: State messages summarize each workflow stage. Hovering over recent messages helps verify the sequence of events and any intermediate insights.
5. **Respond to MoneyBot prompts**: When prompted, choose between confirm, edit, deny, or cancel. Allocation edits open an inline editor; denials collect feedback that feeds a rerun request.
6. **Review recommended allocations**: The allocation card lists per-ticker share counts alongside current pricing and total notional values. Adjust values if needed before confirming.
7. **Iterate or finalize**: Confirm to log the session and conclude, or rerun with updated instructions to explore alternative recommendations.

## 📊 Usage Examples

### Basic Portfolio Analysis

```python
from src.langgraph_workflow.app import app
from src.langgraph_workflow.custom_types import PortfolioData

# Create portfolio data
portfolio = PortfolioData(
    budget=100,
    target_date="2025-12-31",
    holdings={"AAPL": 10, "GOOGL": 5, "TSLA": 3},
    currency="USD",
    risk_tolerance="medium",
    prediction_strength="strong"
)

# Run analysis workflow
result = await app.ainvoke({
    "app_data": {"selected_portfolio": "my_portfolio"},
    "portfolio_data": portfolio
})
```

### Custom Data Collection

```python
from src.data_pipeline.data_fetcher import DataFetcher, NewsFetcher
from src.data_pipeline.data_collector import DataCollector
from src.data_pipeline.llm_provider import LangChainLLMProvider

# Initialize components
llm_provider = LangChainLLMProvider(provider="openai")

# Advanced news fetching with API integration
news_fetcher = NewsFetcher(llm_provider)
news_fetcher.fetch_news("AAPL")  # Uses NewsData API + sentence transformers

# Default news collection from local cache
news_data = DataCollector.collect_news("AAPL")  # Loads from local files
sector_news = DataCollector.collect_sector_news("Technology")
market_news = DataCollector.collect_market_news("US")

# Fetch other data
fundamentals = DataFetcher.fetch_fundamentals("AAPL")
prices = DataFetcher.fetch_historical_prices("AAPL")
```

## 📰 News Collection System

MoneyBot features a sophisticated dual news collection system designed for both real-time analysis and efficient caching:

### Advanced NewsFetcher
- **External API Integration**: Uses NewsData API for real-time financial news
- **AI-Powered Filtering**: Sentence transformers for relevance scoring
- **GKG Theme Integration**: Global Knowledge Graph themes for contextual analysis
- **Multi-Source Aggregation**: Combines ticker-specific, sector, and market news
- **Rate Limiting**: Built-in API quota management

### Default DataCollector
- **Local Data Retrieval**: Efficient loading from cached news files
- **Smart Caching**: Automatic deduplication and freshness validation
- **Sector/Market Support**: Specialized collectors for different news categories
- **Performance Optimized**: Fast local file access for repeated queries

### Usage Patterns
- **Real-time Analysis**: NewsFetcher for fresh data collection
- **Cached Queries**: DataCollector for rapid repeated access
- **RAG Integration**: Both systems feed into vector databases for semantic search

## 🔧 Configuration

### Application presets

Runtime defaults now live under `config/` as layered YAML files:

- `config/base.yml` – baseline defaults used for local development (mirrors the legacy hardcoded values)
- `config/ci.test.yml` – lightweight settings for GitHub Actions test/coverage jobs
- `config/ci.deploy.yml` – high-performance profile for deployment pipelines
- `config/local.yml` – optional developer overrides (ignored by git)

Load a preset explicitly with:

```
python -m src.langgraph_workflow.app --preset ci.test
```

Omit `--preset` to fall back to `MONEYBOT_PRESET` (if set) or `base.yml`. All CLI values (`--host`, `--port`) default to the active preset but can be overridden per run.

### Frontend coordination

Keep the frontend API URL consistent by exporting the backend-configured origin:

```
python scripts/export_frontend_env.py --preset base
```

This writes `src/webapp/.env.development.local` with a matching `VITE_API_BASE_URL` for the selected preset.

### LLM Provider Options

The system supports multiple LLM provider, including self-hosted models.

### Prediction Strength Levels

- **Weak**: Basic trend analysis
- **Medium**: Enhanced feature engineering
- **Strong**: Advanced ML models with GPU acceleration

### Risk Tolerance Settings

- **Low**: Conservative investments, high diversification
- **Medium**: Balanced approach
- **High**: Aggressive growth strategies

## 🔍 API Reference

### Core Endpoints

- `POST /analysis/portfolio/{portfolio_id}/start` - Start analysis workflow
- `POST /analysis/portfolio/{portfolio_id}/respond` - Send user responses
- `GET /utils/load_portfolio_data` - Load portfolio configuration
- `POST /utils/create_portfolio` - Create new portfolio
- `POST /utils/get_shares_price` - Get current stock prices

### WebSocket Events

- `state_update` - Analysis progress updates
- `awaiting_choice` - User decision prompts
- `awaiting_allocation` - Portfolio allocation requests
- `end_analysis` - Analysis completion

## 🧪 Testing & Coverage

MoneyBot uses pytest with adaptive markers so CI can run on GitHub-hosted runners without GPU access.

```bash
# Fast unit tests
pytest -m "unit"

# API-level integration tests
pytest -m "integration"

# End-to-end simulations (confirm/deny/edit/cancel flows)
pytest -m "e2e"

# Full suite with coverage
pytest --cov=src --cov-report=xml --cov-report=term
```

See [docs/testing_and_coverage.md](docs/testing_and_coverage.md) for marker definitions, skip strategies, and CI pipeline details.

## 🚨 Troubleshooting

### Common Issues

1. **API Key Errors**
   - Ensure all required API keys are set in `.env`
   - Verify key permissions and quotas

2. **CUDA/GPU Issues**
   - Install PyTorch with CUDA support: `pip install torch --index-url https://download.pytorch.org/whl/cu118`
   - Set `DEVICE = 'cpu'` in `app.py` for CPU-only mode

3. **Data Loading Errors**
   - Check internet connectivity for API calls
   - Verify ticker symbols are valid
   - Ensure data directories have write permissions

4. **Memory Issues**
   - Reduce batch sizes in workflow configuration
   - Use smaller models for local inference
   - Enable parallel processing only if sufficient RAM

### Performance Optimization

- **GPU Acceleration**: Enable CUDA for faster predictions
- **Parallel Processing**: Set `RUN_PARALLEL = True` for concurrent operations
- **Caching**: Vector databases cache embeddings for faster retrieval
- **Rate Limiting**: Built-in delays prevent API quota exhaustion

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Commit changes: `git commit -am 'Add feature'`
4. Push to branch: `git push origin feature-name`
5. Submit a pull request

## Features in progress
- [ ] Long-term interactions logging to improve the recommendations
- [ ] Real-time portfolio tracking and rebalancing
- [ ] Webapp interface for the portfolio management

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **LangChain**: For LLM integration and workflow management
- **LangGraph**: For sophisticated state management
- **FastAPI**: For high-performance web framework
- **Yahoo Finance**: For market data access
- **NewsData API**: For financial news aggregation
- **ChromaDB**: For vector database functionality

## 📞 Support

For questions, issues, or contributions:

- Create an issue on GitHub
- Check the troubleshooting section
- Review the API documentation
- Examine example code in `src/full_example.py`

---

**Disclaimer**: This tool is for educational and research purposes only. Always consult with qualified financial advisors before making investment decisions. Past performance does not guarantee future results.
