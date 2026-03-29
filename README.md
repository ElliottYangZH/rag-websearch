# RAG WebSearch

A Retrieval-Augmented Generation (RAG) pipeline that combines local documents with live web search to provide grounded, cited answers.

## Features

- **Local Document Retrieval**: Load and index PDFs, Markdown, and text files
- **Live Web Search**: DuckDuckGo integration (no API key required)
- **Ensemble Retrieval**: Combines local (70%) and web (30%) results with Reciprocal Rank Fusion
- **Multi-Provider LLM**: Supports OpenAI, Azure, Anthropic, Google, AWS Bedrock, and Ollama
- **Multi-Provider Embeddings**: Supports OpenAI, Google, and Azure embeddings
- **Query Caching**: In-memory cache for repeated queries
- **Logging**: All queries and retrieved documents are logged
- **Reusable Utilities**: `websearch_utils.py` - standalone web search module for any project

## Quick Start

### Using websearch_utils.py in ANY Project

Copy [`websearch_utils.py`](websearch_utils.py) to your new project:

```python
from websearch_utils import websearch_qa

# Works with any LLM provider
result = websearch_qa("Latest AI news?", provider="anthropic")
print(result["answer"])

# Or create a chain for more control
from websearch_utils import create_websearch_chain

chain = create_websearch_chain(provider="openai", model_name="gpt-4o-mini")
result = chain.invoke({"query": "What is Python?"})
```

Supported providers: `openai`, `anthropic`, `google`, `azure`, `ollama`

### 1. Install Dependencies

```bash
# Use Python 3.12 (required)
py -3.12 -m pip install -r requirements.txt
```

### 2. Configure API Key

Edit the `.env` file and configure your LLM provider:

**OpenAI (default):**
```env
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini
OPENAI_API_KEY=sk-your-openai-api-key-here
```

**Azure OpenAI:**
```env
LLM_PROVIDER=azure
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your-key
AZURE_OPENAI_DEPLOYMENT=gpt-4o-mini
```

**Anthropic Claude:**
```env
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-api-key
```

**Google Gemini:**
```env
LLM_PROVIDER=google
GOOGLE_API_KEY=your-google-api-key
```

**AWS Bedrock:**
```env
LLM_PROVIDER=aws_bedrock
AWS_ACCESS_KEY_ID=your-access-key
AWS_SECRET_ACCESS_KEY=your-secret-key
AWS_REGION=us-east-1
```

**Ollama (local):**
```env
LLM_PROVIDER=ollama
LLM_MODEL=llama2
```

### 3. Add Documents (Optional)

Add your documents to the `docs/` directory. The agent comes with sample documents:
- `docs/langchain_guide.md` - LangChain RAG documentation
- `docs/vscode_tips.md` - VS Code AI tips
- `docs/sample.txt` - Sample text file

### 4. Run the Agent

```bash
# Use Python 3.12 to run
py -3.12 rag_agent.py
```

### 5. Example Queries

```
>>> What is LangChain?
>>> Summarize the VS Code tips
>>> Compare RAG approaches
>>> quit
```

## Usage

### Programmatic Usage

```python
from rag_agent import ask

# Simple usage
result = ask("What is LangChain?")
print(result["answer"])
print(result["sources"])

# Access full result
print(f"Query: {result['query']}")
print(f"Cached: {result['cached']}")
for source in result["sources"]:
    print(f"  [{source['type']}] {source['title']}")
```

### Advanced Usage

```python
from rag_agent import RAGAgent

agent = RAGAgent(
    docs_path="docs",
    vectorstore_path="vectorstore",
    local_weight=0.7,
    web_weight=0.3,
    k=4,
    llm_provider="openai",    # or "azure", "anthropic", "google", "aws_bedrock", "ollama"
    model_name="gpt-4o-mini",  # defaults to LLM_MODEL env var
    use_cache=True,
    cache_ttl=3600
)

# Interactive mode
agent.interactive()

# Single query
result = agent.ask("Your question here")
```

## CLI Commands

When running in interactive mode:
- `quit` / `exit` / `q` - Exit the program
- `cache clear` - Clear the query cache
- `cache stats` - Show cache statistics

## Architecture

```
User Query → Ensemble Retriever → [Local FAISS + DuckDuckGo]
                                       ↓
                              Retrieved Documents
                                       ↓
                         Multi-Provider LLM (configurable)
                              OpenAI / Azure / Anthropic /
                              Google / AWS Bedrock / Ollama
                                       ↓
                           Grounded Answer with Citations
```

### Components

| Component | Description |
|-----------|-------------|
| `src/document_loader.py` | Load PDFs, Markdown, text files |
| `src/vector_store.py` | FAISS vector store with configurable embeddings |
| `src/web_retriever.py` | DuckDuckGo search retriever |
| `src/ensemble_retriever.py` | Combine local + web with weighted RRF |
| `src/rag_chain.py` | RAG chain orchestration |
| `src/llm_provider.py` | Multi-provider LLM factory (OpenAI, Azure, Anthropic, Google, AWS, Ollama) |
| `rag_agent.py` | Main CLI interface with caching and logging |

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `LLM_PROVIDER` | LLM provider (openai, azure, anthropic, google, aws_bedrock, ollama) | "openai" |
| `LLM_MODEL` | Model name (provider-specific) | "gpt-4o-mini" |
| `LLM_TEMPERATURE` | Sampling temperature | 0 |
| `OPENAI_API_KEY` | OpenAI API key | Required (for openai provider) |
| `AZURE_OPENAI_ENDPOINT` | Azure OpenAI endpoint | Required (for azure provider) |
| `AZURE_OPENAI_API_KEY` | Azure OpenAI API key | Required (for azure provider) |
| `AZURE_OPENAI_DEPLOYMENT` | Azure deployment name | Required (for azure provider) |
| `ANTHROPIC_API_KEY` | Anthropic API key | Required (for anthropic provider) |
| `GOOGLE_API_KEY` | Google API key | Required (for google provider) |
| `AWS_ACCESS_KEY_ID` | AWS access key | Required (for aws_bedrock provider) |
| `AWS_SECRET_ACCESS_KEY` | AWS secret key | Required (for aws_bedrock provider) |
| `AWS_REGION` | AWS region | "us-east-1" |
| `EMBEDDING_PROVIDER` | Embeddings provider (openai, google, azure) | "openai" |
| `OPENAI_EMBEDDING_MODEL` | OpenAI embedding model | "text-embedding-3-small" |
| `GOOGLE_EMBEDDING_MODEL` | Google embedding model | "models/embedding-001" |
| `LOG_LEVEL` | Logging level | INFO |

### RAGAgent Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `docs_path` | Path to local documents | "docs" |
| `vectorstore_path` | Path to save/load vector store | "vectorstore" |
| `local_weight` | Weight for local retrieval | 0.7 |
| `web_weight` | Weight for web retrieval | 0.3 |
| `k` | Number of documents to retrieve | 4 |
| `llm_provider` | LLM provider to use | "openai" |
| `model_name` | LLM model (defaults to LLM_MODEL env var) | "gpt-4o-mini" |
| `use_cache` | Enable query caching | True |
| `cache_ttl` | Cache TTL in seconds | 3600 |

## Logging

Logs are stored in `logs/rag_agent.log` and include:
- Timestamp
- Query text
- Retrieved sources
- Answer (truncated)

## File Structure

```
rag-websearch/
├── .venv/                              # Virtual environment
├── .env                                # API keys
├── docs/                               # Local documents
│   ├── langchain_guide.md
│   ├── vscode_tips.md
│   └── sample.txt
├── src/
│   ├── __init__.py
│   ├── document_loader.py
│   ├── vector_store.py
│   ├── web_retriever.py
│   ├── ensemble_retriever.py
│   ├── rag_chain.py
│   └── llm_provider.py                 # Multi-provider LLM factory
├── logs/                               # Log files
├── vectorstore/                        # Persisted vector store
├── rag_agent.py                        # Main CLI
├── requirements.txt
└── README.md
```

## Important Notes

- **Python Version**: Use Python 3.12 with this project (`py -3.12`)
- **LLM Provider**: Configure via `LLM_PROVIDER` env var (default: OpenAI)
- **API Keys**: Required based on your chosen provider (see Configuration section)
- **DuckDuckGo**: No API key required for web search

## License

MIT