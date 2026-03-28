# RAG WebSearch

A Retrieval-Augmented Generation (RAG) pipeline that combines local documents with live web search to provide grounded, cited answers.

## Features

- **Local Document Retrieval**: Load and index PDFs, Markdown, and text files
- **Live Web Search**: DuckDuckGo integration (no API key required)
- **Ensemble Retrieval**: Combines local (70%) and web (30%) results
- **Grounded Answers**: GPT-4o-mini generates answers with citations
- **Query Caching**: In-memory cache for repeated queries
- **Logging**: All queries and retrieved documents are logged

## Quick Start

### 1. Install Dependencies

```bash
# Use Python 3.12 (required)
py -3.12 -m pip install -r requirements.txt
```

### 2. Configure API Key

Edit the `.env` file and add your OpenAI API key:

```env
OPENAI_API_KEY=sk-your-openai-api-key-here
```

Get your API key from: https://platform.openai.com/api-keys

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
    model_name="gpt-4o-mini",
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
                              GPT-4o-mini (LLM)
                                       ↓
                           Grounded Answer with Citations
```

### Components

| Component | Description |
|-----------|-------------|
| `src/document_loader.py` | Load PDFs, Markdown, text files |
| `src/vector_store.py` | FAISS vector store with OpenAI embeddings |
| `src/web_retriever.py` | DuckDuckGo search retriever |
| `src/ensemble_retriever.py` | Combine local + web with weighted RRF |
| `src/rag_chain.py` | RAG chain orchestration |
| `rag_agent.py` | Main CLI interface with caching and logging |

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key | Required |
| `LOG_LEVEL` | Logging level | INFO |

### RAGAgent Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `docs_path` | Path to local documents | "docs" |
| `vectorstore_path` | Path to save/load vector store | "vectorstore" |
| `local_weight` | Weight for local retrieval | 0.7 |
| `web_weight` | Weight for web retrieval | 0.3 |
| `k` | Number of documents to retrieve | 4 |
| `model_name` | OpenAI model | "gpt-4o-mini" |
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
│   └── rag_chain.py
├── logs/                               # Log files
├── vectorstore/                        # Persisted vector store
├── rag_agent.py                        # Main CLI
├── requirements.txt
└── README.md
```

## Important Notes

- **Python Version**: Use Python 3.12 with this project (`py -3.12`)
- **OpenAI API Key**: Required for embeddings and LLM
- **DuckDuckGo**: No API key required for web search

## License

MIT