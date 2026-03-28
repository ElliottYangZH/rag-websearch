# VS Code RAG Tips

## VS Code AI Extensions

### GitHub Copilot
- AI pair programming assistant
- Autocomplete suggestions as you type
- Chat feature for explaining code

### Continue Extension
- Open-source AI coding assistant
- Works with local models or API keys
- RAG capabilities for codebase context

## RAG Development in VS Code

### Recommended Extensions
1. **Python** - Python language support
2. **Jupyter** - Interactive notebooks
3. **GitHub Copilot** - AI assistance
4. **GitLens** - Git history and annotations

### Debugging RAG Applications
- Use print statements for document retrieval debugging
- Check vector store contents with FAISS inspection
- Monitor API calls and token usage

### Environment Setup
```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Run the agent
python rag_agent.py
```

## Testing RAG Queries
1. Check retrieved documents (source, content, scores)
2. Verify LLM response accuracy
3. Validate citation formatting
4. Test edge cases (empty results, long queries)

## Performance Tips
- Cache embeddings for repeated documents
- Use batch processing for large document sets
- Limit retrieved documents to top-k most relevant
- Consider smaller embedding models for large corpora