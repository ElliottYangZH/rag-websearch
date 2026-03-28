# LangChain RAG Guide

## Overview
LangChain is a framework for developing applications powered by language models. It enables applications that:
- Are context-aware (connect LLM to other sources of context)
- Reason (use LLM to reason about how to respond to requests)

## Retrieval-Augmented Generation (RAG)
RAG is a technique for augmenting LLM knowledge with additional context from external sources.

### Key Components
1. **Document Loaders**: Load documents from various sources (PDF, Markdown, etc.)
2. **Text Splitters**: Split large documents into smaller chunks
3. **Embeddings**: Convert text into vector representations
4. **Vector Stores**: Store and search embedded documents (FAISS, Chroma, etc.)
5. **Retrievers**: Fetch relevant documents for a query
6. **Chain**: Orchestrate retrieval + LLM generation

### Basic RAG Chain
```python
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

# Create vector store from documents
vectorstore = FAISS.from_documents(documents, embeddings)

# Create retriever
retriever = vectorstore.as_retriever()

# Create QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model="gpt-4o-mini"),
    chain_type="stuff",
    retriever=retriever
)

# Query
result = qa_chain.invoke("What is LangChain?")
```

### Ensemble Retriever
Combine multiple retrievers with weighted scoring:
```python
from langchain.retrievers import EnsembleRetriever

ensemble = EnsembleRetriever(
    retrievers=[local_retriever, web_retriever],
    weights=[0.7, 0.3]
)
```

## Best Practices
- Use appropriate chunk sizes (1000-2000 chars typically)
- Include overlap between chunks (100-200 chars) to maintain context
- Use metadata to track source of each chunk
- Consider hybrid search (keyword + vector) for better results