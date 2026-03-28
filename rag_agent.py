#!/usr/bin/env python3
"""
RAG Agent - A CLI tool for querying using local documents and web search.

Usage:
    python rag_agent.py
    
    # Or import as a module:
    from rag_agent import ask
    result = ask("What is LangChain?")
"""

import os
import sys
import json
import logging
import hashlib
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv

# Import RAG components
from src.document_loader import DocumentLoader
from src.vector_store import VectorStoreManager
from src.web_retriever import DuckDuckGoRetriever
from src.ensemble_retriever import create_ensemble_retriever
from src.rag_chain import RAGChain

# Load environment variables
load_dotenv()

# Configure logging
LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "rag_agent.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class QueryCache:
    """Simple in-memory cache for query results."""
    
    def __init__(self, ttl_seconds: int = 3600):
        """
        Initialize the cache.
        
        Args:
            ttl_seconds: Time-to-live for cached results (default: 1 hour).
        """
        self.cache: Dict[str, tuple] = {}  # key: (result, timestamp)
        self.ttl = ttl_seconds
    
    def _make_key(self, query: str) -> str:
        """Generate a cache key from query."""
        # Normalize query
        normalized = query.lower().strip()
        return hashlib.sha256(normalized.encode()).hexdigest()
    
    def get(self, query: str) -> Optional[Dict[str, Any]]:
        """Get cached result for query."""
        key = self._make_key(query)
        
        if key in self.cache:
            result, timestamp = self.cache[key]
            
            # Check if still valid
            if time.time() - timestamp < self.ttl:
                logger.info(f"Cache hit for query: {query}")
                return result
            else:
                # Expired
                del self.cache[key]
        
        return None
    
    def set(self, query: str, result: Dict[str, Any]) -> None:
        """Cache a result for query."""
        key = self._make_key(query)
        self.cache[key] = (result, time.time())
        logger.info(f"Cached result for query: {query}")
    
    def clear(self) -> None:
        """Clear all cached results."""
        self.cache.clear()
        logger.info("Cache cleared")
    
    def size(self) -> int:
        """Get number of cached items."""
        return len(self.cache)


class RAGAgent:
    """
    RAG Agent that combines local document retrieval with web search.
    
    This is the main class for the RAG pipeline that:
    - Loads and indexes local documents
    - Performs web search via DuckDuckGo
    - Combines results using ensemble retrieval
    - Generates grounded answers with citations
    """
    
    def __init__(
        self,
        docs_path: str = "docs",
        vectorstore_path: Optional[str] = None,
        local_weight: float = 0.7,
        web_weight: float = 0.3,
        k: int = 4,
        model_name: Optional[str] = None,
        use_cache: bool = True,
        cache_ttl: int = 3600,
        llm_provider: Optional[str] = None
    ):
        """
        Initialize the RAG agent.
        
        Args:
            docs_path: Path to local documents directory.
            vectorstore_path: Optional path to save/load vector store.
            local_weight: Weight for local retrieval (0-1).
            web_weight: Weight for web retrieval (0-1).
            k: Number of documents to retrieve.
            model_name: LLM model to use (defaults to LLM_MODEL env var).
            use_cache: Whether to use query caching.
            cache_ttl: Cache time-to-live in seconds.
            llm_provider: LLM provider to use (openai, azure, anthropic, ollama, google, aws_bedrock).
        """
        self.docs_path = docs_path
        self.vectorstore_path = vectorstore_path
        self.local_weight = local_weight
        self.web_weight = web_weight
        self.k = k
        self.model_name = model_name or os.getenv("LLM_MODEL", "gpt-4o-mini")
        self.llm_provider = llm_provider or os.getenv("LLM_PROVIDER", "openai")
        
        # Initialize cache
        self.use_cache = use_cache
        self.cache = QueryCache(ttl_seconds=cache_ttl) if use_cache else None
        
        # Components (initialized lazily)
        self._document_loader: Optional[DocumentLoader] = None
        self._vectorstore_manager: Optional[VectorStoreManager] = None
        self._web_retriever: Optional[DuckDuckGoRetriever] = None
        self._ensemble_retriever = None
        self._rag_chain: Optional[RAGChain] = None
        
        # Check API key based on provider
        self._check_api_key()
    
    def _check_api_key(self):
        """Check if required API key is set for the current provider."""
        provider = self.llm_provider.lower()
        
        if provider == "openai":
            if not os.getenv("OPENAI_API_KEY"):
                raise ValueError(
                    "OPENAI_API_KEY not found in environment. "
                    "Please set it in .env file or environment."
                )
        elif provider == "azure":
            if not os.getenv("AZURE_OPENAI_ENDPOINT"):
                raise ValueError(
                    "AZURE_OPENAI_ENDPOINT not found in environment. "
                    "Please set it in .env file or environment."
                )
        elif provider == "anthropic":
            if not os.getenv("ANTHROPIC_API_KEY"):
                raise ValueError(
                    "ANTHROPIC_API_KEY not found in environment. "
                    "Please set it in .env file or environment."
                )
        elif provider == "google":
            if not os.getenv("GOOGLE_API_KEY"):
                raise ValueError(
                    "GOOGLE_API_KEY not found in environment. "
                    "Please set it in .env file or environment."
                )
        elif provider == "aws_bedrock":
            if not os.getenv("AWS_ACCESS_KEY_ID") or not os.getenv("AWS_SECRET_ACCESS_KEY"):
                raise ValueError(
                    "AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY not found in environment. "
                    "Please set them in .env file or environment."
                )
        elif provider == "ollama":
            # Ollama runs locally - no API key required
            logger.info("Ollama provider selected - no API key required (runs locally)")
    
    def _initialize_document_loader(self) -> DocumentLoader:
        """Initialize the document loader."""
        if self._document_loader is None:
            self._document_loader = DocumentLoader()
            logger.info("Document loader initialized")
        return self._document_loader
    
    def _initialize_vectorstore(self) -> VectorStoreManager:
        """Initialize the vector store."""
        if self._vectorstore_manager is None:
            self._vectorstore_manager = VectorStoreManager()
            
            # Try to load existing vector store
            if self.vectorstore_path and os.path.exists(self.vectorstore_path):
                try:
                    self._vectorstore_manager.load_vectorstore(self.vectorstore_path)
                    logger.info(f"Loaded vector store from {self.vectorstore_path}")
                except Exception as e:
                    logger.warning(f"Failed to load vector store: {e}")
                    self._create_vectorstore()
            else:
                self._create_vectorstore()
        
        return self._vectorstore_manager
    
    def _create_vectorstore(self) -> None:
        """Create vector store from documents."""
        loader = self._initialize_document_loader()
        
        # Load documents
        if os.path.isdir(self.docs_path):
            documents = loader.load_directory(self.docs_path)
        elif os.path.isfile(self.docs_path):
            documents = loader.load_file(self.docs_path)
        else:
            logger.warning(f"Documents path not found: {self.docs_path}")
            documents = []
        
        if documents:
            # Split into chunks
            chunks = loader.split_documents(documents)
            logger.info(f"Loaded {len(documents)} documents, created {len(chunks)} chunks")
            
            # Create vector store
            self._vectorstore_manager.create_vectorstore(chunks)
            
            # Save if path specified
            if self.vectorstore_path:
                self._vectorstore_manager.save_vectorstore(self.vectorstore_path)
                logger.info(f"Saved vector store to {self.vectorstore_path}")
        else:
            logger.warning("No documents loaded")
    
    def _initialize_web_retriever(self) -> DuckDuckGoRetriever:
        """Initialize the web retriever."""
        if self._web_retriever is None:
            self._web_retriever = DuckDuckGoRetriever(top_k=self.k)
            logger.info("Web retriever initialized")
        return self._web_retriever
    
    def _initialize_ensemble_retriever(self):
        """Initialize the ensemble retriever."""
        if self._ensemble_retriever is None:
            vectorstore_mgr = self._initialize_vectorstore()
            local_retriever = vectorstore_mgr.as_retriever(search_kwargs={"k": self.k})
            web_retriever = self._initialize_web_retriever()
            
            self._ensemble_retriever = create_ensemble_retriever(
                local_retriever=local_retriever,
                web_retriever=web_retriever,
                local_weight=self.local_weight,
                web_weight=self.web_weight,
                k=self.k
            )
            logger.info("Ensemble retriever initialized")
        
        return self._ensemble_retriever
    
    def _initialize_rag_chain(self) -> RAGChain:
        """Initialize the RAG chain."""
        if self._rag_chain is None:
            ensemble = self._initialize_ensemble_retriever()
            self._rag_chain = RAGChain(
                retriever=ensemble,
                model_name=self.model_name,
                llm_provider=self.llm_provider
            )
            logger.info(f"RAG chain initialized with provider: {self.llm_provider}")
        
        return self._rag_chain
    
    def ask(self, query: str, use_cache: Optional[bool] = None) -> Dict[str, Any]:
        """
        Ask a question and get a grounded answer.
        
        Args:
            query: The question to answer.
            use_cache: Override cache setting for this query.
            
        Returns:
            Dictionary containing:
                - answer: The generated answer
                - sources: List of source information
                - query: The original query
                - cached: Whether result was from cache
        """
        # Check cache first
        cache_use = use_cache if use_cache is not None else self.use_cache
        if cache_use and self.cache:
            cached = self.cache.get(query)
            if cached:
                cached["cached"] = True
                return cached
        
        logger.info(f"Processing query: {query}")
        
        # Get RAG chain and process query
        chain = self._initialize_rag_chain()
        result = chain.ask(query)
        result["cached"] = False
        
        # Log the query and sources
        self._log_query(query, result)
        
        # Cache the result
        if cache_use and self.cache:
            self.cache.set(query, result)
        
        return result
    
    def _log_query(self, query: str, result: Dict[str, Any]) -> None:
        """Log query and result."""
        logger.info(f"Query: {query}")
        logger.info(f"Answer: {result['answer'][:200]}...")
        logger.info(f"Sources: {len(result['sources'])} documents")
        
        for i, source in enumerate(result["sources"]):
            source_type = source["type"]
            title = source.get("title", "Unknown")
            if source_type == "web":
                url = source.get("url", "N/A")
                logger.info(f"  [{i+1}] Web: {title} - {url}")
            else:
                logger.info(f"  [{i+1}] Local: {title}")
    
    def interactive(self) -> None:
        """Run interactive CLI loop."""
        print("=" * 60)
        print("RAG WebSearch Agent")
        print("=" * 60)
        print(f"Local documents: {self.docs_path}")
        print(f"Provider: {self.llm_provider}")
        print(f"Model: {self.model_name}")
        print(f"Retrieval weights: local={self.local_weight}, web={self.web_weight}")
        print(f"Caching: {'enabled' if self.use_cache else 'disabled'}")
        print("=" * 60)
        print("Type 'quit', 'exit', or Ctrl+C to exit.")
        print("Type 'cache clear' to clear the query cache.")
        print("Type 'cache stats' to see cache statistics.")
        print("-" * 60)
        
        while True:
            try:
                query = input("\n>>> ").strip()
                
                if not query:
                    continue
                
                if query.lower() in ["quit", "exit", "q"]:
                    print("Goodbye!")
                    break
                
                if query.lower() == "cache clear":
                    if self.cache:
                        self.cache.clear()
                    print("Cache cleared.")
                    continue
                
                if query.lower() == "cache stats":
                    if self.cache:
                        print(f"Cache size: {self.cache.size()} items")
                    else:
                        print("Cache is disabled.")
                    continue
                
                # Process query
                print("\nProcessing...")
                result = self.ask(query)
                
                # Display result
                print("\n" + "=" * 60)
                print("ANSWER:")
                print("-" * 60)
                print(result["answer"])
                print("-" * 60)
                print("SOURCES:")
                for i, source in enumerate(result["sources"]):
                    source_type = source["type"]
                    title = source.get("title", "Unknown")
                    
                    if source_type == "web":
                        url = source.get("url", "N/A")
                        print(f"  [{i+1}] [Web] {title}")
                        print(f"      URL: {url}")
                    else:
                        source_file = source.get("source", "Unknown")
                        print(f"  [{i+1}] [Local] {title}")
                        print(f"      File: {source_file}")
                
                if result.get("cached"):
                    print("-" * 60)
                    print("(Result from cache)")
                
                print("=" * 60)
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except EOFError:
                print("\nGoodbye!")
                break
            except Exception as e:
                logger.error(f"Error processing query: {e}", exc_info=True)
                print(f"\nError: {e}")


# Global agent instance (lazy initialization)
_agent: Optional[RAGAgent] = None


def get_agent(
    docs_path: str = "docs",
    vectorstore_path: str = "vectorstore",
    local_weight: float = 0.7,
    web_weight: float = 0.3,
    k: int = 4,
    model_name: Optional[str] = None,
    llm_provider: Optional[str] = None
) -> RAGAgent:
    """Get or create the global RAG agent instance."""
    global _agent
    
    if _agent is None:
        _agent = RAGAgent(
            docs_path=docs_path,
            vectorstore_path=vectorstore_path,
            local_weight=local_weight,
            web_weight=web_weight,
            k=k,
            model_name=model_name,
            llm_provider=llm_provider
        )
    
    return _agent


def ask(
    query: str,
    docs_path: str = "docs",
    vectorstore_path: str = "vectorstore",
    local_weight: float = 0.7,
    web_weight: float = 0.3,
    k: int = 4,
    llm_provider: Optional[str] = None
) -> Dict[str, Any]:
    """
    Convenience function to ask a question.
    
    Args:
        query: The question to answer.
        docs_path: Path to local documents.
        vectorstore_path: Path to save/load vector store.
        local_weight: Weight for local retrieval.
        web_weight: Weight for web retrieval.
        k: Number of documents to retrieve.
        llm_provider: LLM provider to use.
        
    Returns:
        Dictionary with answer and sources.
    """
    agent = get_agent(
        docs_path=docs_path,
        vectorstore_path=vectorstore_path,
        local_weight=local_weight,
        web_weight=web_weight,
        k=k,
        llm_provider=llm_provider
    )
    return agent.ask(query)


if __name__ == "__main__":
    # Run interactive mode
    agent = RAGAgent()
    agent.interactive()