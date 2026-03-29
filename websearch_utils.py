#!/usr/bin/env python3
"""
WebSearch Utilities - Reusable web search with any LLM.

This module provides a simple interface to enable web search capabilities
for any LLM (OpenAI, Anthropic, Azure, Google, Ollama, etc.).

Usage:
    # Quick Q&A with web search
    from websearch_utils import websearch_qa
    result = websearch_qa("What is the latest news about AI?", provider="openai")
    print(result["answer"])

    # Or create a chain for more control
    from websearch_utils import create_websearch_chain
    chain = create_websearch_chain(provider="anthropic", model_name="claude-3-haiku-20240307")
    result = chain.invoke({"query": "Latest tech news"})

Copy this file to any new project to enable web search with any LLM.
Dependencies: langchain-core, langchain-openai, ddgs
"""

import os
import logging
from typing import Dict, Any, Optional, List
from pydantic import Field
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# =============================================================================
# DuckDuckGo Web Retriever
# =============================================================================

try:
    from ddgs import DDGS
except ImportError:
    raise ImportError(
        "ddgs is required for web search. Install with: pip install duckduckgo-search"
    )

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever


class DuckDuckGoRetriever(BaseRetriever):
    """
    A retriever that searches the web using DuckDuckGo.
    
    This retriever uses the duckduckgo-search library to perform
    web searches and returns results as LangChain Document objects.
    """
    
    top_k: int = Field(default=5, description="Number of top results to retrieve")
    max_snippet_length: int = Field(default=500, description="Maximum length of text snippets")
    region: str = Field(default="wt-wt", description="DuckDuckGo region code")
    safesearch: str = Field(default="moderate", description="Safe search setting ('on', 'moderate', 'off')")
    
    model_config = {
        "arbitrary_types_allowed": True
    }
    
    def _search_duckduckgo(self, query: str) -> List[dict]:
        """Perform a DuckDuckGo search."""
        try:
            results = []
            
            # Try news search first
            with DDGS() as ddgs:
                search_results = ddgs.news(
                    query,
                    max_results=self.top_k,
                    region=self.region,
                    safesearch=self.safesearch
                )
                
                for result in search_results:
                    results.append({
                        "title": result.get("title", ""),
                        "url": result.get("url", ""),
                        "description": result.get("description", ""),
                        "body": result.get("body", "")[:self.max_snippet_length]
                    })
            
            # If no news results, try general search
            if not results:
                with DDGS() as ddgs:
                    search_results = ddgs.text(
                        query,
                        max_results=self.top_k,
                        region=self.region,
                        safesearch=self.safesearch
                    )
                    
                    for result in search_results:
                        results.append({
                            "title": result.get("title", ""),
                            "url": result.get("url", ""),
                            "description": result.get("description", ""),
                            "body": result.get("body", "")[:self.max_snippet_length] if result.get("body") else result.get("description", "")[:self.max_snippet_length]
                        })
            
            return results
            
        except Exception as e:
            logger.error(f"DuckDuckGo search error: {e}")
            return []
    
    def _result_to_document(self, result: dict) -> Document:
        """Convert a search result to a Document."""
        content_parts = []
        if result.get("title"):
            content_parts.append(f"Title: {result['title']}")
        if result.get("description"):
            content_parts.append(f"Description: {result['description']}")
        if result.get("body"):
            content_parts.append(f"Content: {result['body']}")
        
        content = "\n\n".join(content_parts)
        
        return Document(
            page_content=content,
            metadata={
                "source": "web",
                "source_type": "duckduckgo",
                "url": result.get("url", ""),
                "title": result.get("title", ""),
                "description": result.get("description", ""),
            }
        )
    
    def _get_relevant_documents(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForRetrieverRun] = None
    ) -> List[Document]:
        """Get relevant documents for a query."""
        logger.info(f"Searching DuckDuckGo for: {query}")
        
        results = self._search_duckduckgo(query)
        
        documents = [self._result_to_document(result) for result in results]
        
        logger.info(f"Found {len(documents)} web results")
        
        return documents


# =============================================================================
# LLM Provider (Simplified - add more as needed)
# =============================================================================

LLM_PROVIDERS = {
    "openai": {
        "env_key": "OPENAI_API_KEY",
        "default_model": "gpt-4o-mini",
    },
    "anthropic": {
        "env_key": "ANTHROPIC_API_KEY",
        "default_model": "claude-3-haiku-20240307",
    },
    "google": {
        "env_key": "GOOGLE_API_KEY",
        "default_model": "gemini-pro",
    },
    "azure": {
        "env_key": "AZURE_OPENAI_API_KEY",
        "default_model": "gpt-4o-mini",
    },
    "ollama": {
        "env_key": None,  # No API key needed
        "default_model": "llama2",
    },
}


def get_llm(provider: str = "openai", model_name: Optional[str] = None, **kwargs):
    """
    Get an LLM instance for any supported provider.
    
    Args:
        provider: Provider name (openai, anthropic, google, azure, ollama).
        model_name: Model name (defaults to provider's default).
        **kwargs: Additional provider-specific arguments.
    
    Returns:
        LLM instance (ChatModel).
    """
    provider = provider.lower()
    
    if provider not in LLM_PROVIDERS:
        raise ValueError(
            f"Unknown provider: {provider}. Supported: {list(LLM_PROVIDERS.keys())}"
        )
    
    config = LLM_PROVIDERS[provider]
    
    # Check API key
    if config["env_key"]:
        api_key = os.getenv(config["env_key"])
        if not api_key:
            raise ValueError(
                f"{config['env_key']} not found. Set it in .env or environment."
            )
    
    # Get model name
    if model_name is None:
        model_name = os.getenv(f"{provider.upper()}_MODEL") or config["default_model"]
    
    # Import and create LLM based on provider
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model_name,
            api_key=os.getenv("OPENAI_API_KEY"),
            **kwargs
        )
    
    elif provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=model_name,
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
            **kwargs
        )
    
    elif provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        # Strip 'google/' prefix if present
        model_name = model_name.replace("google/", "")
        return ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            **kwargs
        )
    
    elif provider == "azure":
        from langchain_openai import AzureChatOpenAI
        return AzureChatOpenAI(
            model=model_name,
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"),
            **kwargs
        )
    
    elif provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=model_name,
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            **kwargs
        )


# =============================================================================
# WebSearch Chain Creation
# =============================================================================

def create_web_retriever(top_k: int = 5, **kwargs) -> DuckDuckGoRetriever:
    """
    Create a DuckDuckGo web retriever.
    
    Args:
        top_k: Number of results to retrieve.
        **kwargs: Additional arguments for DuckDuckGoRetriever.
    
    Returns:
        DuckDuckGoRetriever instance.
    """
    return DuckDuckGoRetriever(top_k=top_k, **kwargs)


def create_websearch_chain(
    provider: str = "openai",
    model_name: Optional[str] = None,
    top_k: int = 5,
    temperature: float = 0,
    max_tokens: int = 1000,
) -> Any:
    """
    Create a web search Q&A chain with any LLM.
    
    Args:
        provider: LLM provider (openai, anthropic, google, azure, ollama).
        model_name: Model name (defaults to provider's default).
        top_k: Number of web search results.
        temperature: Sampling temperature.
        max_tokens: Maximum tokens in response.
    
    Returns:
        RetrievalQA chain ready to use.
    """
    from langchain_core.chains import RetrievalQA
    
    # Create components
    retriever = create_web_retriever(top_k=top_k)
    llm = get_llm(provider=provider, model_name=model_name, temperature=temperature, max_tokens=max_tokens)
    
    # Create chain
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
    )


def websearch_qa(
    query: str,
    provider: str = "openai",
    model_name: Optional[str] = None,
    top_k: int = 5,
) -> Dict[str, Any]:
    """
    Quick Q&A function using web search and any LLM.
    
    Args:
        query: The question to answer.
        provider: LLM provider (openai, anthropic, google, azure, ollama).
        model_name: Model name (defaults to provider's default).
        top_k: Number of web search results.
    
    Returns:
        Dictionary containing:
            - answer: The generated answer
            - sources: List of source documents
    """
    chain = create_websearch_chain(
        provider=provider,
        model_name=model_name,
        top_k=top_k,
    )
    
    result = chain.invoke({"query": query})
    
    # Format sources
    sources = []
    for doc in result.get("source_documents", []):
        metadata = doc.metadata
        sources.append({
            "title": metadata.get("title", "Unknown"),
            "url": metadata.get("url", ""),
            "snippet": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
        })
    
    return {
        "answer": result["result"] if isinstance(result, dict) else result,
        "sources": sources,
        "query": query,
    }


# =============================================================================
# CLI Interface
# =============================================================================

if __name__ == "__main__":
    import sys
    
    print("=" * 60)
    print("WebSearch Q&A - Web search with any LLM")
    print("=" * 60)
    print("Usage: python websearch_utils.py")
    print()
    
    # Get provider from env or default
    provider = os.getenv("LLM_PROVIDER", "openai")
    model = os.getenv("LLM_MODEL", None)
    top_k = int(os.getenv("WEBSEARCH_TOP_K", "5"))
    
    print(f"Provider: {provider}")
    print(f"Model: {model or 'default'}")
    print(f"Top K: {top_k}")
    print("-" * 60)
    print("Type 'quit' to exit")
    print()
    
    while True:
        try:
            query = input("\n>>> ").strip()
            
            if not query:
                continue
            
            if query.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break
            
            print("\nSearching...")
            result = websearch_qa(query, provider=provider, model_name=model, top_k=top_k)
            
            print("\n" + "=" * 60)
            print("ANSWER:")
            print("-" * 60)
            print(result["answer"])
            print("-" * 60)
            print("SOURCES:")
            for i, source in enumerate(result["sources"], 1):
                print(f"  [{i}] {source['title']}")
                print(f"      {source['url']}")
            print("=" * 60)
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            logger.error(f"Error: {e}", exc_info=True)
            print(f"Error: {e}")
