"""RAG chain implementation using LangChain."""

import os
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path

from langchain_classic.chains import RetrievalQA
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document

from dotenv import load_dotenv
from src.llm_provider import get_llm_provider

load_dotenv()

logger = logging.getLogger(__name__)


class RAGChain:
    """
    RAG chain that combines retrieval with LLM generation.
    
    This class orchestrates document retrieval and LLM-based
    answer generation into a unified pipeline.
    """
    
    def __init__(
        self,
        retriever: BaseRetriever,
        model_name: Optional[str] = None,
        temperature: float = 0,
        max_tokens: int = 1000,
        return_source_documents: bool = True,
        llm_provider: Optional[str] = None
    ):
        """
        Initialize the RAG chain.
        
        Args:
            retriever: Retriever to fetch relevant documents.
            model_name: Model to use (defaults to LLM_MODEL env var or "gpt-4o-mini").
            temperature: Sampling temperature (0 = factual).
            max_tokens: Maximum tokens in response.
            return_source_documents: Whether to return source documents.
            llm_provider: LLM provider to use (defaults to LLM_PROVIDER env var or "openai").
        """
        self.retriever = retriever
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.return_source_documents = return_source_documents
        
        # Initialize LLM using the factory
        llm_provider_instance = get_llm_provider(
            provider=llm_provider,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens
        )
        self.llm = llm_provider_instance.get_llm()
        self.model_name = model_name or os.getenv("LLM_MODEL", "gpt-4o-mini")
        
        # Create the QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=return_source_documents,
            chain_type_kwargs={
                "document_variable_name": "context"
            }
        )
    
    def ask(self, query: str) -> Dict[str, Any]:
        """
        Ask a question and get a grounded answer.
        
        Args:
            query: The question to answer.
            
        Returns:
            Dictionary containing:
                - answer: The generated answer
                - source_documents: List of source documents (if enabled)
                - sources: Formatted list of sources
        """
        logger.info(f"Processing query: {query}")
        
        result = self.qa_chain.invoke({"query": query})
        
        # Format sources
        sources = self._format_sources(result.get("source_documents", []))
        
        return {
            "answer": result["result"] if isinstance(result, dict) else result,
            "source_documents": result.get("source_documents", []),
            "sources": sources,
            "query": query
        }
    
    def _format_sources(self, documents: List[Document]) -> List[Dict[str, str]]:
        """
        Format source documents for display.
        
        Args:
            documents: List of source documents.
            
        Returns:
            List of formatted source dictionaries.
        """
        sources = []
        seen = set()
        
        for doc in documents:
            metadata = doc.metadata
            
            if metadata.get("source_type") == "duckduckgo":
                source_key = metadata.get("url", "")
                source_type = "web"
            else:
                source_key = metadata.get("source_file", "")
                source_type = "local"
            
            # Avoid duplicates
            if source_key in seen:
                continue
            seen.add(source_key)
            
            source_info = {
                "type": source_type,
                "title": metadata.get("title", metadata.get("source_file", "Unknown")),
                "source": source_key if source_type == "web" else metadata.get("source_file", "Unknown"),
                "url": metadata.get("url", None)
            }
            
            # Add snippet from content
            content = doc.page_content
            if len(content) > 200:
                content = content[:200] + "..."
            source_info["snippet"] = content
            
            sources.append(source_info)
        
        return sources
    
    async def aask(self, query: str) -> Dict[str, Any]:
        """
        Async version of ask().
        
        Args:
            query: The question to answer.
            
        Returns:
            Dictionary containing answer and sources.
        """
        logger.info(f"Async processing query: {query}")
        
        result = await self.qa_chain.ainvoke({"query": query})
        
        sources = self._format_sources(result.get("source_documents", []))
        
        return {
            "answer": result["result"] if isinstance(result, dict) else result,
            "source_documents": result.get("source_documents", []),
            "sources": sources,
            "query": query
        }


def create_rag_chain(
    retriever: BaseRetriever,
    model_name: Optional[str] = None,
    temperature: float = 0,
    max_tokens: int = 1000,
    llm_provider: Optional[str] = None
) -> RAGChain:
    """
    Create a RAG chain.
    
    Args:
        retriever: Retriever to fetch relevant documents.
        model_name: Model to use (defaults to LLM_MODEL env var).
        temperature: Sampling temperature.
        max_tokens: Maximum tokens in response.
        llm_provider: LLM provider to use.
        
    Returns:
        RAGChain instance.
    """
    return RAGChain(
        retriever=retriever,
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        llm_provider=llm_provider
    )