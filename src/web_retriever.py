"""Web search retriever using DuckDuckGo."""

from typing import Any, List, Optional
import logging

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import Field
from ddgs import DDGS

logger = logging.getLogger(__name__)


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
    
    def model_post_init(self, __context: Any) -> None:
        """Initialize after Pydantic validation."""
        logger.info(f"DuckDuckGoRetriever init: top_k={self.top_k}, max_snippet_length={self.max_snippet_length}, region={self.region}, safesearch={self.safesearch}")
    
    def _search_duckduckgo(self, query: str) -> List[dict]:
        """
        Perform a DuckDuckGo search.
        
        Args:
            query: Search query.
            
        Returns:
            List of search results.
        """
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
        """
        Convert a search result to a Document.
        
        Args:
            result: Search result dictionary.
            
        Returns:
            LangChain Document.
        """
        # Combine title, description, and body for content
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
        """
        Get relevant documents for a query.
        
        Args:
            query: Search query.
            run_manager: Callback manager for retriever events.
            
        Returns:
            List of relevant documents.
        """
        logger.info(f"Searching DuckDuckGo for: {query}")
        
        results = self._search_duckduckgo(query)
        
        documents = []
        for result in results:
            doc = self._result_to_document(result)
            documents.append(doc)
        
        logger.info(f"Found {len(documents)} web results")
        
        return documents
    
    async def _aget_relevant_documents(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForRetrieverRun] = None
    ) -> List[Document]:
        """
        Async version of _get_relevant_documents.
        
        Args:
            query: Search query.
            run_manager: Callback manager for retriever events.
            
        Returns:
            List of relevant documents.
        """
        # For DuckDuckGo, we use sync search in async context
        return self._get_relevant_documents(query, run_manager)


def create_web_retriever(
    top_k: int = 5,
    max_snippet_length: int = 500
) -> DuckDuckGoRetriever:
    """
    Create a DuckDuckGo web retriever.
    
    Args:
        top_k: Number of results to retrieve.
        max_snippet_length: Maximum snippet length.
        
    Returns:
        DuckDuckGoRetriever instance.
    """
    return DuckDuckGoRetriever(
        top_k=top_k,
        max_snippet_length=max_snippet_length
    )