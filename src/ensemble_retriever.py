"""Ensemble retriever combining local and web retrievers."""

from typing import List, Optional, Tuple
import logging

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

logger = logging.getLogger(__name__)


class WeightedEnsembleRetriever(BaseRetriever):
    """
    An ensemble retriever that combines local and web retrievers.
    
    This retriever uses a weighted combination approach where
    results from multiple retrievers are merged and re-ranked.
    """
    
    def __init__(
        self,
        local_retriever: BaseRetriever,
        web_retriever: BaseRetriever,
        local_weight: float = 0.7,
        web_weight: float = 0.3,
        k: int = 4
    ):
        """
        Initialize the ensemble retriever.
        
        Args:
            local_retriever: Retriever for local documents.
            web_retriever: Retriever for web search results.
            local_weight: Weight for local results (0-1).
            web_weight: Weight for web results (0-1).
            k: Number of results to return.
        """
        super().__init__()
        
        if not 0 <= local_weight <= 1 or not 0 <= web_weight <= 1:
            raise ValueError("Weights must be between 0 and 1")
        
        if abs(local_weight + web_weight - 1.0) > 1e-6:
            raise ValueError("Weights must sum to 1")
        
        self.local_retriever = local_retriever
        self.web_retriever = web_retriever
        self.local_weight = local_weight
        self.web_weight = web_weight
        self.k = k
        
        # Use LangChain's EnsembleRetriever under the hood
        self._ensemble = LangChainEnsembleRetriever(
            retrievers=[local_retriever, web_retriever],
            weights=[local_weight, web_weight]
        )
    
    def _get_relevant_documents(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForRetrieverRun] = None
    ) -> List[Document]:
        """
        Get relevant documents using the ensemble approach.
        
        Args:
            query: Query string.
            run_manager: Callback manager.
            
        Returns:
            Merged and re-ranked documents.
        """
        logger.info(
            f"Ensemble retrieval for query: {query} "
            f"(local_weight={self.local_weight}, web_weight={self.web_weight})"
        )
        
        # Get results from both retrievers
        local_docs = self.local_retriever.invoke(query)
        web_docs = self.web_retriever.invoke(query)
        
        logger.info(
            f"Retrieved {len(local_docs)} local docs, {len(web_docs)} web docs"
        )
        
        # Merge and re-rank using Reciprocal Rank Fusion
        merged = self._merge_and_rerank(local_docs, web_docs)
        
        return merged[:self.k]
    
    def _merge_and_rerank(
        self,
        local_docs: List[Document],
        web_docs: List[Document]
    ) -> List[Document]:
        """
        Merge documents using Reciprocal Rank Fusion.
        
        Args:
            local_docs: Documents from local retriever.
            web_docs: Documents from web retriever.
            
        Returns:
            Merged and re-ranked documents.
        """
        doc_scores = {}
        
        # Score local documents
        for rank, doc in enumerate(local_docs):
            doc_key = self._get_doc_key(doc)
            score = self.local_weight * (1 / (rank + 1))
            doc_scores[doc_key] = doc_scores.get(doc_key, 0) + score
        
        # Score web documents
        for rank, doc in enumerate(web_docs):
            doc_key = self._get_doc_key(doc)
            score = self.web_weight * (1 / (rank + 1))
            doc_scores[doc_key] = doc_scores.get(doc_key, 0) + score
        
        # Sort by score and return documents
        sorted_keys = sorted(doc_scores.keys(), key=lambda k: doc_scores[k], reverse=True)
        
        # Create a mapping from key to document
        doc_map = {}
        for doc in local_docs:
            doc_map[self._get_doc_key(doc)] = doc
        for doc in web_docs:
            doc_map[self._get_doc_key(doc)] = doc
        
        result = [doc_map[key] for key in sorted_keys if key in doc_map]
        
        return result
    
    def _get_doc_key(self, doc: Document) -> str:
        """Generate a unique key for a document."""
        # Use URL for web docs, source_file for local docs
        if doc.metadata.get("source_type") == "duckduckgo":
            return f"web:{doc.metadata.get('url', id(doc))}"
        else:
            return f"local:{doc.metadata.get('source_file', id(doc))}:{doc.page_content[:100]}"
    
    async def _aget_relevant_documents(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForRetrieverRun] = None
    ) -> List[Document]:
        """Async version of _get_relevant_documents."""
        return self._get_relevant_documents(query, run_manager)


def create_ensemble_retriever(
    local_retriever: BaseRetriever,
    web_retriever: BaseRetriever,
    local_weight: float = 0.7,
    web_weight: float = 0.3,
    k: int = 4
) -> WeightedEnsembleRetriever:
    """
    Create an ensemble retriever combining local and web sources.
    
    Args:
        local_retriever: Retriever for local documents.
        web_retriever: Retriever for web search.
        local_weight: Weight for local results (0-1).
        web_weight: Weight for web results (0-1).
        k: Number of results to return.
        
    Returns:
        WeightedEnsembleRetriever instance.
    """
    return WeightedEnsembleRetriever(
        local_retriever=local_retriever,
        web_retriever=web_retriever,
        local_weight=local_weight,
        web_weight=web_weight,
        k=k
    )