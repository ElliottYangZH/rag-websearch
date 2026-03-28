"""Vector store implementation using FAISS with OpenAI embeddings."""

import os
import pickle
from pathlib import Path
from typing import List, Optional, Union

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever


class VectorStoreManager:
    """Manage FAISS vector store with OpenAI embeddings."""
    
    def __init__(
        self,
        embedding_model: str = "text-embedding-3-small",
        vectorstore_path: Optional[str] = None
    ):
        """
        Initialize the vector store manager.
        
        Args:
            embedding_model: OpenAI embedding model to use.
            vectorstore_path: Optional path to load/save vector store.
        """
        self.embedding_model = embedding_model
        self.vectorstore_path = vectorstore_path
        self.embeddings = OpenAIEmbeddings(model=embedding_model)
        self._vectorstore: Optional[FAISS] = None
    
    def create_vectorstore(
        self,
        documents: List[Document]
    ) -> FAISS:
        """
        Create a new FAISS vector store from documents.
        
        Args:
            documents: List of documents to embed and store.
            
        Returns:
            FAISS vector store instance.
        """
        self._vectorstore = FAISS.from_documents(
            documents=documents,
            embedding=self.embeddings
        )
        return self._vectorstore
    
    def load_vectorstore(self, path: str) -> FAISS:
        """
        Load an existing FAISS vector store from disk.
        
        Args:
            path: Path to the vector store directory.
            
        Returns:
            Loaded FAISS vector store.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Vector store not found at: {path}")
        
        self._vectorstore = FAISS.load_local(
            folder_path=path,
            embeddings=self.embeddings,
            allow_dangerous_deserialization=True
        )
        self.vectorstore_path = path
        return self._vectorstore
    
    def save_vectorstore(self, path: str) -> None:
        """
        Save the vector store to disk.
        
        Args:
            path: Path to save the vector store.
        """
        if self._vectorstore is None:
            raise ValueError("No vector store to save. Create or load one first.")
        
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._vectorstore.save_local(folder_path=path)
        self.vectorstore_path = path
    
    def as_retriever(
        self,
        search_type: str = "similarity",
        search_kwargs: Optional[dict] = None
    ) -> BaseRetriever:
        """
        Get a retriever interface for the vector store.
        
        Args:
            search_type: Type of search ('similarity', 'mmr', etc.)
            search_kwargs: Additional search parameters.
            
        Returns:
            Retriever instance.
        """
        if self._vectorstore is None:
            raise ValueError("No vector store available. Create or load one first.")
        
        if search_kwargs is None:
            search_kwargs = {"k": 4}
        
        return self._vectorstore.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs
        )
    
    @property
    def vectorstore(self) -> Optional[FAISS]:
        """Get the underlying vector store."""
        return self._vectorstore
    
    def similarity_search(
        self,
        query: str,
        k: int = 4
    ) -> List[Document]:
        """
        Perform similarity search.
        
        Args:
            query: Query string.
            k: Number of results to return.
            
        Returns:
            List of relevant documents.
        """
        if self._vectorstore is None:
            raise ValueError("No vector store available.")
        
        return self._vectorstore.similarity_search(query, k=k)
    
    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4
    ) -> List[tuple]:
        """
        Perform similarity search with relevance scores.
        
        Args:
            query: Query string.
            k: Number of results to return.
            
        Returns:
            List of (document, score) tuples.
        """
        if self._vectorstore is None:
            raise ValueError("No vector store available.")
        
        return self._vectorstore.similarity_search_with_score(query, k=k)
    
    def add_documents(self, documents: List[Document]) -> None:
        """
        Add documents to the vector store.
        
        Args:
            documents: Documents to add.
        """
        if self._vectorstore is None:
            raise ValueError("No vector store available. Create one first.")
        
        self._vectorstore.add_documents(documents)


def create_vectorstore(
    documents: List[Document],
    embedding_model: str = "text-embedding-3-small"
) -> FAISS:
    """
    Convenience function to create a vector store.
    
    Args:
        documents: Documents to embed.
        embedding_model: OpenAI embedding model.
        
    Returns:
        FAISS vector store.
    """
    manager = VectorStoreManager(embedding_model=embedding_model)
    return manager.create_vectorstore(documents)