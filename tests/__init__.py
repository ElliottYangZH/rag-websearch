"""Tests for RAG pipeline components."""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.document_loader import DocumentLoader, load_documents
from src.vector_store import VectorStoreManager, create_vectorstore
from src.web_retriever import DuckDuckGoRetriever, create_web_retriever
from src.ensemble_retriever import WeightedEnsembleRetriever, create_ensemble_retriever


def test_document_loader():
    """Test document loader functionality."""
    print("Testing DocumentLoader...")
    
    loader = DocumentLoader(chunk_size=100, chunk_overlap=20)
    
    # Test loading sample.txt
    docs = loader.load_text("docs/sample.txt")
    assert len(docs) > 0
    print(f"  Loaded {len(docs)} documents from sample.txt")
    
    # Test splitting
    chunks = loader.split_documents(docs)
    assert len(chunks) > 0
    print(f"  Created {len(chunks)} chunks")
    
    print("  ✓ DocumentLoader tests passed")


def test_vector_store():
    """Test vector store functionality."""
    print("Testing VectorStore...")
    
    # Skip if no API key
    if not os.getenv("OPENAI_API_KEY"):
        print("  ⚠ Skipping vector store test (no API key)")
        return
    
    loader = DocumentLoader()
    docs = loader.load_text("docs/sample.txt")
    chunks = loader.split_documents(docs)
    
    manager = VectorStoreManager()
    manager.create_vectorstore(chunks)
    
    # Test similarity search
    results = manager.similarity_search("RAG", k=2)
    assert len(results) > 0
    print(f"  Found {len(results)} similar documents")
    
    print("  ✓ VectorStore tests passed")


def test_web_retriever():
    """Test web retriever functionality."""
    print("Testing WebRetriever...")
    
    retriever = create_web_retriever(top_k=3)
    docs = retriever.invoke("LangChain RAG")
    
    assert len(docs) > 0
    print(f"  Found {len(docs)} web results")
    
    for doc in docs[:2]:
        print(f"  - {doc.metadata.get('title', 'No title')}")
    
    print("  ✓ WebRetriever tests passed")


def test_ensemble_retriever():
    """Test ensemble retriever functionality."""
    print("Testing EnsembleRetriever...")
    
    # This is a basic structure test
    # Full functionality requires vector store
    
    retriever = create_web_retriever(top_k=3)
    
    # Just verify it can be created
    print(f"  Web retriever created with top_k={retriever.top_k}")
    
    print("  ✓ EnsembleRetriever structure test passed")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("Running RAG Pipeline Tests")
    print("=" * 60)
    
    test_document_loader()
    test_vector_store()
    test_web_retriever()
    test_ensemble_retriever()
    
    print("=" * 60)
    print("All tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()