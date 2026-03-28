"""Document loader for various file types (PDF, Markdown, TXT)."""

import os
from typing import List, Optional, Union
from pathlib import Path

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
    DirectoryLoader,
)
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class DocumentLoader:
    """Load documents from various file formats and split into chunks."""
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        encoding: str = "utf-8"
    ):
        """
        Initialize the document loader.
        
        Args:
            chunk_size: Maximum size of each text chunk.
            chunk_overlap: Number of overlapping characters between chunks.
            encoding: Text encoding for plain text files.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.encoding = encoding
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
    
    def load_pdf(self, file_path: str) -> List[Document]:
        """
        Load a PDF file and return its contents as documents.
        
        Args:
            file_path: Path to the PDF file.
            
        Returns:
            List of Document objects, one per page.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF file not found: {file_path}")
        
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        
        # Add metadata about source
        for doc in documents:
            doc.metadata["source_type"] = "pdf"
            doc.metadata["source_file"] = os.path.basename(file_path)
        
        return documents
    
    def load_markdown(self, file_path: str) -> List[Document]:
        """
        Load a Markdown file and return its contents as documents.
        
        Args:
            file_path: Path to the Markdown file.
            
        Returns:
            List of Document objects.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Markdown file not found: {file_path}")
        
        loader = UnstructuredMarkdownLoader(file_path)
        documents = loader.load()
        
        # Add metadata about source
        for doc in documents:
            doc.metadata["source_type"] = "markdown"
            doc.metadata["source_file"] = os.path.basename(file_path)
        
        return documents
    
    def load_text(self, file_path: str) -> List[Document]:
        """
        Load a plain text file and return its contents as documents.
        
        Args:
            file_path: Path to the text file.
            
        Returns:
            List of Document objects.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Text file not found: {file_path}")
        
        loader = TextLoader(file_path, encoding=self.encoding)
        documents = loader.load()
        
        # Add metadata about source
        for doc in documents:
            doc.metadata["source_type"] = "text"
            doc.metadata["source_file"] = os.path.basename(file_path)
        
        return documents
    
    def load_file(self, file_path: str) -> List[Document]:
        """
        Load a file based on its extension.
        
        Args:
            file_path: Path to the file.
            
        Returns:
            List of Document objects.
            
        Raises:
            ValueError: If the file type is not supported.
        """
        path = Path(file_path)
        extension = path.suffix.lower()
        
        if extension == ".pdf":
            return self.load_pdf(file_path)
        elif extension in [".md", ".markdown"]:
            return self.load_markdown(file_path)
        elif extension == ".txt":
            return self.load_text(file_path)
        else:
            raise ValueError(f"Unsupported file type: {extension}")
    
    def load_directory(
        self,
        directory_path: str,
        recursive: bool = True,
        glob_pattern: str = "**/*"
    ) -> List[Document]:
        """
        Load all supported files from a directory.
        
        Args:
            directory_path: Path to the directory.
            recursive: Whether to search recursively.
            glob_pattern: Glob pattern for matching files.
            
        Returns:
            List of all loaded documents.
        """
        if not os.path.exists(directory_path):
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        
        # Count files by type
        supported_types = [".pdf", ".md", ".markdown", ".txt"]
        
        if recursive:
            pattern = f"{glob_pattern}"
        else:
            pattern = "*"
        
        all_documents = []
        
        for root, _, files in os.walk(directory_path):
            for filename in files:
                filepath = os.path.join(root, filename)
                ext = Path(filename).suffix.lower()
                
                if ext in supported_types:
                    try:
                        docs = self.load_file(filepath)
                        all_documents.extend(docs)
                    except Exception as e:
                        print(f"Warning: Failed to load {filepath}: {e}")
            
            if not recursive:
                break
        
        return all_documents
    
    def split_documents(
        self,
        documents: List[Document]
    ) -> List[Document]:
        """
        Split documents into smaller chunks.
        
        Args:
            documents: List of documents to split.
            
        Returns:
            List of chunked documents.
        """
        return self.text_splitter.split_documents(documents)
    
    def load_and_split(
        self,
        source: str,
        is_directory: bool = False
    ) -> List[Document]:
        """
        Load documents from a file or directory and split into chunks.
        
        Args:
            source: Path to file or directory.
            is_directory: Whether source is a directory.
            
        Returns:
            List of chunked documents.
        """
        if is_directory:
            documents = self.load_directory(source)
        else:
            documents = self.load_file(source)
        
        return self.split_documents(documents)


def load_documents(
    source: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    is_directory: bool = False
) -> List[Document]:
    """
    Convenience function to load and split documents.
    
    Args:
        source: Path to file or directory.
        chunk_size: Maximum size of each text chunk.
        chunk_overlap: Number of overlapping characters between chunks.
        is_directory: Whether source is a directory.
        
    Returns:
        List of chunked documents.
    """
    loader = DocumentLoader(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return loader.load_and_split(source, is_directory=is_directory)