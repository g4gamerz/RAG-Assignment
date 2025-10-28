"""
Document loader for PDF, HTML, Markdown, and text files.
"""
import os
from typing import List, Dict
from pathlib import Path
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    TextLoader
)

class DocumentLoader:
    """Load documents from various file formats."""

    def __init__(self, supported_formats: List[str] = None):
        """
        Initialize the document loader.

        Args:
            supported_formats: List of supported file extensions (default: pdf, html, md, txt)
        """
        self.supported_formats = supported_formats or ['pdf', 'html', 'md', 'txt']

    def load_document(self, file_path: str) -> List[Document]:
        """
        Load a single document.

        Args:
            file_path: Path to the document file

        Returns:
            List of Document objects
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        extension = file_path.suffix.lower().lstrip('.')

        if extension not in self.supported_formats:
            raise ValueError(
                f"Unsupported file format: {extension}. "
                f"Supported formats: {', '.join(self.supported_formats)}"
            )

        if extension == 'pdf':
            loader = PyPDFLoader(str(file_path))
        elif extension == 'html':
            loader = UnstructuredHTMLLoader(str(file_path))
        elif extension in ['md', 'markdown']:
            loader = UnstructuredMarkdownLoader(str(file_path))
        elif extension == 'txt':
            loader = TextLoader(str(file_path), encoding='utf-8')
        else:
            raise ValueError(f"No loader available for: {extension}")

        documents = loader.load()

        for doc in documents:
            doc.metadata['source'] = str(file_path)
            doc.metadata['filename'] = file_path.name
            doc.metadata['extension'] = extension

        return documents

    def load_directory(self, directory_path: str) -> List[Document]:
        """
        Load all supported documents from a directory.

        Args:
            directory_path: Path to directory containing documents

        Returns:
            List of all loaded Document objects
        """
        directory = Path(directory_path)

        if not directory.exists() or not directory.is_dir():
            raise NotADirectoryError(f"Directory not found: {directory}")

        all_documents = []

        for file_path in directory.rglob('*'):
            if file_path.is_file():
                extension = file_path.suffix.lower().lstrip('.')
                if extension in self.supported_formats:
                    try:
                        documents = self.load_document(str(file_path))
                        all_documents.extend(documents)
                        print(f"[OK] Loaded: {file_path.name} ({len(documents)} pages/sections)")
                    except Exception as e:
                        print(f"[X] Error loading {file_path.name}: {str(e)}")

        print(f"\nTotal documents loaded: {len(all_documents)}")
        return all_documents

    def get_document_stats(self, documents: List[Document]) -> Dict:
        """
        Get statistics about loaded documents.

        Args:
            documents: List of Document objects

        Returns:
            Dictionary with document statistics
        """
        if not documents:
            return {"total": 0}

        total_chars = sum(len(doc.page_content) for doc in documents)
        sources = set(doc.metadata.get('source', 'unknown') for doc in documents)
        extensions = set(doc.metadata.get('extension', 'unknown') for doc in documents)

        return {
            "total_documents": len(documents),
            "total_characters": total_chars,
            "avg_chars_per_doc": total_chars // len(documents),
            "unique_sources": len(sources),
            "file_types": list(extensions)
        }

if __name__ == "__main__":
    loader = DocumentLoader()

    try:
        docs = loader.load_directory("../data")
        stats = loader.get_document_stats(docs)
        print("\nDocument Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
    except Exception as e:
        print(f"Error: {e}")
        print("\nPlace your documents in the 'data/' directory to get started.")
