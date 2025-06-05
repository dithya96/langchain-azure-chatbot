import os
import gzip  # For handling .gz files
from dotenv import load_dotenv
from embed_and_store import process_and_store_chunks

from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader,
)
# For custom loader, we need BaseLoader
from langchain_community.document_loaders.base import BaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from langchain_core.documents import Document

# Load environment variables
load_dotenv()


# --- Custom Gzip Text Loader ---
class GzipTextLoader(BaseLoader):
    """Loads a .gz file as text."""

    def __init__(self, file_path: str, encoding: str = "utf-8", errors: str = "ignore"):
        """
        Initialize with file path.
        Args:
            file_path: Path to the .gz file.
            encoding: The encoding to use when decoding the file.
            errors: How to handle decoding errors ('strict', 'ignore', 'replace').
        """
        self.file_path = file_path
        self.encoding = encoding
        self.errors = errors

    def load(self) -> list[Document]:
        """Load from file path."""
        try:
            with gzip.open(self.file_path, "rt", encoding=self.encoding, errors=self.errors) as f:
                text = f.read()
        except Exception as e:
            raise RuntimeError(f"Error loading {self.file_path}: {e}") from e

        metadata = {"source": self.file_path}
        return [Document(page_content=text, metadata=metadata)]


# --- 1. Document Loading ---
def load_documents():
    documents = []
    source_code_path = "src"
    log_path = "logs"
    docs_path = "documentation"

    # --- Loading Source Code (Python and Java files) ---
    print(f"Loading source code from '{source_code_path}' directory...")
    if os.path.exists(source_code_path):
        code_loader = DirectoryLoader(
            f'./{source_code_path}/',
            glob="**/*[.py|.java]",
            loader_cls=TextLoader,
            show_progress=True,
            use_multithreading=True,
            silent_errors=True
        )
        try:
            loaded_code_files = code_loader.load()
            if loaded_code_files:
                documents.extend(loaded_code_files)
                print(f"Loaded {len(loaded_code_files)} source code files (.py, .java).")
            else:
                print(f"No .py or .java files found in '{source_code_path}'.")
        except Exception as e:
            print(f"Error loading source code from '{source_code_path}': {e}")
    else:
        print(f"'{source_code_path}' directory not found. Skipping source code loading.")

    # --- Loading plain .log files ---
    print(f"Loading plain .log files from '{log_path}' directory...")
    if os.path.exists(log_path):
        plain_log_loader = DirectoryLoader(
            f'./{log_path}/',
            glob="**/*.log",  # Only .log files
            loader_cls=TextLoader,
            show_progress=True,
            use_multithreading=True,
            silent_errors=True
        )
        try:
            loaded_plain_log_files = plain_log_loader.load()
            if loaded_plain_log_files:
                documents.extend(loaded_plain_log_files)
                print(f"Loaded {len(loaded_plain_log_files)} plain .log files.")
            else:
                print(f"No plain .log files found in '{log_path}'.")
        except Exception as e:
            print(f"Error loading .log files from '{log_path}': {e}")

        # --- Loading gzipped .gz log files ---
        print(f"Loading gzipped .gz log files from '{log_path}' directory...")
        # Assuming .gz files in the log directory are gzipped logs (e.g. *.log.gz or *.gz)
        gzipped_log_loader = DirectoryLoader(
            f'./{log_path}/',
            glob="**/*.gz",  # Handles .log.gz and other .gz files
            loader_cls=GzipTextLoader,  # Use our custom loader
            show_progress=True,
            use_multithreading=True,  # GzipTextLoader is simple, MT should be fine
            silent_errors=True
        )
        try:
            loaded_gzipped_log_files = gzipped_log_loader.load()
            if loaded_gzipped_log_files:
                documents.extend(loaded_gzipped_log_files)
                print(f"Loaded {len(loaded_gzipped_log_files)} gzipped .gz log files.")
            else:
                print(f"No gzipped .gz log files found in '{log_path}'.")
        except Exception as e:
            print(f"Error loading .gz files from '{log_path}': {e}")
    else:
        print(f"'{log_path}' directory not found. Skipping log file loading.")

    # --- Loading Markdown documentation files ---
    print(f"Loading Markdown files from '{docs_path}' directory...")
    if os.path.exists(docs_path):
        md_loader = DirectoryLoader(
            f'./{docs_path}/',
            glob="**/*.md",
            loader_cls=TextLoader,
            show_progress=True,
            use_multithreading=True,
            silent_errors=True
        )
        try:
            loaded_md_files = md_loader.load()
            if loaded_md_files:
                documents.extend(loaded_md_files)
                print(f"Loaded {len(loaded_md_files)} Markdown files.")
            else:
                print(f"No .md files found in '{docs_path}'.")
        except Exception as e:
            print(f"Error loading markdown files from '{docs_path}': {e}")
    else:
        print(f"'{docs_path}' directory not found. Skipping Markdown file loading.")

    print(f"Total documents loaded: {len(documents)}.")
    return documents


# --- 2. Document Splitting / Chunking & Metadata Enrichment ---
def split_documents(documents):
    if not documents:
        print("No documents to split.")
        return []

    print("Splitting documents and enriching metadata...")
    general_text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    python_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON, chunk_size=2000, chunk_overlap=200
    )
    java_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.JAVA, chunk_size=2000, chunk_overlap=200
    )
    markdown_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.MARKDOWN, chunk_size=1000, chunk_overlap=150
    )

    all_processed_chunks = []
    for doc in documents:
        original_metadata = doc.metadata.copy()
        file_path = original_metadata.get("source", "")
        file_name = os.path.basename(file_path) if file_path else "unknown_source"
        base_metadata_updates = {"file_name": file_name}
        chunks_from_doc = []

        if file_path.endswith(".py"):
            chunks_from_doc = python_splitter.split_documents([doc])
            base_metadata_updates.update({"file_type": "python_code", "language": "python"})
        elif file_path.endswith(".java"):
            chunks_from_doc = java_splitter.split_documents([doc])
            base_metadata_updates.update({"file_type": "java_code", "language": "java"})
        elif file_path.endswith(".md"):
            chunks_from_doc = markdown_splitter.split_documents([doc])
            base_metadata_updates.update({"file_type": "markdown_doc"})
        # Check for both .log and .gz (assuming .gz in log dir are logs)
        elif file_path.endswith((".log", ".gz")):  # MODIFIED HERE
            chunks_from_doc = general_text_splitter.split_documents([doc])
            base_metadata_updates.update({"file_type": "log"})
        else:
            chunks_from_doc = general_text_splitter.split_documents([doc])
            base_metadata_updates.update({"file_type": "other_text"})

        for chunk in chunks_from_doc:
            updated_metadata = original_metadata.copy()
            updated_metadata.update(base_metadata_updates)
            chunk.metadata = updated_metadata
        all_processed_chunks.extend(chunks_from_doc)

    print(f"Split into {len(all_processed_chunks)} chunks with enriched metadata.")