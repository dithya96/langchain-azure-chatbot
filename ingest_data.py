import os
from dotenv import load_dotenv
from embed_and_store import process_and_store_chunks

from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader, # TextLoader can be used for Markdown files as well
     CSVLoader,
     SharePointLoader,
     AzureBlobStorageContainerLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

# Load environment variables
load_dotenv()

# --- 1. Document Loading ---
def load_documents():
    documents = []
    source_code_path = "src"
    log_path = "logs"
    docs_path = "documentation"

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
        loaded_code_files = code_loader.load()
        if loaded_code_files:
            documents.extend(loaded_code_files)
            print(f"Loaded {len(loaded_code_files)} source code files (.py, .java).")
        else:
            print(f"No .py or .java files found in '{source_code_path}'.")
    else:
        print(f"'{source_code_path}' directory not found. Skipping source code loading.")

    # --- Loading log files ---
    print(f"Loading log files from '{log_path}' directory...")
    if os.path.exists(log_path):
        log_loader = DirectoryLoader(
            f'./{log_path}/',
            glob="**/*.log",
            loader_cls=TextLoader,
            show_progress=True,
            use_multithreading=True,
            silent_errors=True
        )
        loaded_log_files = log_loader.load()
        if loaded_log_files:
            documents.extend(loaded_log_files)
            print(f"Loaded {len(loaded_log_files)} log files.")
        else:
            print(f"No .log files found in '{log_path}'.")
    else:
        print(f"'{log_path}' directory not found. Skipping log file loading.")

    # --- Loading Markdown documentation files ---
    print(f"Loading Markdown files from '{docs_path}' directory...")
    if os.path.exists(docs_path):
        md_loader = DirectoryLoader(
            f'./{docs_path}/',
            glob="**/*.md",
            loader_cls=TextLoader, # TextLoader works well for .md files
            show_progress=True,
            use_multithreading=True,
            silent_errors=True
        )
        loaded_md_files = md_loader.load()
        if loaded_md_files:
            documents.extend(loaded_md_files)
            print(f"Loaded {len(loaded_md_files)} Markdown files.")
        else:
            print(f"No .md files found in '{docs_path}'.")
    else:
        print(f"'{docs_path}' directory not found. Skipping Markdown file loading.")

    # --- (Your Azure Blob Storage and SharePoint loading logic can remain here if needed) ---

    print(f"Total documents loaded: {len(documents)}.")
    return documents

# Loading from Azure Blob Storage (if you've uploaded files there)
    # AZURE_BLOB_CONNECTION_STRING = os.getenv("AZURE_BLOB_CONNECTION_STRING")
    # AZURE_BLOB_CONTAINER_NAME = "your-blob-container-for-docs" # Replace with your container
    # if AZURE_BLOB_CONNECTION_STRING and AZURE_BLOB_CONTAINER_NAME:
    #     print(f"Loading documents from Azure Blob Storage container: {AZURE_BLOB_CONTAINER_NAME}...")
    #     blob_loader = AzureBlobStorageContainerLoader(
    #         conn_str=AZURE_BLOB_CONNECTION_STRING,
    #         container=AZURE_BLOB_CONTAINER_NAME
    #     )
    #     documents.extend(blob_loader.load())
    # else:
    #     print("Azure Blob Storage environment variables not set. Skipping Blob loading.")


# --- 2. Document Splitting / Chunking ---
def split_documents(documents):
    if not documents:
        print("No documents to split.")
        return []

    print("Splitting documents...")
    general_text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)

    # Code splitter for Python
    python_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON, chunk_size=2000, chunk_overlap=200
    )

    # Code splitter for Java
    java_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.JAVA, chunk_size=2000, chunk_overlap=200
    )

    markdown_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.MARKDOWN,
        chunk_size=1000, # Adjust as needed for your Markdown content
        chunk_overlap=150
    )

    split_docs = []
    for doc in documents:
        file_path = doc.metadata.get("source", "") # Get the source file path
        if file_path.endswith(".py"):
            print(f"Splitting Python file: {file_path}")
            split_docs.extend(python_splitter.split_documents([doc]))
        elif file_path.endswith(".java"):
            print(f"Splitting Java file: {file_path}")
            split_docs.extend(java_splitter.split_documents([doc]))
        elif file_path.endswith(".md"):
            print(f"Splitting Markdown file: {file_path}")
            split_docs.extend(markdown_splitter.split_documents([doc]))
        elif file_path.endswith(".log"):
            print(f"Splitting log file: {file_path}")
            split_docs.extend(general_text_splitter.split_documents([doc])) # Or a custom log splitter
        else:
            print(f"Splitting with general text splitter: {file_path}")
            split_docs.extend(general_text_splitter.split_documents([doc]))

    print(f"Split into {len(split_docs)} chunks.")
    return split_docs

if __name__ == '__main__':
    src_dir = "src"
    logs_dir = "logs"
    docs_dir = "documentation" # New directory for Markdown docs

    if not os.path.exists(src_dir): os.makedirs(src_dir)
    if not os.path.exists(logs_dir): os.makedirs(logs_dir)
    if not os.path.exists(docs_dir): os.makedirs(docs_dir) # Create docs directory

    # Dummy Python file
    if not os.path.exists(os.path.join(src_dir, "example.py")):
        with open(os.path.join(src_dir, "example.py"), "w") as f:
            f.write("def hello_python():\n    print('Hello, Python world!')\n# This is a sample Python file.")

    # Dummy Log file
    if not os.path.exists(os.path.join(logs_dir, "application.log")):
        with open(os.path.join(logs_dir, "application.log"), "w") as f:
            f.write("2025-06-03 10:05:15.120 INFO --- [main] MyApp - Application started successfully.\n")

    # Dummy Markdown file
    if not os.path.exists(os.path.join(docs_dir, "api_guide.md")):
        with open(os.path.join(docs_dir, "api_guide.md"), "w") as f:
            f.write("# API Guide\n\nThis document describes the API endpoints.\n\n## Endpoint 1\n- Method: GET\n- Path: /data\n\n## Endpoint 2\n- Method: POST\n- Path: /submit")

    loaded_docs = load_documents()
    if loaded_docs:
        chunked_documents = split_documents(loaded_docs)
        if chunked_documents:
            print("Data ingestion and splitting complete. Now embedding and storing...")
            process_and_store_chunks(chunked_documents)
            print("All processing, embedding, and storing complete.")
        else:
            print("No chunks were generated after splitting. Nothing to store.")
    else:
        print("No documents were loaded. Nothing to process or store.")