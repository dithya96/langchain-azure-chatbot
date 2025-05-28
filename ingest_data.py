import os
from dotenv import load_dotenv
from embed_and_store import process_and_store_chunks # Assuming embed_and_store.py is in the same directory


from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader,
    CSVLoader,
    SharePointLoader, # Requires careful setup for auth
    AzureBlobStorageContainerLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

# Load environment variables
load_dotenv()

# --- 1. Document Loading ---
def load_documents():
    documents = []

    # Example: Loading source code (Python files from a 'src' directory)
    # Ensure you have a directory named 'src' with some .py files for this to work
    print("Loading source code...")
    if os.path.exists("src"):
        code_loader = DirectoryLoader('./src/', glob="**/*.py", loader_cls=TextLoader, show_progress=True, use_multithreading=True)
        documents.extend(code_loader.load())
    else:
        print("'src' directory not found. Skipping source code loading.")

    # Example: Loading log files (from a 'logs' directory)
    # Ensure you have a directory named 'logs' with some .log files
    print("Loading log files...")
    if os.path.exists("logs"):
        log_loader = DirectoryLoader('./logs/', glob="**/*.log", loader_cls=TextLoader, show_progress=True, use_multithreading=True)
        documents.extend(log_loader.load())
    else:
        print("'logs' directory not found. Skipping log file loading.")


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


    print(f"Loaded {len(documents)} documents.")
    return documents

# --- 2. Document Splitting / Chunking ---
def split_documents(documents):
    print("Splitting documents...")
    # General text splitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)

    # Code splitter (example for Python)
    code_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON, chunk_size=2000, chunk_overlap=200
    )

    split_docs = []
    for doc in documents:
        # Simple heuristic: if 'source' in metadata likely refers to a file path
        if doc.metadata.get("source") and ".py" in doc.metadata["source"]:
            split_docs.extend(code_splitter.split_documents([doc]))
        else:
            split_docs.extend(text_splitter.split_documents([doc]))

    print(f"Split into {len(split_docs)} chunks.")
    return split_docs

if __name__ == '__main__':
    # Create dummy files for testing if they don't exist
    if not os.path.exists("src"): os.makedirs("src")
    if not os.path.exists("logs"): os.makedirs("logs")
    if not os.path.exists("src/example.py"):
        with open("src/example.py", "w") as f:
            f.write("def hello():\n    print('Hello, world!')\n# This is a sample Python file.")
    if not os.path.exists("logs/app.log"):
        with open("logs/app.log", "w") as f:
            f.write("INFO: Application started.\nERROR: An error occurred.\nINFO: Process completed.")

    loaded_docs = load_documents()
    if loaded_docs:
        chunked_documents = split_documents(loaded_docs)
        if chunked_documents:
            print("Data ingestion and splitting complete. Now embedding and storing...")
            process_and_store_chunks(chunked_documents) # <-- CALL THE NEW FUNCTION HERE
            print("All processing, embedding, and storing complete.")
        else:
            print("No chunks were generated after splitting. Nothing to store.")
    else:
        print("No documents were loaded. Nothing to process or store.")