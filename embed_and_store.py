import os
import concurrent.futures # Import for ThreadPoolExecutor
from dotenv import load_dotenv
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores.azuresearch import AzureSearch
# Assuming Document objects are used, ensure this import if needed elsewhere
# from langchain_core.documents import Document


load_dotenv()

def process_and_store_chunks(document_chunks):
    """
    Initializes the Azure OpenAI embedding model and Azure AI Search vector store,
    then embeds the provided document chunks and stores them, using concurrency.

    Args:
        document_chunks (list): A list of LangChain Document objects (chunks)
                                 produced by the splitting process.
    """
    if not document_chunks:
        print("No document chunks provided to embed and store. Exiting.")
        return

    # --- 1. Initialize Azure OpenAI Embeddings Model ---
    print("Initializing Azure OpenAI Embeddings model...")
    try:
        azure_openai_api_key = os.getenv("AZURE_OPENAI_API_KEY")
        azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        azure_openai_embedding_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME")
        azure_openai_api_version = os.getenv("AZURE_OPENAI_API_VERSION")

        if not all([azure_openai_api_key, azure_openai_endpoint, azure_openai_embedding_deployment, azure_openai_api_version]):
            print("Error: Azure OpenAI environment variables are not fully set in .env file.")
            # ... (rest of your error message)
            return

        # This embeddings object will be shared across threads.
        # AzureOpenAIEmbeddings is designed to be usable in this way.
        embeddings = AzureOpenAIEmbeddings(
            azure_deployment=azure_openai_embedding_deployment,
            azure_endpoint=azure_openai_endpoint,
            api_key=azure_openai_api_key,
            api_version=azure_openai_api_version,
            chunk_size=16 # Max batch size for Azure OpenAI's text-embedding-ada-002 for optimal API call.
        )
        print("Azure OpenAI Embeddings model initialized successfully.")
    except Exception as e:
        print(f"Error initializing AzureOpenAIEmbeddings: {e}")
        return

    # --- 2. Initialize Azure AI Search as Vector Store ---
    # We initialize one vector_store instance. The add_documents method will be called
    # by multiple threads, but each call is a self-contained operation.
    # The SearchClient used by AzureSearch is generally thread-safe.
    print("Initializing Azure AI Search vector store...")
    azure_ai_search_endpoint = os.getenv("AZURE_AI_SEARCH_ENDPOINT")
    azure_ai_search_api_key = os.getenv("AZURE_AI_SEARCH_API_KEY")
    azure_ai_search_index_name = os.getenv("AZURE_AI_SEARCH_INDEX_NAME")

    if not all([azure_ai_search_endpoint, azure_ai_search_api_key, azure_ai_search_index_name]):
        print("Error: Azure AI Search environment variables are not fully set in .env file.")
        # ... (rest of your error message)
        return

    try:
        # This vector_store object configuration will be used by multiple threads.
        vector_store = AzureSearch(
            azure_search_endpoint=azure_ai_search_endpoint,
            azure_search_key=azure_ai_search_api_key,
            index_name=azure_ai_search_index_name,
            embedding_function=embeddings.embed_query # LangChain will use embed_documents for lists
        )
        print(f"Azure AI Search vector store initialized successfully for index: {azure_ai_search_index_name}")
    except Exception as e:
        print(f"Error initializing AzureSearch vector store: {e}")
        return

    # --- 3. Add Documents to Vector Store using Concurrency ---
    print(f"Preparing to add {len(document_chunks)} chunks to Azure AI Search index: {azure_ai_search_index_name}...")

    # Define meta-batch size for parallel processing by threads.
    # Each thread will call vector_store.add_documents() with one meta_batch.
    # LangChain's AzureSearch will then internally batch uploads to Azure (default 1000 docs).
    # LangChain's AzureOpenAIEmbeddings will internally batch calls to OpenAI (chunk_size=16 docs).
    META_BATCH_SIZE = 2000  # Number of chunks per parallel task. Tune this based on performance.
                           # Too small, and thread overhead dominates. Too large, and less parallelism.
    MAX_WORKERS = 4        # Number of concurrent threads. Tune based on client CPU, network,
                           # and observed throttling from Azure services. Start small (e.g., 2-5).

    meta_batches = [
        document_chunks[i:i + META_BATCH_SIZE]
        for i in range(0, len(document_chunks), META_BATCH_SIZE)
    ]

    print(f"Split {len(document_chunks)} chunks into {len(meta_batches)} meta-batches of up to {META_BATCH_SIZE} chunks each for concurrent processing.")

    successful_chunks_count = 0

    # Function to be executed by each thread
    def process_single_meta_batch(docs_batch):
        try:
            # The vector_store object is from the outer scope.
            # Its add_documents method will handle embedding and uploading this batch.
            vector_store.add_documents(documents=docs_batch)
            print(f"Successfully processed a meta-batch of {len(docs_batch)} chunks.")
            return len(docs_batch)
        except Exception as ex:
            print(f"Error processing a meta-batch of {len(docs_batch)} chunks: {ex}")
            # Depending on requirements, you might want to collect failed docs_batch
            # or specific error details for later retry.
            return 0 # Indicate 0 chunks successfully processed for this failed batch

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all meta-batches to the executor
        future_to_batch_size = {
            executor.submit(process_single_meta_batch, meta_batch): len(meta_batch)
            for meta_batch in meta_batches
        }

        for future in concurrent.futures.as_completed(future_to_batch_size):
            try:
                num_successfully_processed = future.result()
                successful_chunks_count += num_successfully_processed
                if num_successfully_processed < future_to_batch_size[future]:
                    print(f"A meta-batch had partial success or failure. Successfully processed {num_successfully_processed} of {future_to_batch_size[future]} chunks.")
                else:
                     print(f"A meta-batch of {num_successfully_processed} chunks completed.")
            except Exception as exc: # Catch exceptions from the task itself if not caught inside
                print(f"A meta-batch generated an exception: {exc}")
            print(f"Total chunks successfully processed so far: {successful_chunks_count}")

    print(f"All meta-batches processed. Total successfully added chunks: {successful_chunks_count} out of {len(document_chunks)}.")

