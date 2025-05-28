import os
from dotenv import load_dotenv
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores.azuresearch import AzureSearch


def process_and_store_chunks(document_chunks):
    """
    Initializes the Azure OpenAI embedding model and Azure AI Search vector store,
    then embeds the provided document chunks and stores them.

    Args:
        document_chunks (list): A list of LangChain Document objects (chunks)
                                 produced by the splitting process.
    """
    if not document_chunks:
        print("No document chunks provided to embed and store. Exiting.")
        return

    load_dotenv() # Load environment variables from .env file

    # --- 1. Initialize Azure OpenAI Embeddings Model ---
    print("Initializing Azure OpenAI Embeddings model...")
    try:
        azure_openai_api_key = os.getenv("AZURE_OPENAI_API_KEY")
        azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        azure_openai_embedding_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME")
        azure_openai_api_version = os.getenv("AZURE_OPENAI_API_VERSION")

        if not all([azure_openai_api_key, azure_openai_endpoint, azure_openai_embedding_deployment, azure_openai_api_version]):
            print("Error: Azure OpenAI environment variables are not fully set in .env file.")
            print("Please ensure AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME, and AZURE_OPENAI_API_VERSION are defined.")
            return

        embeddings = AzureOpenAIEmbeddings(
            azure_deployment=azure_openai_embedding_deployment,
            azure_endpoint=azure_openai_endpoint,
            api_key=azure_openai_api_key, # For production, consider DefaultAzureCredential
            api_version=azure_openai_api_version,
            chunk_size=16 # Max batch size for Azure OpenAI's text-embedding-ada-002 is 16. Adjust if using other models.
        )
        print("Azure OpenAI Embeddings model initialized successfully.")
    except Exception as e:
        print(f"Error initializing AzureOpenAIEmbeddings: {e}")
        return

    # --- 2. Initialize Azure AI Search as Vector Store ---
    print("Initializing Azure AI Search vector store...")
    azure_ai_search_endpoint = os.getenv("AZURE_AI_SEARCH_ENDPOINT")
    azure_ai_search_api_key = os.getenv("AZURE_AI_SEARCH_API_KEY") # For production, consider DefaultAzureCredential
    azure_ai_search_index_name = os.getenv("AZURE_AI_SEARCH_INDEX_NAME")

    if not all([azure_ai_search_endpoint, azure_ai_search_api_key, azure_ai_search_index_name]):
        print("Error: Azure AI Search environment variables are not fully set in .env file.")
        print("Please ensure AZURE_AI_SEARCH_ENDPOINT, AZURE_AI_SEARCH_API_KEY, and AZURE_AI_SEARCH_INDEX_NAME are defined.")
        return

    try:

        vector_store = AzureSearch(
            azure_search_endpoint=azure_ai_search_endpoint,
            azure_search_key=azure_ai_search_api_key,
            index_name=azure_ai_search_index_name,
            embedding_function=embeddings.embed_query
        )
        print(f"Azure AI Search vector store initialized successfully for index: {azure_ai_search_index_name}")
    except Exception as e:
        print(f"Error initializing AzureSearch vector store: {e}")
        return

    # --- 3. Add Documents to Vector Store ---
    # This will create the index if it doesn't exist, based on LangChain's default schema
    # for AzureSearch, and embed the documents using the model via `embedding_function`.
    print(f"Adding {len(document_chunks)} chunks to Azure AI Search index: {azure_ai_search_index_name}...")
    try:
        vector_store.add_documents(documents=document_chunks)
        print(f"Successfully added {len(document_chunks)} chunks to the vector store.")
    except Exception as e:
        print(f"Error adding documents to AzureSearch: {e}")




