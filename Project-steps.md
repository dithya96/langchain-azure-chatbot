Set Up Core Azure Resources 🏗️
Even with a code-centric LangChain approach, you'll need these foundational Azure services. You can create them via the Azure portal:
Azure OpenAI Service:
Deployments: Once created, go to Azure OpenAI Studio. You'll need to deploy:
A Chat Model: e.g., gpt-35-turbo-16k or gpt-4. Note your deployment name.
An Embedding Model: e.g., text-embedding-ada-002. Note your deployment name.
Credentials: Note down your Azure OpenAI endpoint, API key, and the deployment names.
Azure AI Search Service:


Azure Blob Storage Account (Optional but Recommended):
Creation: In the Azure portal, search for "Storage accounts" and create one. Create containers within it as needed (e.g., raw-documents, source-code, logs).
Credentials: Note your storage account connection string if you plan to use it with LangChain's Azure Blob Storage loaders.

Create a Project Directory:


mkdir langchain-azure-chatbot
cd langchain-azure-chatbot

 Set up a Python Virtual Environment:
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

Install Necessary Libraries:
pip install langchain langchain-openai langchain-community langchain-text-splitters \
            azure-ai-search-documents azure-identity python-dotenv streamlit \
            langchain_community


Create a .env File:
Store your Azure credentials securely in a .env file in your project root. 

AZURE_OPENAI_API_KEY="YOUR_AZURE_OPENAI_API_KEY"
AZURE_OPENAI_ENDPOINT="YOUR_AZURE_OPENAI_ENDPOINT"
AZURE_OPENAI_CHAT_DEPLOYMENT_NAME="YOUR_CHAT_MODEL_DEPLOYMENT_NAME"
AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME="YOUR_EMBEDDING_MODEL_DEPLOYMENT_NAME"
AZURE_OPENAI_API_VERSION="2024-02-01" # Or your desired API version

AZURE_AI_SEARCH_SERVICE_NAME="YOUR_AZURE_AI_SEARCH_SERVICE_NAME" # Just the name, not the full endpoint
AZURE_AI_SEARCH_ENDPOINT="YOUR_AZURE_AI_SEARCH_ENDPOINT" # Full endpoint e.g. https://your-search-service.search.windows.net
AZURE_AI_SEARCH_API_KEY="YOUR_AZURE_AI_SEARCH_ADMIN_KEY"
AZURE_AI_SEARCH_INDEX_NAME="your-langchain-index" # Choose an index name

