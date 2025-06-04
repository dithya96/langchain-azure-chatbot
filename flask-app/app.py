import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.vectorstores.azuresearch import AzureSearch
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from openai import RateLimitError # Import specific error for handling
import logging

# --- Initialize Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Load Environment Variables ---
load_dotenv()

# --- Global Variable to Hold the Cached RAG Chain ---
# This will be initialized once when the app starts
rag_chain_instance_global = None

# --- Get variables for Embeddings Resource ---
AZURE_OPENAI_ENDPOINT_EMBEDDINGS = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY_EMBEDDINGS = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME")
AZURE_OPENAI_API_VERSION_EMBEDDINGS = os.getenv("AZURE_OPENAI_EMBEDDING_API_VERSION")

# --- Get variables for Chat Model Resource ---
AZURE_OPENAI_ENDPOINT_CHAT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY_CHAT = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_CHAT_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME")
AZURE_OPENAI_API_VERSION_CHAT = os.getenv("AZURE_OPENAI_API_VERSION")

# --- Azure AI Search ---
AZURE_AI_SEARCH_ENDPOINT = os.getenv("AZURE_AI_SEARCH_ENDPOINT")
AZURE_AI_SEARCH_API_KEY = os.getenv("AZURE_AI_SEARCH_API_KEY")
AZURE_AI_SEARCH_INDEX_NAME = os.getenv("AZURE_AI_SEARCH_INDEX_NAME")



### Statefully manage chat history ###
store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


def create_rag_chain():
    """
    Initializes and returns the RAG chain.
    This function is called once at app startup to "cache" the chain.
    """
    logger.info("Attempting to initialize RAG chain...")

    # Simple error checking for environment variables
    if not all([AZURE_OPENAI_ENDPOINT_EMBEDDINGS, AZURE_OPENAI_API_KEY_EMBEDDINGS, AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME, AZURE_OPENAI_API_VERSION_EMBEDDINGS]):
        logger.error("Azure OpenAI Embeddings environment variables are not fully set.")
        return None # Or raise an exception
    if not all([AZURE_OPENAI_ENDPOINT_CHAT, AZURE_OPENAI_API_KEY_CHAT, AZURE_OPENAI_CHAT_DEPLOYMENT_NAME, AZURE_OPENAI_API_VERSION_CHAT]):
        logger.error("Azure OpenAI Chat environment variables are not fully set.")
        return None
    if not all([AZURE_AI_SEARCH_ENDPOINT, AZURE_AI_SEARCH_API_KEY, AZURE_AI_SEARCH_INDEX_NAME]):
        logger.error("Azure AI Search environment variables are not fully set.")
        return None

    try:
        # 1. Embeddings
        embeddings = AzureOpenAIEmbeddings(
            azure_deployment=AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME,
            azure_endpoint=AZURE_OPENAI_ENDPOINT_EMBEDDINGS,
            api_key=AZURE_OPENAI_API_KEY_EMBEDDINGS,
            api_version=AZURE_OPENAI_API_VERSION_EMBEDDINGS,
        )
        # 2. Vector Store
        vector_store = AzureSearch(
            azure_search_endpoint=AZURE_AI_SEARCH_ENDPOINT,
            azure_search_key=AZURE_AI_SEARCH_API_KEY,
            index_name=AZURE_AI_SEARCH_INDEX_NAME,
            embedding_function=embeddings.embed_query
        )
        # 3. LLM
        llm = AzureChatOpenAI(
            azure_deployment=AZURE_OPENAI_CHAT_DEPLOYMENT_NAME,
            azure_endpoint=AZURE_OPENAI_ENDPOINT_CHAT,
            api_key=AZURE_OPENAI_API_KEY_CHAT,
            api_version=AZURE_OPENAI_API_VERSION_CHAT,
            temperature=0.3,
            max_tokens=500
        )
        # 4. Retriever
        retriever = vector_store.as_retriever() # Add search_kwargs if needed e.g. {"k": 3}

        ### Contextualize question ###
        contextualize_q_system_prompt = """Given a chat history and the latest user question \
        which might reference context in the chat history, formulate a standalone question \
        which can be understood without the chat history. Do NOT answer the question, \
        just reformulate it if needed and otherwise return it as is."""
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        history_aware_retriever = create_history_aware_retriever(
            llm, retriever, contextualize_q_prompt
        )


        ### Answer question ###
        qa_system_prompt = """You are an assistant for question-answering tasks. \
        Use the following pieces of retrieved context to answer the question. \
        If you don't know the answer, just say that you don't know. \
        Use three sentences maximum and keep the answer concise.\

        {context}"""
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", qa_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        return rag_chain
    except Exception as e:
        logger.error(f"Failed to initialize RAG chain: {e}", exc_info=True)
        return None


# --- Flask App Definition ---
app = Flask(__name__)
CORS(app) # Enable CORS for all routes, for development. For production, configure origins.

# --- Initialize RAG chain on application startup ---
# This simulates the caching behavior. The chain is created once.
with app.app_context(): # Ensures this runs within the application context
    rag_chain_instance_global = create_rag_chain()
    if rag_chain_instance_global is None:
        logger.critical("CRITICAL: RAG Chain could not be initialized on startup. The /chat endpoint will not work.")


@app.route('/chat', methods=['POST'])
def chat_endpoint():
    global rag_chain_instance_global # Access the globally cached chain

    if rag_chain_instance_global is None:
        logger.error("RAG chain is not initialized. Cannot process request.")
        return jsonify({"error": "RAG chain not initialized. Server configuration issue."}), 500

    try:
        data = request.get_json()
        if not data or 'query' not in data:
            logger.warning("Received bad request: 'query' field missing.")
            return jsonify({"error": "Missing 'query' field in request body"}), 400

        user_query = data['query']
        logger.info(f"Received query: {user_query}")

        session_id = data['session_id']
        logger.info(f"Received session id: {session_id}")

        # Invoke the LangChain RAG chain
        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain_instance_global,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )
        assistant_response = conversational_rag_chain.invoke(
            {"input": user_query},
            config={"configurable": {"session_id": session_id}
            },
        )["answer"]

        # assistant_response = rag_chain_instance_global.invoke(user_query)
        logger.info(f"Generated response for query '{user_query}'")

        return jsonify({"response": assistant_response})

    except RateLimitError as rle:
        logger.warning(f"Rate limit exceeded: {rle}")
        return jsonify({"error": "API rate limit exceeded. Please try again later."}), 429
    except Exception as e:
        logger.error(f"Error processing chat request: {e}", exc_info=True)
        return jsonify({"error": f"An internal error occurred: {str(e)}"}), 500


@app.route('/health', methods=['GET'])
def health_check():
    global rag_chain_instance_global  # Access the globally cached chain

    if rag_chain_instance_global is not None:
        # Basic check: RAG chain was initialized successfully at startup
        # This implies dependencies were reachable at startup.
        return jsonify({
            "status": "ok",
            "message": "Flask application is running and RAG chain is initialized.",
            "rag_chain_initialized": True
        }), 200
    else:
        # RAG chain failed to initialize during startup
        logger.error("Health check failed: RAG chain is not initialized.")
        return jsonify({
            "status": "error",
            "message": "Flask application is running, but the RAG chain failed to initialize. Check server logs for details.",
            "rag_chain_initialized": False
        }), 503  # 503 Service Unavailable is appropriate here

if __name__ == '__main__':
    # This is for local development.
    # For production, use a WSGI server like Gunicorn or uWSGI.
    # Example: gunicorn -w 4 -b 0.0.0.0:5000 app:app
    app.run(debug=True, host='0.0.0.0', port=5001) # Using port 5001 to avoid common conflicts


#Previous RAG chain implemenation
#         # 5. Prompt Template
#         template = """You are an AI assistant. Answer the question based ONLY on the following context.
# If the context doesn't contain the answer, state that you don't have enough information from the provided documents.
# Be concise and cite source file names or document titles from the metadata if available (e.g., from a 'source' field in metadata).
#
# Context:
# {context}
#
# Question: {question}
#
# Answer:
# """
#         prompt = ChatPromptTemplate.from_template(template)
#
#         # 6. RAG Chain
#         def format_docs(docs):
#             return "\n\n".join(
#                 f"Source: {doc.metadata.get('source', 'Unknown')}\nContent: {doc.page_content}" for doc in docs)
#
#         rag_chain = (
#                 {"context": retriever | format_docs, "question": RunnablePassthrough()}
#                 | prompt
#                 | llm
#                 | StrOutputParser()
#         )
#         logger.info("RAG chain initialized successfully.")
