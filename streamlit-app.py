import streamlit as st
import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.vectorstores.azuresearch import AzureSearch
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()

# --- Get variables for Embeddings Resource ---
AZURE_OPENAI_ENDPOINT_EMBEDDINGS = os.getenv("AZURE_OPENAI_ENDPOINT_EMBEDDINGS")
AZURE_OPENAI_API_KEY_EMBEDDINGS = os.getenv("AZURE_OPENAI_API_KEY_EMBEDDINGS")
AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME")
AZURE_OPENAI_API_VERSION_EMBEDDINGS = os.getenv("AZURE_OPENAI_API_VERSION_EMBEDDINGS")

# --- Get variables for Chat Model Resource ---
AZURE_OPENAI_ENDPOINT_CHAT = os.getenv("AZURE_OPENAI_ENDPOINT_CHAT")
AZURE_OPENAI_API_KEY_CHAT = os.getenv("AZURE_OPENAI_API_KEY_CHAT")
AZURE_OPENAI_CHAT_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME")
AZURE_OPENAI_API_VERSION_CHAT = os.getenv("AZURE_OPENAI_API_VERSION_CHAT")

# --- Azure AI Search (remains the same) ---
AZURE_AI_SEARCH_ENDPOINT = os.getenv("AZURE_AI_SEARCH_ENDPOINT")
AZURE_AI_SEARCH_API_KEY = os.getenv("AZURE_AI_SEARCH_API_KEY")
AZURE_AI_SEARCH_INDEX_NAME = os.getenv("AZURE_AI_SEARCH_INDEX_NAME")

@st.cache(allow_output_mutation=True, suppress_st_warning=True) # Added suppress_st_warning for st.cache
def get_cached_rag_chain():
    # Simple error checking for environment variables
    if not all([AZURE_OPENAI_ENDPOINT_EMBEDDINGS, AZURE_OPENAI_API_KEY_EMBEDDINGS, AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME, AZURE_OPENAI_API_VERSION_EMBEDDINGS]):
        st.error("Azure OpenAI Embeddings environment variables are not fully set.")
        return None
    if not all([AZURE_OPENAI_ENDPOINT_CHAT, AZURE_OPENAI_API_KEY_CHAT, AZURE_OPENAI_CHAT_DEPLOYMENT_NAME, AZURE_OPENAI_API_VERSION_CHAT]):
        st.error("Azure OpenAI Chat environment variables are not fully set.")
        return None
    if not all([AZURE_AI_SEARCH_ENDPOINT, AZURE_AI_SEARCH_API_KEY, AZURE_AI_SEARCH_INDEX_NAME]):
        st.error("Azure AI Search environment variables are not fully set.")
        return None

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
        max_tokens=500 # Kept your reduction
    )
    #Retriever
    retriever = vector_store.as_retriever(search_type="hybrid")
    #Prompt Template
    template = """You are an expert AI assistant for our application. Your goal is to be insightful and proactive.
Answer the question based on the following context.
If it is relevant, answer the question based ONLY on the provided context and cite sources if possible.
If the context is not relevant or doesn't contain the answer, try to answer the question using your general knowledge.
If you are using general knowledge because the context was not relevant or sufficient, mention that you are doing so.
If you cannot answer the question using either the context or your general knowledge, state that you are unable to answer.


Context:
{context}

Question: {question}

---
Primary Answer:
[Provide a direct answer to the question based on the context. Cite sources if possible, e.g., (Source: application_docs.md)]

---
Predictive Insights & Next Steps:
Based on the question and the provided context:
1. Are there any related topics or proactive suggestions you can offer?
2. If the context discusses an issue or a process, what are the key implications or next logical steps?
(If no specific predictive insights are apparent from the context, state "No specific predictive insights or next steps apparent from this context.")
"""
    prompt = ChatPromptTemplate.from_template(template)

    #RAG Chain
    def format_docs(docs):
        return "\n\n".join(
            f"Source: {doc.metadata.get('source', 'Unknown')}\nContent: {doc.page_content}" for doc in docs)

    rag_chain_instance = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
    )
    return rag_chain_instance

# --- UI Starts Here ---
st.title("📚 Gen AI Chatbot (Streamlit 1.12.0)")
st.caption("Ask about your application's source code, logs, or documentation!")

# Get the RAG chain (cached)
rag_chain = get_cached_rag_chain()

if not rag_chain:
    st.warning("RAG chain could not be initialized. Please check environment variables and Azure service status.")
    st.stop()

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! How can I help you today?"}]

# Display chat messages from history
# We will display messages first, then the input form at the bottom.
for message in st.session_state.messages:
    if message["role"] == "user":
        st.markdown(f"<div style='text-align: right; margin-left: 20%; margin-bottom: 10px; padding: 10px; background-color: #DCF8C6; color: #003366; border-radius: 10px;'><b>You:</b> {message['content']}</div>", unsafe_allow_html=True)
    elif message["role"] == "assistant":
        st.markdown(f"<div style='text-align: left; margin-right: 20%; margin-bottom: 10px; padding: 10px; background-color: #E8E8E8; color: #003366; border-radius: 10px;'><b>Assistant:</b> {message['content']}</div>", unsafe_allow_html=True)
    else: # Should not happen with current logic but good for robustness
        st.text(message['content'])

# Input form for user query
# Using a form helps manage submission and clearing the input on submit.
with st.form(key="chat_input_form", clear_on_submit=True):
    user_query = st.text_input("Your question:", key="user_query_input", placeholder="Ask your question here...")
    submit_button = st.form_submit_button(label="Send")

if submit_button and user_query:
    # Append user's query to chat history
    st.session_state.messages.append({"role": "user", "content": user_query})

    # Show a spinner while processing
    with st.spinner("Assistant is thinking..."):
        try:
            # Invoke the LangChain RAG chain
            assistant_response = rag_chain.invoke(user_query)
            st.session_state.messages.append({"role": "assistant", "content": assistant_response})
        except Exception as e:
            error_message = f"Sorry, an error occurred: {e}"
            st.session_state.messages.append({"role": "assistant", "content": error_message})
            # The error will be displayed in the chat history loop.
            # You could also st.error(error_message) here for immediate feedback if desired.

    # Rerun the app to display the new messages and clear the form (due to clear_on_submit=True)
    st.experimental_rerun()

# Add a small footer or instruction
st.markdown("---")
st.markdown("Type your question above and click 'Send'.")