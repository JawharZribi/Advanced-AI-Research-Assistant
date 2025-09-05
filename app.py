# app.py
import streamlit as st
import os
from dotenv import load_dotenv

from my_rag_app.pipeline import create_rag_pipeline, create_router_chain, create_web_search_chain
from my_rag_app.data_loader import load_and_split_documents
from my_rag_app.vector_store import create_vector_store
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory

# Load environment variables from .env file
load_dotenv()

# Define paths
DATA_PATH = "documents"
CHROMA_PATH = "chroma_db"

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in st.session_state.store:
        st.session_state.store[session_id] = ChatMessageHistory()
    return st.session_state.store[session_id]

@st.cache_resource
def load_chains_and_tools():
    """
    Load and cache all the necessary AI components.
    """
    print("Loading AI models, chains, and tools...")
    embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    
    ollama_base_url = "http://host.docker.internal:11434"
    
    # Initialize LLMs
    llm_rag = OllamaLLM(model="llama3", base_url=ollama_base_url)
    llm_router = OllamaLLM(model="llama3", base_url=ollama_base_url)
    
    # Create chains and tools
    rag_chain_with_history = create_rag_pipeline(db, llm_rag, get_session_history)
    router_chain = create_router_chain(llm_router)
    search_tool = TavilySearchResults()
    web_search_chain = create_web_search_chain(llm_rag, search_tool)
    
    print("Finished loading.")
    return rag_chain_with_history, router_chain, search_tool, llm_router, web_search_chain

def main():
    st.set_page_config(
        page_title="Advanced AI Assistant",
        page_icon="🤖",
        layout="wide"
    )

    st.title("🤖 Advanced AI Research Assistant")
    st.info("Ask about your documents or general knowledge questions.")

    rag_chain_with_history, router_chain, search_tool, llm_router, web_search_chain = load_chains_and_tools()

    if "store" not in st.session_state:
        st.session_state.store = {}
    
    # --- SIDEBAR FOR FILE UPLOADS ---
    with st.sidebar:
        st.header("Manage Knowledge Base")
        uploaded_files = st.file_uploader(
            "Upload files to add to the knowledge base.",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=True
        )
        if uploaded_files:
            if st.button("Process Documents"):
                with st.spinner("Processing documents..."):
                    os.makedirs(DATA_PATH, exist_ok=True)
                    for uploaded_file in uploaded_files:
                        file_path = os.path.join(DATA_PATH, uploaded_file.name)
                        with open(file_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                    
                    chunks = load_and_split_documents()
                    create_vector_store(chunks)
                    st.success("Documents processed successfully!")
                    st.cache_resource.clear()
                    st.rerun()

    # --- MAIN CHAT INTERFACE ---
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Route the query
                classification = router_chain.invoke(prompt)
                
                # Act based on the classification
                if "web_search" in classification.lower():
                    search_results = search_tool.invoke({"query": prompt})
                    response = web_search_chain.invoke({"context": search_results, "question": prompt})
                elif "informational" in classification.lower():
                    response = rag_chain_with_history.invoke(
                        {"question": prompt},
                        config={"configurable": {"session_id": "main_session"}}
                    )
                else: # conversational
                    response = "Hello! How can I assist you today?"
                
                st.markdown(response)
        
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()