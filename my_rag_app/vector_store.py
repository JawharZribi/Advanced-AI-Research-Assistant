# my_rag_app/vector_store.py
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Define the path for the persistent ChromaDB database
CHROMA_PATH = "chroma_db"

def create_vector_store(documents):
    """
    Creates a persistent ChromaDB vector store from a list of documents.

    Args:
        documents (list): A list of LangChain document objects to be indexed.

    Returns:
        Chroma: The ChromaDB vector store object.
    """
    print("Initializing local embedding model...")
    # This now uses the class from the new package
    embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    print(f"Creating and saving a new vector store at: {CHROMA_PATH}")
    db = Chroma.from_documents(
        documents,
        embedding_function,
        persist_directory=CHROMA_PATH
    )

    print("Vector store created successfully.")
    return db