from my_rag_app.data_loader import load_and_split_documents
from my_rag_app.vector_store import create_vector_store 


def main():
    print("Loading the documents...")
    
    # Load and split documents
    docs = load_and_split_documents()
    
    if not docs:
        print("No documents to process. Exiting.")
        return

    # Create the vector store
    vector_store = create_vector_store(docs)
    
    print("Database creation complete. You can now run the main application.")
    
    
if __name__ == "__main__":
    main()