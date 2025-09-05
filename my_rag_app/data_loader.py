# my_rag_app/data_loader.py
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Define the path for the documents directory
DATA_PATH = "documents"

def load_and_split_documents():
    """
    Loads documents from the DATA_PATH using DirectoryLoader and splits them into chunks.
    This version uses the default UnstructuredFileLoader which handles multiple types.

    Returns:
        list: A list of split document chunks.
    """
    print(f"Loading documents from {DATA_PATH}...")
    
    # The DirectoryLoader will automatically use UnstructuredFileLoader
    # for common file types like .pdf, .docx, and .txt.
    loader = DirectoryLoader(
        DATA_PATH,
        glob="**/*",  # Search all subdirectories for any file
        show_progress=True,
        use_multithreading=True
    )
    
    documents = loader.load()

    if not documents:
        print("No documents found in the specified directory.")
        return []

    print(f"Splitting {len(documents)} documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splits = text_splitter.split_documents(documents)
    
    print(f"Finished splitting documents into {len(splits)} chunks.")
    return splits