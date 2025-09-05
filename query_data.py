# query_data.py

# Notice we are now importing our function again!
from my_rag_app.pipeline import create_rag_pipeline
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM

CHROMA_PATH = "chroma_db"

def main():
    embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    llm_rag = OllamaLLM(model="phi3") # Using tinyllama for speed
    # llm_router = OllamaLLM(model="phi3") # Using tinyllama for speed

    # Here we call our factory function from pipeline.py
    rag_chain = create_rag_pipeline(db, llm_rag)
    # router_chain = create_router_chain(llm_router)

    print("\n✅ Setup complete. You can now ask questions.")
    print("---------------------------------------------")

    while True:
        user_question = input("Ask a question (or type 'exit' to quit): ")

        if user_question.lower() in ["exit", "quit", "q"]:
            print("Exiting application. Goodbye!")
            break
        
        response = rag_chain.invoke(user_question)

        print("\n--- Answer ---")
        print(response)
        print("--------------\n")
        
if __name__ == "__main__":
    main()