# Advanced AI Research Assistant 🤖

This project is a sophisticated, full-stack AI agent designed for advanced document analysis and question-answering. It leverages a multi-tool architecture to provide context-aware answers from a private, persistent knowledge base of local documents, or by searching the live internet for up-to-the-minute information. The entire application is containerized with Docker for easy and reproducible deployment.

## Key Features

- **Multi-Tool Agent:** An intelligent router classifies user intent to delegate tasks to the appropriate tool:
    - **Document RAG:** For in-depth questions about an internal knowledge base.
    - **Web Search:** For general knowledge and current events.
    - **Conversational:** For simple greetings and small talk.
- **Advanced RAG Pipeline:** The core of the assistant, featuring:
    - **Persistent Knowledge Base:** Uses **ChromaDB** to store and manage document embeddings, allowing the knowledge base to grow over time.
    - **Multi-Document Support:** Ingests and processes multiple file types, including `.pdf`, `.docx`, and `.txt`.
    - **High-Quality Retrieval:** A **Flashrank re-ranker** filters and improves search results to ensure the most relevant context is used, mitigating model hallucinations.
    - **Sourced Answers:** Cites the source document for information retrieved from the knowledge base.
- **Local & Private:** Powered by open-source, locally-hosted LLMs via **Ollama** (`Llama 3`, `Phi-3`), ensuring 100% privacy and no API costs.
- **Web Interface:** A user-friendly, interactive chat interface built with **Streamlit**, featuring a tool for uploading new documents to the knowledge base.
- **Containerized:** Fully containerized with **Docker** and **Docker Compose** for easy, one-command setup and deployment.

## Tech Stack

- **Backend:** Python, LangChain
- **Frontend:** Streamlit
- **LLMs:** Ollama (Llama 3, Phi-3)
- **Vector Database:** ChromaDB
- **Embedding & Re-ranking:** Hugging Face Transformers, Flashrank
- **Web Search:** Tavily AI
- **Deployment:** Docker, Docker Compose

## Getting Started

### Prerequisites

- [Docker](https://www.docker.com/get-started) and Docker Compose installed.
- [Ollama](https://ollama.com/) installed and running.

### Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
    cd your-repo-name
    ```

2.  **Download Local LLMs:**
    Pull the necessary models using Ollama.
    ```bash
    ollama pull llama3
    ollama pull phi3
    ```

3.  **Create the Environment File:**
    Create a `.env` file in the root directory and add your Tavily API key:
    ```
    TAVILY_API_KEY="your-tavily-api-key"
    ```

4.  **Add Documents:**
    Place your `.pdf`, `.docx`, or `.txt` files into the `documents/` directory.

5.  **Build the Knowledge Base:**
    Before running the main app for the first time, you must process your documents to create the persistent vector store.
    ```bash
    python create_database.py
    ```

### Usage

1.  **Start the application:**
    ```bash
    docker-compose up --build
    ```

2.  **Access the UI:**
    Open your web browser and navigate to `http://localhost:8501`.
