# my_rag_app/pipeline.py
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
# CHANGE: Import the correct local re-ranker
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_community.document_compressors import FlashrankRerank
from flashrank import Ranker
from langchain_core.prompts import MessagesPlaceholder


# This is our new helper function
def print_and_pass_prompt(prompt_input):
    print("\n--- INJECTED PROMPT ---")
    # The actual prompt object is in the 'messages' attribute
    for message in prompt_input.messages:
        print(message.content)
    print("-----------------------\n")
    return prompt_input   
    
def create_rag_pipeline(vector_store, llm, get_session_history_func):
    print("Creating RAG pipeline with Local Re-ranking and Memory...")
    
    # 1. Create a base retriever
    base_retriever = vector_store.as_retriever(search_kwargs={"k": 10})

    # 2. Create the Flashrank Re-ranker compressor
    ranker_client = Ranker()
    compressor = FlashrankRerank(client=ranker_client)

    # 3. Create the compression retriever
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=base_retriever
    )

    # Create a Question Rephrasing Chain
    rephrase_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Given a chat history and the latest user question, formulate a standalone question that can be understood without the chat history. Do NOT answer the question, just reformulate it if needed and otherwise return it as is."),
            MessagesPlaceholder("chat_history"),
            ("human", "{question}"),
        ]
    )
    question_rephraser = rephrase_prompt | llm | StrOutputParser()

    rag_chain_template = """
You are an expert research assistant. Your job is to answer questions based ONLY on the provided context.
Cite the source document for each piece of information you use. The context will include metadata with a 'source' field for each document.

If the information to answer the question is not in the context, you MUST say "I cannot answer this question from the provided documents."
Do not make up information or use outside knowledge.

Context:
{context}

Question:
{question}

Final Answer:
"""
    rag_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", rag_chain_template),
            MessagesPlaceholder("chat_history"),
            ("human", "{question}"),
        ]
    )

    def format_docs(docs):
        return "\n\n".join(f"Source: {doc.metadata.get('source', 'Unknown')}\n\n{doc.page_content}" for doc in docs)
    
    # 4. Define the final RAG chain using the new compression retriever
    rag_chain = (
        RunnablePassthrough.assign(
            context=question_rephraser | compression_retriever | format_docs
        )
        | rag_prompt
        | llm
        | StrOutputParser()
    )

    # 3. Wrap the RAG chain with memory
    rag_with_memory = RunnableWithMessageHistory(
        rag_chain,
        get_session_history_func,
        input_messages_key="question",
        history_messages_key="chat_history",
    )
    
    return rag_with_memory

def create_router_chain(llm):
    print("Creating router chain...")
    
    # Update the template with the new 'web_search' option
    template = """You are an expert at routing a user's request.
Classify the user's request as 'web_search', 'informational', or 'conversational'.
Output a JSON object with a single key 'classification' and one of the three values.
For example: {{"classification": "conversational"}}

User Request:
{request}
"""
    
    prompt = ChatPromptTemplate.from_template(template)
    
    router_chain = (
        {"request": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return router_chain

def create_web_search_chain(llm, search_tool):
    """
    Creates a chain to summarize web search results.
    """
    template = """You are an expert research assistant.
Summarize the provided context to answer the user's question.

Context:
{context}

Question:
{question}

Answer:"""
    
    prompt = ChatPromptTemplate.from_template(template)
    
    # This chain now correctly maps the 'question' to the 'query' for the tool
    web_search_chain = (
        {"context": RunnableLambda(lambda x: search_tool.invoke({"query": x["question"]})), "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return web_search_chain