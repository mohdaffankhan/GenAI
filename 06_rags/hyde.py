import os
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_qdrant import QdrantVectorStore

def load_split(file_path:Path, chunk_size:int =1000, chunk_overlap:int=200):
    "Load a PDF and split it into smaller chunks for embedding and retrieval."
    loader = PyPDFLoader(file_path=file_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(documents=docs)

def init_vector_store(docs:list, db_url:str, collection_name:str, embeddings):
    "Try to load an existing Qdrant collection; if it doesn't exist, create & index it."
    try:
        return QdrantVectorStore.from_existing_collection(
            url=db_url,
            collection_name=collection_name,
            embedding=embeddings
        )
    except Exception:
        return QdrantVectorStore.from_documents(
            documents=[],
            url=db_url,
            collection_name=collection_name,
            embedding=embeddings
        ).add_documents(documents=docs)

from langchain.prompts import ChatPromptTemplate

# HyDE document generation chain
template = """Please write a scientific paper passage to answer the question
Question: {question}
Passage:"""
prompt_hyde = ChatPromptTemplate.from_template(template)

from langchain_core.output_parsers import StrOutputParser

generate_docs_for_retrieval = (
    prompt_hyde | ChatGoogleGenerativeAI(model="gemini-2.0-flash-001", api_key=os.getenv("GEMINI_API_KEY")) | StrOutputParser() 
)

# RAG chain -> get answer based on retrieved context
template = """Answer the following question based on this context:

{context}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-001", api_key=os.getenv("GEMINI_API_KEY"))

final_rag_chain = (
    prompt
    | llm
    | StrOutputParser()
)

def main():
    GEMINI_KEY = os.getenv("GEMINI_API_KEY")
    if not GEMINI_KEY:
        raise RuntimeError("Set GEMINI_API_KEY in your environment.")

    pdf_path = Path(__file__).parent / "linux-cookbook.pdf"
    qdrant_url = "http://localhost:6333"
    collection = "learning_langchain"

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=GEMINI_KEY
    )

    split_docs = load_split(pdf_path)
    vector_store = init_vector_store(
        split_docs, db_url=qdrant_url, collection_name=collection, embeddings=embeddings
    )
    retriever = vector_store.as_retriever()

    print("Ready!  (Ctrl‑C to quit)")
    try:
        while True:
            user_q = input("\nQuery> ").strip()
            if not user_q:
                continue
            
            generate_docs_for_retrieval.invoke({"question":user_q})
            retrieval_chain = generate_docs_for_retrieval | retriever 
            retireved_docs = retrieval_chain.invoke({"question":user_q})
            # print(retireved_docs)
            final_docs=final_rag_chain.invoke({"context":retireved_docs,"question":user_q})
            
            print("\nFinal Answer:\n", final_docs)
    except (KeyboardInterrupt, EOFError):
        print("\nGoodbye!")


if __name__ == "__main__":
    main()




