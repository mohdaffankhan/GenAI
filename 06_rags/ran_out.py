import os
import concurrent.futures
from typing import List
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore

from google import genai
from google.genai import types

# load & split
def load_and_split(pdf_path: Path, chunk_size: int = 1000, chunk_overlap: int = 200) -> list:
    """
    Load a PDF and split it into smaller chunks for embedding and retrieval.
    """
    loader = PyPDFLoader(file_path=pdf_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap )
    return splitter.split_documents(documents=docs)

# embeddings store in vector db
def init_vector_store(docs: list, url: str, collection_name: str, embeddings):
    """
    Try to load an existing Qdrant collection; if it doesn't exist, create & index it.
    """
    try:
        # if the collection is already there, this will succeed
        return QdrantVectorStore.from_existing_collection(
            url=url,
            collection_name=collection_name,
            embedding=embeddings
        )
    except Exception:
        # otherwise, build it from scratch
        return QdrantVectorStore.from_documents(
            documents=[],
            url=url,
            collection_name=collection_name,
            embedding=embeddings
        ).add_documents(documents=docs)

# generate parallel queries
def generate_parallel_queries(client: genai.Client, original_query: str, n_variants: int = 5) -> list[str]:
    """
    Ask the LLM to produce multiple specific reformulations of the user's query.
    """
    system_instructions = (
        f"""
        You are a helpful AI assistant tasked with reformulating user queries to improve retrieval in a RAG system. 
        You are an assistant that generates {n_variants} distinct, detailed reformulations of the {original_query} to improve retrieval. Return *only* the bullet‑or‑numbered list.

        Guidelines:
        - Rephrase the original
        - Break into sub-questions
        - Expand with related concepts
        - Clarify any ambiguity

        Examples:
        1. Original query: "what is fs module?"
            Rewritten query:
            1. "what is filesystem?"
            2. "what is a module?"
            3. "what is file system module?"
        2. Original query: "how does event loop work in Node.js?"
            Rewritten queries:
            1. "how does the event loop function?"  
            2. "explain event loop in JavaScript"  
            3. "how does Node.js handle asynchronous operations?"  
        3. Original query: "what is npm?"  
            Rewritten queries:  
            1. "what does npm stand for?"  
            2. "what is node package manager?"  
            3. "how do you use npm in a project?"  
        4. Original query: "how to create a server in express?"  
            Rewritten queries:  
            1. "how do you make a web server using Express.js?"  
            2. "express server setup tutorial"  
            3. "steps to create express server in Node.js"  
        5. Original query: "difference between var, let and const?"  
            Rewritten queries:
            1. "how do var, let, and const differ in JavaScript?"  
            2. "when to use var vs let vs const?"  
        6. Original query: "how to handle errors in async/await?"
            Rewritten queries:
            1. "best way to catch errors with async await"  
            2. "async await error handling example"  
            3. "how to use try-catch with async functions?"  
        7. Original query: "what is a promise in JavaScript?"
            Rewritten queries:
            1. "what is a promise?"  
            2. "what is a promise in JavaScript?"
            3. "how to create a promise in JavaScript?
            """
    )
    response = client.models.generate_content(
        model='gemini-2.0-flash-001',
        config=types.GenerateContentConfig(system_instruction=system_instructions),
        contents=[original_query]
    )
    lines = [ln.strip() for ln in response.text.splitlines() if ln.strip()]
    # strip any leading numbering ("1. …", "- ", etc.)
    queries = [ln.split(maxsplit=1)[-1].lstrip(".- ") for ln in lines]
    return queries

# retrieve
def retrieve_for_queries(store: QdrantVectorStore, queries: List[str], top_k: int = 3) -> List:
    """
    Perform similarity search for each reformulated query in parallel and dedupe results.
    """
    all_docs = []
    with concurrent.futures.ThreadPoolExecutor() as pool:
        future_to_q = {
            pool.submit(store.similarity_search, q, top_k): q
            for q in queries
        }
        for fut in concurrent.futures.as_completed(future_to_q):
            try:
                all_docs.extend(fut.result())
            except Exception as e:
                print(f"[Warning] retrieval failed for '{future_to_q[fut]}': {e}")

    seen = set()
    unique_docs = []
    for d in all_docs:
        if d.page_content not in seen:
            seen.add(d.page_content)
            unique_docs.append(d)
    return unique_docs

# similarity search 
def answer_query(client: genai.Client, user_query: str, context_docs) -> str:
    """
    Given the aggregated context, prompt the LLM to answer or say 'no info'.
    """
    system_instruction = (
        "You are a helpful assistant that answers based on provided context. "
        "If the answer isn't in the context, reply 'There is no information about this in the pdf.'"
    )
    context = "\n---\n".join(d.page_content for d in context_docs)
    prompt = f"Context:\n{context}\n\nQuestion: {user_query}\nAnswer:"
    response = client.models.generate_content(
        model='gemini-2.0-flash-001',
        config=types.GenerateContentConfig(system_instruction=system_instruction),
        contents=[prompt]
    )
    return response.text.strip()

def main():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Please set GEMINI_API_KEY in your environment.")

    pdf_path = Path(__file__).parent / "linux-cookbook.pdf"
    qdrant_url = "http://localhost:6333"
    collection = "parallel_queries"

    client = genai.Client(api_key=api_key)
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=api_key
    )

    # One‑time load & index
    split_docs = load_and_split(pdf_path)
    vector_store = init_vector_store(split_docs, qdrant_url, collection, embeddings)
    print("vector_store is a", type(vector_store))

    try:
        while True:
            user_q = input("Query> ").strip()
            if not user_q:
                continue

            # 1) Rewrite
            variants = generate_parallel_queries(client, user_q)
            print("\nRewritten queries:")
            for i, q in enumerate(variants, 1):
                print(f"  {i}. {q}")

            # 2) Retrieve
            retrieved = retrieve_for_queries(vector_store, variants)
            print(f"\nRetrieved {len(retrieved)} unique chunks.")
            print(type(retrieved))

            # 3) Answer
            ans = answer_query(client, user_q, retrieved)
            print(f"\nAnswer:\n{ans}\n")

    except (KeyboardInterrupt, EOFError):
        print("\nGoodbye!")  # graceful exit


if __name__ == "__main__":
    main()