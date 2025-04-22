import os
import concurrent.futures
from typing import List, Dict, Tuple
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain.schema import Document

from google import genai
from google.genai import types

def load_split(file_path:Path, chunk_size:int =1000, chunk_overlap:int=200):
    "Load a PDF and split it into smaller chunks for embedding and retrieval."
    loader = PyPDFLoader(file_path=file_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(documents=docs)

# store embeddings in qdrant db
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

# generate parallel queries
def generate_parallel_queries(user_query:str, client:genai.Client, n_variants:int=5)->list:
    "Ask Gemini to rewrite the user's query into `n_variants` distinct reformulations."
    system_instructions = f"""
        You are a helpful AI assistant tasked with reformulating user queries to improve retrieval in a RAG system. 
        You are an assistant that generates {n_variants} distinct, detailed reformulations of the {user_query} to improve retrieval. Return *only* the bullet‑or‑numbered list.

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
    response = client.models.generate_content(
        model='gemini-2.0-flash-001',
        config=types.GenerateContentConfig(
            system_instruction=system_instructions,
        ),
        contents = user_query
    )
    lines = [ln.strip() for ln in response.text.splitlines() if ln.strip()]
    # strip any leading numbering ("1. …", "- ", etc.)
    queries = [ln.split(maxsplit=1)[-1].lstrip(".- ") for ln in lines]
    return queries

# rrf logic
def reciprocal_rank_fusion(
    rankings: List[List[str]],
    id_to_doc: Dict[str, Document],
    k: int = 60,
    top_n: int = 10
) -> List[Document]:
    """
    1) Fuse multiple ID‐rankings into a single sorted list of doc_ids with RRF.
    2) Rehydrate the top_n IDs back to Document objects (attaching the score).
    """
    # 1) accumulate RRF scores
    scores: Dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)

    # 2) sort IDs by descending score
    fused_list: List[Tuple[str, float]] = sorted(
        scores.items(),
        key=lambda pair: pair[1],
        reverse=True
    )

    # 3) rehydrate and attach score
    fused_docs: List[Document] = []
    for doc_id, score in fused_list[:top_n]:
        doc = id_to_doc.get(doc_id)
        if doc:
            doc.metadata["rrf_score"] = score
            fused_docs.append(doc)

    return fused_docs

# parallel retrieval -> ranking retrievals -> return rankings & retrieved docs
def retrieve_rankings(
    store: QdrantVectorStore,
    rewritten_queries: List[str],
    top_k: int = 3,
) -> Tuple[List[List[str]], Dict[str, Document]]:
    """
    1) Run top_k retrieval in parallel for each query.
    2) Build:
       - rankings: List of lists of doc_ids (one list per query)
       - id_to_doc: map doc_id -> Document
    """
    # 1) parallel fetch
    with concurrent.futures.ThreadPoolExecutor() as exec:
        futures = {
            exec.submit(store.similarity_search, q, top_k): q
            for q in rewritten_queries
        }

        raw_hits: Dict[str, List[Document]] = {}
        for fut in concurrent.futures.as_completed(futures):
            q = futures[fut]
            try:
                raw_hits[q] = fut.result()
            except Exception:
                raw_hits[q] = []

    # 2) build rankings + map
    rankings: List[List[str]] = []
    id_to_doc: Dict[str, Document] = {}
    for docs in raw_hits.values():
        ranking_ids: List[str] = []
        for doc in docs:
            doc_id = doc.metadata.get("_id", repr(doc))
            id_to_doc[doc_id] = doc
            ranking_ids.append(doc_id)
        rankings.append(ranking_ids)

    fused_docs = reciprocal_rank_fusion(rankings, id_to_doc)
    return fused_docs
def answer_query(
        client: genai.Client,
        user_query: str,
        context_docs: List[Document]
        )->str:
    
    system_instruction = (
        "You are a helpful assistant that answers based on provided context."
        "If the answer isn't in the context, reply 'There is no information about this in the pdf.'"
    )
    context = "\n---\n".join(d.page_content for d in context_docs)
    prompt = f"Context:\n{context}\n\nQuestion: {user_query}\nAnswer:"
    response = client.models.generate_content(
        model="gemini-2.0-flash-001",
        config=types.GenerateContentConfig(system_instruction=system_instruction),
        contents=prompt
    )
    return response.text.strip()

def main():
    GEMINI_KEY = os.getenv("GEMINI_API_KEY")
    if not GEMINI_KEY:
        raise RuntimeError("Set GEMINI_API_KEY in your environment.")

    pdf_path = Path(__file__).parent / "linux-cookbook.pdf"
    qdrant_url = "http://localhost:6333"
    collection = "learning_langchain"

    client = genai.Client(api_key=GEMINI_KEY)
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=GEMINI_KEY
    )

    # 1) Load & split, 2) init or load vector store
    split_docs = load_split(pdf_path)
    vector_store = init_vector_store(
        split_docs, db_url=qdrant_url, collection_name=collection, embeddings=embeddings
    )

    try:
        while True:
            user_q = input("Query> ").strip()
            if not user_q:
                continue

            # 3) Generate & print rewrites
            query_variants = generate_parallel_queries(user_q, client)
            print("\nRewritten queries:")
            for i, q in enumerate(query_variants, 1):
                print(f"  {i}. {q}")

            # 4) Retrieve + RRF-fuse + print fused hits
            fused_hits = retrieve_rankings(
                vector_store, query_variants, top_k=3
            )
            print(f"\nFused retrieval → {len(fused_hits)} chunks:")
            for i, doc in enumerate(fused_hits, 1):
                snippet = doc.page_content[:100].replace("\n", " ")
                print(f"  {i}. {snippet}…")

            # 5) Answer
            ans = answer_query(client, user_q, fused_hits)
            print(f"\nAnswer:\n{ans}\n")

    except (KeyboardInterrupt, EOFError):
        print("\nGoodbye!")


if __name__ == "__main__":
    main()
