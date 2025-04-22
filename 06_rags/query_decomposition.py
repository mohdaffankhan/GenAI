import os
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore

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

def decompose_queries(user_query:str, client:genai.Client):
    system_instructions = """You are a helpful assistant that generates multiple sub-questions related to an input question. \n
    The goal is to break down the input into a set of sub-problems / sub-questions that can be answers in isolation. \n
    Generate multiple search queries related to: {question} \n
    Output (3 queries):
    
    Example:
    1. What is machine learning?
    -Sub-questions
    1. What is machine?
    2. What is learning?
    3. What is machine learning?
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

def get_context_for(question: str, retriever, k: int = 5) -> str:
    docs = retriever.get_relevant_documents(question)[:k]
    # stitch their text together
    return "\n\n".join(d.page_content for d in docs)

PROMPT = """
Here is the (sub‑)question:
{question}

Previously answered Q&A pairs (to give you context):
{q_a_pairs}

Relevant document excerpts:
{context}

Answer the question above, drawing on the excerpts and prior Q&A.
"""

def answer_query(
    question: str,
    q_a_pairs: str,
    context: str,
    client: genai.Client
) -> str:
    # fill in your prompt
    full_prompt = PROMPT.format(
        question=question, q_a_pairs=q_a_pairs or "None", context=context or "None"
    )
    # call the LLM
    resp = client.models.generate_content(
        model="gemini-2.0-flash-001",
        config=types.GenerateContentConfig(),
        contents=full_prompt
    )
    return resp.text.strip()


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
    retriever = vector_store.as_retriever()
    print("Ready!  (Ctrl‑C to quit)")
    try:
        while True:
            user_q = input("\nQuery> ").strip()
            if not user_q:
                continue

            # decompose
            query_variants = decompose_queries(user_q, client)
            print("Decomposed queries:")
            for i, q in enumerate(query_variants, 1):
                print(f"  {i}. {q}")
            
            q_a_pairs = ""
            for q in query_variants:
                ctx = get_context_for(q, retriever)
                ans = answer_query(q, q_a_pairs, ctx, client)
                pair = f"Question: {q}\nAnswer: {ans}\n"
                q_a_pairs += pair + " \n"

            # Final synthesis on the original question
            final_ctx = get_context_for(user_q, retriever)
            final_ans = answer_query(user_q, q_a_pairs, final_ctx, client)
            print("\nFinal Answer:\n", final_ans)
    except (KeyboardInterrupt, EOFError):
        print("\nGoodbye!")


if __name__ == "__main__":
    main()