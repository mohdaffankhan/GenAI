import os
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_qdrant import QdrantVectorStore

from google import genai
from google.genai import types

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

def generate_stepback_queries(client: genai.Client, original_query: str, n_variants: int = 5) -> list[str]:
    system_instructions = """
You are an expert at world knowledge. Your task is to step back and paraphrase a question to a more generic step-back question, which is easier to answer. Here are a few examples:
1. Input: Which position did Knox Cunningham hold from May 1955 to Apr 1956? 
    Output: Which positions have Knox Cunningham held in his career?

2. Input: when was the last time a team from canada won the stanley cup as of 2002
    Output: which years did a team from canada won the stanley cup as of 2002

3. Input: What city is the person who broadened the doctrine of philosophy of language from?
    Output: who broadened the doctrine of philosophy of language

4. Input: Is shrimp scampi definitely free of plastic?
    Output: what is shrimp scampi made of?

5. Input: Do the anchors on Rede Globo speak Chinese?
    Output: What languages do the anchors on Rede Globo speak?

"""
    response = client.models.generate_content(
        model='gemini-2.0-flash-001',
        config=types.GenerateContentConfig(system_instruction=system_instructions),
        contents=[original_query]
    )
    lines = [ln.strip() for ln in response.text.splitlines() if ln.strip()]
    # strip any leading numbering ("1. …", "- ", etc.)
    queries = [ln.split(maxsplit=1)[-1].lstrip(".- ") for ln in lines]
    return queries

from langchain_core.prompts import ChatPromptTemplate
response_prompt_template = """You are an expert of world knowledge. I am going to ask you a question. Your response should be comprehensive and not contradicted with the following context if they are relevant. Otherwise, ignore them if they are not relevant.

# {normal_context}
# {step_back_context}

# Original Question: {question}
# Answer:"""
response_prompt = ChatPromptTemplate.from_template(response_prompt_template)

from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
def rag_chain(question: str, retriever: RunnableLambda):
    chain = (
        {
            # Retrieve context using the normal question
            "normal_context": RunnableLambda(lambda x: x["question"]) | retriever,
            # Retrieve context using the step-back question
            "step_back_context": generate_stepback_queries | RunnableLambda(lambda x: x["question"]) | retriever,
            # Pass on the question
            "question": lambda x: x["question"],
        }
        | response_prompt
        | ChatGoogleGenerativeAI(model="gemini-2.0-flash-001", api_key=os.getenv("GEMINI_API_KEY"))
        | StrOutputParser()
    )
    return chain.invoke({"question": question})

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
            
            generate_stepback_queries(client, user_q)

            final_docs = rag_chain(user_q, retriever)
            print("\nFinal Answer:\n", final_docs)
    except (KeyboardInterrupt, EOFError):
        print("\nGoodbye!")


if __name__ == "__main__":
    main()
