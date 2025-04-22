import os
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore

from google import genai
from google.genai import types

pdf_path = Path(__file__).parent/"linux-cookbook.pdf"

# loader = PyPDFLoader(file_path=pdf_path)
# docs = loader.load()

# text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
# split_docs = text_splitter.split_documents(documents=docs)

embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=os.getenv("GEMINI_API_KEY"))

# vector_store = QdrantVectorStore.from_documents(
#     documents=[],
#     url="http://localhost:6333",
#     collection_name="learning_langchain",
#     embedding=embeddings
# )

# vector_store.add_documents(documents=split_docs)

print("Injection Done")

retriever = QdrantVectorStore.from_existing_collection(
    url="http://localhost:6333",
    collection_name="learning_langchain",
    embedding=embeddings
)

# search_result = retriever.similarity_search(
#     query="what is vim?"
# )

# print("Relevant chunks", search_result)

system_instructions = """
You are an helpfull AI Assistant who responds base on the available context
If the answer is not found in the context, reply with "There is no information about this in the pdf".

Context: {context}
"""

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


while True:

    user_query = input("> ")

    relevant_chunks = retriever.similarity_search(
        query=user_query
    )
    relevant_docs = "\n".join([doc.page_content for doc in relevant_chunks])
    response = client.models.generate_content(
        model='gemini-2.0-flash-001',
        config=types.GenerateContentConfig(
            system_instruction=system_instructions,
        ),
        contents = relevant_docs
    )

    print(response.text)