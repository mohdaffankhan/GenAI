import os
from google import genai
from dotenv import load_dotenv

load_dotenv(dotenv_path="../.env")
api_key= os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

text = "What is the meaning of life?"

result = client.models.embed_content(
        model="gemini-embedding-exp-03-07",
        contents=text)

print(result.embeddings)