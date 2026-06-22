
import os

import chromadb
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, util
import google.generativeai as genai

load_dotenv()
_api_key= os.getenv("GEMINI_API_KEY")
model = SentenceTransformer("all-MiniLM-L6-v2")

genai.configure(api_key=_api_key)
genmodel = genai.GenerativeModel("gemini-2.5-flash")

client = chromadb.PersistentClient(path='./studybuddy_db')
collection = client.get_or_create_collection(name="study_notes")

query = input("Ask something:")
query_embedding = model.encode(query)
results = collection.query(query_embeddings=[query_embedding.tolist()],n_results=3)

top_notes = results["documents"][0]


context = "\n\n".join(top_notes)

prompt = f"""
You are my study assistant.

Use ONLY the information provided below.

Context:
{context}

Question:
{query}

Answer:
"""

response = genmodel.generate_content(prompt)

print(response.text)
