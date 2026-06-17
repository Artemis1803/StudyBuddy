from sentence_transformers import SentenceTransformer
import os
from sentence_transformers import util
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
_api_key= os.getenv("GEMINI_API_KEY")
model = SentenceTransformer("all-MiniLM-L6-v2")

genai.configure(api_key=_api_key)

genmodel = genai.GenerativeModel("gemini-2.5-flash")
notes = []
files = os.listdir("notes")

for file in files:
    with open(f"notes/{file}","r") as text:
        content = text.read()
        chunks = content.split("\n\n")
        # print(chunks)
        for chunk in chunks:
            notes.append(chunk)

# print(notes)
notes_embedding = model.encode(notes)

query = input("Ask something:")

query_embedding = model.encode(query)

scores = util.cos_sim(query_embedding,notes_embedding)
print(scores)
score_list = scores[0].tolist()

results = list(zip(notes,score_list))
results.sort(key=lambda x: x[1], reverse=True)

top_notes = []
for note, score in results[:3]:
    top_notes.append(note)

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
# print(type(notes_embedding))
# print(notes_embedding.shape)

# print(type(query_embedding))
# print(query_embedding.shape)

# print(type(scores))
# print(scores.shape)