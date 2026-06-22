import os
import hashlib
from dotenv import load_dotenv
import chromadb
from sentence_transformers import SentenceTransformer
import json


load_dotenv()
model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path='./studybuddy_db')
collection = client.get_or_create_collection(name="study_notes")
chunks = []
files = [
    file
    for file in os.listdir("notes")
    if file.endswith(".txt")
]
REGISTRY_FILE = "file_registry.json"

if os.path.exists(REGISTRY_FILE):
    with open(REGISTRY_FILE,"r") as file:
        registry = json.load(file)
else:
    registry = {}

def get_file_hash(filepath):
    with open(filepath,"rb") as file:
        content = file.read()
    return hashlib.md5(content).hexdigest()



for file in files:
    file_path = f"notes/{file}"
    current_hash = get_file_hash(file_path)
    stored_hash = registry.get(file)
    if stored_hash == current_hash:
        print(f"skipping {file}")
        continue
    print(f"Processing {file}")
    with open(f"notes/{file}","r") as text:
        content = text.read()
        file_chunks = content.split("\n\n")
        for index,chunk in enumerate(file_chunks, start=1):
            chunk_record = {
                "id":f"{file}_{index}",
                "text":chunk,
                "source":file
            
            }
            chunks.append(chunk_record)
    registry[file]=current_hash

chunk_texts = [chunk["text"] for chunk in chunks]
if not chunks:
    print("No files changed")
    exit()
notes_embedding = model.encode(chunk_texts)

ids = []
documents = []
metadatas = []

for chunk in chunks:
    ids.append(chunk["id"])
    documents.append(chunk["text"])
    metadatas.append({
        "source":chunk["source"]
    })

collection.upsert(ids=ids,documents=documents,embeddings=notes_embedding.tolist()
               ,metadatas=metadatas)

with open(REGISTRY_FILE,"w") as file:
    json.dump(registry,file,indent=4)

print(f"Indexed {len(ids)} chunks")