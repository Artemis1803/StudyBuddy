from sentence_transformers import SentenceTransformer
from sentence_transformers import util

model = SentenceTransformer("all-MiniLM-L6-v2")

notes = [
    "RAG uses vector search",
    "Embeddings convert text into vectors",
    "Transformers use attention mechanisms",
    "Python lists are mutable"
]

query = "How does retrieval augmented generation work?"

note_embeddings = model.encode(notes)

query_embeddings = model.encode(query)

scores = util.cos_sim(query_embeddings,note_embeddings)
best_index = scores.argmax()
print("Best Match:")
print(notes[best_index])

print("Scores:")
print(scores[0][best_index])