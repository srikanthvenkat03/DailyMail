from datasets import load_dataset
import faiss
import pickle
from sentence_transformers import SentenceTransformer
import numpy as np
import os

# Load Dataset
dataset = load_dataset("cnn_dailymail", "3.0.0", split="train[:10000]")  
documents = [d["article"] for d in dataset]   

# Embedding Model
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
embeddings = embedder.encode(documents, show_progress_bar=True, convert_to_numpy=True)

# Build FAISS Index
dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(np.array(embeddings))

# Save Index + Docs
os.makedirs("index", exist_ok=True)
faiss.write_index(index, "index/faiss_index.bin")

with open("index/documents.pkl", "wb") as f:
    pickle.dump(documents, f)   

print("FAISS index and documents saved!")
