import os
import pickle
import faiss
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
import threading
from dotenv import load_dotenv

load_dotenv()  

HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if HF_TOKEN is None:
    raise ValueError("Hugging Face token not found. Please set HF_TOKEN in your environment.")

# Load FAISS index and documents
INDEX_DIR = "index"
faiss_index = faiss.read_index(os.path.join(INDEX_DIR, "faiss_index.bin"))
with open(os.path.join(INDEX_DIR, "documents.pkl"), "rb") as f:
    documents = pickle.load(f)
if not isinstance(documents, list):
    documents = list(documents)

# Initialize embedding model
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Load LLaMA-2 model
model_id = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=HF_TOKEN)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype="auto",
    load_in_4bit=True,
    use_auth_token=HF_TOKEN
)

# Function to generate answer with streaming tokens
def rag_answer_stream(query: str, top_k: int = 3):
    query_vec = embedder.encode([query])
    _, I = faiss_index.search(query_vec, top_k)
    retrieved_docs = [documents[i] for i in I[0]]
    context = "\n".join(retrieved_docs)
    prompt = f"<s>[INST] Answer the question based on the context:\n\n{context}\n\nQuestion: {query} [/INST]"

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    # Run generation in a background thread
    thread = threading.Thread(
        target=model.generate,
        kwargs={
            "input_ids": tokenizer(prompt, return_tensors="pt").input_ids.to(model.device),
            "max_new_tokens": 1024,
            "do_sample": True,
            "temperature": 0.7,
            "streamer": streamer
        }
    )
    thread.start()
    return streamer
