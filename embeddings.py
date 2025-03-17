import os
import numpy as np
import torch
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModel

# Load API keys from .env
load_dotenv()
hf_api_key = os.getenv("HUGGINGFACE_API_KEY")

# Load Hugging Face model
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

def embed_text(text):
    """Generate embeddings for a given text using Hugging Face model."""
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    return outputs.last_hidden_state[:, 0, :].numpy()  # Extract CLS token embeddings

def create_faiss_index(text_chunks):
    """Creates a FAISS index from text embeddings."""
    import faiss
    embeddings = np.array([embed_text(chunk) for chunk in text_chunks]).squeeze()

    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)
    
    return index, text_chunks
