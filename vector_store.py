import os
import faiss
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

# Embedding model
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

FAISS_INDEX_PATH = "faiss_index"

def create_faiss_index(text_chunks):
    """
    Creates a FAISS vector store from text chunks of multiple PDFs.
    """
    vector_store = FAISS.from_texts(text_chunks, embeddings)
    vector_store.save_local(FAISS_INDEX_PATH)

def load_faiss_index():
    """
    Loads an existing FAISS index.
    """
    if os.path.exists(FAISS_INDEX_PATH):
        return FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    return None
