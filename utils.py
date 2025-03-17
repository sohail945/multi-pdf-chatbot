from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Function to embed text chunks using OpenAI's embeddings
def embed_chunks(chunks):
    """
    Embed text chunks using OpenAI's embeddings.

    Args:
        chunks (list of str): List of text chunks to embed.

    Returns:
        OpenAIEmbeddings: Embedding object initialized with OpenAI API key.
    """
    openai_api_key = os.getenv("OPENAI_API_KEY")  # Load API key from environment
    if not openai_api_key:
        raise ValueError("OpenAI API key not found. Please set it in the .env file.")

    # Initialize OpenAI embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    return embeddings

# Function to create a FAISS index
def create_faiss_index(chunks):
    """
    Create a FAISS index from the given text chunks.

    Args:
        chunks (list of str): List of text chunks.

    Returns:
        FAISS: FAISS index for the given chunks.
    """
    embeddings = embed_chunks(chunks)
    # Convert text chunks into Document objects required by LangChain's FAISS implementation
    documents = [Document(page_content=chunk) for chunk in chunks]
    # Create the FAISS index
    faiss_index = FAISS.from_documents(documents, embeddings)
    return faiss_index
