import streamlit as st
from pdf_loader import load_pdfs
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from groq import Groq
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Ensure 'temp_files' directory exists
TEMP_DIR = "temp_files"
os.makedirs(TEMP_DIR, exist_ok=True)

# Set Streamlit page configuration
st.set_page_config(page_title="Multi-PDF Chatbot", page_icon="📄", layout="wide")

# Initialize session state variables
if "conversation_history" not in st.session_state:
    st.session_state["conversation_history"] = []
if "clear_chat" not in st.session_state:
    st.session_state["clear_chat"] = False

# Sidebar for PDF Upload
with st.sidebar:
    st.title("📂 Upload PDFs")
    uploaded_files = st.file_uploader("Upload one or more PDFs:", type="pdf", accept_multiple_files=True)
    st.markdown("---")
    st.subheader("⚙️ Settings")
    if st.button("🗑️ Clear Chat History"):
        st.session_state["conversation_history"] = []
        st.session_state["clear_chat"] = True
        st.rerun()

# Main Chat Interface
st.title("💬 Chat with Your PDFs")

# Process uploaded PDFs
if uploaded_files:
    file_paths = []
    for uploaded_file in uploaded_files:
        file_path = os.path.join(TEMP_DIR, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())
        file_paths.append(file_path)

    try:
        # Load and process PDFs
        pdf_text = load_pdfs(file_paths)
        text_chunks = [pdf_text[i:i+500] for i in range(0, len(pdf_text), 450)]

        # Embed text chunks and create FAISS index
        model = SentenceTransformer('all-mpnet-base-v2')
        embeddings = model.encode(text_chunks)
        dimension = embeddings.shape[1]
        faiss_index = faiss.IndexFlatL2(dimension)
        faiss_index.add(np.array(embeddings))
    except Exception as e:
        st.error(f"❌ Error processing PDFs: {e}")
        st.stop()

    # Initialize Groq client
    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    # Chat Input Box
    question = st.chat_input("Ask a question about the PDFs...", key="user_question")

    if question:
        st.session_state["clear_chat"] = False  # Reset clear flag
        
        try:
            def retrieve_relevant_chunks(question, faiss_index, text_chunks, k=5):
                question_embedding = model.encode([question])
                distances, indices = faiss_index.search(question_embedding, k)
                return [text_chunks[i] for i in indices[0]]

            relevant_chunks = retrieve_relevant_chunks(question, faiss_index, text_chunks, k=10)
            context = "\n\n".join(relevant_chunks) if relevant_chunks else ""
            history_context = "\n".join([f"User: {entry['user']}\nBot: {entry['bot']}" for entry in st.session_state["conversation_history"][-5:]])
            
            full_input = f"""
            System: You are an AI assistant that must only answer questions based on the given document context.
            - If the context does not contain relevant information, respond with:
            "I'm sorry, I don't know. The document does not provide enough information to answer this."
            - Do not make assumptions.
            - Do not generate answers from general knowledge.
            - Only use the document content for responses.

            Conversation History:
            {history_context}

            Context from Document:
            {context}

            User Question:
            {question}

            Answer:
            """
            
            response = groq_client.chat.completions.create(
                messages=[{"role": "user", "content": full_input}],
                model="mixtral-8x7b-32768",
                max_tokens=200,
                temperature=0.7,
            )
            
            answer = response.choices[0].message.content

            # Display user question
            with st.chat_message("user", avatar="🧑"):
                st.markdown(f"{question}")

            # Display bot response
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(f"**{answer}**")

            # Save conversation history
            st.session_state["conversation_history"].append({"user": question, "bot": answer})
        except Exception as e:
            st.error(f"❌ Error generating answer: {e}")

# Display chat history in a natural chat format
for entry in st.session_state["conversation_history"]:
    with st.chat_message("user", avatar="🧑"):
        st.markdown(entry["user"])
    with st.chat_message("assistant", avatar="🤖"):
        st.markdown(entry["bot"])