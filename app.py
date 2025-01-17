import streamlit as st
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

# Loading FAISS index and text chunks from a .pkl file
@st.cache_resource
def load_faiss_index():
    with open("faiss_index.pkl", "rb") as f:
        data = pickle.load(f)
    index = data["index"]  # FAISS index
    text_chunks = data["chunks"]  # Text chunks
    return index, text_chunks

# Loading the embedding model
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

# Searching FAISS index for the query
def search_index(query, index, model, text_chunks, top_k=3):
    query_embedding = model.encode([query])
    distances, indices = index.search(np.array(query_embedding), top_k)
    results = [text_chunks[i] for i in indices[0]]
    return results

# Streamlit App Layout
st.title("FAISS-based Question-Answer System")
st.markdown("""
Welcome to the FAISS-based question-answering app. 
Type a question in the box below to retrieve relevant content.
""")

# User Query Input
user_query = st.text_input("Enter your question:", "")

# Performing search on submission
if user_query:
    st.write("Searching for relevant information...")
    
    # Loading resources
    index, text_chunks = load_faiss_index()
    model = load_embedding_model()
    
    # Searching for relevant chunks
    top_chunks = search_index(user_query, index, model, text_chunks)
    
    # Displaying results
    st.write("### Top Relevant Chunks:")
    for i, chunk in enumerate(top_chunks, start=1):
        st.write(f"**Result {i}:** {chunk}")
