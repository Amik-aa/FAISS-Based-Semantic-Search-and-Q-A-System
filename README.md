# FAISS-Based-Semantic-Search-and-Q-A-System
FAISS-Based Semantic Search and Q&A System is a Streamlit based application that enables semantic search and question answering over text datasets. It uses FAISS for efficient similarity search and Sentence Transformers for embedding generation.
This project implements a question answer system by embedding textual data, indexing it using FAISS, and retrieving answers to user queries. Below are the detailed steps performed.
Step 1: Initial Setup
Install Required Libraries
We installed the necessary libraries to support the project
Step 2: Data Preparation
Loading the Data
We loaded textual data from a file for embedding and querying. The file was read and the content extracted
Step 3: Data Cleaning
The raw text underwent cleaning to make it suitable for embedding. This included:
Removing unwanted spaces, newlines, and formatting issues.
Filtering non text elements.
Step 4: Splitting Text into Chunks
We split the cleaned text into manageable chunks for embedding. The chunks were created based on a fixed size
Step 5: Generating Text Embeddings
We used the SentenceTransformers library to generate embeddings for the chunks
Step 6: Creating a FAISS Index
The embeddings were indexed using FAISS for efficient similarity-based querying.
Step 7: Creating a Streamlit app
Created a fully functional app for querying our FAISS index and generating answers.
