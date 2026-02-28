RAG Chatbot

A Retrieval Augmented Generation (RAG) system that lets you upload any PDF or text document and ask questions about it in natural language. The AI answers based on YOUR documents, not general knowledge.

🔴 Live Demo
https://rag-chatbot-9qy4alfydz4cscu7exyooa.streamlit.app/

You can use your own Groq key :

Follow this steps:

Go to console.groq.com

Sign up with Google

Click API Keys → Create API Key



What is RAG?

Normal ChatGPT only knows what it was trained on. RAG gives the AI access to YOUR specific documents.


How It Works — Step by Step

Step 1 — Document Processing

- Upload any PDF or TXT file
  
- Document is split into 500-word chunks with 50-word overlap
  
- Each chunk is converted to a 384-dimensional vector using Sentence Transformers (all-MiniLM-L6-v2)
  
- Vectors stored in ChromaDB (vector database)

Step 2 — Semantic Search

- Your question is converted to a vector using the same model
  
- ChromaDB finds the 4 most similar chunks using cosine similarity
  
- This is semantic search — it understands meaning, not just keywords

Step 3 — AI Generation

- Top 4 chunks and your question sent to LLaMA 3.1 via Groq API
  
- LLM reads the context and generates an accurate answer
  
- Sources shown so you can verify exactly which chunks were used



Features

- Upload multiple PDFs or TXT files at once
  
- Real-time question answering
  
- Shows which document chunks were used to answer
  
- Conversation history for follow-up questions
  
- Session stats (questions asked, chunks indexed)
  
- Clean dark UI built with Streamlit



Tech Stack

LLM : LLaMA 3.1 8B (via Groq API) 

Vector Database : ChromaDB 

Embeddings : Sentence Transformers (all-MiniLM-L6-v2)

PDF Processing : PyPDF

Frontend : Streamlit, Plotly

Language : Python


