# GovAssist AI

**Your Intelligent Guide to U.S. Government Services**

A domain-specialized RAG-powered chatbot that simplifies access to Social Security, Immigration, and Passport services through real-time information retrieval and AI-powered responses.

---

## Overview

GovAssist AI combines a comprehensive knowledge base of 18,000+ government document chunks with real-time web search capabilities to provide accurate, trustworthy answers about U.S. government services. The system uses advanced retrieval-augmented generation (RAG) to ground responses in verified sources while maintaining conversational context.

---

## Key Features

- **Comprehensive Knowledge Base**: 18,483 indexed document chunks from SSA, USCIS, and Travel.State sources
- **Hybrid Search**: Combines vectorstore retrieval with real-time web search for current information
- **Semantic Search**: Uses FAISS vector database for fast, meaning-based document retrieval
- **Conversation Memory**: Maintains context across multi-turn conversations
- **Source Citations**: Provides transparent attribution for all information
- **Adjustable Parameters**: User-controlled retrieval settings (top-k, similarity threshold)

---

## Technology Stack

**Backend:**
- FastAPI - REST API server
- FAISS - Vector similarity search
- Sentence Transformers - Text embeddings (all-MiniLM-L6-v2)
- Gemini 2.0 Flash - Large language model
- Tavily - Real-time web search

**Frontend:**
- Streamlit - Interactive web interface
- Custom CSS - Styled components

**Data Processing:**
- PyPDF2/pdfplumber - PDF text extraction
- JSON parsing - Structured data handling
- Custom chunking - Document segmentation

---

## Architecture

```
User Query
    |
    v
[Streamlit Frontend]
    |
    v
[FastAPI Backend]
    |
    +-- [Query Classification]
    |
    +-- [FAISS Vector Search] --> [18,483 document chunks]
    |
    +-- [Tavily Web Search] --> [Real-time results]
    |
    +-- [Context Building]
    |
    v
[Gemini 2.0 Flash] --> [Generated Answer + Sources]
    |
    v
[User Response with Citations]
```

---

## Installation


3. Install dependencies:
```bash
pip install -r requirements.txt
```

 Build the vectorstore (first time only):
```bash
# Place your documents in data/ folders:
# - data/text_data/
# - data/json_data/
# - data/ssn_pdf/

# The vectorstore will build automatically on first backend startup
```

---

## Usage

### Start the Backend:
```bash
cd backend
uvicorn app:app --reload
```
Backend will be available at: http://localhost:8000

### Start the Frontend:
```bash
cd frontend
streamlit run app.py
```
Frontend will be available at: http://localhost:8501

### Using the Application:

1. Open http://localhost:8501 in your browser
2. Adjust search settings in the sidebar (optional):
   - Results to retrieve (1-10)
   - Similarity threshold (0.0-1.0)
   - Enable/disable web search
3. Type your question about government services
4. View answer with cited sources
5. Click "View Sources" to see document references

---

## Project Structure

```
GovAssist-AI/
├── backend/
│   └── app.py              # FastAPI server
├── frontend/
│   └── app.py              # Streamlit interface
├── src/
│   ├── data_loader.py      # Document loading and parsing
│   ├── embedding.py        # Embedding model management
│   ├── vectorstore.py      # FAISS index operations
│   └── search.py           # RAG pipeline and search logic
├── tools/
│   └── web_search.py       # Tavily web search integration
├── data/
│   ├── text_data/          # Plain text documents
│   ├── json_data/          # Structured JSON files
│   ├── ssn_pdf/            # PDF documents
│   └── vector_store/       # FAISS index (generated)
├── .env                    # Environment variables
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

---

## Configuration

### RAG Parameters (in sidebar):

- **Results to retrieve (top_k)**: Number of document chunks to retrieve (default: 5)
- **Similarity threshold**: Minimum similarity score for results (default: 0.4)
- **Web search**: Enable real-time web search fallback (default: enabled)

### System Defaults:

- **Embedding model**: all-MiniLM-L6-v2 (384 dimensions)
- **LLM**: Gemini 2.0 Flash
- **Chunk sizes**: 
  - SSA: 800 characters (150 overlap)
  - USCIS: 900 characters (150 overlap)
  - Travel.State: 700 characters (100 overlap)
- **Conversation history**: Last 5 Q&A pairs

---

## Data Sources

The knowledge base includes official documents from:

- **Social Security Administration (SSA)**: Benefits, applications, office information
- **U.S. Citizenship and Immigration Services (USCIS)**: Visa forms, green cards, citizenship
- **U.S. Department of State (Travel.State)**: Passport services, international travel

---

## Performance

- **Startup time**: 2-3 seconds (cached vectorstore)
- **Query latency**: 3-5 seconds average
- **Retrieval speed**: 0.1 seconds (FAISS search)
- **LLM generation**: 2-3 seconds
- **Vectorstore size**: ~500 MB on disk

## Team

- **Rubash Mali** (A02464927@usu.edu)
- **Sulove [Last Name]** (A02474149@usu.edu)

---

**Built with care to make government services accessible to everyone.**
