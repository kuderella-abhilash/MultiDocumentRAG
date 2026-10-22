
# Multi-Document RAG System

A Retrieval-Augmented Generation (RAG) application that
answers questions using information retrieved from multiple PDF documents.

## Features

- Load multiple PDF documents
- Split documents into overlapping text chunks
- Generate embeddings using Hugging Face
- Store vectors in ChromaDB
- Perform semantic similarity search
- Generate context-grounded answers using Llama 3 and Ollama
- Display source document and page metadata

## Tech Stack

- Python
- LangChain
- Hugging Face Sentence Transformers
- ChromaDB
- Ollama
- Llama 3
- PyPDF

## Architecture

PDF Documents → Chunking → Embeddings → ChromaDB
→ Semantic Retrieval → Llama 3 → Answer

## Prerequisites

- Python 3.10 or a compatible version
- Git
- Ollama
- The Llama 3 model

## Setup

Clone this repository:

```bash
git clone YOUR_REPOSITORY_URL
cd MultiDocumentRAG
```

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Download the model:

```bash
ollama pull llama3
```

Create a `data` directory and add your own PDF documents.

Run the application:

```bash
python main.py
```

## Current Progress

- [x] PDF document loading
- [x] Text chunking
- [x] Embedding generation
- [x] ChromaDB vector storage
- [x] Semantic retrieval
- [x] LLM answer generation
- [ ] Retrieval quality evaluation
- [ ] Streamlit chat interface

## Privacy

PDF documents, local vector databases, virtual environments,
and secret credentials are excluded from version control.
