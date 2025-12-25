# Policy Question Answering Assistant
This repository contains a lightweight **Retrieval-Augmented Generation (RAG)** system built to answer questions based on a small set of company policy documents.  
The project focuses on **prompt engineering, retrieval quality, hallucination avoidance, and evaluation**.

---

## Features

- Open-source embeddings (**BAAI/bge-small-en-v1.5**)
- Local FAISS vector store (privacy-friendly)
- Lightweight hybrid retrieval (dense + keyword re-ranking)
- Hallucination control using similarity thresholding
- Structured, citation-based answers
- Graceful handling of missing or out-of-scope queries
- CLI-based interface

---

## Setup Instructions

### 1. Clone the repository

```bash
git clone <your-github-repo-url>
cd policy-rag-assistant
```

### 2. Create and activate a virtual environment
```
python -m venv venv
source venv/binactivate        # Linux / Mac
venv\Scripts\activate         # Windows
```

### 3. Install dependencies
```
pip install -r requirements.txt
```

### 4. Set up environment variables
Create a .env file in the project root
```
GROQ_API_KEY=your_groq_api_key
```

### 5. Run the application
```
python main.py
```

## Architecture Overview
The system follows a standard RAG pipeline with a clear separation between indexing and query-time execution.

### Indexing Phase (One-time at startup)
Load policy documents

Chunk text using character-based chunking

Generate embeddings

Build FAISS vector index

### Query Phase (Per user question)
Embed user query

Retrieve relevant chunks from FAISS

Apply similarity threshold

Re-rank results using keyword overlap

Generate grounded answer using LLaMA-3.1 (Groq)

Return a refusal if context is insufficient

### High-level Flow
```
User Question
     ↓
Query Embedding
     ↓
FAISS Retrieval
     ↓
Threshold Check
     ↓
Context Injection
     ↓
LLaMA-3.1 Answer Generation
```

## Prompt Engineering
### Initial Prompt
The initial prompt instructs the model to answer using only the retrieved context and to refuse when information is missing. However, it does not enforce structure or citations.
```
Use ONLY the information provided in the context below to answer the question.
If the answer is not present, say:
"No relevant policy information was found in the provided documents."
```
### Improved Prompt (Final)
The improved prompt introduces stricter rules to reduce hallucinations and enforce structured output with explicit citations.
```
Rules:
- Answer ONLY using the provided context.
- Do NOT use external knowledge.
- Do NOT make assumptions or guesses.
- If insufficient information exists, respond exactly with:
  "No relevant policy information was found in the provided documents."

Output format:

Answer:
- Bullet points

Sources:
- filename, chunk index
```

### What Changed and Why

Added explicit constraints to prevent over-inference

Enforced bullet-point answers for clarity

Required source citations (filename + chunk index)

Standardized refusal behavior for missing information

## Key Trade-offs
Character-based chunking chosen for simplicity over token-based chunking

In-memory FAISS indexing rebuilt on each application run

Manual evaluation used instead of automated metrics

Lightweight hybrid retrieval balances performance and simplicity

## Future Improvements

With more time, the system could be improved by:

Persisting the FAISS index to disk

Adding intent detection for greetings and vague queries

Using BM25 or cross-encoder re-ranking

Adding output schema validation

Implementing automated evaluation metrics
