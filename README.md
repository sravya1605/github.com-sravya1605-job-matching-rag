# Job-Matching RAG Pipeline

A small Retrieval-Augmented Generation (RAG) retrieval pipeline that matches a candidate's description to the most relevant job descriptions.

## How it works
1. Job descriptions are loaded as LangChain `Document` objects.
2. Each document is embedded with `sentence-transformers/all-MiniLM-L6-v2` (Hugging Face).
3. Embeddings are stored in a ChromaDB vector store.
4. A retriever returns the top-3 closest job descriptions for a candidate query.
5. An evaluation set of test queries measures retrieval quality (hit@3).

## Tech stack
Python, LangChain, ChromaDB, Hugging Face sentence-transformers

## Run it
```
python -m venv venv
venv\Scripts\activate        # Windows  (Mac/Linux: source venv/bin/activate)
pip install -r requirements.txt
python rag_job_matcher.py
```
The first run downloads the embedding model (about 90 MB).

## Result
Retrieval hit@3: **8/8 (100%)** on 8 test queries over 8 job descriptions.

This is a small prototype dataset. Next steps: expand to 30-50 real job descriptions with harder queries, add an LLM generation step on top of the retrieved context, and expose a FastAPI `/match` endpoint.
