# Create a README.md file as initial documentation index
readme_content = '''# Advanced Conversational Retrieval-Augmented Generation (RAG) System

## Overview
This repository implements a production-grade RAG system designed for advanced conversational document Q&A over technical corpora, with support for multi-document ingestion, hierarchical + semantic chunking, hybrid retrieval (vector + keyword), conversational memory, validation of LLM hallucination, HyDE, and more.

## Major Features
- **Data Ingestion:** Upload and parse PDFs, extract logical/semantic chunks
- **Embeddings:** OpenAI Embedding API (text-embedding-3-large)
- **Retrieval:** Hybrid search using vector (Milvus/Chroma/FAISS) + BM25 keyword search
- **Multi-Query Generation, HyDE:** Expand queries for broader context, generate hypothetical answers for improved intent capture
- **Response Generation:** Contextual compression, LLM-based reranking, hallucination/relevance validation
- **Conversational Memory:** Vector store for dialogue history; LLM-powered query transformation
- **Caching:** Fast Redis/in-memory caching for repeated/overlapping queries
- **Long-context Handling:** Smart context expansion when retrieval is low-quality

## Quick Start
1. Clone this repo and install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
2. Set your OpenAI API key as `OPENAI_API_KEY` environment variable.
3. Start the API server:
   ```sh
   uvicorn main_api:app --reload
   ```
4. Upload PDFs and ask questions via `/upload_pdfs/` and `/ask/` endpoints.

---

For **full developer and architecture documentation**, see the `docs/` directory.
'''

with open(base_dir / "README.md", "w") as f:
    f.write(readme_content)
print("README created!")