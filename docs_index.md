# Advanced RAG System: Documentation Index

## Directory Structure

- `src/`: Source code
    - `ingestion/`: Data ingestion, chunking (hierarchical & semantic)
    - `retrieval/`: Hybrid retrieval pipeline (vector + BM25, multi-query/HyDE)
    - `database/`: Embedding service, vector store/DB
    - `generation/`: LLM answer generation, hallucination/relevance validation, reranker, memory
    - `utils/`: Helpers and utilities
    - `utils/cache.py`: Caching layer
- `config/`: All configuration and schema
- `data/`: Uploaded PDFs and chunked splits
- `main_api.py`: Entrypoint for FastAPI server
- `requirements.txt`: All dependencies
- `README.md`: Quick start
- `docs/`: All system architecture, flow diagrams, and developer usage docs (full documentation)

## Documentation Plan

1. **System Overview:** System architecture, major components, design principles (found in docs/system_overview.md)
2. **Ingestion & Chunking:** Full detail and examples for hierarchical and semantic chunking
3. **Vector Stores & Embedding:** Setup, configuration, and tuning for Milvus/FAISS/Chroma/OpenAI embeddings
4. **Retrieval Flow:** Query expansion, Hybrid search, BM25, HNSW, PQ, HyDE, scoring, re-ranking
5. **Response Generation Module:** LLM pipeline, compression, validation steps
6. **Conversational Memory:** Persistent vector history, follow-up transformation
7. **Caching:** Redis/in-memory design, TTL
8. **Advanced Optimizations:** Secondary retrieval, context expansion, RAFT finetuning

---

_Detail for each module plus ready-to-run examples in the module docs (see docs/ for extended developer documentation)_

