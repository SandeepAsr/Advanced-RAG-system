# Create main pipeline and API entrypoint
main_content = '''"""Main API and Pipeline Entrypoint for Advanced Conversational RAG"""
import os
import asyncio
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException
from pathlib import Path
import shutil
from .src.ingestion.document_processor import DocumentIngestion
from .src.database.embeddings import embedding_service
from .src.database.vector_store import get_vector_store
from .src.retrieval.retrieval_pipeline import HybridRetriever
from .src.generation.response_generator import AnswerGenerator
from .src.generation.conversation_memory import ConversationMemoryStore, QueryTransformer
from .src.utils.cache import CacheLayer
from .config.config import config

app = FastAPI()
cache = CacheLayer()
doc_ingestion = DocumentIngestion()
mem_store = ConversationMemoryStore()
query_transformer = QueryTransformer()
ans_gen = AnswerGenerator()

@app.post("/upload_pdfs/")
async def upload_pdfs(files: List[UploadFile] = File(...)):
    pdf_dir = Path(config.pdf_path)
    pdf_dir.mkdir(exist_ok=True)
    file_paths = []
    for file in files:
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Supports only PDF uploads.")
        dest = pdf_dir / file.filename
        with open(dest, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        file_paths.append(dest)
    chunks = await doc_ingestion.process_documents(file_paths)
    # Embed chunks
    texts = [chunk.content for chunk in chunks]
    embeddings = await embedding_service.embed_texts(texts)
    for chunk, emb in zip(chunks, embeddings):
        chunk.embedding = emb
    # Persist to vector store
    vs = get_vector_store()
    await vs.add_documents(chunks)
    return {"num_chunks": len(chunks), "status": "success"}

@app.post("/ask/")
async def ask_question(question: str, conversation_history: Optional[List[dict]] = None):
    # Cache lookup
    cache_key = question
    ans = cache.get(cache_key)
    if ans:
        return {"answer": ans["answer"], "cached": True}
    # Transform query if follow-up
    if conversation_history:
        question = await query_transformer.transform(question, conversation_history)
        # Retrieve conversation memory
        relevant_snippets = await mem_store.search(question)
    else:
        relevant_snippets = []
    # Retrieval
    vs = get_vector_store()
    # Optionally: expose a method to get all chunks for BM25, but for prototype-load all
    retrieved_chunks = await HybridRetriever(vs.collection).retrieve(question)
    # Use top memory-summary chunks too if conversation context
    if relevant_snippets:
        for mem in relevant_snippets:
            retrieved_chunks.append({"content": mem["text"], "metadata": mem["metadata"]})
    # Answer generation and validation
    result = await ans_gen.generate(question, retrieved_chunks)
    if result["supported"]:
        cache.set(cache_key, result)
    return result
'''
with open(base_dir / "main_api.py", "w") as f:
    f.write(main_content)
print("Main API created!")