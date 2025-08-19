# Create retrieval pipeline including multi-query generation, BM25, HyDE, Hybrid Search
retrieval_pipeline_content = '''"""Advanced Retrieval Pipeline for RAG System"""

import asyncio
from typing import List, Dict, Any, Tuple, Optional
import re
from rank_bm25 import BM25Okapi
from openai import AsyncOpenAI
from ..database.embeddings import embedding_service
from ..database.vector_store import get_vector_store
from ..ingestion.document_processor import DocumentChunk
from ..utils.helpers import logger, count_tokens
from ..config.config import config

class BM25Retriever:
    """BM25 keyword search over chunk contents"""
    
    def __init__(self, chunks: List[DocumentChunk]):
        self.corpus = [chunk.content for chunk in chunks]
        self.tokens = [doc.split(" ") for doc in self.corpus]
        self.bm25 = BM25Okapi(self.tokens)
        self.chunk_map = {i: chunk for i, chunk in enumerate(chunks)}
    
    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        # Tokenize query
        query_tokens = query.split(" ")
        scores = self.bm25.get_scores(query_tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: -scores[i])[:top_k]
        results = []
        for idx in top_indices:
            chunk = self.chunk_map[idx]
            results.append({
                **chunk.to_dict(),
                'score': float(scores[idx])
            })
        return results

class MultiQueryGenerator:
    """Generate multiple queries for a single input using LLM"""
    def __init__(self):
        self.client = AsyncOpenAI(api_key=config.openai_api_key)
        self.model = config.llm.model_name
    async def generate_queries(self, user_query: str, n: int = 3) -> List[str]:
        prompt = f"""
        Given the user's question, generate {n} alternative queries phrased from different perspectives and approaches that may retrieve different but relevant information.\n\nQuestion: {user_query}\nAlternative queries:
        """
        try:
            result = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=500
            )
            text = result.choices[0].message.content
            # Extract list
            lines = [line.strip("- ") for line in text.strip().split("\n") if line.strip()]
            # Remove lines like 'Alternative queries:'
            lines = [line for line in lines if len(line) > 0 and not line.lower().startswith("alternative queries")]
            if len(lines) >= n:
                return lines[:n]
            return [user_query] * n  # fallback: repeat original
        except Exception as e:
            logger.warning(f"MultiQuery failed: {e}")
            return [user_query] * n

class HyDEQueryGenerator:
    """Generate a hypothetical answer for HyDE and embed it"""
    def __init__(self):
        self.client = AsyncOpenAI(api_key=config.openai_api_key)
        self.model = config.llm.model_name
    async def generate_hypothetical(self, user_query: str) -> str:
        prompt = f"""
        Hypothetically answer this question as if you were an expert, only in a few sentences:\n\n{user_query}\n\n--\nHypothetical answer:
        """
        try:
            result = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=256
            )
            text = result.choices[0].message.content.strip()
            return text
        except Exception as e:
            logger.warning(f"HyDE generation failed: {e}")
            return user_query

class HybridRetriever:
    """Hybrid Search: Vector + BM25 keyword search"""
    def __init__(self, chunks: List[DocumentChunk]):
        self.chunks = chunks
        self.vector_store = get_vector_store()
        self.bm25 = BM25Retriever(self.chunks)
    async def hybrid_search(self, query: str, hyde_enabled: bool=True, top_k: int=10, alpha: float=0.5) -> List[Dict[str, Any]]:
        # Multi-Query Expansion
        mq = MultiQueryGenerator()
        queries = await mq.generate_queries(query, n=config.retrieval.num_queries) if config.retrieval.use_multi_query else [query]
        
        # HyDE embedding
        query_embeddings = []
        if hyde_enabled and config.retrieval.use_hyde:
            hyde = HyDEQueryGenerator()
            hypo_query = await hyde.generate_hypothetical(query)
            emb = await embedding_service.embed_single(hypo_query)
            query_embeddings.append(emb)
        else:
            for q in queries:
                emb = await embedding_service.embed_single(q)
                query_embeddings.append(emb)
        
        # Vector search
        vector_results = []
        for emb in query_embeddings:
            chunks = await self.vector_store.search(emb, top_k=top_k)
            vector_results.extend(chunks)
        # BM25 keyword search
        bm25_results = []
        for q in queries:
            chunks = self.bm25.search(q, top_k=top_k)
            bm25_results.extend(chunks)
        
        # Deduplicate by ID, combine scores
        combined = {}
        for doc in vector_results:
            key = doc["id"]
            combined[key] = {**doc, 'vector_score': doc['score'], 'bm25_score': 0}
        for doc in bm25_results:
            key = doc["chunk_id"] if 'chunk_id' in doc else doc["id"]
            # Update or insert bm25_score
            if key in combined:
                combined[key]['bm25_score'] = doc['score']
            else:
                combined[key] = {**doc, 'vector_score': 0, 'bm25_score': doc['score']}
        # Combine scores by weighted alpha
        for key, val in combined.items():
            val['hybrid_score'] = alpha * val['vector_score'] + (1 - alpha) * val['bm25_score']
        # Sort by hybrid score
        sorted_docs = sorted(combined.values(), key=lambda d: -d['hybrid_score'])
        return sorted_docs[:top_k]

    async def retrieve(self, query: str, hyde_enabled: bool=True, top_k: int=10):
        """Hybrid retrieval interface"""
        return await self.hybrid_search(query, hyde_enabled=hyde_enabled, top_k=top_k, alpha=config.retrieval.hybrid_alpha)
'''

with open(base_dir / "src" / "retrieval" / "retrieval_pipeline.py", "w") as f:
    f.write(retrieval_pipeline_content)
    
print("Retrieval pipeline module created!")