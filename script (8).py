# Create response generation and hallucination validation
response_gen_content = '''"""Response Generation and Validation for RAG system"""

import asyncio
from typing import List, Dict, Any, Optional
from openai import AsyncOpenAI
from ..config.config import config
from ..utils.helpers import logger

class Reranker:
    """Re-rank the results using LLM (or model scores)"""
    def __init__(self):
        self.client = AsyncOpenAI(api_key=config.openai_api_key)
        self.model = config.llm.model_name
    async def rerank(self, query: str, docs: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        if not docs:
            return []
        chunks_prompt = "\n\n".join([d["content"][:1000] for d in docs])
        prompt = f"Given the following retrieved document chunks, and this user query: {query}\n\nRank the chunks by how likely they are to directly answer the question. Return a JSON list of chunk indices in decreasing relevance.\n\nChunks:\n{chunks_prompt}\n---\nReturn JSON list of indices only:"
        try:
            resp = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=256
            )
            import json as pyjson
            idxs = pyjson.loads(resp.choices[0].message.content.strip().replace('```json','').replace('```',''))
            ordered_docs = [docs[i] for i in idxs[:top_k] if i < len(docs)]
            return ordered_docs
        except Exception as e:
            logger.warning(f"Reranker failed: {e}")
            return docs[:top_k]

class ContextCompressor:
    """Extract only most relevant sentences from top chunks"""
    def __init__(self):
        self.client = AsyncOpenAI(api_key=config.openai_api_key)
        self.model = config.llm.model_name
    async def compress(self, query: str, chunks: List[Dict[str, Any]]) -> str:
        prompt = f"""
        Given the following chunks, extract only the sentences that are directly relevant to the user question:\n\nQuestion: {query}\n\nChunks:\n{'\n---\n'.join(chunk['content'] for chunk in chunks)}
        """
        try:
            resp = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=1024
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            logger.warning(f"Context compression failed: {e}")
            # fallback: concatenate first N sentences
            results = []
            for chunk in chunks:
                results.extend(chunk["content"].split(". ")[:5])
            return ". ".join(results)

class AnswerValidator:
    """Validate if generated answer is supported by context and relevant to original question"""
    def __init__(self):
        self.client = AsyncOpenAI(api_key=config.openai_api_key)
        self.model = config.llm.model_name
    async def validate(self, query: str, context: str, answer: str) -> Tuple[bool, str]:
        prompt = f"""
        Based only on the given context, is the following answer correct AND directly supported by the context?\n\nContext:\n{context}\n\nAnswer:\n{answer}\n\nRespond strictly with either YES or NO and a reason."
        try:
            resp = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=256
            )
            txt = resp.choices[0].message.content
            if txt.strip().upper().startswith("YES"):
                return True, txt
            else:
                return False, txt
        except Exception as e:
            logger.warning(f"Answer validation failed: {e}")
            return False, "Validation failed"

class AnswerGenerator:
    """Main answer generator with context/hallucination checks"""
    def __init__(self):
        self.client = AsyncOpenAI(api_key=config.openai_api_key)
        self.model = config.llm.model_name
        self.compressor = ContextCompressor()
        self.reranker = Reranker()
        self.validator = AnswerValidator()
    async def generate(self, user_query: str, retrieved_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Contextual compression
        compressed_context = await self.compressor.compress(user_query, retrieved_chunks)
        # Reranking
        reranked = await self.reranker.rerank(user_query, retrieved_chunks, top_k=config.retrieval.rerank_top_k)
        # Prepare context for LLM
        unified_context = f"\n\n".join(chunk['content'] for chunk in reranked)
        # Generate answer
        prompt = f"Answer the following question using only the provided context.\n\nQuestion: {user_query}\n\nContext:\n{unified_context}\n\nAnswer:\n"
        answer_resp = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=config.llm.temperature,
            max_tokens=config.llm.max_tokens
        )
        answer = answer_resp.choices[0].message.content.strip()
        # Validate hallucination
        is_supported, reason = await self.validator.validate(user_query, unified_context, answer)
        if not is_supported:
            answer = "[RAG System]: The answer cannot be supported by the provided documents/context. Please consult the documents directly for your query."
        return {
            "answer": answer,
            "supported": is_supported,
            "validation": reason,
            "chunks_used": reranked
        }
'''

with open(base_dir / "src" / "generation" / "response_generator.py", "w") as f:
    f.write(response_gen_content)
    
print("Response generator module created!")